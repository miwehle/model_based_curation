from __future__ import annotations

import csv
import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from math import ceil
from pathlib import Path
from typing import Any, Protocol

import yaml
from lab_infrastructure.compute_metrics import get_gpu_util

Example = dict[str, Any]
_CSV_FIELDS = ("id", "keep", "loss", "src", "tgt")
_LOG = logging.getLogger(__name__)


class _FlowSeq(list):
    pass


class BatchLossScorer(Protocol):
    def score_batch(self, examples: list[Mapping[str, Any]]) -> list[float]: ...


def _load_dataset(path: str | Path):
    from datasets import load_from_disk

    return load_from_disk(str(path))


def _bucket_index(loss: float, upper_bounds: Sequence[float]) -> int:
    for index, upper_bound in enumerate(upper_bounds):
        if loss < upper_bound:
            return index
    return len(upper_bounds)


def _format_bound(bound: float) -> str:
    if bound.is_integer():
        return str(int(bound))
    return f"{bound:g}".replace("-", "m").replace(".", "_")


def _bucket_filename(bucket_index: int, upper_bounds: Sequence[float]) -> str:
    del upper_bounds
    return f"{bucket_index + 1}.csv"


yaml.SafeDumper.add_representer(
    _FlowSeq, lambda dumper, data: dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)
)


class Splitter:
    def __init__(
        self,
        upper_bounds: Sequence[float],
        output_dir: str | Path,
        *,
        decode_src_text: Callable[[list[int]], str],
        decode_tgt_text: Callable[[list[int]], str],
        csv_delimiter: str = ",",
        loss_decimal_separator: str = ".",
        decode_from_loss: float | None = None,
        decode_at_least: int = 10,
        log_every_batches: int = 1,
    ) -> None:
        self._bounds = [float(bound) for bound in upper_bounds]
        self._output_dir = Path(output_dir)
        self._decode_src_text = decode_src_text
        self._decode_tgt_text = decode_tgt_text
        self._csv_delimiter = csv_delimiter
        self._loss_decimal_separator = loss_decimal_separator
        self._decode_from_loss = decode_from_loss
        self._decode_at_least = decode_at_least
        self._log_every_batches = log_every_batches

    def split_dataset(
        self, dataset_path: str | Path, scorer: BatchLossScorer, *, batch_size: int
    ) -> list[Path]:
        _LOG.info("Opening dataset from %s", dataset_path)
        dataset = _load_dataset(dataset_path)
        total_examples = len(dataset)
        total_batches = ceil(total_examples / batch_size) if total_examples else 0
        _LOG.info("Loaded dataset with %s examples; writing buckets to %s", total_examples, self._output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = self._bucket_paths()
        batch: list[Example] = []
        processed_examples = 0
        batch_index = 0

        with ExitStack() as stack:
            files = [
                stack.enter_context(path.open("w", encoding="utf-8", newline="")) for path in output_paths
            ]
            writers = [
                csv.DictWriter(file, fieldnames=_CSV_FIELDS, delimiter=self._csv_delimiter) for file in files
            ]
            for writer in writers:
                writer.writeheader()
            examples_per_bucket = [0] * len(writers)
            decoded_per_bucket = [0] * len(writers)
            max_losses = [None] * len(writers)
            for row in dataset:
                batch.append(dict(row))
                if len(batch) == batch_size:
                    batch_index += 1
                    if batch_index % self._log_every_batches == 0:
                        _LOG.info(
                            self._batch_message(batch_index, total_batches, len(batch), processed_examples)
                        )
                    self._flush_batch(
                        batch, scorer, writers, examples_per_bucket, decoded_per_bucket, max_losses
                    )
                    processed_examples += len(batch)
                    batch.clear()
            if batch:
                batch_index += 1
                _LOG.info(self._batch_message(batch_index, total_batches, len(batch), processed_examples))
            self._flush_batch(batch, scorer, writers, examples_per_bucket, decoded_per_bucket, max_losses)
            processed_examples += len(batch)
        self._write_bucket_stats(examples_per_bucket, max_losses[-1])

        _LOG.info(
            "Finished split with %s processed examples into %s buckets", processed_examples, len(output_paths)
        )
        return output_paths

    def _batch_message(
        self, batch_index: int, total_batches: int, batch_len: int, processed_examples: int
    ) -> str:
        batch_label = (
            f"Scoring batch {batch_index}/{total_batches}"
            if total_batches
            else f"Scoring batch {batch_index}"
        )
        start_index = processed_examples + 1
        end_index = processed_examples + batch_len
        gpu_util = get_gpu_util()
        gpu_text = f"{gpu_util}%" if gpu_util is not None else "-"
        return f"{batch_label} ({batch_len} examples; rows {start_index}-{end_index}; gpu={gpu_text})"

    def _bucket_paths(self) -> list[Path]:
        return [
            self._output_dir / _bucket_filename(bucket_index, self._bounds)
            for bucket_index in range(len(self._bounds) + 1)
        ]

    def _flush_batch(
        self,
        batch: list[Example],
        scorer: BatchLossScorer,
        writers: Sequence[csv.DictWriter[str]],
        examples_per_bucket: list[int],
        decoded_per_bucket: list[int],
        max_losses: list[float | None],
    ) -> None:
        if not batch:
            return
        losses = scorer.score_batch(batch)
        if len(losses) != len(batch):
            raise ValueError("score_batch must return one loss per example.")
        for example, loss in zip(batch, losses, strict=True):
            writer_index = _bucket_index(float(loss), self._bounds)
            examples_per_bucket[writer_index] += 1
            max_losses[writer_index] = (
                loss if max_losses[writer_index] is None else max(max_losses[writer_index], loss)
            )
            decode_text = (
                decoded_per_bucket[writer_index] < self._decode_at_least
                or self._decode_from_loss is None
                or loss >= self._decode_from_loss
            )
            row = self._csv_row(example, float(loss), decode_text=decode_text)
            writers[writer_index].writerow(row)
            decoded_per_bucket[writer_index] += int(decode_text)

    def _write_bucket_stats(
        self, examples_per_bucket: Sequence[int], max_loss_in_last_bucket: float | None
    ) -> None:
        buckets = [
            _FlowSeq(
                [
                    i + 1,
                    0.0 if i == 0 else self._bounds[i - 1],
                    None if i == len(self._bounds) else self._bounds[i],
                    count,
                ]
            )
            for i, count in enumerate(examples_per_bucket)
        ]
        if buckets and buckets[-1][2] is None:
            if buckets[-1][3] == 0:
                buckets.pop()
            else:
                buckets[-1][2] = max_loss_in_last_bucket
        stats = {"buckets": buckets}
        with (self._output_dir / "bucket_stats.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(stats, handle, sort_keys=False)

    def _csv_row(self, example: Mapping[str, Any], loss: float, *, decode_text: bool) -> dict[str, str | int]:
        if "src_ids" not in example or "tgt_ids" not in example:
            raise ValueError("Examples must define 'src_ids' and 'tgt_ids' for CSV bucket output.")
        src_ids = [int(token_id) for token_id in example["src_ids"]]
        tgt_ids = [int(token_id) for token_id in example["tgt_ids"]]
        return {
            "id": int(example["id"]),
            "keep": "",
            "loss": self._format_loss(loss),
            "src": self._decode_src_text(src_ids) if decode_text else "",
            "tgt": self._decode_tgt_text(tgt_ids) if decode_text else "",
        }

    def _format_loss(self, loss: float) -> str:
        formatted = str(loss)
        if self._loss_decimal_separator == ".":
            return formatted
        return formatted.replace(".", ",")

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
import logging
from math import ceil
from pathlib import Path
import subprocess
from typing import Any, Protocol

Example = dict[str, Any]
_CSV_FIELDS = ("id", "keep", "loss", "src", "tgt")
_NOT_DECODED = "(not decoded)"
_LOG = logging.getLogger(__name__)


class BatchLossScorer(Protocol):
    def score_batch(self, examples: list[Mapping[str, Any]]) -> list[float]: ...


def _load_dataset(path: str | Path):
    from datasets import load_from_disk

    return load_from_disk(str(path))


def _get_gpu_util() -> int | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    try:
        return int(out.strip())
    except ValueError:
        return None


def _validate_upper_bounds(upper_bounds: Sequence[float]) -> list[float]:
    bounds = [float(bound) for bound in upper_bounds]
    if any(bound < 0 for bound in bounds):
        raise ValueError("upper_bounds must be non-negative.")
    if any(left >= right for left, right in zip(bounds, bounds[1:])):
        raise ValueError("upper_bounds must be strictly increasing.")
    return bounds


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
    bucket_id = bucket_index + 1
    lower = 0.0 if bucket_index == 0 else upper_bounds[bucket_index - 1]
    upper = (
        "inf"
        if bucket_index == len(upper_bounds)
        else _format_bound(upper_bounds[bucket_index])
    )
    return f"{bucket_id:02d}_loss_{_format_bound(lower)}_to_{upper}.csv"


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
        log_every_batches: int = 1,
    ) -> None:
        self._bounds = _validate_upper_bounds(upper_bounds)
        self._output_dir = Path(output_dir)
        self._decode_src_text = decode_src_text
        self._decode_tgt_text = decode_tgt_text
        self._csv_delimiter = csv_delimiter
        self._loss_decimal_separator = loss_decimal_separator
        self._decode_from_loss = decode_from_loss
        self._log_every_batches = log_every_batches

    def split_dataset(
        self,
        dataset_path: str | Path,
        scorer: BatchLossScorer,
        *,
        batch_size: int,
    ) -> list[Path]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        _LOG.info("Opening dataset from %s", dataset_path)
        dataset = _load_dataset(dataset_path)
        total_examples = len(dataset)
        total_batches = ceil(total_examples / batch_size) if total_examples else 0
        _LOG.info(
            "Loaded dataset with %s examples; writing buckets to %s",
            total_examples,
            self._output_dir,
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = self._bucket_paths()
        batch: list[Example] = []
        processed_examples = 0
        batch_index = 0

        with ExitStack() as stack:
            files = [
                stack.enter_context(path.open("w", encoding="utf-8", newline=""))
                for path in output_paths
            ]
            writers = [
                csv.DictWriter(file, fieldnames=_CSV_FIELDS, delimiter=self._csv_delimiter)
                for file in files
            ]
            for writer in writers:
                writer.writeheader()
            for row in dataset:
                batch.append(dict(row))
                if len(batch) == batch_size:
                    batch_index += 1
                    if batch_index % self._log_every_batches == 0:
                        _LOG.info(
                            self._batch_message(
                                batch_index, total_batches, len(batch), processed_examples
                            )
                        )
                    self._flush_batch(batch, scorer, writers)
                    processed_examples += len(batch)
                    batch.clear()
            if batch:
                batch_index += 1
                _LOG.info(
                    self._batch_message(
                        batch_index, total_batches, len(batch), processed_examples
                    )
                )
            self._flush_batch(batch, scorer, writers)
            processed_examples += len(batch)

        _LOG.info(
            "Finished split with %s processed examples into %s buckets",
            processed_examples,
            len(output_paths),
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
        gpu_util = _get_gpu_util()
        gpu_text = f"{gpu_util}%" if gpu_util is not None else "-"
        return (
            f"{batch_label} ({batch_len} examples; rows {start_index}-{end_index}; "
            f"gpu={gpu_text})"
        )

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
    ) -> None:
        if not batch:
            return
        losses = scorer.score_batch(batch)
        if len(losses) != len(batch):
            raise ValueError("score_batch must return one loss per example.")
        for example, loss in zip(batch, losses, strict=True):
            writer_index = _bucket_index(float(loss), self._bounds)
            row = self._csv_row(example, float(loss))
            writers[writer_index].writerow(row)

    def _csv_row(
        self, example: Mapping[str, Any], loss: float
    ) -> dict[str, str | int]:
        if "src_ids" not in example or "tgt_ids" not in example:
            raise ValueError(
                "Examples must define 'src_ids' and 'tgt_ids' for CSV bucket output."
            )
        src_ids = [int(token_id) for token_id in example["src_ids"]]
        tgt_ids = [int(token_id) for token_id in example["tgt_ids"]]
        decode_text = self._decode_from_loss is None or loss >= self._decode_from_loss
        return {
            "id": int(example["id"]),
            "keep": "",
            "loss": self._format_loss(loss),
            "src": self._decode_src_text(src_ids) if decode_text else _NOT_DECODED,
            "tgt": self._decode_tgt_text(tgt_ids) if decode_text else _NOT_DECODED,
        }

    def _format_loss(self, loss: float) -> str:
        formatted = str(loss)
        if self._loss_decimal_separator == ".":
            return formatted
        return formatted.replace(".", ",")

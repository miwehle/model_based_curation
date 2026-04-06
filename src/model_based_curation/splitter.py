from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
import logging
from math import ceil
from pathlib import Path
from typing import Any, Protocol

Example = dict[str, Any]
_CSV_FIELDS = ("id", "loss", "src", "tgt")
_LOG = logging.getLogger(__name__)


class BatchLossScorer(Protocol):
    def score_batch(self, examples: list[Mapping[str, Any]]) -> list[float]: ...


def _load_dataset(path: str | Path):
    from datasets import load_from_disk

    return load_from_disk(str(path))


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
        sort_by_loss_desc: bool = False,
    ) -> None:
        self._bounds = _validate_upper_bounds(upper_bounds)
        self._output_dir = Path(output_dir)
        self._decode_src_text = decode_src_text
        self._decode_tgt_text = decode_tgt_text
        self._sort_by_loss_desc = sort_by_loss_desc

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
            writers = [csv.DictWriter(file, fieldnames=_CSV_FIELDS) for file in files]
            for writer in writers:
                writer.writeheader()
            for row in dataset:
                batch.append(dict(row))
                if len(batch) == batch_size:
                    batch_index += 1
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

        if self._sort_by_loss_desc:
            _LOG.info("Sorting %s bucket files by loss descending", len(output_paths))
            for path in output_paths:
                self._sort_bucket_file(path)
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
        return f"{batch_label} ({batch_len} examples; rows {start_index}-{end_index})"

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
            writers[writer_index].writerow(self._csv_row(example, float(loss)))

    def _csv_row(
        self, example: Mapping[str, Any], loss: float
    ) -> dict[str, str | int | float]:
        if "src_ids" not in example or "tgt_ids" not in example:
            raise ValueError(
                "Examples must define 'src_ids' and 'tgt_ids' for CSV bucket output."
            )
        src_ids = [int(token_id) for token_id in example["src_ids"]]
        tgt_ids = [int(token_id) for token_id in example["tgt_ids"]]
        return {
            "id": int(example["id"]),
            "loss": loss,
            "src": self._decode_src_text(src_ids),
            "tgt": self._decode_tgt_text(tgt_ids),
        }

    def _sort_bucket_file(self, path: Path) -> None:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows.sort(key=lambda row: float(row["loss"]), reverse=True)
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        tmp_path.replace(path)

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

from datasets import Dataset

from model_based_curation.splitter import Splitter

_TMP_DIR = Path(__file__).resolve().parents[1] / ".local_tmp"


def _temp_dir(prefix: str) -> Path:
    path = _TMP_DIR / f"{prefix}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class _FakeBatchScorer:
    def __init__(self) -> None:
        self.seen_batches: list[list[int]] = []

    def score_batch(self, examples: list[Mapping[str, object]]) -> list[float]:
        ids = [int(example["id"]) for example in examples]
        self.seen_batches.append(ids)
        mapping = {1: 0.2, 2: 0.9, 3: 2.3}
        return [mapping[ex_id] for ex_id in ids]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _decode_text(token_ids: list[int]) -> str:
    return "|".join(str(token_id) for token_id in token_ids)


def test_splitter_writes_csv_buckets_named_by_bucket_interval():
    dataset_dir = _temp_dir("mapped_dataset_splitter")
    output_dir = _temp_dir("bucket_output_splitter")
    ds = Dataset.from_list(
        [
            {"id": 1, "src_ids": [11, 12], "tgt_ids": [21, 22]},
            {"id": 2, "src_ids": [13], "tgt_ids": [23]},
            {"id": 3, "src_ids": [14, 15, 16], "tgt_ids": [24]},
        ]
    )
    ds.save_to_disk(str(dataset_dir))
    scorer = _FakeBatchScorer()

    output_paths = Splitter(
        [0.5, 1.5], output_dir, decode_text=_decode_text
    ).split_dataset(dataset_dir, scorer, batch_size=2)

    assert scorer.seen_batches == [[1, 2], [3]]
    assert [path.name for path in output_paths] == [
        "01_loss_0_to_0_5.csv",
        "02_loss_0_5_to_1_5.csv",
        "03_loss_1_5_to_inf.csv",
    ]

    bucket_1 = _read_rows(output_paths[0])
    bucket_2 = _read_rows(output_paths[1])
    bucket_3 = _read_rows(output_paths[2])

    assert bucket_1 == [{"id": "1", "loss": "0.2", "src": "11|12", "tgt": "21|22"}]
    assert bucket_2 == [{"id": "2", "loss": "0.9", "src": "13", "tgt": "23"}]
    assert bucket_3 == [{"id": "3", "loss": "2.3", "src": "14|15|16", "tgt": "24"}]


def test_splitter_sorts_rows_within_each_bucket_by_loss_desc():
    dataset_dir = _temp_dir("mapped_dataset_sorted")
    output_dir = _temp_dir("bucket_output_sorted")
    ds = Dataset.from_list(
        [
            {"id": 1, "src_ids": [11], "tgt_ids": [21]},
            {"id": 2, "src_ids": [12], "tgt_ids": [22]},
            {"id": 3, "src_ids": [13], "tgt_ids": [23]},
        ]
    )
    ds.save_to_disk(str(dataset_dir))

    class _SortingScorer:
        def score_batch(self, examples: list[Mapping[str, object]]) -> list[float]:
            mapping = {1: 0.9, 2: 0.2, 3: 0.7}
            return [mapping[int(example["id"])] for example in examples]

    output_paths = Splitter(
        [1.5], output_dir, decode_text=_decode_text, sort_by_loss_desc=True
    ).split_dataset(dataset_dir, _SortingScorer(), batch_size=2)

    bucket_rows = _read_rows(output_paths[0])
    assert [row["id"] for row in bucket_rows] == ["1", "3", "2"]
    assert [row["loss"] for row in bucket_rows] == ["0.9", "0.7", "0.2"]

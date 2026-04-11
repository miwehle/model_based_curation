from __future__ import annotations

import csv
import logging
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

from datasets import Dataset
import yaml

from model_based_curation.split.splitter import Splitter

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


def _read_rows(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _decode_text(token_ids: list[int]) -> str:
    return "|".join(str(token_id) for token_id in token_ids)


def _read_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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
        [0.5, 1.5], output_dir, decode_src_text=_decode_text, decode_tgt_text=_decode_text
    ).split_dataset(dataset_dir, scorer, batch_size=2)

    assert scorer.seen_batches == [[1, 2], [3]]
    assert [path.name for path in output_paths] == ["1.csv", "2.csv", "3.csv"]

    bucket_1 = _read_rows(output_paths[0])
    bucket_2 = _read_rows(output_paths[1])
    bucket_3 = _read_rows(output_paths[2])

    assert bucket_1 == [{"id": "1", "keep": "", "loss": "0.2", "src": "11|12", "tgt": "21|22"}]
    assert bucket_2 == [{"id": "2", "keep": "", "loss": "0.9", "src": "13", "tgt": "23"}]
    assert bucket_3 == [{"id": "3", "keep": "", "loss": "2.3", "src": "14|15|16", "tgt": "24"}]
    assert _read_yaml(output_dir / "bucket_stats.yaml") == {
        "buckets": [[1, 0.0, 0.5, 1], [2, 0.5, 1.5, 1], [3, 1.5, 2.3, 1]]
    }
    assert _read_text(output_dir / "bucket_stats.yaml") == (
        "buckets:\n" "- [1, 0.0, 0.5, 1]\n" "- [2, 0.5, 1.5, 1]\n" "- [3, 1.5, 2.3, 1]\n"
    )


def test_splitter_can_decode_src_and_tgt_with_different_rules():
    dataset_dir = _temp_dir("mapped_dataset_decoder_rules")
    output_dir = _temp_dir("bucket_output_decoder_rules")
    ds = Dataset.from_list([{"id": 1, "src_ids": [11, 12], "tgt_ids": [58101, 21, 0]}])
    ds.save_to_disk(str(dataset_dir))

    class _SingleExampleScorer:
        def score_batch(self, examples: list[Mapping[str, object]]) -> list[float]:
            del examples
            return [0.2]

    output_paths = Splitter(
        [0.5],
        output_dir,
        decode_src_text=_decode_text,
        decode_tgt_text=lambda token_ids: _decode_text(token_ids[1:]),
    ).split_dataset(dataset_dir, _SingleExampleScorer(), batch_size=1)

    bucket_rows = _read_rows(output_paths[0])
    assert bucket_rows == [{"id": "1", "keep": "", "loss": "0.2", "src": "11|12", "tgt": "21|0"}]


def test_splitter_logs_progress(caplog):
    dataset_dir = _temp_dir("mapped_dataset_logged")
    output_dir = _temp_dir("bucket_output_logged")
    ds = Dataset.from_list(
        [{"id": 1, "src_ids": [11], "tgt_ids": [21]}, {"id": 2, "src_ids": [12], "tgt_ids": [22]}]
    )
    ds.save_to_disk(str(dataset_dir))

    class _LoggedScorer:
        def score_batch(self, examples: list[Mapping[str, object]]) -> list[float]:
            del examples
            return [0.1, 0.2]

    with caplog.at_level(logging.INFO):
        Splitter([1.5], output_dir, decode_src_text=_decode_text, decode_tgt_text=_decode_text).split_dataset(
            dataset_dir, _LoggedScorer(), batch_size=2
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any("Opening dataset from" in message for message in messages)
    assert any("Scoring batch 1/1" in message for message in messages)
    assert any("gpu=" in message for message in messages)


def test_splitter_can_write_semicolon_csv_with_german_decimal_separator():
    dataset_dir = _temp_dir("mapped_dataset_german_csv")
    output_dir = _temp_dir("bucket_output_german_csv")
    ds = Dataset.from_list(
        [{"id": 1, "src_ids": [11], "tgt_ids": [21]}, {"id": 2, "src_ids": [12], "tgt_ids": [22]}]
    )
    ds.save_to_disk(str(dataset_dir))

    class _GermanCsvScorer:
        def score_batch(self, examples: list[Mapping[str, object]]) -> list[float]:
            mapping = {1: 0.9, 2: 0.2}
            return [mapping[int(example["id"])] for example in examples]

    output_paths = Splitter(
        [1.5],
        output_dir,
        decode_src_text=_decode_text,
        decode_tgt_text=_decode_text,
        csv_delimiter=";",
        loss_decimal_separator=",",
    ).split_dataset(dataset_dir, _GermanCsvScorer(), batch_size=2)

    bucket_rows = _read_rows(output_paths[0], delimiter=";")
    assert [row["id"] for row in bucket_rows] == ["1", "2"]
    assert [row["loss"] for row in bucket_rows] == ["0,9", "0,2"]
    assert _read_yaml(output_dir / "bucket_stats.yaml") == {"buckets": [[1, 0.0, 1.5, 2]]}


def test_splitter_decodes_at_least_n_examples_per_bucket_before_threshold():
    dataset_dir = _temp_dir("mapped_dataset_decode_rules")
    output_dir = _temp_dir("bucket_output_decode_rules")
    ds = Dataset.from_list(
        [
            {"id": 1, "src_ids": [11], "tgt_ids": [21]},
            {"id": 2, "src_ids": [12], "tgt_ids": [22]},
            {"id": 3, "src_ids": [13], "tgt_ids": [23]},
            {"id": 4, "src_ids": [14], "tgt_ids": [24]},
        ]
    )
    ds.save_to_disk(str(dataset_dir))

    class _ThresholdScorer:
        def score_batch(self, examples: list[Mapping[str, object]]) -> list[float]:
            mapping = {1: 0.1, 2: 0.2, 3: 1.7, 4: 0.3}
            return [mapping[int(example["id"])] for example in examples]

    output_paths = Splitter(
        [1.5],
        output_dir,
        decode_src_text=_decode_text,
        decode_tgt_text=_decode_text,
        decode_from_loss=1.0,
        decode_at_least=2,
    ).split_dataset(dataset_dir, _ThresholdScorer(), batch_size=4)

    assert _read_rows(output_paths[0]) == [
        {"id": "1", "keep": "", "loss": "0.1", "src": "11", "tgt": "21"},
        {"id": "2", "keep": "", "loss": "0.2", "src": "12", "tgt": "22"},
        {"id": "4", "keep": "", "loss": "0.3", "src": "", "tgt": ""},
    ]
    assert _read_rows(output_paths[1]) == [{"id": "3", "keep": "", "loss": "1.7", "src": "13", "tgt": "23"}]

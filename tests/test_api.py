from __future__ import annotations

import csv
import logging
import sys
import types
from pathlib import Path
from uuid import uuid4

import pytest
from datasets import Dataset, load_from_disk

from model_based_curation import FilterRunConfig, SplitRunConfig, filter, split

_TMP_DIR = Path(__file__).resolve().parents[1] / ".local_tmp"


def _read_rows(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _temp_dir(prefix: str) -> Path:
    path = _TMP_DIR / f"{prefix}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_dataset(path: Path, rows: list[dict[str, object]]) -> None:
    Dataset.from_list(rows).save_to_disk(str(path))


class _SplitPaths:
    def __init__(self, root_dir: Path) -> None:
        self.drive_artifacts = root_dir / "drive_artifacts"
        self.local_artifacts = root_dir / "local_artifacts"
        self.dataset_dir = self.drive_artifacts / "datasets" / "europarl" / "d1"
        self.local_dataset_dir = self.local_artifacts / "datasets" / "europarl" / "d1"
        self.output_dir = self.local_dataset_dir / "loss_buckets" / "r1"
        self.drive_dir = self.dataset_dir / "loss_buckets" / "r1"


class _FilterPaths:
    def __init__(self, root_dir: Path) -> None:
        self.drive_artifacts = root_dir / "drive_artifacts"
        self.local_artifacts = root_dir / "local_artifacts"
        self.dataset_dir = self.drive_artifacts / "datasets" / "europarl" / "d1"
        self.local_dataset_dir = self.local_artifacts / "datasets" / "europarl" / "d1"
        self.bucket_dir = self.dataset_dir / "loss_buckets" / "r1"
        self.local_bucket_dir = self.local_dataset_dir / "loss_buckets" / "r1"
        self.drive_bucket_dir = self.bucket_dir
        self.output_dir = self.local_artifacts / "datasets" / "europarl" / "d2"
        self.drive_dir = self.drive_artifacts / "datasets" / "europarl" / "d2"


def _value_property(value):
    return property(lambda self: value)


def _patch_attr(monkeypatch, cls, name: str, value) -> None:
    monkeypatch.setattr(cls, name, _value_property(value))


class _FakeTokenizer:
    def decode(self, token_ids: list[int]) -> str:
        return "|".join(str(token_id) for token_id in token_ids)


class _FakeModel:
    src_pad_idx = 0
    tgt_pad_idx = 0


class _FakeTranslator:
    device = "cpu"

    def __init__(self) -> None:
        self.model = _FakeModel()
        self.device = self.__class__.device
        self.tokenizer = _FakeTokenizer()
        self.tgt_bos_id = 99

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device):
        del checkpoint_path, device
        return cls()


class _StaticScorer:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def score_batch(self, examples):
        del examples
        return self._scores


def _patch_split_runtime(monkeypatch, scorer, *, device: str = "cpu") -> None:
    _FakeTranslator.device = device
    fake_translator_module = types.ModuleType("translator.inference")
    fake_translator_module.Translator = _FakeTranslator
    monkeypatch.setitem(sys.modules, "translator.inference", fake_translator_module)
    monkeypatch.setattr("model_based_curation.api.BatchSeq2SeqLossScorer", lambda *args, **kwargs: scorer)


def _write_bucket(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "keep", "loss", "src", "tgt"), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def test_split_copies_buckets_to_drive(monkeypatch, caplog):
    root_dir = _temp_dir("split_api")
    paths = _SplitPaths(root_dir)
    _write_dataset(paths.dataset_dir, [{"id": 1, "src_ids": [11], "tgt_ids": [99, 21, 0]}])
    (paths.dataset_dir / "root.txt").write_text("root", encoding="utf-8")
    (paths.dataset_dir / "loss_buckets").mkdir()
    (paths.dataset_dir / "loss_buckets" / "bucket.gsheet").write_text("skip", encoding="utf-8")
    _patch_split_runtime(monkeypatch, _StaticScorer([0.2]))

    config = SplitRunConfig(
        dataset="europarl/d1",
        checkpoint="run",
        upper_bounds=(0.5,),
        artifacts_dir=paths.drive_artifacts,
        local_artifacts_dir=paths.local_artifacts,
    )

    with caplog.at_level(logging.INFO):
        output_paths = split(config)

    assert output_paths[0].is_file()
    assert _read_rows(output_paths[0], delimiter=";") == [
        {"id": "1", "keep": "", "loss": "0,2", "src": "11", "tgt": "21|0"}
    ]
    assert _read_rows(paths.drive_dir / output_paths[0].name, delimiter=";") == [
        {"id": "1", "keep": "", "loss": "0,2", "src": "11", "tgt": "21|0"}
    ]
    assert (paths.drive_dir / "bucket_stats.yaml").is_file()
    assert (paths.output_dir / "split_config.yaml").is_file()
    assert (paths.drive_dir / "split_config.yaml").is_file()
    assert (paths.local_dataset_dir / "root.txt").read_text(encoding="utf-8") == "root"
    assert not (paths.local_dataset_dir / "loss_buckets" / "bucket.gsheet").exists()
    messages = [record.getMessage() for record in caplog.records]
    assert any("Copying bucket files to" in message for message in messages)


def test_split_uses_next_bucket_run(monkeypatch):
    root_dir = _temp_dir("split_api_next_bucket_run")
    paths = _SplitPaths(root_dir)
    paths.drive_dir.mkdir(parents=True, exist_ok=True)
    _write_dataset(paths.dataset_dir, [{"id": 1, "src_ids": [11], "tgt_ids": [99, 21, 0]}])
    _patch_split_runtime(monkeypatch, _StaticScorer([0.2]))

    output_paths = split(
        SplitRunConfig(
            dataset="europarl/d1",
            checkpoint="run",
            upper_bounds=(0.5,),
            artifacts_dir=paths.drive_artifacts,
            local_artifacts_dir=paths.local_artifacts,
        )
    )

    assert output_paths[0].parent.name == "r2"


def test_split_can_write_german_csv_format(monkeypatch):
    root_dir = _temp_dir("split_api_german_csv")
    paths = _SplitPaths(root_dir)
    _write_dataset(paths.dataset_dir, [{"id": 1, "src_ids": [11], "tgt_ids": [99, 21, 0]}])
    _patch_split_runtime(monkeypatch, _StaticScorer([0.2]))

    config = SplitRunConfig(
        dataset="europarl/d1",
        checkpoint="run",
        upper_bounds=(0.5,),
        artifacts_dir=paths.drive_artifacts,
        local_artifacts_dir=paths.local_artifacts,
        csv_delimiter=";",
        loss_decimal_separator=",",
    )

    output_paths = split(config)
    assert _read_rows(output_paths[0], delimiter=";") == [
        {"id": "1", "keep": "", "loss": "0,2", "src": "11", "tgt": "21|0"}
    ]


def test_split_passes_bf16_setting_to_batch_scorer(monkeypatch):
    root_dir = _temp_dir("split_api_bf16")
    paths = _SplitPaths(root_dir)
    _write_dataset(paths.dataset_dir, [{"id": 1, "src_ids": [11], "tgt_ids": [99, 21, 0]}])

    scorer_kwargs = {}

    def _make_scorer(*args, **kwargs):
        del args
        scorer_kwargs.update(kwargs)
        return _StaticScorer([0.2])

    _FakeTranslator.device = "cuda"
    fake_translator_module = types.ModuleType("translator.inference")
    fake_translator_module.Translator = _FakeTranslator
    monkeypatch.setitem(sys.modules, "translator.inference", fake_translator_module)
    monkeypatch.setattr("model_based_curation.api.BatchSeq2SeqLossScorer", _make_scorer)

    config = SplitRunConfig(
        dataset="europarl/d1",
        checkpoint="run",
        upper_bounds=(0.5,),
        artifacts_dir=paths.drive_artifacts,
        local_artifacts_dir=paths.local_artifacts,
        use_bf16=True,
    )
    split(config)

    assert scorer_kwargs["use_bf16"] is True


def test_filter_writes_log_and_copies_dataset_to_drive(monkeypatch, caplog):
    root_dir = _temp_dir("filter_api")
    paths = _FilterPaths(root_dir)

    _write_dataset(
        paths.dataset_dir,
        [{"id": 1, "src_ids": [11], "tgt_ids": [21]}, {"id": 2, "src_ids": [12], "tgt_ids": [22]}],
    )
    paths.bucket_dir.mkdir(parents=True, exist_ok=True)
    _write_bucket(
        paths.bucket_dir / "1.csv", [{"id": "2", "keep": "", "loss": "3,1", "src": "12", "tgt": "22"}]
    )

    with caplog.at_level(logging.INFO):
        result = filter(
            FilterRunConfig(
                dataset="europarl/d1",
                bucket_run="r1",
                bucket_files=(1,),
                artifacts_dir=paths.drive_artifacts,
                local_artifacts_dir=paths.local_artifacts,
            )
        )

    assert result == paths.output_dir
    assert [int(row["id"]) for row in load_from_disk(str(paths.output_dir))] == [1]
    assert [int(row["id"]) for row in load_from_disk(str(paths.drive_dir))] == [1]
    assert (paths.output_dir / "filter_config.yaml").is_file()
    assert (paths.drive_dir / "filter_config.yaml").is_file()
    assert (paths.drive_dir / "filter.log").is_file()
    register_text = (paths.drive_artifacts / "datasets" / "dataset_register.csv").read_text(encoding="utf-8")
    assert ";europarl/d2;curate;europarl/d1;" in register_text
    messages = [record.getMessage() for record in caplog.records]
    assert any("Filter completed successfully" in message for message in messages)


def test_filter_can_use_explicit_bucket_files_subset(monkeypatch):
    root_dir = _temp_dir("filter_api_bucket_subset")
    paths = _FilterPaths(root_dir)

    _write_dataset(
        paths.dataset_dir,
        [{"id": 1, "src_ids": [11], "tgt_ids": [21]}, {"id": 2, "src_ids": [12], "tgt_ids": [22]}],
    )
    paths.bucket_dir.mkdir(parents=True, exist_ok=True)
    _write_bucket(
        paths.bucket_dir / "1.csv", [{"id": "1", "keep": "", "loss": "0,4", "src": "11", "tgt": "21"}]
    )
    _write_bucket(
        paths.bucket_dir / "2.csv", [{"id": "2", "keep": "", "loss": "0,9", "src": "12", "tgt": "22"}]
    )

    filter(
        FilterRunConfig(
            dataset="europarl/d1",
            bucket_run="r1",
            bucket_files=(2,),
            artifacts_dir=paths.drive_artifacts,
            local_artifacts_dir=paths.local_artifacts,
        )
    )
    assert [int(row["id"]) for row in load_from_disk(str(paths.output_dir))] == [1]


def test_filter_copies_missing_local_bucket_files_from_drive(monkeypatch):
    root_dir = _temp_dir("filter_api_copy_buckets")
    paths = _FilterPaths(root_dir)

    _write_dataset(
        paths.dataset_dir,
        [{"id": 1, "src_ids": [11], "tgt_ids": [21]}, {"id": 2, "src_ids": [12], "tgt_ids": [22]}],
    )
    paths.drive_bucket_dir.mkdir(parents=True, exist_ok=True)
    _write_bucket(
        paths.drive_bucket_dir / "1.csv", [{"id": "2", "keep": "", "loss": "3,1", "src": "12", "tgt": "22"}]
    )

    filter(
        FilterRunConfig(
            dataset="europarl/d1",
            bucket_run="r1",
            bucket_files=(1,),
            artifacts_dir=paths.drive_artifacts,
            local_artifacts_dir=paths.local_artifacts,
        )
    )

    assert (paths.local_bucket_dir / "1.csv").is_file()
    assert [int(row["id"]) for row in load_from_disk(str(paths.output_dir))] == [1]


def test_filter_uses_next_curate_dataset(monkeypatch):
    root_dir = _temp_dir("filter_api_next_curate")
    paths = _FilterPaths(root_dir)
    paths.drive_dir.mkdir(parents=True, exist_ok=True)
    _write_dataset(paths.dataset_dir, [{"id": 1, "src_ids": [11], "tgt_ids": [21]}])
    paths.bucket_dir.mkdir(parents=True, exist_ok=True)
    _write_bucket(
        paths.bucket_dir / "1.csv", [{"id": "1", "keep": "x", "loss": "0,1", "src": "11", "tgt": "21"}]
    )

    result = filter(
        FilterRunConfig(
            dataset="europarl/d1",
            bucket_run="r1",
            bucket_files=(1,),
            artifacts_dir=paths.drive_artifacts,
            local_artifacts_dir=paths.local_artifacts,
        )
    )

    assert result.name == "d3"


def test_filter_fails_when_no_bucket_files_exist(monkeypatch):
    root_dir = _temp_dir("filter_api_no_buckets")
    paths = _FilterPaths(root_dir)
    _write_dataset(paths.dataset_dir, [{"id": 1, "src_ids": [11], "tgt_ids": [21]}])

    with pytest.raises(ValueError, match="No bucket files found"):
        filter(
            FilterRunConfig(
                dataset="europarl/d1",
                bucket_run="r1",
                artifacts_dir=paths.drive_artifacts,
                local_artifacts_dir=paths.local_artifacts,
            )
        )


def test_filter_fails_when_bucket_file_is_missing_locally_and_on_drive(monkeypatch):
    root_dir = _temp_dir("filter_api_missing_bucket")
    paths = _FilterPaths(root_dir)
    _write_dataset(paths.dataset_dir, [{"id": 1, "src_ids": [11], "tgt_ids": [21]}])

    with pytest.raises(ValueError, match=r"Bucket file not found: .*1\.csv"):
        filter(
            FilterRunConfig(
                dataset="europarl/d1",
                bucket_run="r1",
                bucket_files=(1,),
                artifacts_dir=paths.drive_artifacts,
                local_artifacts_dir=paths.local_artifacts,
            )
        )

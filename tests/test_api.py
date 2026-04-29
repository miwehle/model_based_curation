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
    def __init__(self, root_dir: Path, *, dataset_dir_name: str = "dataset") -> None:
        self.dataset_dir = root_dir / dataset_dir_name
        self.local_dataset_dir = root_dir / "local_artifacts" / "dataset"
        self.local_curation_dir = self.local_dataset_dir / "curation"
        self.output_dir = self.local_curation_dir / "loss_buckets"
        self.drive_dir = root_dir / "drive_artifacts" / "dataset" / "curation" / "loss_buckets"


class _FilterPaths:
    def __init__(self, root_dir: Path) -> None:
        self.dataset_dir = root_dir / "dataset"
        self.local_dataset_dir = root_dir / "local_artifacts" / "dataset"
        self.local_curation_dir = self.local_dataset_dir / "curation"
        self.drive_curation_dir = root_dir / "drive_artifacts" / "dataset" / "curation"
        self.bucket_dir = self.dataset_dir / "curation" / "loss_buckets"
        self.local_bucket_dir = self.local_curation_dir / "loss_buckets"
        self.drive_bucket_dir = self.drive_curation_dir / "loss_buckets"
        self.output_dir = self.local_curation_dir / "curated_dataset"
        self.drive_dir = self.drive_curation_dir / "curated_dataset"


def _value_property(value):
    return property(lambda self: value)


def _patch_attr(monkeypatch, cls, name: str, value) -> None:
    monkeypatch.setattr(cls, name, _value_property(value))


def _patch_config_paths(
    monkeypatch, *, dataset_dir: Path, output_dir: Path, drive_dir: Path, checkpoint_file: Path
) -> None:
    _patch_attr(monkeypatch, SplitRunConfig, "dataset_drive_path", dataset_dir)
    _patch_attr(monkeypatch, SplitRunConfig, "dataset_local_path", dataset_dir)
    _patch_attr(monkeypatch, SplitRunConfig, "output_path", output_dir)
    _patch_attr(monkeypatch, SplitRunConfig, "drive_output_path", drive_dir)
    _patch_attr(monkeypatch, SplitRunConfig, "checkpoint_file", checkpoint_file)


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


def _patch_filter_config_paths(
    monkeypatch, *, dataset_dir: Path, output_dir: Path, drive_dir: Path, drive_bucket_dir: Path | None = None
) -> None:
    bucket_dir = dataset_dir / "curation" / "loss_buckets"
    resolved_drive_bucket_dir = drive_bucket_dir or bucket_dir
    _patch_attr(monkeypatch, FilterRunConfig, "dataset_drive_path", dataset_dir)
    _patch_attr(monkeypatch, FilterRunConfig, "dataset_local_path", dataset_dir)
    _patch_attr(monkeypatch, FilterRunConfig, "bucket_dir", bucket_dir)
    _patch_attr(monkeypatch, FilterRunConfig, "drive_bucket_dir", resolved_drive_bucket_dir)
    _patch_attr(monkeypatch, FilterRunConfig, "output_path", output_dir)
    _patch_attr(monkeypatch, FilterRunConfig, "drive_output_path", drive_dir)


def test_split_copies_buckets_to_drive(monkeypatch, caplog):
    root_dir = _temp_dir("split_api")
    paths = _SplitPaths(root_dir, dataset_dir_name="drive_dataset")
    _write_dataset(paths.dataset_dir, [{"id": 1, "src_ids": [11], "tgt_ids": [99, 21, 0]}])
    (paths.dataset_dir / "root.txt").write_text("root", encoding="utf-8")
    (paths.dataset_dir / "curation").mkdir()
    (paths.dataset_dir / "curation" / "bucket.gsheet").write_text("skip", encoding="utf-8")
    _patch_split_runtime(monkeypatch, _StaticScorer([0.2]))
    _patch_config_paths(
        monkeypatch,
        dataset_dir=paths.dataset_dir,
        output_dir=paths.output_dir,
        drive_dir=paths.drive_dir,
        checkpoint_file=root_dir / "checkpoint.pt",
    )
    _patch_attr(monkeypatch, SplitRunConfig, "dataset_local_path", paths.local_dataset_dir)

    config = SplitRunConfig(dataset="dataset", checkpoint="run", upper_bounds=(0.5,))

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
    assert not (paths.local_curation_dir / "bucket.gsheet").exists()
    messages = [record.getMessage() for record in caplog.records]
    assert any("Copying bucket files to" in message for message in messages)


def test_split_fails_early_when_drive_output_dir_exists(monkeypatch):
    root_dir = _temp_dir("split_api_drive_exists")
    paths = _SplitPaths(root_dir)
    paths.drive_dir.mkdir(parents=True, exist_ok=True)
    _patch_config_paths(
        monkeypatch,
        dataset_dir=paths.dataset_dir,
        output_dir=paths.output_dir,
        drive_dir=paths.drive_dir,
        checkpoint_file=root_dir / "checkpoint.pt",
    )

    with pytest.raises(ValueError, match="Drive output directory already exists"):
        split(SplitRunConfig(dataset="dataset", checkpoint="run", upper_bounds=(0.5,)))


def test_split_can_write_german_csv_format(monkeypatch):
    root_dir = _temp_dir("split_api_german_csv")
    paths = _SplitPaths(root_dir)
    _write_dataset(paths.dataset_dir, [{"id": 1, "src_ids": [11], "tgt_ids": [99, 21, 0]}])
    _patch_split_runtime(monkeypatch, _StaticScorer([0.2]))
    _patch_config_paths(
        monkeypatch,
        dataset_dir=paths.dataset_dir,
        output_dir=paths.output_dir,
        drive_dir=root_dir / "drive",
        checkpoint_file=root_dir / "checkpoint.pt",
    )

    config = SplitRunConfig(
        dataset="dataset",
        checkpoint="run",
        upper_bounds=(0.5,),
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
    _patch_config_paths(
        monkeypatch,
        dataset_dir=paths.dataset_dir,
        output_dir=paths.output_dir,
        drive_dir=root_dir / "drive",
        checkpoint_file=root_dir / "checkpoint.pt",
    )

    def _make_scorer(*args, **kwargs):
        del args
        scorer_kwargs.update(kwargs)
        return _StaticScorer([0.2])

    _FakeTranslator.device = "cuda"
    fake_translator_module = types.ModuleType("translator.inference")
    fake_translator_module.Translator = _FakeTranslator
    monkeypatch.setitem(sys.modules, "translator.inference", fake_translator_module)
    monkeypatch.setattr("model_based_curation.api.BatchSeq2SeqLossScorer", _make_scorer)

    config = SplitRunConfig(dataset="dataset", checkpoint="run", upper_bounds=(0.5,), use_bf16=True)
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
    _patch_filter_config_paths(
        monkeypatch, dataset_dir=paths.dataset_dir, output_dir=paths.output_dir, drive_dir=paths.drive_dir
    )

    with caplog.at_level(logging.INFO):
        result = filter(FilterRunConfig(dataset="dataset", bucket_files=(1,)))

    assert result == paths.output_dir
    assert [int(row["id"]) for row in load_from_disk(str(paths.output_dir))] == [1]
    assert [int(row["id"]) for row in load_from_disk(str(paths.drive_dir))] == [1]
    assert (paths.output_dir / "filter_config.yaml").is_file()
    assert (paths.drive_dir / "filter_config.yaml").is_file()
    assert (paths.local_curation_dir / "filter.log").is_file()
    assert (paths.drive_dir / "filter.log").is_file()
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
    _patch_filter_config_paths(
        monkeypatch, dataset_dir=paths.dataset_dir, output_dir=paths.output_dir, drive_dir=paths.drive_dir
    )

    filter(FilterRunConfig(dataset="dataset", bucket_files=(2,)))
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
    _patch_filter_config_paths(
        monkeypatch,
        dataset_dir=paths.dataset_dir,
        output_dir=paths.output_dir,
        drive_dir=paths.drive_dir,
        drive_bucket_dir=paths.drive_bucket_dir,
    )
    _patch_attr(monkeypatch, FilterRunConfig, "dataset_local_path", paths.local_dataset_dir)
    _patch_attr(monkeypatch, FilterRunConfig, "bucket_dir", paths.local_bucket_dir)

    filter(FilterRunConfig(dataset="dataset", bucket_files=(1,)))

    assert (paths.local_bucket_dir / "1.csv").is_file()
    assert [int(row["id"]) for row in load_from_disk(str(paths.output_dir))] == [1]


def test_filter_fails_early_when_drive_output_dir_exists(monkeypatch):
    root_dir = _temp_dir("filter_api_drive_exists")
    paths = _FilterPaths(root_dir)
    paths.drive_dir.mkdir(parents=True, exist_ok=True)
    _patch_filter_config_paths(
        monkeypatch, dataset_dir=paths.dataset_dir, output_dir=paths.output_dir, drive_dir=paths.drive_dir
    )

    with pytest.raises(ValueError, match="Drive output directory already exists"):
        filter(FilterRunConfig(dataset="dataset", bucket_files=(1,)))


def test_filter_fails_when_no_bucket_files_exist(monkeypatch):
    root_dir = _temp_dir("filter_api_no_buckets")
    paths = _FilterPaths(root_dir)
    _write_dataset(paths.dataset_dir, [{"id": 1, "src_ids": [11], "tgt_ids": [21]}])
    _patch_filter_config_paths(
        monkeypatch, dataset_dir=paths.dataset_dir, output_dir=paths.output_dir, drive_dir=paths.drive_dir
    )

    with pytest.raises(ValueError, match="No bucket files found"):
        filter(FilterRunConfig(dataset="dataset"))


def test_filter_fails_when_bucket_file_is_missing_locally_and_on_drive(monkeypatch):
    root_dir = _temp_dir("filter_api_missing_bucket")
    paths = _FilterPaths(root_dir)
    _write_dataset(paths.dataset_dir, [{"id": 1, "src_ids": [11], "tgt_ids": [21]}])
    _patch_filter_config_paths(
        monkeypatch,
        dataset_dir=paths.dataset_dir,
        output_dir=paths.output_dir,
        drive_dir=paths.drive_dir,
        drive_bucket_dir=paths.drive_bucket_dir,
    )
    _patch_attr(monkeypatch, FilterRunConfig, "dataset_local_path", paths.local_dataset_dir)
    _patch_attr(monkeypatch, FilterRunConfig, "bucket_dir", paths.local_bucket_dir)

    with pytest.raises(ValueError, match=r"Bucket file not found: .*1\.csv"):
        filter(FilterRunConfig(dataset="dataset", bucket_files=(1,)))

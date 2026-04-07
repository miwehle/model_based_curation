from __future__ import annotations

import csv
import logging
from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from datasets import Dataset, load_from_disk

from model_based_curation import FilterConfig, filter

_TMP_DIR = Path(__file__).resolve().parent / ".local_tmp"


def _temp_dir(prefix: str) -> Path:
    path = _TMP_DIR / f"{prefix}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_bucket(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("id", "keep", "loss", "src", "tgt"), delimiter=";"
        )
        writer.writeheader()
        writer.writerows(rows)


def _patch_config_paths(
    monkeypatch, *, dataset_dir: Path, output_dir: Path, drive_dir: Path
) -> None:
    monkeypatch.setattr(
        FilterConfig, "dataset_drive_path", property(lambda self: dataset_dir)
    )
    monkeypatch.setattr(
        FilterConfig, "dataset_local_path", property(lambda self: dataset_dir)
    )
    monkeypatch.setattr(
        FilterConfig,
        "bucket_dir",
        property(lambda self: dataset_dir / "curation" / "loss_buckets"),
    )
    monkeypatch.setattr(
        FilterConfig,
        "drive_bucket_dir",
        property(lambda self: dataset_dir / "curation" / "loss_buckets"),
    )
    monkeypatch.setattr(FilterConfig, "output_path", property(lambda self: output_dir))
    monkeypatch.setattr(
        FilterConfig, "drive_output_path", property(lambda self: drive_dir)
    )


def test_filter_writes_log_and_copies_dataset_to_drive(monkeypatch, caplog):
    root_dir = _temp_dir("filter_api")
    dataset_dir = root_dir / "dataset"
    bucket_dir = dataset_dir / "curation" / "loss_buckets"
    output_dir = (
        root_dir / "local_artifacts" / "dataset" / "curation" / "filtered_dataset"
    )
    drive_dir = (
        root_dir / "drive_artifacts" / "dataset" / "curation" / "filtered_dataset"
    )

    Dataset.from_list(
        [
            {"id": 1, "src_ids": [11], "tgt_ids": [21]},
            {"id": 2, "src_ids": [12], "tgt_ids": [22]},
        ]
    ).save_to_disk(str(dataset_dir))
    with (dataset_dir / "dataset_manifest.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"num_examples": 2, "id_field": "id"}, handle, sort_keys=False)
    bucket_dir.mkdir(parents=True, exist_ok=True)
    _write_bucket(
        bucket_dir / "01_loss.csv",
        [{"id": "2", "keep": "", "loss": "3,1", "src": "12", "tgt": "22"}],
    )
    _patch_config_paths(
        monkeypatch, dataset_dir=dataset_dir, output_dir=output_dir, drive_dir=drive_dir
    )

    with caplog.at_level(logging.INFO):
        result = filter(FilterConfig(dataset="dataset"))

    assert result == output_dir
    assert [int(row["id"]) for row in load_from_disk(str(output_dir))] == [1]
    assert [int(row["id"]) for row in load_from_disk(str(drive_dir))] == [1]
    assert (output_dir.parent / "filter.log").is_file()
    assert (drive_dir / "filter.log").is_file()
    assert any(
        "Filter completed successfully" in record.getMessage()
        for record in caplog.records
    )


def test_filter_fails_early_when_drive_output_dir_exists(monkeypatch):
    root_dir = _temp_dir("filter_api_drive_exists")
    drive_dir = (
        root_dir / "drive_artifacts" / "dataset" / "curation" / "filtered_dataset"
    )
    drive_dir.mkdir(parents=True, exist_ok=True)
    _patch_config_paths(
        monkeypatch,
        dataset_dir=root_dir / "dataset",
        output_dir=(
            root_dir
            / "local_artifacts"
            / "dataset"
            / "curation"
            / "filtered_dataset"
        ),
        drive_dir=drive_dir,
    )

    with pytest.raises(ValueError, match="Drive output directory already exists"):
        filter(FilterConfig(dataset="dataset"))


def test_filter_fails_when_no_bucket_files_exist(monkeypatch):
    root_dir = _temp_dir("filter_api_no_buckets")
    dataset_dir = root_dir / "dataset"
    Dataset.from_list([{"id": 1, "src_ids": [11], "tgt_ids": [21]}]).save_to_disk(
        str(dataset_dir)
    )
    _patch_config_paths(
        monkeypatch,
        dataset_dir=dataset_dir,
        output_dir=(
            root_dir
            / "local_artifacts"
            / "dataset"
            / "curation"
            / "filtered_dataset"
        ),
        drive_dir=(
            root_dir
            / "drive_artifacts"
            / "dataset"
            / "curation"
            / "filtered_dataset"
        ),
    )

    with pytest.raises(ValueError, match="No bucket files found"):
        filter(FilterConfig(dataset="dataset"))

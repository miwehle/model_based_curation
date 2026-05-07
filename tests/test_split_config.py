from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from model_based_curation import FilterRunConfig, SplitRunConfig


def _dataset_roots() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1] / ".local_tmp" / "tests" / uuid4().hex
    drive = root / "drive_artifacts"
    local = root / "local_artifacts"
    (drive / "datasets" / "europarl" / "d1_preprocess").mkdir(parents=True)
    return drive, local


def test_configs_use_local_datasets_directory():
    drive, local = _dataset_roots()

    assert SplitRunConfig(
        dataset="europarl/d1",
        checkpoint="run",
        upper_bounds=(0.5, 1.5),
        artifacts_dir=drive,
        local_artifacts_dir=local,
    ).dataset_local_path == local / "datasets" / "europarl" / "d1_preprocess"
    assert FilterRunConfig(
        dataset="europarl/d1", bucket_run="r1", artifacts_dir=drive, local_artifacts_dir=local
    ).dataset_local_path == local / "datasets" / "europarl" / "d1_preprocess"


def test_filter_run_config_uses_bucket_run_directory():
    drive, local = _dataset_roots()
    config = FilterRunConfig(
        dataset="europarl/d1", bucket_run="r1", artifacts_dir=drive, local_artifacts_dir=local
    )

    assert config.bucket_dir == local / "datasets" / "europarl" / "d1_preprocess" / "loss_buckets" / "r1"
    expected_drive_bucket_dir = drive / "datasets" / "europarl" / "d1_preprocess" / "loss_buckets" / "r1"
    assert config.drive_bucket_dir == expected_drive_bucket_dir


def test_split_run_config_validates_csv_format_options():
    with pytest.raises(ValueError, match="log_every_batches"):
        SplitRunConfig(dataset="dataset", checkpoint="run", upper_bounds=(0.5, 1.5), log_every_batches=0)

    with pytest.raises(ValueError, match="csv_delimiter"):
        SplitRunConfig(dataset="dataset", checkpoint="run", upper_bounds=(0.5, 1.5), csv_delimiter="|")

    with pytest.raises(ValueError, match="loss_decimal_separator"):
        SplitRunConfig(
            dataset="dataset", checkpoint="run", upper_bounds=(0.5, 1.5), loss_decimal_separator=":"
        )

    with pytest.raises(ValueError, match="decode_from_loss"):
        SplitRunConfig(dataset="dataset", checkpoint="run", upper_bounds=(0.5, 1.5), decode_from_loss=-0.1)

    with pytest.raises(ValueError, match="decode_at_least"):
        SplitRunConfig(dataset="dataset", checkpoint="run", upper_bounds=(0.5, 1.5), decode_at_least=-1)


def test_split_run_config_validates_upper_bounds():
    with pytest.raises(ValueError, match="upper_bounds"):
        SplitRunConfig(dataset="dataset", checkpoint="run", upper_bounds=(0.5, 0.5))

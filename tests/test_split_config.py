from __future__ import annotations

from pathlib import Path

import pytest

from model_based_curation import FilterConfig, SplitConfig


def test_configs_use_local_datasets_directory():
    assert SplitConfig(
        dataset="dataset", checkpoint="run", upper_bounds=(0.5, 1.5)
    ).dataset_local_path == Path("/content/nmt_lab/artifacts/datasets/dataset")
    assert FilterConfig(dataset="dataset").dataset_local_path == Path(
        "/content/nmt_lab/artifacts/datasets/dataset"
    )


def test_filter_config_uses_curated_dataset_output_directory():
    config = FilterConfig(dataset="dataset")

    assert config.output_path == Path("/content/nmt_lab/artifacts/datasets/dataset/curation/curated_dataset")
    assert config.drive_output_path == Path(
        "/content/drive/MyDrive/nmt_lab/artifacts/datasets/dataset/curation/curated_dataset"
    )


def test_split_config_validates_csv_format_options():
    with pytest.raises(ValueError, match="log_every_batches"):
        SplitConfig(dataset="dataset", checkpoint="run", upper_bounds=(0.5, 1.5), log_every_batches=0)

    with pytest.raises(ValueError, match="csv_delimiter"):
        SplitConfig(dataset="dataset", checkpoint="run", upper_bounds=(0.5, 1.5), csv_delimiter="|")

    with pytest.raises(ValueError, match="loss_decimal_separator"):
        SplitConfig(dataset="dataset", checkpoint="run", upper_bounds=(0.5, 1.5), loss_decimal_separator=":")

    with pytest.raises(ValueError, match="decode_from_loss"):
        SplitConfig(dataset="dataset", checkpoint="run", upper_bounds=(0.5, 1.5), decode_from_loss=-0.1)

    with pytest.raises(ValueError, match="decode_at_least"):
        SplitConfig(dataset="dataset", checkpoint="run", upper_bounds=(0.5, 1.5), decode_at_least=-1)


def test_split_config_validates_upper_bounds():
    with pytest.raises(ValueError, match="upper_bounds"):
        SplitConfig(dataset="dataset", checkpoint="run", upper_bounds=(0.5, 0.5))

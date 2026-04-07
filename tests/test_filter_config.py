from __future__ import annotations

from pathlib import Path

from model_based_curation import FilterConfig


def test_filter_config_derives_conventional_paths_from_dataset():
    cfg = FilterConfig(dataset="europarl_de-en_train")

    assert cfg.dataset_drive_path == Path(
        "/content/drive/MyDrive/nmt_lab/artifacts/datasets/europarl_de-en_train"
    )
    assert cfg.dataset_local_path == Path(
        "/content/nmt_lab/artifacts/europarl_de-en_train"
    )
    assert cfg.bucket_dir == Path(
        "/content/nmt_lab/artifacts/europarl_de-en_train/curation/loss_buckets"
    )
    assert cfg.drive_bucket_dir == Path(
        "/content/drive/MyDrive/nmt_lab/artifacts/datasets/europarl_de-en_train/curation/loss_buckets"
    )
    assert cfg.output_path == Path(
        "/content/nmt_lab/artifacts/europarl_de-en_train/curation/filtered_dataset"
    )
    assert cfg.drive_output_path == Path(
        "/content/drive/MyDrive/nmt_lab/artifacts/datasets/europarl_de-en_train/curation/filtered_dataset"
    )

from __future__ import annotations

from pathlib import Path

import torch

from model_based_curation import SplitConfig


def test_split_config_resolves_optional_local_dataset_and_drive_paths():
    cfg = SplitConfig(
        dataset_path="/drive/datasets/train",
        checkpoint_path="/drive/checkpoints/model.pt",
        output_dir="/content/loss_buckets",
        upper_bounds=(0.5, 1.5),
        batch_size=64,
        sort_by_loss_desc=True,
        device=torch.device("cuda"),
        local_dataset_dir="/content/datasets/train",
        copy_buckets_to_drive_dir="/drive/output/buckets",
        overwrite_output=True,
    )

    assert cfg.resolved_dataset_path == Path("/content/datasets/train")
    assert cfg.output_path == Path("/content/loss_buckets")
    assert cfg.checkpoint_file == Path("/drive/checkpoints/model.pt")
    assert cfg.copy_buckets_to_drive_path == Path("/drive/output/buckets")
    assert cfg.sort_by_loss_desc is True


def test_split_config_uses_original_dataset_path_when_no_local_copy_is_set():
    cfg = SplitConfig(
        dataset_path="/drive/datasets/train",
        checkpoint_path="/drive/checkpoints/model.pt",
        output_dir="/content/loss_buckets",
        upper_bounds=(0.5, 1.5),
    )

    assert cfg.resolved_dataset_path == Path("/drive/datasets/train")
    assert cfg.copy_buckets_to_drive_path is None

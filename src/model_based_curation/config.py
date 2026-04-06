from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True, kw_only=True)
class SplitConfig:
    dataset_path: str
    checkpoint_path: str
    output_dir: str
    upper_bounds: tuple[float, ...]
    batch_size: int = 32
    sort_by_loss_desc: bool = False
    device: str | torch.device | None = None
    local_dataset_dir: str | None = None
    copy_buckets_to_drive_dir: str | None = None
    overwrite_output: bool = False

    @property
    def resolved_dataset_path(self) -> Path:
        return Path(self.local_dataset_dir or self.dataset_path)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def checkpoint_file(self) -> Path:
        return Path(self.checkpoint_path)

    @property
    def copy_buckets_to_drive_path(self) -> Path | None:
        return (
            None
            if self.copy_buckets_to_drive_dir is None
            else Path(self.copy_buckets_to_drive_dir)
        )

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass

_CONFIG = ConfigDict(extra="forbid")


@dataclass(frozen=True, kw_only=True, config=_CONFIG)
class SplitConfig:
    dataset: str
    checkpoint: str
    upper_bounds: tuple[float, ...]
    csv_delimiter: Literal[",", ";"] = ";"
    loss_decimal_separator: Literal[".", ","] = ","
    batch_size: int = Field(default=32, gt=0)
    log_every_batches: int = Field(default=50, gt=0)
    use_bf16: bool = False
    decode_from_loss: float | None = Field(default=None, ge=0)
    decode_at_least: int = Field(default=10, ge=0)

    @model_validator(mode="after")
    def validate_upper_bounds(self) -> SplitConfig:
        if any(bound < 0 for bound in self.upper_bounds):
            raise ValueError("upper_bounds must be non-negative.")
        if any(left >= right for left, right in zip(self.upper_bounds, self.upper_bounds[1:])):
            raise ValueError("upper_bounds must be strictly increasing.")
        return self

    @property
    def dataset_drive_path(self) -> Path:
        return Path("/content/drive/MyDrive/nmt_lab/artifacts/datasets") / self.dataset

    @property
    def dataset_local_path(self) -> Path:
        return Path("/content") / "nmt_lab" / "artifacts" / "datasets" / self.dataset

    @property
    def output_path(self) -> Path:
        return self.dataset_local_path / "curation" / "loss_buckets"

    @property
    def drive_output_path(self) -> Path:
        return (
            Path("/content/drive/MyDrive/nmt_lab/artifacts")
            / "datasets"
            / self.dataset
            / "curation"
            / "loss_buckets"
        )

    @property
    def checkpoint_file(self) -> Path:
        return (
            Path("/content/drive/MyDrive/nmt_lab/artifacts/training_runs") / self.checkpoint / "checkpoint.pt"
        )


@dataclass(frozen=True, kw_only=True, config=_CONFIG)
class FilterConfig:
    dataset: str
    bucket_files: tuple[int, ...] = ()

    @property
    def dataset_drive_path(self) -> Path:
        return Path("/content/drive/MyDrive/nmt_lab/artifacts/datasets") / self.dataset

    @property
    def dataset_local_path(self) -> Path:
        return Path("/content") / "nmt_lab" / "artifacts" / "datasets" / self.dataset

    @property
    def bucket_dir(self) -> Path:
        return self.dataset_local_path / "curation" / "loss_buckets"

    @property
    def drive_bucket_dir(self) -> Path:
        return self.dataset_drive_path / "curation" / "loss_buckets"

    @property
    def output_path(self) -> Path:
        return self.dataset_local_path / "curation" / "curated_dataset"

    @property
    def drive_output_path(self) -> Path:
        return self.dataset_drive_path / "curation" / "curated_dataset"

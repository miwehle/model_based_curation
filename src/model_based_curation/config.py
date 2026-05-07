from __future__ import annotations

from pathlib import Path
from typing import Literal

from lab_infrastructure.dataset_artifacts import resolve_dataset
from pydantic import ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass

_CONFIG = ConfigDict(extra="forbid")
_DRIVE_ARTIFACTS = Path("/content/drive/MyDrive/nmt_lab/artifacts")
_LOCAL_ARTIFACTS = Path("/content/nmt_lab/artifacts")


def _dataset_path(artifacts_dir: str | Path, dataset: str) -> Path:
    return resolve_dataset(Path(artifacts_dir) / "datasets", dataset)


def _local_dataset_path(
    local_artifacts_dir: str | Path, drive_artifacts_dir: str | Path, drive_path: Path
) -> Path:
    relative_path = drive_path.relative_to(Path(drive_artifacts_dir) / "datasets")
    return Path(local_artifacts_dir) / "datasets" / relative_path


@dataclass(frozen=True, kw_only=True, config=_CONFIG)
class SplitRunConfig:
    dataset: str
    checkpoint: str
    upper_bounds: tuple[float, ...]
    artifacts_dir: str | Path = _DRIVE_ARTIFACTS
    local_artifacts_dir: str | Path = _LOCAL_ARTIFACTS
    csv_delimiter: Literal[",", ";"] = ";"
    loss_decimal_separator: Literal[".", ","] = ","
    batch_size: int = Field(default=32, gt=0)
    log_every_batches: int = Field(default=50, gt=0)
    use_bf16: bool = False
    decode_from_loss: float | None = Field(default=None, ge=0)
    decode_at_least: int = Field(default=10, ge=0)

    @model_validator(mode="after")
    def validate_upper_bounds(self) -> SplitRunConfig:
        if any(bound < 0 for bound in self.upper_bounds):
            raise ValueError("upper_bounds must be non-negative.")
        if any(left >= right for left, right in zip(self.upper_bounds, self.upper_bounds[1:])):
            raise ValueError("upper_bounds must be strictly increasing.")
        return self

    @property
    def dataset_drive_path(self) -> Path:
        return _dataset_path(self.artifacts_dir, self.dataset)

    @property
    def dataset_local_path(self) -> Path:
        return _local_dataset_path(self.local_artifacts_dir, self.artifacts_dir, self.dataset_drive_path)

    @property
    def bucket_root_path(self) -> Path:
        return self.dataset_local_path / "loss_buckets"

    @property
    def drive_bucket_root_path(self) -> Path:
        return self.dataset_drive_path / "loss_buckets"

    @property
    def checkpoint_file(self) -> Path:
        return Path(self.artifacts_dir) / "training_runs" / self.checkpoint / "checkpoint.pt"


@dataclass(frozen=True, kw_only=True, config=_CONFIG)
class FilterRunConfig:
    dataset: str
    bucket_run: str
    bucket_files: tuple[int, ...] = ()
    artifacts_dir: str | Path = _DRIVE_ARTIFACTS
    local_artifacts_dir: str | Path = _LOCAL_ARTIFACTS

    @property
    def dataset_drive_path(self) -> Path:
        return _dataset_path(self.artifacts_dir, self.dataset)

    @property
    def dataset_local_path(self) -> Path:
        return _local_dataset_path(self.local_artifacts_dir, self.artifacts_dir, self.dataset_drive_path)

    @property
    def bucket_dir(self) -> Path:
        return self.dataset_local_path / "loss_buckets" / self.bucket_run

    @property
    def drive_bucket_dir(self) -> Path:
        return self.dataset_drive_path / "loss_buckets" / self.bucket_run

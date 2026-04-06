from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class SplitConfig:
    dataset: str
    checkpoint: str
    upper_bounds: tuple[float, ...]
    csv_delimiter: str = ";"
    loss_decimal_separator: str = ","
    batch_size: int = 32
    sort_by_loss_desc: bool = False
    use_bf16: bool = False
    decode_from_loss: float | None = None
    overwrite_output: bool = False

    def __post_init__(self) -> None:
        if self.csv_delimiter not in {",", ";"}:
            raise ValueError("csv_delimiter must be ',' or ';'.")
        if self.loss_decimal_separator not in {".", ","}:
            raise ValueError("loss_decimal_separator must be '.' or ','.")
        if self.decode_from_loss is not None and self.decode_from_loss < 0:
            raise ValueError("decode_from_loss must be non-negative.")

    @property
    def dataset_drive_path(self) -> Path:
        return Path("/content/drive/MyDrive/nmt_lab/artifacts/datasets") / self.dataset

    @property
    def dataset_local_path(self) -> Path:
        return Path("/content") / "nmt_lab" / "artifacts" / self.dataset

    @property
    def output_path(self) -> Path:
        return self.dataset_local_path / "curation" / "loss_buckets"

    @property
    def drive_output_path(self) -> Path:
        return (
            Path("/content/drive/MyDrive/nmt_lab/artifacts")
            / self.dataset
            / "curation"
            / "loss_buckets"
        )

    @property
    def checkpoint_file(self) -> Path:
        return (
            Path("/content/drive/MyDrive/nmt_lab/artifacts/training_runs")
            / self.checkpoint
            / "checkpoint.pt"
        )

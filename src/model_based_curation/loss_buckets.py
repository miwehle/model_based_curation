from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

import yaml


Example = dict[str, Any]
LossFn = Callable[[Mapping[str, Any]], float]


def _load_dataset(path: str | Path):
    from datasets import load_from_disk

    return load_from_disk(str(path))


def _validate_upper_bounds(upper_bounds: Sequence[float]) -> list[float]:
    bounds = [float(bound) for bound in upper_bounds]
    if any(bound < 0 for bound in bounds):
        raise ValueError("upper_bounds must be non-negative.")
    if any(left >= right for left, right in zip(bounds, bounds[1:])):
        raise ValueError("upper_bounds must be strictly increasing.")
    return bounds


def _bucket_index(loss: float, upper_bounds: Sequence[float]) -> int:
    for index, upper_bound in enumerate(upper_bounds):
        if loss < upper_bound:
            return index
    return len(upper_bounds)


def _format_bound(bound: float) -> str:
    if bound.is_integer():
        return str(int(bound))
    return f"{bound:g}".replace("-", "m").replace(".", "_")


def _bucket_filename(bucket_index: int, upper_bounds: Sequence[float]) -> str:
    bucket_id = bucket_index + 1
    lower = 0.0 if bucket_index == 0 else upper_bounds[bucket_index - 1]
    upper = "inf" if bucket_index == len(upper_bounds) else _format_bound(upper_bounds[bucket_index])
    return f"{bucket_id:02d}_loss_{_format_bound(lower)}_to_{upper}.yaml"


def _open_bucket_files(output_dir: Path, upper_bounds: Sequence[float]) -> tuple[list[TextIO], list[Path]]:
    files: list[TextIO] = []
    paths: list[Path] = []
    for bucket_index in range(len(upper_bounds) + 1):
        path = output_dir / _bucket_filename(bucket_index, upper_bounds)
        files.append(path.open("w", encoding="utf-8"))
        paths.append(path)
    return files, paths


def split_by_loss_streaming(
    dataset_path: str | Path,
    upper_bounds: Sequence[float],
    output_dir: str | Path,
    loss_fn: LossFn,
) -> list[Path]:
    """Split a mapped Arrow dataset into YAML bucket files based on per-example loss.

    Args:
        dataset_path: Path to the Arrow dataset saved by the final
            ``data_preprocessor`` stage.
        upper_bounds: Strictly increasing finite bucket upper bounds. For
            bounds ``[l_1, ..., l_(n-1)]`` the produced intervals are
            ``[0, l_1)``, ``[l_1, l_2)``, ..., ``[l_(n-1), inf)``.
        output_dir: Directory receiving one YAML file per bucket.
        loss_fn: Callable computing a scalar loss from one dataset example.
    """
    bounds = _validate_upper_bounds(upper_bounds)
    dataset = _load_dataset(dataset_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files, paths = _open_bucket_files(out_dir, bounds)
    written = [False] * len(files)

    try:
        for row in dataset:
            loss = float(loss_fn(row))
            example: Example = dict(row)
            example["loss"] = loss
            bucket_index = _bucket_index(loss, bounds)
            yaml.safe_dump([example], files[bucket_index], sort_keys=False, allow_unicode=True)
            written[bucket_index] = True
    finally:
        for was_written, file in zip(written, files):
            if not was_written:
                file.write("[]\n")
            file.close()

    return paths


split_by_loss = split_by_loss_streaming

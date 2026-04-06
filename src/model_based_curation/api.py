from __future__ import annotations

import shutil
from pathlib import Path

from .batch_seq2seq_loss_scorer import BatchSeq2SeqLossScorer
from .config import CurationConfig
from .loss_buckets import split_by_loss_batched


def _resolve_device(device):
    import torch

    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    if str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise ValueError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _copy_dataset_if_requested(config: CurationConfig) -> Path:
    source = Path(config.dataset_path)
    target = config.resolved_dataset_path
    if config.local_dataset_dir is None:
        return source
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    return target


def _copy_buckets_to_drive(output_dir: Path, drive_dir: Path) -> None:
    drive_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*.yaml"):
        shutil.copy2(path, drive_dir / path.name)


def curate(config: CurationConfig) -> list[Path]:
    from translator.inference import Translator

    dataset_path = _copy_dataset_if_requested(config)
    output_dir = config.output_path
    _prepare_output_dir(output_dir, overwrite=config.overwrite_output)

    resolved_device = _resolve_device(config.device)
    translator = Translator.from_checkpoint(config.checkpoint_file, resolved_device)
    scorer = BatchSeq2SeqLossScorer(
        translator.model,
        device=translator.device,
        src_pad_id=translator.model.src_pad_idx,
        tgt_pad_id=translator.model.tgt_pad_idx,
    )
    output_paths = split_by_loss_batched(
        dataset_path,
        list(config.upper_bounds),
        output_dir,
        scorer,
        batch_size=config.batch_size,
    )

    if config.copy_buckets_to_drive_path is not None:
        _copy_buckets_to_drive(output_dir, config.copy_buckets_to_drive_path)
    return output_paths

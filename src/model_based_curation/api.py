from __future__ import annotations

import logging
import shutil
from dataclasses import asdict
from pathlib import Path

from lab_infrastructure import write_run_config

from .config import FilterConfig, SplitConfig
from .filter import Filter
from .split.batch_seq2seq_loss_scorer import BatchSeq2SeqLossScorer
from .split.splitter import Splitter

_LOG = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_GIT_KEY_PREFIX = "model_based_curation"


def _resolve_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fail_if_dir_exists(path: Path, *, label: str) -> None:
    if path.exists():
        raise ValueError(f"{label} already exists: {path}")


def _copy_dataset_to_local_artifacts(config: SplitConfig) -> Path:
    source = config.dataset_drive_path
    target = config.dataset_local_path
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, target / path.name)
    return target


def _copy_buckets_to_drive(output_dir: Path, drive_dir: Path) -> None:
    drive_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*"):
        if not path.is_file():
            continue
        shutil.copy2(path, drive_dir / path.name)


def _copy_dataset_to_drive(output_dir: Path, drive_dir: Path) -> None:
    shutil.copytree(output_dir, drive_dir)


def _strip_leading_token(token_ids: list[int], token_id: int | None) -> list[int]:
    if token_id is None or not token_ids or token_ids[0] != token_id:
        return token_ids
    return token_ids[1:]


def split(config: SplitConfig) -> list[Path]:
    """Split a dataset into multiple bucket files based on example loss.

    The dataset comes from ``config.dataset``, and ``config.upper_bounds`` defines
    the bucket boundaries.
    """
    output_dir = config.output_path
    drive_output_dir = config.drive_output_path
    _fail_if_dir_exists(output_dir, label="Local output directory")
    _fail_if_dir_exists(drive_output_dir, label="Drive output directory")
    from translator.inference import Translator

    dataset_path = _copy_dataset_to_local_artifacts(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_payload = {"split_config": asdict(config)}
    write_run_config(
        output_dir / "split_config.yaml", split_payload, repo_root=_REPO_ROOT, git_key_prefix=_GIT_KEY_PREFIX
    )
    _LOG.info("Preparing split for dataset %s", config.dataset)
    resolved_device = _resolve_device()
    _LOG.info("Loading checkpoint %s on device %s", config.checkpoint_file, resolved_device)
    translator = Translator.from_checkpoint(config.checkpoint_file, resolved_device)
    scorer = BatchSeq2SeqLossScorer(
        translator.model,
        device=translator.device,
        src_pad_id=translator.model.src_pad_idx,
        tgt_pad_id=translator.model.tgt_pad_idx,
        use_bf16=config.use_bf16,
    )
    output_paths = Splitter(
        config.upper_bounds,
        output_dir,
        decode_src_text=lambda token_ids: translator.tokenizer.decode(token_ids),
        decode_tgt_text=lambda token_ids: translator.tokenizer.decode(
            _strip_leading_token(token_ids, translator.tgt_bos_id)
        ),
        csv_delimiter=config.csv_delimiter,
        loss_decimal_separator=config.loss_decimal_separator,
        decode_from_loss=config.decode_from_loss,
        decode_at_least=config.decode_at_least,
        log_every_batches=config.log_every_batches,
    ).split_dataset(dataset_path, scorer, batch_size=config.batch_size)
    _LOG.info("Copying bucket files to %s", drive_output_dir)
    _copy_buckets_to_drive(output_dir, drive_output_dir)
    return output_paths


def filter(config: FilterConfig) -> Path:
    output_dir = config.output_path
    drive_output_dir = config.drive_output_path
    _fail_if_dir_exists(output_dir, label="Local output directory")
    _fail_if_dir_exists(drive_output_dir, label="Drive output directory")

    dataset_path = _copy_dataset_to_local_artifacts(config)
    bucket_paths = [config.bucket_dir / f"{bucket_file}.csv" for bucket_file in config.bucket_files]
    if not bucket_paths:
        raise ValueError(f"No bucket files found in {config.bucket_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_dir.parent / "filter.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    logging.getLogger().addHandler(handler)
    try:
        _LOG.info("Preparing filter for dataset %s", config.dataset)
        _LOG.info("Filtering %s bucket files from %s", len(bucket_paths), config.bucket_dir)
        filtered_dataset_path = Filter().filter_dataset(bucket_paths, dataset_path, output_dir)
        filter_payload = {"filter_config": asdict(config)}
        write_run_config(
            filtered_dataset_path / "filter_config.yaml",
            filter_payload,
            repo_root=_REPO_ROOT,
            git_key_prefix=_GIT_KEY_PREFIX,
        )
        _LOG.info("Copying filtered dataset to %s", drive_output_dir)
        _copy_dataset_to_drive(filtered_dataset_path, drive_output_dir)
        _LOG.info("Filter completed successfully")
        shutil.copy2(log_path, drive_output_dir / log_path.name)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    return filtered_dataset_path


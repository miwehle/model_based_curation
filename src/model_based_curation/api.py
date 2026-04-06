from __future__ import annotations

from contextlib import contextmanager
import logging
import shutil
from pathlib import Path

from .batch_seq2seq_loss_scorer import BatchSeq2SeqLossScorer
from .config import SplitConfig
from .splitter import Splitter

_LOG = logging.getLogger(__name__)
_PACKAGE_LOG = logging.getLogger("model_based_curation")
_LOG_FILE_NAME = "split.log"


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


def _copy_dataset_if_requested(config: SplitConfig) -> Path:
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
    for path in output_dir.glob("*.csv"):
        shutil.copy2(path, drive_dir / path.name)


def _copy_log_to_drive(log_path: Path, drive_dir: Path) -> None:
    drive_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(log_path, drive_dir / log_path.name)


@contextmanager
def _attach_file_logger(log_path: Path):
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    _PACKAGE_LOG.addHandler(handler)
    try:
        yield
    finally:
        handler.flush()
        handler.close()
        _PACKAGE_LOG.removeHandler(handler)


def _strip_leading_token(token_ids: list[int], token_id: int | None) -> list[int]:
    if token_id is None or not token_ids or token_ids[0] != token_id:
        return token_ids
    return token_ids[1:]


def split(config: SplitConfig) -> list[Path]:
    from translator.inference import Translator

    dataset_path = _copy_dataset_if_requested(config)
    output_dir = config.output_path
    _prepare_output_dir(output_dir, overwrite=config.overwrite_output)
    log_path = output_dir / _LOG_FILE_NAME

    with _attach_file_logger(log_path):
        _LOG.info("Preparing split for dataset %s", config.dataset_path)
        resolved_device = _resolve_device(config.device)
        _LOG.info(
            "Loading checkpoint %s on device %s", config.checkpoint_file, resolved_device
        )
        translator = Translator.from_checkpoint(config.checkpoint_file, resolved_device)
        _LOG.info("Checkpoint loaded; creating batch loss scorer")
        scorer = BatchSeq2SeqLossScorer(
            translator.model,
            device=translator.device,
            src_pad_id=translator.model.src_pad_idx,
            tgt_pad_id=translator.model.tgt_pad_idx,
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
            sort_by_loss_desc=config.sort_by_loss_desc,
        ).split_dataset(dataset_path, scorer, batch_size=config.batch_size)

        if config.copy_buckets_to_drive_path is not None:
            _LOG.info("Copying bucket files to %s", config.copy_buckets_to_drive_path)
            _copy_buckets_to_drive(output_dir, config.copy_buckets_to_drive_path)
        _LOG.info("Split completed successfully")
        if config.copy_buckets_to_drive_path is not None:
            _LOG.info(
                "Copying split log to %s", config.copy_buckets_to_drive_path / log_path.name
            )
            _copy_log_to_drive(log_path, config.copy_buckets_to_drive_path)
    return output_paths

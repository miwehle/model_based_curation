from __future__ import annotations

import csv
import logging
import sys
import types
from pathlib import Path
from uuid import uuid4

from datasets import Dataset

from model_based_curation import SplitConfig, split

_TMP_DIR = Path(__file__).resolve().parents[1] / ".local_tmp"


def _read_rows(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _temp_dir(prefix: str) -> Path:
    path = _TMP_DIR / f"{prefix}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_split_writes_log_file_and_copies_it_to_drive(monkeypatch, caplog):
    root_dir = _temp_dir("split_api")
    dataset_dir = root_dir / "dataset"
    output_dir = root_dir / "output"
    drive_dir = root_dir / "drive"
    Dataset.from_list([{"id": 1, "src_ids": [11], "tgt_ids": [99, 21, 0]}]).save_to_disk(
        str(dataset_dir)
    )

    class _FakeTokenizer:
        def decode(self, token_ids: list[int]) -> str:
            return "|".join(str(token_id) for token_id in token_ids)

    class _FakeModel:
        src_pad_idx = 0
        tgt_pad_idx = 0

    class _FakeTranslator:
        def __init__(self) -> None:
            self.model = _FakeModel()
            self.device = "cpu"
            self.tokenizer = _FakeTokenizer()
            self.tgt_bos_id = 99

        @classmethod
        def from_checkpoint(cls, checkpoint_path, device):
            del checkpoint_path, device
            return cls()

    class _FakeScorer:
        def score_batch(self, examples):
            del examples
            return [0.2]

    fake_translator_module = types.ModuleType("translator.inference")
    fake_translator_module.Translator = _FakeTranslator
    monkeypatch.setitem(sys.modules, "translator.inference", fake_translator_module)
    monkeypatch.setattr(
        "model_based_curation.api.BatchSeq2SeqLossScorer",
        lambda *args, **kwargs: _FakeScorer(),
    )

    config = SplitConfig(
        dataset_path=str(dataset_dir),
        checkpoint_path=str(root_dir / "checkpoint.pt"),
        output_dir=str(output_dir),
        upper_bounds=(0.5,),
        copy_buckets_to_drive_dir=str(drive_dir),
    )

    with caplog.at_level(logging.INFO):
        output_paths = split(config)

    local_log_path = output_dir / "split.log"
    drive_log_path = drive_dir / "split.log"

    assert output_paths[0].is_file()
    assert _read_rows(output_paths[0]) == [
        {"id": "1", "loss": "0.2", "src": "11", "tgt": "21|0"}
    ]
    assert local_log_path.is_file()
    assert drive_log_path.is_file()
    assert "Preparing split for dataset" in local_log_path.read_text(encoding="utf-8")
    assert "Split completed successfully" in drive_log_path.read_text(encoding="utf-8")
    assert any("Copying split log to" in record.getMessage() for record in caplog.records)


def test_split_can_write_german_csv_format(monkeypatch):
    root_dir = _temp_dir("split_api_german_csv")
    dataset_dir = root_dir / "dataset"
    output_dir = root_dir / "output"
    Dataset.from_list([{"id": 1, "src_ids": [11], "tgt_ids": [99, 21, 0]}]).save_to_disk(
        str(dataset_dir)
    )

    class _FakeTokenizer:
        def decode(self, token_ids: list[int]) -> str:
            return "|".join(str(token_id) for token_id in token_ids)

    class _FakeModel:
        src_pad_idx = 0
        tgt_pad_idx = 0

    class _FakeTranslator:
        def __init__(self) -> None:
            self.model = _FakeModel()
            self.device = "cpu"
            self.tokenizer = _FakeTokenizer()
            self.tgt_bos_id = 99

        @classmethod
        def from_checkpoint(cls, checkpoint_path, device):
            del checkpoint_path, device
            return cls()

    class _FakeScorer:
        def score_batch(self, examples):
            del examples
            return [0.2]

    fake_translator_module = types.ModuleType("translator.inference")
    fake_translator_module.Translator = _FakeTranslator
    monkeypatch.setitem(sys.modules, "translator.inference", fake_translator_module)
    monkeypatch.setattr(
        "model_based_curation.api.BatchSeq2SeqLossScorer",
        lambda *args, **kwargs: _FakeScorer(),
    )

    config = SplitConfig(
        dataset_path=str(dataset_dir),
        checkpoint_path=str(root_dir / "checkpoint.pt"),
        output_dir=str(output_dir),
        upper_bounds=(0.5,),
        csv_delimiter=";",
        loss_decimal_separator=",",
    )

    output_paths = split(config)
    assert _read_rows(output_paths[0], delimiter=";") == [
        {"id": "1", "loss": "0,2", "src": "11", "tgt": "21|0"}
    ]


def test_split_passes_bf16_setting_to_batch_scorer(monkeypatch):
    root_dir = _temp_dir("split_api_bf16")
    dataset_dir = root_dir / "dataset"
    output_dir = root_dir / "output"
    Dataset.from_list([{"id": 1, "src_ids": [11], "tgt_ids": [99, 21, 0]}]).save_to_disk(
        str(dataset_dir)
    )

    class _FakeTokenizer:
        def decode(self, token_ids: list[int]) -> str:
            return "|".join(str(token_id) for token_id in token_ids)

    class _FakeModel:
        src_pad_idx = 0
        tgt_pad_idx = 0

    class _FakeTranslator:
        def __init__(self) -> None:
            self.model = _FakeModel()
            self.device = "cuda"
            self.tokenizer = _FakeTokenizer()
            self.tgt_bos_id = 99

        @classmethod
        def from_checkpoint(cls, checkpoint_path, device):
            del checkpoint_path, device
            return cls()

    scorer_kwargs = {}

    class _FakeScorer:
        def score_batch(self, examples):
            del examples
            return [0.2]

    fake_translator_module = types.ModuleType("translator.inference")
    fake_translator_module.Translator = _FakeTranslator
    monkeypatch.setitem(sys.modules, "translator.inference", fake_translator_module)

    def _make_scorer(*args, **kwargs):
        del args
        scorer_kwargs.update(kwargs)
        return _FakeScorer()

    monkeypatch.setattr("model_based_curation.api.BatchSeq2SeqLossScorer", _make_scorer)

    split(
        SplitConfig(
            dataset_path=str(dataset_dir),
            checkpoint_path=str(root_dir / "checkpoint.pt"),
            output_dir=str(output_dir),
            upper_bounds=(0.5,),
            use_bf16=True,
        )
    )

    assert scorer_kwargs["use_bf16"] is True

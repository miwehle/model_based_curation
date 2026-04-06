from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

import yaml
from datasets import Dataset

from model_based_curation.loss_buckets import split_by_loss_batched


_TMP_DIR = Path(__file__).resolve().parents[1] / ".local_tmp"


def _temp_dir(prefix: str) -> Path:
    path = _TMP_DIR / f"{prefix}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class _FakeBatchScorer:
    def __init__(self) -> None:
        self.seen_batches: list[list[int]] = []

    def score_batch(self, examples: list[Mapping[str, object]]) -> list[float]:
        ids = [int(example["id"]) for example in examples]
        self.seen_batches.append(ids)
        mapping = {1: 0.2, 2: 0.9, 3: 2.3}
        return [mapping[ex_id] for ex_id in ids]


def test_split_by_loss_batched_scores_and_writes_in_batches():
    dataset_dir = _temp_dir("mapped_dataset_batched")
    output_dir = _temp_dir("bucket_output_batched")
    ds = Dataset.from_list([
        {"id": 1, "src_ids": [11, 12], "tgt_ids": [21, 22], "src_text": "eins", "tgt_text": "one"},
        {"id": 2, "src_ids": [13], "tgt_ids": [23], "src_text": "zwei", "tgt_text": "two"},
        {"id": 3, "src_ids": [14, 15, 16], "tgt_ids": [24], "src_text": "drei", "tgt_text": "three"},
    ])
    ds.save_to_disk(str(dataset_dir))
    scorer = _FakeBatchScorer()

    output_paths = split_by_loss_batched(
        dataset_dir,
        [0.5, 1.5],
        output_dir,
        scorer,
        batch_size=2,
    )

    assert scorer.seen_batches == [[1, 2], [3]]
    assert [path.name for path in output_paths] == [
        "01_loss_0_to_0_5.yaml",
        "02_loss_0_5_to_1_5.yaml",
        "03_loss_1_5_to_inf.yaml",
    ]

    bucket_1 = yaml.safe_load(output_paths[0].read_text(encoding="utf-8"))
    bucket_2 = yaml.safe_load(output_paths[1].read_text(encoding="utf-8"))
    bucket_3 = yaml.safe_load(output_paths[2].read_text(encoding="utf-8"))

    assert [example["id"] for example in bucket_1] == [1]
    assert bucket_1[0]["loss"] == 0.2
    assert [example["id"] for example in bucket_2] == [2]
    assert [example["id"] for example in bucket_3] == [3]

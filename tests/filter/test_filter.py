from __future__ import annotations

import csv
from pathlib import Path
from uuid import uuid4

import yaml
from datasets import load_from_disk

from model_based_curation.filter import Filter

_TMP_DIR = Path(__file__).resolve().parents[1] / ".local_tmp"


def _temp_dir(prefix: str) -> Path:
    path = _TMP_DIR / f"{prefix}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_bucket(path: Path, rows: list[dict[str, str]], *, delimiter: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "keep", "loss", "src", "tgt"), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def test_filter_filters_examples_by_bucket_ids_and_updates_manifest():
    root_dir = _temp_dir("filter_dataset")
    dataset_dir = root_dir / "dataset"
    output_dir = root_dir / "filtered"
    bucket_1 = root_dir / "01_loss.csv"
    bucket_2 = root_dir / "02_loss.csv"

    source_rows = [
        {"id": 1, "src_ids": [11], "tgt_ids": [21]},
        {"id": 2, "src_ids": [12], "tgt_ids": [22]},
        {"id": 3, "src_ids": [13], "tgt_ids": [23]},
        {"id": 4, "src_ids": [14], "tgt_ids": [24]},
    ]
    from datasets import Dataset

    Dataset.from_list(source_rows).save_to_disk(str(dataset_dir))
    (dataset_dir / "preprocess_config.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    with (dataset_dir / "dataset_manifest.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id_field": "id",
                "src_field": "src_ids",
                "tgt_field": "tgt_ids",
                "num_examples": 4,
            },
            handle,
            sort_keys=False,
            allow_unicode=True,
        )

    _write_bucket(bucket_1, [{"id": "2", "keep": "", "loss": "2,1", "src": "12", "tgt": "22"}], delimiter=";")
    _write_bucket(
        bucket_2,
        [
            {"id": "4", "keep": "", "loss": "3.4", "src": "14", "tgt": "24"},
            {"id": "2", "keep": "", "loss": "2.1", "src": "12", "tgt": "22"},
        ],
        delimiter=",",
    )

    result_path = Filter().filter_dataset([bucket_1, bucket_2], dataset_dir, output_dir)

    filtered = load_from_disk(str(result_path))
    assert [int(row["id"]) for row in filtered] == [1, 3]
    assert (output_dir / "dataset_info.json").is_file()
    assert (output_dir / "state.json").is_file()
    assert (output_dir / "preprocess_config.yaml").read_text(encoding="utf-8") == ("schema_version: 1\n")
    with (output_dir / "dataset_manifest.yaml").open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    assert manifest["num_examples"] == 2
    assert manifest["id_field"] == "id"


def test_filter_keeps_dataset_unchanged_when_no_bucket_ids_match():
    root_dir = _temp_dir("filter_no_matches")
    dataset_dir = root_dir / "dataset"
    output_dir = root_dir / "filtered"
    bucket_path = root_dir / "01_loss.csv"

    from datasets import Dataset

    Dataset.from_list(
        [{"id": 10, "src_ids": [11], "tgt_ids": [21]}, {"id": 20, "src_ids": [12], "tgt_ids": [22]}]
    ).save_to_disk(str(dataset_dir))
    _write_bucket(
        bucket_path, [{"id": "30", "keep": "", "loss": "1,5", "src": "13", "tgt": "23"}], delimiter=";"
    )

    Filter().filter_dataset([bucket_path], dataset_dir, output_dir)

    filtered = load_from_disk(str(output_dir))
    assert [int(row["id"]) for row in filtered] == [10, 20]


def test_filter_keeps_examples_with_non_empty_keep_marker():
    root_dir = _temp_dir("filter_keep_marker")
    dataset_dir = root_dir / "dataset"
    output_dir = root_dir / "filtered"
    bucket_path = root_dir / "01_loss.csv"

    from datasets import Dataset

    Dataset.from_list(
        [{"id": 10, "src_ids": [11], "tgt_ids": [21]}, {"id": 20, "src_ids": [12], "tgt_ids": [22]}]
    ).save_to_disk(str(dataset_dir))
    _write_bucket(
        bucket_path,
        [
            {"id": "10", "keep": "x", "loss": "1,5", "src": "11", "tgt": "21"},
            {"id": "20", "keep": "   ", "loss": "1,8", "src": "12", "tgt": "22"},
        ],
        delimiter=";",
    )

    Filter().filter_dataset([bucket_path], dataset_dir, output_dir)

    filtered = load_from_disk(str(output_dir))
    assert [int(row["id"]) for row in filtered] == [10]

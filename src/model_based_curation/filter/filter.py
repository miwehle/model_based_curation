from __future__ import annotations

import csv
import shutil
from collections.abc import Sequence
from pathlib import Path

import yaml


class Filter:
    def filter_dataset(
        self, bucket_paths: Sequence[Path], dataset_path: str | Path, output_path: str | Path
    ) -> Path:
        from datasets import load_from_disk

        removed_ids = self._load_removed_ids(bucket_paths)
        source_dataset_path = Path(dataset_path)
        target_dataset_path = Path(output_path)
        ds = load_from_disk(str(source_dataset_path))
        kept_indices = self._kept_indices(removed_ids, ds)
        ds.select(kept_indices).save_to_disk(str(target_dataset_path))
        self._copy_metadata(source_dataset_path, target_dataset_path, num_examples=len(kept_indices))
        return target_dataset_path

    def _load_removed_ids(self, bucket_paths: Sequence[Path]) -> list[int]:
        ids: list[int] = []
        for bucket_path in bucket_paths:
            with Path(bucket_path).open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter=self._resolve_csv_delimiter(handle))
                ids.extend(int(row["id"]) for row in reader if not self._is_kept_bucket_row(row))
        return sorted(set(ids))

    def _is_kept_bucket_row(self, row: dict[str, str]) -> bool:
        return bool((row.get("keep") or "").strip())

    def _resolve_csv_delimiter(self, handle) -> str:
        sample = handle.read(1024)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        except csv.Error:
            return ";"
        return dialect.delimiter

    def _kept_indices(self, removed_ids: Sequence[int], ds) -> list[int]:
        kept: list[int] = []
        i = 0
        n = len(removed_ids)
        for index, example in enumerate(ds):
            example_id = int(example["id"])
            while i < n and removed_ids[i] < example_id:
                i += 1
            if i < n and removed_ids[i] == example_id:
                continue
            kept.append(index)
        return kept

    def _copy_metadata(
        self, source_dataset_path: Path, target_dataset_path: Path, *, num_examples: int
    ) -> None:
        for path in source_dataset_path.iterdir():
            if not path.is_file() or self._is_hf_dataset_file(path):
                continue
            if path.name == "dataset_manifest.yaml":
                self._copy_dataset_manifest(path, target_dataset_path / path.name, num_examples)
                continue
            shutil.copy2(path, target_dataset_path / path.name)

    def _copy_dataset_manifest(self, source_path: Path, target_path: Path, num_examples: int) -> None:
        with source_path.open("r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle) or {}
        manifest["num_examples"] = int(num_examples)
        with target_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(manifest, handle, sort_keys=False, allow_unicode=True)

    def _is_hf_dataset_file(self, path: Path) -> bool:
        return (
            path.name in {"dataset_info.json", "state.json"}
            or path.suffix == ".arrow"
            or path.name.startswith("data-")
        )

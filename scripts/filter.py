from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )


def main() -> int:
    from model_based_curation import FilterConfig, filter

    _configure_logging()
    if len(sys.argv) != 2:
        print("Usage: python scripts/filter.py <config-path>")
        return 1

    config_path = Path(sys.argv[1])
    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"Failed to load config: {exc}")
        return 1

    try:
        filter(FilterConfig(**{**cfg, "bucket_files": tuple(cfg.get("bucket_files") or ())}))
    except Exception as exc:
        print(f"Filter failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

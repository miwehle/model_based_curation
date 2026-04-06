from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
TRANSLATOR_SRC_DIR = REPO_ROOT.parent / "translator" / "src"
if TRANSLATOR_SRC_DIR.is_dir() and str(TRANSLATOR_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSLATOR_SRC_DIR))


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )


def main() -> int:
    from model_based_curation import SplitConfig, split

    _configure_logging()
    if len(sys.argv) != 2:
        print("Usage: python scripts/split.py <config-path>")
        return 1

    config_path = Path(sys.argv[1])
    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"Failed to load config: {exc}")
        return 1

    try:
        raw_upper_bounds = cfg.get("upper_bounds") or ()
        split_config = SplitConfig(
            **{**cfg, "upper_bounds": tuple(float(x) for x in raw_upper_bounds)}
        )
        split(split_config)
    except Exception as exc:
        print(f"Split failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

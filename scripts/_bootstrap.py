from __future__ import annotations

import logging
import sys
from pathlib import Path


def add_src_dirs(script_file: str) -> None:
    """Add local src directories so direct script execution can resolve imports."""
    repo_root = Path(script_file).resolve().parents[1]
    src_dirs = (
        repo_root / "src",
        repo_root.parent / "lab_infrastructure" / "src",
        repo_root.parent / "translator" / "src",
    )
    for src_dir in src_dirs:
        if src_dir.is_dir() and str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )


from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_SHARED_SRC_DIR = Path(__file__).resolve().parent.parent / "nmt_lab_shared" / "src"
if str(_SHARED_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_SRC_DIR))

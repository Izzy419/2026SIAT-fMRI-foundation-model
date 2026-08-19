"""Portable path configuration for the SLIM/JEPA experiments.

All paths can be overridden with environment variables.  Defaults are kept
inside ``SLIM_JEPA/data`` so that the scripts do not depend on a particular
server layout.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def _path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


DATA_DIR = _path("SLIM_NSD_DATA", PROJECT_DIR / "data")
NSD_MIN_DIR = _path("SLIM_NSD_MIN_DATA", DATA_DIR / "nsd_subj01_min")
RAW_BOLD_DIR = _path("SLIM_NSD_RAW_BOLD", DATA_DIR / "rawdata_sub01")
CHECKPOINT_PATH = _path(
    "SLIM_BRAIN_CHECKPOINT", PROJECT_DIR / "checkpoints" / "best_model.pth"
)

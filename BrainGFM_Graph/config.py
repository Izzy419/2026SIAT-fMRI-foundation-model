"""Portable path configuration for the BrainGFM graph experiments."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def _path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


DATA_DIR = _path("BRAINGFM_NSD_DATA", PROJECT_DIR / "data")
PROCESSED_DIR = _path("BRAINGFM_PROCESSED_DATA", DATA_DIR / "processed")
MODEL_DIR = PROJECT_DIR / "model"
NSD_MASK_PATH = _path(
    "NSD_MASK_PATH", DATA_DIR / "nsd_subj01_min" / "nsdgeneral.nii.gz"
)
WARP_PATH = _path(
    "NSD_MNI_TO_FUNC_WARP", DATA_DIR / "R1" / "MNI-to-func1pt8.nii.gz"
)
ATLAS_CACHE_DIR = _path("BRAINGFM_ATLAS_CACHE", DATA_DIR / "atlas_cache")

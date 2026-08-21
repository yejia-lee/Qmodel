"""
Central configuration for paths that live outside the repository
(shared clinical data on HPC, model checkpoints) or are otherwise
account/node-specific.

All values can be overridden via environment variables so this repo
runs unmodified on any account/node/local clone:

    export QMODEL_DATA_PATH=/path/to/mds_ed.csv
    export QMODEL_CKPT_ROOT=/path/to/checkpoints

Everything else (script outputs under results/, csv/, source imports
under src/) is resolved relative to each script's own location or to
this file's location, so it is portable by construction and does not
need overriding.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = os.environ.get(
    "QMODEL_DATA_PATH",
    "/user/gaad2403/MDS-ED/src/data/memmap/mds_ed.csv",
)

CKPT_ROOT = os.environ.get(
    "QMODEL_CKPT_ROOT",
    str(PROJECT_ROOT / "checkpoints"),
)

SRC_DIR = str(PROJECT_ROOT / "src")

"""Central place for paths to the real (private, not-committed) datasets on disk.

By default this assumes a `data/` folder at the repo root (see the top-level README for
what goes in it). Override with the SCG_DATA_ROOT environment variable to point at data
stored somewhere else, without editing any of the run scripts.
"""
from __future__ import annotations

import os

DATA_ROOT = os.environ.get(
    "SCG_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
)

# Chelten (+ Chuck/Dasty) parquet recordings and the qdaq-gui hand-click annotations.
DATA_DIR = os.path.join(DATA_ROOT, "Training Data")
ANNOTATION_SCG_PATH = os.path.join(DATA_ROOT, "qdaq-gui-annotations", "Annotation Chelten SCG")
ANNOTATION_ECG_PATH = os.path.join(DATA_ROOT, "qdaq-gui-annotations", "Annotation Chelten")

# Unlabeled pretraining pool (autoencoder pipeline only).
EPI_BASE = os.path.join(
    DATA_ROOT,
    "epicardially-attached-cardiac-accelerometer-data-from-canines-and-porcines-1.0.0",
    "accelerometer_data",
)
HUMAN_BASE = os.path.join(DATA_ROOT, "Human-SCG-ECG-Data")

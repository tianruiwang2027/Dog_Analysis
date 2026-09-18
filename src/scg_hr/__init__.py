"""scg_hr -- dog SCG-vs-ECG heart-rate detection pipeline.

Quick tour:
  constants        frozen thresholds/hyperparameters, per-dog channel conventions
  signal_utils      Stage 1: bandpass, Shannon-energy envelope, local sharpening
  models            SCGNet / S2Net (+ proposed SCGNetFused) architectures
  scg_pipeline      Stages 1-4 end to end: candidates -> CNN confirm -> S2 recovery -> gap-aware HR
  ecg_pantompkins   fully-tuned Pan-Tompkins R-peak detector
  compare_utils     smoothing/alignment/stats helpers for comparing two HR traces
  io_hive           loading raw signals out of the project's hive parquet layout

See docs/PIPELINE_SUMMARY.md for the full narrative, validation numbers, and
the standing hard constraint: the CORAL (`coral-st`) binary's own source is
never modified by anything in this repo -- everything here is downstream,
read-only analysis of raw recordings and (optionally) CORAL's own CSV output.
"""
from . import constants, signal_utils, models, scg_pipeline, ecg_pantompkins, compare_utils, io_hive

__all__ = [
    "constants", "signal_utils", "models", "scg_pipeline",
    "ecg_pantompkins", "compare_utils", "io_hive",
]

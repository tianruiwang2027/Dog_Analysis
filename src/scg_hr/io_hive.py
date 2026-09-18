"""Loading raw signals out of the project's hive parquet layout.

Expected layout (not included in this repo -- see README.md "Data" section):
  <hive_dir>/username=<stream>/device=<dog>/stream=<n>/date=*/data_0.parquet
    username=ecg_polar   stream=0    columns: ts, c1                (Chelten only)
    username=ecg_biopac  stream=0    columns: ts, c1..c4             (Chelten, Dasty, Chuck)
    username=scg_mwd     stream=45   columns: ts, c1 (or c1/c2/c3)   (all dogs)

`ts` is int64 epoch MICROSECONDS. Column selection per stream/dog follows
constants.AXIS / constants.LEAD.
"""
import glob
import os
from pathlib import Path

import numpy as np
import polars as pl

DEFAULT_HIVE_DIR = os.environ.get("SCG_HIVE_DIR", "./hive")


def _glob_one(pattern):
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"no file matched: {pattern}")
    return matches[0]


def stream_path(username, device, stream, hive_dir=DEFAULT_HIVE_DIR):
    pattern = f"{hive_dir}/username={username}/device={device}/stream={stream}/date=*/data_0.parquet"
    return _glob_one(pattern)


def load_window(username, device, stream, column, t0=None, t1=None, pad_s=0.0,
                 hive_dir=DEFAULT_HIVE_DIR):
    """Load a (optionally time-restricted, padded) column from one hive stream.
    t0/t1 are int64 epoch microseconds; None loads the full stream."""
    p = stream_path(username, device, stream, hive_dir=hive_dir)
    df = pl.read_parquet(p, columns=["ts", column])
    ts = df["ts"].to_numpy().astype("int64")
    x = df[column].to_numpy().astype(float)
    if t0 is None and t1 is None:
        return ts, x
    a = np.searchsorted(ts, (t0 if t0 is not None else ts[0]) - int(pad_s * 1e6))
    b = np.searchsorted(ts, (t1 if t1 is not None else ts[-1]) + int(pad_s * 1e6))
    return ts[a:b], x[a:b]


def load_scg_window(dog, t0=None, t1=None, axis=None, pad_s=0.0, hive_dir=DEFAULT_HIVE_DIR):
    """Load Chelten/Dasty/Chuck's scg_mwd signal for the given axis convention.

    axis="mag" is NOT a literal parquet column -- it's the mean-centered 3-axis
    magnitude sqrt(sum((c_i - mean(c_i))**2 for c1,c2,c3)), matching CORAL's
    own coral_scg.py convention (constants.AXIS). Any other axis value (e.g.
    "c1"/"c2"/"c3") is loaded as a literal column.

    Defaults to constants.SCG_CHANNEL[dog] when axis is not given -- THIS is
    the channel this repo's SCGNet/S2Net pipeline was actually built/validated
    against (Chelten: literal "c1" -- verified to reproduce
    constants.CHELTEN_SCG_REF_MAX exactly; NOT the same as CORAL's own "mag"
    convention in constants.AXIS, which is unrelated and only relevant if
    you're comparing against CORAL's own separate output).

    Returns (ts, x, axis_used).
    """
    from . import constants as C
    axis = axis or C.SCG_CHANNEL.get(dog) or C.AXIS[dog]
    p = stream_path("scg_mwd", dog, 45, hive_dir=hive_dir)
    if axis == "mag":
        df = pl.read_parquet(p, columns=["ts", "c1", "c2", "c3"])
        ts = df["ts"].to_numpy().astype("int64")
        x = np.sqrt(sum((df[c].to_numpy().astype(float) - df[c].to_numpy().astype(float).mean()) ** 2
                         for c in ("c1", "c2", "c3")))
    else:
        df = pl.read_parquet(p, columns=["ts", axis])
        ts = df["ts"].to_numpy().astype("int64")
        x = df[axis].to_numpy().astype(float)
    if t0 is not None or t1 is not None:
        a = np.searchsorted(ts, (t0 if t0 is not None else ts[0]) - int(pad_s * 1e6))
        b = np.searchsorted(ts, (t1 if t1 is not None else ts[-1]) + int(pad_s * 1e6))
        ts, x = ts[a:b], x[a:b]
    return ts, x, axis


def full_span(username, device, stream, hive_dir=DEFAULT_HIVE_DIR):
    p = stream_path(username, device, stream, hive_dir=hive_dir)
    df = pl.read_parquet(p, columns=["ts"])
    ts = df["ts"].to_numpy().astype("int64")
    return int(ts.min()), int(ts.max())


def fs_estimate(ts):
    """Robust local sample-rate estimate (median dt). Prefer this over the
    total-span/(n-1) estimate for a stream that may contain internal gaps
    -- see docs/PIPELINE_SUMMARY.md Section 10.1 for why the naive mean/span
    estimate can be badly biased by even a handful of large gaps or
    packetization artifacts."""
    return 1e6 / np.median(np.diff(ts))

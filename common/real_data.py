"""
Loader for the real hive parquet files + qdaq-gui annotation databases.

Produces objects with the same fields `teacher/pretrain_matcher.extract_features` and
`teacher/autolabel.py` already consume (x_ecg/ts_ecg/fs_ecg, x_scg/ts_scg/fs_scg), so
the rest of the pipeline (built and tested against `teacher/synth_data.py`) runs
unchanged against real data. No hand-click information is read here except by
`load_chelten_scg_clicks`, which is used ONLY for final validation, never fed to the
matcher or the auto-labeling pipeline itself.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class RealRecording:
    dog: str
    x_ecg: np.ndarray
    ts_ecg: np.ndarray  # us, int64
    fs_ecg: float
    x_scg: np.ndarray
    ts_scg: np.ndarray  # us, int64
    fs_scg: float
    # kept for API parity with SynthRecording; real recordings have no ground truth
    true_beat_times_ecg_clock: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    true_lag_us: float = 0.0
    scg_scale: float = float("nan")
    # multi-domain pooling (teacher/multi_domain_data.py, teacher/mil_matcher.py):
    # which kind of recording this is, so a shared model can be conditioned on it
    domain: str = "dog_noninvasive"
    # a caller-trusted ECG beat-time source (e.g. a human dataset's own hand-verified
    # R-peak annotations) to use INSTEAD of running detect_qrs -- None means "run
    # detect_qrs as usual". Never hand-clicked SCG labels; this is always the
    # ECG-side anchor the whole design already treats as reliable, just sometimes
    # given to us directly rather than algorithmically detected.
    ecg_beats_hint: np.ndarray | None = None

    @property
    def duration_s(self) -> float:
        return (self.ts_ecg[-1] - self.ts_ecg[0]) / 1e6


def _ts_to_us_int64(ts: pd.Series) -> np.ndarray:
    if str(ts.dtype).startswith("datetime64"):
        return ts.astype("int64").to_numpy()
    return ts.to_numpy().astype(np.int64)


def _read_window(path: str, cols: list[str], t_lo_us: float | None, t_hi_us: float | None) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["ts"] + cols)
    ts = _ts_to_us_int64(df["ts"])
    df = df.assign(_ts_us=ts)
    if t_lo_us is not None:
        df = df[df["_ts_us"] >= t_lo_us]
    if t_hi_us is not None:
        df = df[df["_ts_us"] <= t_hi_us]
    return df


def _estimate_fs(ts_us: np.ndarray) -> float:
    dt = np.median(np.diff(ts_us[: min(len(ts_us), 20000)]))
    return 1e6 / dt


def load_ecg(path: str, col: str, t_lo_us: float | None = None, t_hi_us: float | None = None):
    df = _read_window(path, [col], t_lo_us, t_hi_us)
    ts = df["_ts_us"].to_numpy()
    x = df[col].to_numpy().astype(np.float64)
    return x, ts, _estimate_fs(ts)


def load_scg_mag(path: str, t_lo_us: float | None = None, t_hi_us: float | None = None):
    """'mag' convention (Chelten, Chuck per PIPELINE_SUMMARY.md's AXIS dict): magnitude
    of the 3-axis accelerometer. Mean-centered per axis first so the DC/gravity
    component doesn't dominate the magnitude before the Stage-1 bandpass removes it
    anyway."""
    df = _read_window(path, ["c1", "c2", "c3"], t_lo_us, t_hi_us)
    ts = df["_ts_us"].to_numpy()
    c1 = df["c1"].to_numpy().astype(np.float64)
    c2 = df["c2"].to_numpy().astype(np.float64)
    c3 = df["c3"].to_numpy().astype(np.float64)
    c1 -= c1.mean()
    c2 -= c2.mean()
    c3 -= c3.mean()
    mag = np.sqrt(c1**2 + c2**2 + c3**2)
    return mag, ts, _estimate_fs(ts)


def load_scg_axis(path: str, axis: str, t_lo_us: float | None = None, t_hi_us: float | None = None):
    """'c2' convention (Dasty per PIPELINE_SUMMARY.md's AXIS dict): one raw axis."""
    df = _read_window(path, [axis], t_lo_us, t_hi_us)
    ts = df["_ts_us"].to_numpy()
    x = df[axis].to_numpy().astype(np.float64)
    return x, ts, _estimate_fs(ts)


def load_chelten(t_lo_us: float | None, t_hi_us: float | None, data_dir: str) -> RealRecording:
    x_ecg, ts_ecg, fs_ecg = load_ecg(f"{data_dir}/ecg_polar_chelten.parquet", "c1", t_lo_us, t_hi_us)
    x_scg, ts_scg, fs_scg = load_scg_mag(f"{data_dir}/MWD_Chelten.parquet", t_lo_us, t_hi_us)
    return RealRecording("Chelten", x_ecg, ts_ecg, fs_ecg, x_scg, ts_scg, fs_scg)


def load_chuck(t_lo_us: float | None, t_hi_us: float | None, data_dir: str) -> RealRecording:
    x_ecg, ts_ecg, fs_ecg = load_ecg(f"{data_dir}/ecg_biopac_Chuck.parquet", "c4", t_lo_us, t_hi_us)
    x_scg, ts_scg, fs_scg = load_scg_mag(f"{data_dir}/MWD_Chuck.parquet", t_lo_us, t_hi_us)
    return RealRecording("Chuck", x_ecg, ts_ecg, fs_ecg, x_scg, ts_scg, fs_scg)


def load_dasty(t_lo_us: float | None, t_hi_us: float | None, data_dir: str) -> RealRecording:
    x_ecg, ts_ecg, fs_ecg = load_ecg(f"{data_dir}/ecg_Dasty_Biopac.parquet", "c4", t_lo_us, t_hi_us)
    x_scg, ts_scg, fs_scg = load_scg_axis(f"{data_dir}/MWD_Dasty.parquet", "c2", t_lo_us, t_hi_us)
    return RealRecording("Dasty", x_ecg, ts_ecg, fs_ecg, x_scg, ts_scg, fs_scg)


def load_chelten_scg_clicks(sqlite_path: str) -> np.ndarray:
    """Hand-clicked SCG peak timestamps (the empty-label rows -- 'Good/Bad Data
    Start'/'Stop' rows are data-quality region markers, not beat clicks). Used ONLY
    for final validation, per the user's instructions -- never fed to the matcher or
    the auto-labeling pipeline."""
    con = sqlite3.connect(sqlite_path)
    try:
        cur = con.cursor()
        cur.execute("select timestamp from events where label='' order by timestamp")
        return np.array([r[0] for r in cur.fetchall()], dtype=np.int64)
    finally:
        con.close()

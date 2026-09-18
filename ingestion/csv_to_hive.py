#!/usr/bin/env python3
"""Convert the dog ECG CSVs into the qsib hive-parquet layout so the
qsib-timeseries static harness (plot_qsib.py) can plot them with all its
publication rules intact. Raw stays raw: ECG_uV is written verbatim as c1.

Per-sample absolute time is built from a single wall-clock anchor
(first Host_Timestamp, treated as America/Chicago local -> UTC) plus the
exact per-sample device clock deltas (Device_Time_s), which preserves true
inter-sample timing and any acquisition gaps."""
import os, sys
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/Chicago")
UTC = ZoneInfo("UTC")
CACHE = os.path.join(os.path.dirname(__file__), "ecg_cache")
USER = "ecg-test"

SRC = {
    "Dog1": "ECG_Test_Dog1_20260625_141055.csv",
    "Dog2": "ECG_Test_Dog2_20260625_153820.csv",
}

for device, fname in SRC.items():
    path = os.path.join(os.path.dirname(__file__), "raw_data", fname)
    df = pd.read_csv(path)
    n = len(df)

    # wall-clock anchor: first host timestamp (naive local) -> UTC
    anchor_local = pd.Timestamp(df["Host_Timestamp"].iloc[0]).tz_localize(TZ)
    anchor_utc = anchor_local.tz_convert(UTC)

    dev = df["Device_Time_s"].to_numpy(dtype=float)
    rel = dev - dev[0]                      # seconds since first sample (device clock)
    ts_utc = anchor_utc.value + (rel * 1e6).round().astype(np.int64) * 1000  # ns
    ts_us = (ts_utc // 1000).astype(np.int64)                                # int64 µs UTC

    out = pd.DataFrame({"ts": ts_us, "c1": df["ECG_uV"].to_numpy()})

    # --- diagnostics ---
    dt = np.diff(rel)
    dt = dt[dt > 0]
    q1, q3 = np.percentile(dt, [25, 75])
    fs = 1.0 / dt[(dt >= q1) & (dt <= q3)].mean()
    gaps = np.where(dt > 5 * (1.0 / fs))[0]
    span_local = (anchor_local, anchor_local + pd.Timedelta(seconds=rel[-1]))
    phases = df["Phase"].astype(str)
    pchg = phases.ne(phases.shift()).to_numpy()
    pchg[0] = True
    print(f"\n[{device}] {n:,} samples  fs≈{fs:.2f} Hz  "
          f"ECG range [{out.c1.min()}, {out.c1.max()}] µV")
    print(f"  local span {span_local[0]:%H:%M:%S} -> {span_local[1]:%H:%M:%S} "
          f"(UTC date {anchor_utc:%Y%m%d})")
    print(f"  acquisition gaps (>5x period): {len(gaps)}"
          + (f"  max gap {dt.max():.2f}s" if len(gaps) else ""))
    for i in np.where(pchg)[0]:
        t = anchor_local + pd.Timedelta(seconds=rel[i])
        print(f"  phase '{phases.iloc[i]}' starts {t:%H:%M:%S} (sample {i:,})")

    part = os.path.join(CACHE, f"username={USER}", f"device={device}",
                        "stream=0", f"date={anchor_utc:%Y%m%d}")
    os.makedirs(part, exist_ok=True)
    out.to_parquet(os.path.join(part, "data_0.parquet"), index=False)
    print(f"  -> {part}/data_0.parquet")

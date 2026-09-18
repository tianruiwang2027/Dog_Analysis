#!/usr/bin/env python3
"""Convert the Polar dog ECG .xlsx exports into (a) the qsib hive-parquet layout
for plot_qsib.py and (b) a flat ts,data parquet for qdaq-gui. Same schema and
timing logic as csv_to_hive.py: raw ECG_uV is written verbatim, and a true UTC
per-sample timestamp is built from the first Host_Timestamp anchor (America/
Chicago local -> UTC) plus exact Device_Time_s deltas, so gaps stay honest."""
import os
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/Chicago")
UTC = ZoneInfo("UTC")
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "ecg_cache")
USER = "ecg-test"

# device label -> source xlsx (device label also names the hive partition)
SRC = {
    "Chelten": "ECG_Polar_Chelten_Day2_50D1_20260626_120333.xlsx",
    "Chuck":   "ECG_Polar_Chuck_Day2_7D82_20260626_143946.xlsx",
}

for device, fname in SRC.items():
    df = pd.read_excel(os.path.join(HERE, "raw_data", fname), sheet_name="in")
    n = len(df)

    anchor_local = pd.Timestamp(df["Host_Timestamp"].iloc[0]).tz_localize(TZ)
    anchor_utc = anchor_local.tz_convert(UTC)

    dev = df["Device_Time_s"].to_numpy(dtype=float)
    rel = dev - dev[0]
    ts_us = (anchor_utc.value // 1000 + (rel * 1e6).round().astype(np.int64)).astype(np.int64)

    ecg = df["ECG_uV"].to_numpy()

    # --- diagnostics ---
    dt = np.diff(rel); dt = dt[dt > 0]
    q1, q3 = np.percentile(dt, [25, 75])
    fs = 1.0 / dt[(dt >= q1) & (dt <= q3)].mean()
    gaps = np.where(dt > 5 * (1.0 / fs))[0]
    end_local = anchor_local + pd.Timedelta(seconds=rel[-1])
    phases = df["Phase"].astype(str)
    pchg = phases.ne(phases.shift()).to_numpy(); pchg[0] = True
    print(f"\n[{device}] {n:,} samples  fs≈{fs:.2f} Hz  "
          f"ECG range [{ecg.min()}, {ecg.max()}] µV")
    print(f"  local span {anchor_local:%H:%M:%S} -> {end_local:%H:%M:%S} "
          f"(UTC date {anchor_utc:%Y%m%d})")
    print(f"  acquisition gaps (>5x period): {len(gaps)}"
          + (f"  max gap {dt.max():.2f}s" if len(gaps) else ""))
    for i in np.where(pchg)[0]:
        t = anchor_local + pd.Timedelta(seconds=rel[i])
        print(f"  phase '{phases.iloc[i]}' starts {t:%H:%M:%S} (sample idx {i:,})")

    # (a) hive parquet for plot_qsib.py: ts, c1
    part = os.path.join(CACHE, f"username={USER}", f"device={device}",
                        "stream=0", f"date={anchor_utc:%Y%m%d}")
    os.makedirs(part, exist_ok=True)
    pd.DataFrame({"ts": ts_us, "c1": ecg}).to_parquet(
        os.path.join(part, "data_0.parquet"), index=False)

    # (b) flat ts,data parquet for qdaq-gui
    flat = os.path.join(HERE, os.path.splitext(fname)[0] + ".parquet")
    pd.DataFrame({"ts": ts_us, "data": ecg}).to_parquet(flat, index=False)
    print(f"  -> hive: {part}/data_0.parquet")
    print(f"  -> flat: {flat}")

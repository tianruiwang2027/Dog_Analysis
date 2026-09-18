#!/usr/bin/env python3
"""Convert BioPac ACQ text exports (multi-lead canine ECG) into the unified
qsib hive-parquet layout. Raw stays raw: every channel is written verbatim in
its native units (mV) as c1..cN. A channels.json sidecar records each channel's
name + unit so the picture/plot code knows what c1..cN mean (the lake itself
only knows c1..cN).

Timing: BioPac exports carry no per-sample clock, only a fixed sample interval
('0.5 msec/sample' -> 2000 Hz) and a wall-clock 'Recording on' anchor. The anchor
is treated as America/Chicago local -> UTC (same convention as the other
converters here), and per-sample ts is the exact nominal grid from that anchor.
"""
import os, json, re
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/Chicago")
UTC = ZoneInfo("UTC")
HERE = os.path.dirname(os.path.abspath(__file__))
HIVE = os.path.join(HERE, "hive")
USER = "ecg_biopac"

# dog -> source file (the 4-hex prefix is the MWD device suffix for that dog)
SRC = {
    "Dasty":   "0868_Dasty_BioPacECG_20260626.txt",
    "Chelten": "50D1_Chelten_BioPacECG_20260626.txt",
    "Chuck":   "7D82_Chuck_BioPacECG_20260626.txt",
}
DEVID = {"Dasty": "0868", "Chelten": "50D1", "Chuck": "7D82"}


def parse_header(path, nhead=80):
    with open(path) as f:
        head = [next(f).rstrip("\n") for _ in range(nhead)]
    rec = next(l for l in head if l.startswith("Recording on:"))
    anchor_local = pd.Timestamp(rec.split("Recording on:", 1)[1].strip()).tz_localize(TZ)
    interval = next(l for l in head if "msec/sample" in l)
    dt_ms = float(interval.split()[0])
    fs = 1000.0 / dt_ms
    nch_line = next(l for l in head if l.strip().endswith("channels"))
    nch = int(nch_line.split()[0])
    ci = head.index(nch_line) + 1                       # name/unit pairs follow
    names, units = [], []
    for k in range(nch):
        names.append(head[ci + 2 * k].strip())
        units.append(head[ci + 2 * k + 1].strip())
    chrow = next(i for i, l in enumerate(head) if re.match(r"^CH\d", l))
    sep = "\t" if "\t" in head[chrow] else ","          # delimiter varies per export
    skiprows = chrow + 2                                 # skip 'CHx,...' + 'N samples,...'
    return anchor_local, fs, nch, names, units, skiprows, sep


for dog, fname in SRC.items():
    path = os.path.join(HERE, "raw_data", fname)
    anchor_local, fs, nch, names, units, skiprows, sep = parse_header(path)
    anchor_utc = anchor_local.tz_convert(UTC)

    part = os.path.join(HIVE, f"username={USER}", f"device={dog}",
                        "stream=0", f"date={anchor_utc:%Y%m%d}")
    if os.path.exists(os.path.join(part, "data_0.parquet")):
        print(f"[{dog}] already converted -> skip")
        continue

    df = pd.read_csv(path, skiprows=skiprows, header=None, usecols=list(range(nch)), sep=sep)
    n = len(df)
    ts_us = (anchor_utc.value // 1000
             + np.round(np.arange(n, dtype=np.int64) * 1e6 / fs).astype(np.int64)).astype(np.int64)

    out = pd.DataFrame({"ts": ts_us})
    for k in range(nch):
        out[f"c{k+1}"] = df.iloc[:, k].to_numpy()

    # --- diagnostics ---
    end_local = anchor_local + pd.Timedelta(seconds=(n - 1) / fs)
    # saturation fraction per channel (BioPac doesn't rail like the int16 SCG, but
    # report extremes anyway for the picture)
    rng = {f"c{k+1}": [float(np.nanmin(df.iloc[:, k])), float(np.nanmax(df.iloc[:, k]))]
           for k in range(nch)}
    print(f"\n[{dog}] dev={DEVID[dog]}  {n:,} samples  fs={fs:.1f} Hz  {nch} channels")
    print(f"  local {anchor_local:%Y-%m-%d %H:%M:%S} -> {end_local:%H:%M:%S} "
          f"(UTC date {anchor_utc:%Y%m%d})  dur={(n-1)/fs/60:.2f} min")
    for k in range(nch):
        print(f"    c{k+1:<2} {names[k]:<18} [{units[k]}]  range {rng[f'c{k+1}']}")

    part = os.path.join(HIVE, f"username={USER}", f"device={dog}",
                        "stream=0", f"date={anchor_utc:%Y%m%d}")
    os.makedirs(part, exist_ok=True)
    out.to_parquet(os.path.join(part, "data_0.parquet"), index=False)

    meta = {
        "dog": dog, "mwd_device_suffix": DEVID[dog], "instrument": "BioPac ACQ export",
        "modality": "ECG", "fs_hz": fs, "n_samples": int(n),
        "anchor_local": str(anchor_local), "anchor_utc": str(anchor_utc),
        "tz": "America/Chicago",
        "channels": {f"c{k+1}": {"name": names[k], "unit": units[k]} for k in range(nch)},
        "ranges": rng,
        "note": ("ts is the nominal fixed-rate grid from the 'Recording on' anchor "
                 "(no per-sample clock in BioPac export). c%d is BioPac-computed HR." % nch),
    }
    with open(os.path.join(part, "channels.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  -> {part}/data_0.parquet  (+channels.json)")

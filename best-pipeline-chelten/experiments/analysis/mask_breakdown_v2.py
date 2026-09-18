#!/usr/bin/env python3
"""Same breakdown as before, but with anti-coasting AND exact-repeat-run OFF --
only hardware mask + band-specific SQI + global SQI floor remain."""
import os
import numpy as np, polars as pl
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.collections import LineCollection
import datetime

HERE = "/tmp/dog-test-ecg/dog-test-ecg-code/analysis"
SQI_THR = 0.10
ATTRACTOR_BAND = (95, 125)
BAND_THR = 0.50
UTC = datetime.timezone.utc

CATS = ["hardware mask", "weak in attractor band", "low SQI (<0.10)", "valid"]
COLORS = {"hardware mask": "#7f7f7f", "weak in attractor band": "#ff7f0e",
          "low SQI (<0.10)": "#bcbd22", "valid": "#2ca02c"}


def dn(us):
    return mdates.date2num([datetime.datetime.fromtimestamp(t/1e6, tz=UTC) for t in np.atleast_1d(us)])


def classify(path, maskpath=None):
    k = pl.read_csv(path)
    ts = k["ts"].to_numpy().astype("datetime64[us]").astype("int64")
    bpm = k["bpm"].to_numpy().astype(float); sqi = k["sqi"].to_numpy().astype(float)

    in_hw_mask = np.zeros(len(ts), bool)
    if maskpath and os.path.exists(maskpath):
        m = pl.read_parquet(maskpath)
        for a, b in zip(m["mask_start"].to_numpy().astype("datetime64[us]").astype("int64"),
                         m["mask_end"].to_numpy().astype("datetime64[us]").astype("int64")):
            in_hw_mask |= (ts >= a) & (ts <= b)

    inband = (bpm >= ATTRACTOR_BAND[0]) & (bpm <= ATTRACTOR_BAND[1])
    weak_band = inband & (sqi < BAND_THR)
    low_sqi = sqi < SQI_THR

    cat = np.full(len(ts), "valid", dtype=object)
    cat[low_sqi] = "low SQI (<0.10)"
    cat[weak_band] = "weak in attractor band"
    cat[in_hw_mask] = "hardware mask"
    return ts, bpm, sqi, cat


t0 = np.datetime64("2026-06-26T17:22:30").astype("datetime64[us]").astype(int)
t1 = np.datetime64("2026-06-26T17:26:30").astype("datetime64[us]").astype(int)
pad = 20_000_000

fig, axs = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
for ax, (label, csvp, maskp) in zip(axs, [
    ("SCG CORAL HR", f"{HERE}/coral_scg/Chelten/out.csv", f"{HERE}/coral_scg/Chelten/mask.parquet"),
    ("Polar CORAL HR", f"{HERE}/coral_ecg/Chelten_ecg_polar_maskfix/out.csv",
     f"{HERE}/coral_ecg/Chelten_ecg_polar_maskfix/mask.parquet"),
]):
    ts, bpm, sqi, cat = classify(csvp, maskp)
    m = (ts >= t0 - pad) & (ts <= t1 + pad)
    ts, bpm, cat = ts[m], bpm[m], cat[m]
    x = dn(ts)
    for c in CATS:
        sel = cat == c
        if sel.sum() == 0:
            continue
        ax.scatter(x[sel], bpm[sel], s=14, color=COLORS[c], label=c, zorder=3, edgecolor="none")
    ax.plot(x, bpm, color="0.85", lw=0.8, zorder=1)
    ax.set_ylabel(f"{label}\n(bpm)"); ax.set_ylim(40, 150)
    ax.grid(True, alpha=0.3)
    counts = {c: int((cat == c).sum()) for c in CATS}
    total = len(cat)
    breakdown = "  ".join(f"{c}: {100*n/total:.0f}%" for c, n in counts.items() if n > 0)
    ax.set_title(f"{label} -- {breakdown}", fontsize=9.5)

axs[0].legend(loc="upper right", ncol=2, fontsize=8, framealpha=0.9)
axs[1].xaxis.set_major_formatter(DateFormatter("%H:%M:%S", tz=UTC))
axs[1].set_xlabel("time (UTC)")
axs[0].set_xlim(dn(t0 - pad)[0], dn(t1 + pad)[0])
fig.suptitle("[Chelten] masked vs unmasked, NO anti-coasting / NO exact-repeat-run -- 17:22:30-17:26:30", fontsize=12)
fig.tight_layout()
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_mask_breakdown_window_v2.png"
fig.savefig(out, dpi=150)
print("->", out)

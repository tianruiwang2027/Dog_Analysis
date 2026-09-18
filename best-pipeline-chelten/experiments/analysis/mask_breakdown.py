#!/usr/bin/env python3
"""Breaks down exactly WHY each hop is masked/kept for SCG and Polar over
17:22:30-17:26:30, per-criterion, in priority order:
  1. hardware/lead-off mask (mask.parquet, fed to coral-st --mask)
  2. exact-repeat-run (frozen, bit-identical >=8 hops)
  3. anti-coasting (rolling peak-to-peak <=2bpm over 90s)
  4. weak-in-attractor-band (95-125bpm candidate but SQI<0.50)
  5. low SQI (<0.10)
  6. valid
"""
import os
import numpy as np, polars as pl
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.collections import LineCollection
from scipy.ndimage import maximum_filter1d, minimum_filter1d
import datetime

HERE = "/tmp/dog-test-ecg/dog-test-ecg-code/analysis"
SQI_THR = 0.10
ATTRACTOR_BAND = (95, 125)
BAND_THR = 0.50
UTC = datetime.timezone.utc

CATS = ["hardware mask", "frozen/repeat", "coasting (low variability)",
        "weak in attractor band", "low SQI (<0.10)", "valid"]
COLORS = {"hardware mask": "#7f7f7f", "frozen/repeat": "#9467bd",
          "coasting (low variability)": "#8c564b", "weak in attractor band": "#ff7f0e",
          "low SQI (<0.10)": "#bcbd22", "valid": "#2ca02c"}


def dn(us):
    return mdates.date2num([datetime.datetime.fromtimestamp(t/1e6, tz=UTC) for t in np.atleast_1d(us)])


def rolling_ptp(x, w):
    return maximum_filter1d(x, w, mode="nearest") - minimum_filter1d(x, w, mode="nearest")


def frozen_runs_mask(bpm, min_hops=8):
    same = np.concatenate([[False], bpm[1:] == bpm[:-1]])
    run_id = np.cumsum(~same)
    run_len = np.zeros(len(bpm))
    for rid in np.unique(run_id):
        m = run_id == rid
        run_len[m] = m.sum()
    return run_len >= min_hops


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

    frozen = frozen_runs_mask(bpm, min_hops=8)
    hopdt = np.median(np.diff(ts)) / 1e6 if len(ts) > 2 else 0.25
    w = max(5, int(round(90.0 / max(hopdt, 1e-3))) | 1)
    coasting = rolling_ptp(bpm, w) <= 2.0
    inband = (bpm >= ATTRACTOR_BAND[0]) & (bpm <= ATTRACTOR_BAND[1])
    weak_band = inband & (sqi < BAND_THR)
    low_sqi = sqi < SQI_THR

    cat = np.full(len(ts), "valid", dtype=object)
    # priority order (later overwrites earlier -> apply from lowest to highest priority)
    cat[low_sqi] = "low SQI (<0.10)"
    cat[weak_band] = "weak in attractor band"
    cat[coasting] = "coasting (low variability)"
    cat[frozen] = "frozen/repeat"
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

axs[0].legend(loc="upper right", ncol=3, fontsize=8, framealpha=0.9)
axs[1].xaxis.set_major_formatter(DateFormatter("%H:%M:%S", tz=UTC))
axs[1].set_xlabel("time (UTC)")
axs[0].set_xlim(dn(t0 - pad)[0], dn(t1 + pad)[0])
fig.suptitle("[Chelten] masked vs unmasked, by reason -- 17:22:30-17:26:30", fontsize=13)
fig.tight_layout()
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_mask_breakdown_window.png"
fig.savefig(out, dpi=150)
print("->", out)

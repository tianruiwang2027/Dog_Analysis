#!/usr/bin/env python3
"""Final SCG vs Polar CORAL HR comparison: adds a band-specific SQI floor
(SQI>=0.60 required only for candidates landing in [95,125] bpm -- coral-st's
documented noise-attractor zone) on top of the gap-bridged mask + exact-repeat-run
filter, and shows it removes the two circled outlier clusters."""
import os
import numpy as np, polars as pl
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter1d, minimum_filter1d

HERE = "/tmp/dog-test-ecg/dog-test-ecg-code/analysis"
SQI_THR = 0.10
ATTRACTOR_BAND = (95, 125)


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


def load_coral(path, maskpath=None, band_thr=None):
    k = pl.read_csv(path)
    ts = k["ts"].to_numpy().astype("datetime64[us]").astype("int64")
    bpm = k["bpm"].to_numpy().astype(float); sqi = k["sqi"].to_numpy().astype(float)
    valid = sqi >= SQI_THR
    if band_thr is not None:
        inband = (bpm >= ATTRACTOR_BAND[0]) & (bpm <= ATTRACTOR_BAND[1])
        valid &= ~(inband & (sqi < band_thr))
    hopdt = np.median(np.diff(ts)) / 1e6 if len(ts) > 2 else 0.25
    w = max(5, int(round(90.0 / max(hopdt, 1e-3))) | 1)
    valid &= rolling_ptp(bpm, w) > 2.0
    if maskpath and os.path.exists(maskpath):
        m = pl.read_parquet(maskpath)
        for a, b in zip(m["mask_start"].to_numpy().astype("datetime64[us]").astype("int64"),
                         m["mask_end"].to_numpy().astype("datetime64[us]").astype("int64")):
            valid &= ~((ts >= a) & (ts <= b))
    valid &= ~frozen_runs_mask(bpm, min_hops=8)
    return ts, bpm, sqi, valid


def align(tsA, hrA, vA, tsB, hrB, vB, win_us=2_000_000):
    bts, bhr = tsB[vB], hrB[vB]
    A = np.where(vA)[0]; xa, yb = [], []
    for i in A:
        t = tsA[i]; sel = (bts >= t - win_us) & (bts <= t + win_us)
        if sel.sum() >= 2:
            xa.append(hrA[i]); yb.append(bhr[sel].mean())
    return np.array(xa), np.array(yb)


def stats(a, b):
    d = a - b
    return dict(n=int(len(a)), bias=float(d.mean()), mae=float(np.abs(d).mean()),
                within10=float(np.mean(np.abs(d) <= 10)),
                r=float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan"))


configs = [
    ("gap-bridged mask only\n(same as last time)", None),
    ("+ SQI≥0.60 required\nin the 95–125bpm attractor band", 0.60),
]

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
for ax, (label, band_thr) in zip(axs, configs):
    scg_ts, scg_bpm, scg_sqi, scg_v = load_coral(
        f"{HERE}/coral_scg/Chelten/out.csv", f"{HERE}/coral_scg/Chelten/mask.parquet", band_thr)
    pts, pbpm, psqi, pv = load_coral(
        f"{HERE}/coral_ecg/Chelten_ecg_polar_maskfix/out.csv",
        f"{HERE}/coral_ecg/Chelten_ecg_polar_maskfix/mask.parquet", band_thr)
    xa, yb = align(scg_ts, scg_bpm, scg_v, pts, pbpm, pv)
    R = stats(xa, yb)
    ax.axvspan(*ATTRACTOR_BAND, color="orange", alpha=0.08, zorder=0)
    ax.axhspan(*ATTRACTOR_BAND, color="orange", alpha=0.08, zorder=0)
    ax.scatter(yb, xa, s=6, alpha=0.35, color="#2ca02c", edgecolor="none", zorder=2)
    lim = [40, 210]; ax.plot(lim, lim, "k-", lw=1, alpha=0.6, zorder=1)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("Polar CORAL HR"); ax.set_ylabel("SCG CORAL HR")
    ax.set_title(label, fontsize=10)
    ax.text(0.03, 0.97, f"MAE {R['mae']:.2f}  bias {R['bias']:+.2f}\n"
                          f"≤10bpm {100*R['within10']:.1f}%  r={R['r']:.3f}\nn={R['n']}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.9))
    ax.grid(True, alpha=0.3)

fig.suptitle("[Chelten] SCG vs Polar CORAL HR -- orange band = coral-st's documented\n"
             "95-125bpm noise-attractor zone (confirmed via the pure-white-noise test earlier)", fontsize=11)
fig.tight_layout()
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_final_scatter.png"
fig.savefig(out, dpi=150)
print("->", out)

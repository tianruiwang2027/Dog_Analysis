#!/usr/bin/env python3
"""Scatter of HRV from CORAL vs HRV from RRi, with agreement stats.

Reuses hrv_60s's rolling-60 s SDNN/RMSSD (RRi = Pan-Tompkins beats; CORAL = HR
sampled at the same beats). Rolling windows overlap heavily and are autocorrelated,
so for HONEST stats we collapse to ONE point per non-overlapping 60 s bin (median
of the rolling values in the bin) before correlating. Grid: rows = {RMSSD, SDNN},
cols = CORAL sources. Each panel: identity line, OLS fit, and stats (n, Pearson r,
bias = CORAL-RRi, MAE, slope, and recovered-fraction = median CORAL/RRi).

Usage: hrv_scatter.py <Dog>
"""
import sys, os
import numpy as np, polars as pl
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hrv_60s as H   # load_coral_valid, coral_at_beats, rolling_hrv, COL, HERE, FIGS

BIN_US = 60_000_000


def bin_pairs(t_us, rri, coral):
    """One (rri, coral) point per non-overlapping 60 s bin: median of finite values."""
    both = np.isfinite(rri) & np.isfinite(coral)
    if both.sum() < 4:
        return np.array([]), np.array([])
    t, r, c = t_us[both], rri[both], coral[both]
    b = ((t - t[0]) // BIN_US).astype(int)
    xs, ys = [], []
    for bb in np.unique(b):
        m = b == bb
        xs.append(np.median(r[m])); ys.append(np.median(c[m]))
    return np.array(xs), np.array(ys)


def stats(x, y):
    d = y - x
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 0 and y.std() > 0 else float("nan")
    slope = float(np.polyfit(x, y, 1)[0]) if len(x) > 2 else float("nan")
    frac = float(np.median(y / x)) if np.all(x > 0) else float(np.median(y) / np.median(x))
    return dict(n=len(x), r=r, bias=float(d.mean()), mae=float(np.abs(d).mean()),
                rmse=float(np.sqrt((d*d).mean())), slope=slope, frac=frac)


def main():
    dog = sys.argv[1]
    # --- beats (RRi ground truth), unified across modalities ---
    beats = {}
    for mod, name in [("ecg_biopac", "BioPac"), ("ecg_polar", "Polar")]:
        f = os.path.join(H.HERE, "ecg_hr", f"{dog}_{mod}.parquet")
        if os.path.exists(f):
            e = pl.read_parquet(f); t = e["ts"].to_numpy().astype("int64"); bpm = e["bpm"].to_numpy().astype(float)
            o = np.argsort(t); beats[name] = (t[o], 60000.0/bpm[o])
    bt = np.concatenate([beats[n][0] for n in beats]); bn = np.concatenate([beats[n][1] for n in beats])
    o = np.argsort(bt); bt, bn = bt[o], bn[o]
    sdnn_rri, rmssd_rri = H.rolling_hrv(bt, bn)
    rri = {"RMSSD": rmssd_rri, "SDNN": sdnn_rri}

    # --- CORAL sources sampled at the same beats ---
    coral = {}
    sp = os.path.join(H.HERE, "coral_scg", dog, "out.csv")
    if os.path.exists(sp):
        coral["SCG (MWD)"] = H.load_coral_valid(sp, os.path.join(H.HERE, "coral_scg", dog, "mask.parquet"))
    for mod, name in [("ecg_biopac", "BioPac"), ("ecg_polar", "Polar")]:
        cp = os.path.join(H.HERE, "coral_ecg", f"{dog}_{mod}", "out.csv")
        if os.path.exists(cp):
            coral[name] = H.load_coral_valid(cp, os.path.join(H.HERE, "coral_ecg", f"{dog}_{mod}", "mask.parquet"))
    chrv = {}
    for name, (cts, cbpm) in coral.items():
        nn = 60000.0 / H.coral_at_beats(bt, cts, cbpm)
        s, r = H.rolling_hrv(bt, nn); chrv[name] = {"RMSSD": r, "SDNN": s}

    srcs = list(coral)
    metrics = ["RMSSD", "SDNN"]
    lim = {m: max(1.0, np.nanpercentile(rri[m], 99)) * 1.05 for m in metrics}

    fig, axes = plt.subplots(len(metrics), len(srcs), figsize=(4.6*len(srcs), 9), squeeze=False)
    print(f"[{dog}] HRV RRi-vs-CORAL agreement (one point per 60 s bin):")
    for i, m in enumerate(metrics):
        for j, s in enumerate(srcs):
            ax = axes[i][j]
            x, y = bin_pairs(bt, rri[m], chrv[s][m])
            col = H.COL.get(s if s == "SCG (MWD)" else s.split()[0], "0.4")
            L = lim[m]
            ax.plot([0, L], [0, L], "k-", lw=1, alpha=0.5)
            if len(x):
                ax.scatter(x, y, s=18, alpha=0.6, color=col, edgecolor="none")
                sg = np.polyfit(x, y, 1); xs = np.array([0, L])
                ax.plot(xs, sg[0]*xs + sg[1], "--", color=col, lw=1.3, alpha=0.9)
                S = stats(x, y)
                ax.text(0.04, 0.96,
                        f"n={S['n']}  r={S['r']:.2f}\nbias {S['bias']:+.0f} ms  MAE {S['mae']:.0f}\n"
                        f"slope {S['slope']:.2f}  recovers {100*S['frac']:.0f}%",
                        transform=ax.transAxes, va="top", fontsize=8,
                        bbox=dict(boxstyle="round", fc="white", alpha=0.9))
                print(f"  {m:5} {s:11}: n={S['n']:3d} r={S['r']:+.2f} bias={S['bias']:+6.0f} "
                      f"MAE={S['mae']:5.0f} slope={S['slope']:+.2f} recovers={100*S['frac']:.0f}%")
            ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            if j == 0: ax.set_ylabel(f"CORAL {m} (ms)")
            ax.set_xlabel(f"RRi {m} (ms)")
            if i == 0: ax.set_title(s, fontsize=11)
    fig.suptitle(f"[{dog}] HRV: CORAL vs RRi (60 s windows) — points below the identity line = CORAL "
                 f"under-reads HRV; slope≈0 / low recovered-% = it doesn't track it", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(H.FIGS, f"hrv_scatter_{dog}.png"); fig.savefig(out, dpi=200); print("->", out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verification: SCG vs Polar CORAL HR, current best filter pipeline
(SQI>=0.10 + anti-coasting + exact-repeat-run filter + gap-bridged mask +
band-specific SQI>=0.50 in the 95-125bpm attractor zone), zoomed to a single
window: 17:22:30-17:26:30."""
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


def load_coral(path, maskpath=None):
    k = pl.read_csv(path)
    ts = k["ts"].to_numpy().astype("datetime64[us]").astype("int64")
    bpm = k["bpm"].to_numpy().astype(float); sqi = k["sqi"].to_numpy().astype(float)
    valid = sqi >= SQI_THR
    inband = (bpm >= ATTRACTOR_BAND[0]) & (bpm <= ATTRACTOR_BAND[1])
    valid &= ~(inband & (sqi < BAND_THR))
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
    A = np.where(vA)[0]; xa, yb, ta = [], [], []
    for i in A:
        t = tsA[i]; sel = (bts >= t - win_us) & (bts <= t + win_us)
        if sel.sum() >= 2:
            xa.append(hrA[i]); yb.append(bhr[sel].mean()); ta.append(t)
    return np.array(xa), np.array(yb), np.array(ta, dtype="int64")


def stats(a, b):
    d = a - b
    return dict(n=int(len(a)), bias=float(d.mean()), mae=float(np.abs(d).mean()),
                within10=float(np.mean(np.abs(d) <= 10)),
                r=float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan"))


scg_ts, scg_bpm, scg_sqi, scg_v = load_coral(f"{HERE}/coral_scg/Chelten/out.csv", f"{HERE}/coral_scg/Chelten/mask.parquet")
pts, pbpm, psqi, pv = load_coral(f"{HERE}/coral_ecg/Chelten_ecg_polar_maskfix/out.csv",
                                  f"{HERE}/coral_ecg/Chelten_ecg_polar_maskfix/mask.parquet")

t0 = np.datetime64("2026-06-26T17:22:30").astype("datetime64[us]").astype(int)
t1 = np.datetime64("2026-06-26T17:26:30").astype("datetime64[us]").astype(int)

xa, yb, ta = align(scg_ts, scg_bpm, scg_v, pts, pbpm, pv)
inwin = (ta >= t0) & (ta <= t1)
Rwin = stats(xa[inwin], yb[inwin])
Rall = stats(xa, yb)
print("window stats:", Rwin)
print("overall (for reference):", Rall)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.6, 1]})

for ts_, bpm_, sqi_, v_, color, label in [
    (scg_ts, scg_bpm, scg_sqi, scg_v, "#d62728", "SCG CORAL HR"),
    (pts, pbpm, psqi, pv, "#2ca02c", "Polar CORAL HR"),
]:
    m = (ts_ >= t0 - 30_000_000) & (ts_ <= t1 + 30_000_000)
    x = dn(ts_[m]); y = bpm_[m]; vv = v_[m]
    pts_xy = np.column_stack([x, y]); segs = np.stack([pts_xy[:-1], pts_xy[1:]], axis=1)
    a = np.where(vv[:-1] & vv[1:], 0.9, 0.15)
    c = np.array(plt.matplotlib.colors.to_rgba(color))[None, :].repeat(len(segs), 0); c[:, 3] = a
    ax0.add_collection(LineCollection(segs, colors=c, linewidths=1.8, zorder=3, label=label))

ax0.set_xlim(dn(t0)[0], dn(t1)[0]); ax0.set_ylim(40, 150)
ax0.xaxis.set_major_formatter(DateFormatter("%H:%M:%S", tz=UTC))
ax0.set_ylabel("Heart rate (bpm)"); ax0.set_xlabel("time (UTC)")
ax0.legend(loc="upper right"); ax0.grid(True, alpha=0.3)
ax0.set_title("17:22:30–17:26:30 -- solid=valid (passes filter), faint=invalid/excluded")

ax1.scatter(yb[inwin], xa[inwin], s=22, alpha=0.6, color="#1f77b4", edgecolor="none")
lim = [40, 150]; ax1.plot(lim, lim, "k-", lw=1, alpha=0.6)
ax1.set_xlim(lim); ax1.set_ylim(lim); ax1.set_aspect("equal")
ax1.set_xlabel("Polar CORAL HR"); ax1.set_ylabel("SCG CORAL HR")
ax1.set_title("paired points in this window", fontsize=10)
ax1.text(0.03, 0.97, f"MAE {Rwin['mae']:.2f}  bias {Rwin['bias']:+.2f}\n"
                       f"≤10bpm {100*Rwin['within10']:.1f}%  r={Rwin['r']:.3f}\nn={Rwin['n']}",
         transform=ax1.transAxes, va="top", fontsize=10,
         bbox=dict(boxstyle="round", fc="white", alpha=0.9))
ax1.grid(True, alpha=0.3)

fig.suptitle("[Chelten] SCG vs Polar CORAL HR -- current best filter pipeline, one 4-minute window", fontsize=12)
fig.tight_layout()
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_window_17_22_30_to_17_26_30.png"
fig.savefig(out, dpi=150)
print("->", out)

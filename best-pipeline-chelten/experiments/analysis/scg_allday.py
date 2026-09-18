#!/usr/bin/env python3
"""SCG heart rate over the full recording session (ECG only over a short
validation window). Row 1: SCG-CORAL HR over the whole recording (opacity ∝ SQI,
gaps honest, invalid blanked); the ECG-validated window is shaded gold with the
ECG RRi HR overlaid as ground truth. Row 2: MWD tri-axial accel activity (60 s
RMS, dB). Reads coral_scg/<Dog>_full/. Usage: scg_allday.py <Dog>"""
import sys, os, datetime
import numpy as np, polars as pl
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from scipy.ndimage import median_filter
import compare_all as C
import combined_hrv_figure as CF

dog = sys.argv[1]
AN = C.HERE
sp = os.path.join(AN, "coral_scg", dog + "_full", "out.csv")
mp = os.path.join(AN, "coral_scg", dog + "_full", "mask.parquet")
ts, bpm, sqi, valid = C.load_coral(sp, mp)

# ECG beats (validation window)
bts, bbs = [], []
for mod in ("ecg_biopac", "ecg_polar"):
    f = os.path.join(AN, "ecg_hr", f"{dog}_{mod}.parquet")
    if os.path.exists(f):
        e = pl.read_parquet(f); bts.append(e["ts"].to_numpy().astype("int64")); bbs.append(e["bpm"].to_numpy().astype(float))
bt = np.concatenate(bts) if bts else np.array([], "int64")
bbpm = np.concatenate(bbs) if bbs else np.array([])
if len(bt):
    o = np.argsort(bt); bt, bbpm = bt[o], bbpm[o]

tmm, dbm = CF.motion_db(dog)

fig = plt.figure(figsize=(16, 9), layout="constrained")
gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.0])
ax = fig.add_subplot(gs[0]); axm = fig.add_subplot(gs[1], sharex=ax)
x0, x1 = C.dn(ts.min())[0], C.dn(ts.max())[0]

# ECG-validated window (gold) + RRi HR ground truth
if len(bt):
    ax.axvspan(C.dn(bt.min())[0], C.dn(bt.max())[0], color="gold", alpha=0.18, zorder=0, label="ECG-validated window")
    CF.seg(ax, bt, median_filter(bbpm, 5), "black", lw=1.3, zorder=4, label="ECG RRi HR")

# SCG-CORAL HR — opacity ∝ SQI, invalid blanked (gaps/coast/motion honest)
xs = C.dn(ts); pts = np.column_stack([xs, bpm]); segs = np.stack([pts[:-1], pts[1:]], 1)
a = np.where(valid[:-1], 0.30 + 0.70 * np.clip(sqi[:-1] / 0.25, 0, 1), 0.0)
col = np.array(to_rgba(C.COL["SCG (MWD)"]))[None, :].repeat(len(segs), 0); col[:, 3] = a
ax.add_collection(LineCollection(segs, colors=col, linewidths=1.6, zorder=5, label="SCG-CORAL HR (opacity ∝ SQI)"))
ax.set_ylim(40, 210); ax.set_xlim(x0, x1); ax.set_ylabel("heart rate (bpm)")
ax.grid(True, alpha=0.3); ax.legend(loc="upper right", ncol=3, framealpha=0.95, fontsize=9); ax.tick_params(labelbottom=False)
ax.set_title(f"[{dog}] — SCG heart rate, full session  ·  valid HR on {100*valid.mean():.0f}% of hops",
             fontsize=12)

# activity
if tmm is not None:
    floor = float(np.nanpercentile(dbm, 1)); top = float(np.nanpercentile(dbm, 99.7)) + 1
    axm.fill_between(C.dn(tmm), floor, dbm, color="#b3873a", alpha=0.45, lw=0)
    axm.plot(C.dn(tmm), dbm, color="#6b4f1d", lw=0.7); axm.set_ylim(floor, top)
axm.set_ylabel("MWD accel\nRMS, 60 s (dB)", fontsize=9); axm.grid(True, alpha=0.3)
axm.xaxis.set_major_formatter(DateFormatter("%H:%M", tz=C.TZ))
date_str = datetime.datetime.fromtimestamp(ts.min() / 1e6, tz=datetime.timezone.utc).astimezone(C.TZ).strftime("%Y-%m-%d")
axm.set_xlabel(f"America/Chicago local time · {date_str}")

out = os.path.join(C.FIGS, f"scg_allday_{dog}.png"); fig.savefig(out, dpi=200); print("->", out)

#!/usr/bin/env python3
"""17:24:20-17:26:30 -- put ALL FOUR comparisons together:
  1. CORAL SCG   vs CORAL ECG   (algorithmic pipeline, current best filters)
  2. Hand SCG    vs Hand ECG    (3s-smoothed hand-clicked beats, joint-good)
  3. CORAL SCG   vs Hand SCG    (does the algorithm track the human on the SCG channel?)
  4. CORAL ECG   vs Hand ECG    (does the algorithm track the human on the ECG channel?)
"""
import os, sqlite3
import numpy as np, polars as pl
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from scipy.ndimage import maximum_filter1d, minimum_filter1d
import datetime

HERE = "/tmp/dog-test-ecg/dog-test-ecg-code/analysis"
SQI_THR = 0.10
ATTRACTOR_BAND = (95, 125)
BAND_THR = 0.50
UTC = datetime.timezone.utc
SMOOTH_S = 3.0

t0 = np.datetime64("2026-06-26T17:24:20").astype("datetime64[us]").astype(int)
t1 = np.datetime64("2026-06-26T17:26:30").astype("datetime64[us]").astype(int)


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
    return ts, bpm, valid


# ---- CORAL algorithmic series ----
scg_ts, scg_bpm, scg_v = load_coral(f"{HERE}/coral_scg/Chelten/out.csv", f"{HERE}/coral_scg/Chelten/mask.parquet")
ecg_ts, ecg_bpm, ecg_v = load_coral(f"{HERE}/coral_ecg/Chelten_ecg_polar_maskfix/out.csv",
                                     f"{HERE}/coral_ecg/Chelten_ecg_polar_maskfix/mask.parquet")

# ---- hand-annotated peaks ----
con = sqlite3.connect("/tmp/annotation_chelten_ecg.db")
cur = con.cursor()
cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = '' ORDER BY timestamp")
h_ecg_pk = np.array([r[0] for r in cur.fetchall()], dtype="int64")

con2 = sqlite3.connect("/tmp/annotation_chelten_scg.db")
cur2 = con2.cursor()
cur2.execute("SELECT DISTINCT timestamp FROM events WHERE label = '' ORDER BY timestamp")
h_scg_pk = np.array([r[0] for r in cur2.fetchall()], dtype="int64")
cur2.execute("SELECT DISTINCT timestamp, label FROM events WHERE label != '' ORDER BY timestamp")
scg_events = cur2.fetchall()
scg_ev_ts = np.array([r[0] for r in scg_events], dtype="int64")
scg_ev_lab = [r[1].lower() for r in scg_events]


def scg_hand_good(ts):
    good = np.zeros(len(ts), bool)
    for i in range(len(scg_ev_ts)):
        if "good" in scg_ev_lab[i] and "start" in scg_ev_lab[i]:
            t0_, t1_ = scg_ev_ts[i], (scg_ev_ts[i + 1] if i + 1 < len(scg_ev_ts) else scg_ev_ts[i])
            good |= (ts >= t0_) & (ts < t1_)
    return good


def beat_hr(peaks):
    rr = np.diff(peaks) / 1e6
    hr = 60.0 / rr
    tmid = peaks[:-1] + np.diff(peaks) // 2
    return tmid, hr


def smooth(tmid, hr, win_s=SMOOTH_S):
    win_us = int(win_s * 1e6)
    out = np.empty(len(tmid))
    for i, t in enumerate(tmid):
        sel = (tmid >= t - win_us) & (tmid <= t + win_us)
        out[i] = hr[sel].mean()
    return out


h_ecg_tmid, h_ecg_hr_raw = beat_hr(h_ecg_pk)
h_scg_tmid, h_scg_hr_raw = beat_hr(h_scg_pk)
h_ecg_hr = smooth(h_ecg_tmid, h_ecg_hr_raw)
h_scg_hr = smooth(h_scg_tmid, h_scg_hr_raw)
h_ecg_v = (h_ecg_tmid >= h_ecg_pk.min()) & (h_ecg_tmid <= h_ecg_pk.max())
h_scg_v = scg_hand_good(h_scg_tmid)


def align(tsA, hrA, vA, tsB, hrB, vB, win_us=2_000_000):
    bts, bhr = tsB[vB], hrB[vB]
    A = np.where(vA)[0]; xa, yb, ta = [], [], []
    for i in A:
        t = tsA[i]; sel = (bts >= t - win_us) & (bts <= t + win_us)
        if sel.sum() >= 1:
            xa.append(hrA[i]); yb.append(bhr[sel].mean()); ta.append(t)
    return np.array(xa), np.array(yb), np.array(ta, dtype="int64")


def stats(a, b):
    d = a - b
    return dict(n=int(len(a)), bias=float(d.mean()), mae=float(np.abs(d).mean()),
                within10=float(np.mean(np.abs(d) <= 10)),
                r=float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan"))


results = {}

# 1. CORAL SCG vs CORAL ECG
xa, yb, ta = align(scg_ts, scg_bpm, scg_v, ecg_ts, ecg_bpm, ecg_v)
inw = (ta >= t0) & (ta <= t1)
results["CORAL SCG vs CORAL ECG"] = (xa[inw], yb[inw], stats(xa[inw], yb[inw]))

# 2. Hand SCG vs Hand ECG (smoothed, joint-good)
xa, yb, ta = align(h_scg_tmid, h_scg_hr, h_scg_v, h_ecg_tmid, h_ecg_hr, h_ecg_v)
inw = (ta >= t0) & (ta <= t1)
results["Hand SCG vs Hand ECG"] = (xa[inw], yb[inw], stats(xa[inw], yb[inw]))

# 3. CORAL SCG vs Hand SCG (restricted to scg-hand-good)
xa, yb, ta = align(scg_ts, scg_bpm, scg_v, h_scg_tmid, h_scg_hr, h_scg_v)
inw = (ta >= t0) & (ta <= t1)
results["CORAL SCG vs Hand SCG"] = (xa[inw], yb[inw], stats(xa[inw], yb[inw]))

# 4. CORAL ECG vs Hand ECG
xa, yb, ta = align(ecg_ts, ecg_bpm, ecg_v, h_ecg_tmid, h_ecg_hr, h_ecg_v)
inw = (ta >= t0) & (ta <= t1)
results["CORAL ECG vs Hand ECG"] = (xa[inw], yb[inw], stats(xa[inw], yb[inw]))

print(f"{'comparison':28s}  n     MAE    bias    within10   r")
for label, (xa, yb, R) in results.items():
    print(f"{label:28s}  {R['n']:4d}  {R['mae']:6.2f}  {R['bias']:+6.2f}  {100*R['within10']:6.1f}%   {R['r']:.3f}")

# ---- figure ----
fig = plt.figure(figsize=(17, 11))
gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 1.0], hspace=0.4, wspace=0.35)
axT = fig.add_subplot(gs[0, :])
pad = 5_000_000
mS = (scg_ts >= t0 - pad) & (scg_ts <= t1 + pad)
mE = (ecg_ts >= t0 - pad) & (ecg_ts <= t1 + pad)
axT.plot(dn(scg_ts[mS]), scg_bpm[mS], color="0.85", lw=1, zorder=1)
axT.plot(dn(ecg_ts[mE]), ecg_bpm[mE], color="0.85", lw=1, zorder=1)
SvC = scg_v & mS; EvC = ecg_v & mE
axT.plot(dn(scg_ts[SvC]), scg_bpm[SvC], color="#d62728", lw=1.6, alpha=0.85, label="CORAL SCG (filtered)", zorder=3)
axT.plot(dn(ecg_ts[EvC]), ecg_bpm[EvC], color="#2ca02c", lw=1.6, alpha=0.85, label="CORAL Polar/ECG (filtered)", zorder=3)
mHS = (h_scg_tmid >= t0 - pad) & (h_scg_tmid <= t1 + pad) & h_scg_v
mHE = (h_ecg_tmid >= t0 - pad) & (h_ecg_tmid <= t1 + pad) & h_ecg_v
axT.scatter(dn(h_scg_tmid[mHS]), h_scg_hr[mHS], s=14, color="#8c1515", marker="D", zorder=4, label="Hand SCG (smoothed, good)")
axT.scatter(dn(h_ecg_tmid[mHE]), h_ecg_hr[mHE], s=14, color="#145214", marker="D", zorder=4, label="Hand ECG (smoothed, good)")
axT.set_xlim(dn(t0 - pad)[0], dn(t1 + pad)[0]); axT.set_ylim(40, 130)
axT.xaxis.set_major_formatter(DateFormatter("%H:%M:%S", tz=UTC))
axT.set_ylabel("HR (bpm)"); axT.legend(loc="upper right", ncol=2, fontsize=9)
axT.grid(True, alpha=0.3)
axT.set_title("CORAL algorithmic (lines) vs hand-annotated (diamonds), 17:24:20-17:26:30")

colors = ["#1f77b4", "#9467bd", "#ff7f0e", "#17becf"]
for i, (label, (xa, yb, R)) in enumerate(results.items()):
    ax = fig.add_subplot(gs[1, i])
    ax.scatter(yb, xa, s=18, alpha=0.6, color=colors[i], edgecolor="none")
    lim = [40, 130]; ax.plot(lim, lim, "k-", lw=1, alpha=0.6)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    parts = label.split(" vs ")
    ax.set_xlabel(parts[1]); ax.set_ylabel(parts[0])
    ax.set_title(label, fontsize=10)
    ax.text(0.03, 0.97, f"MAE {R['mae']:.2f}  bias {R['bias']:+.2f}\n"
                          f"≤10bpm {100*R['within10']:.1f}%  r={R['r']:.3f}\nn={R['n']}",
            transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="white", alpha=0.9))
    ax.grid(True, alpha=0.3)

fig.suptitle("[Chelten] CORAL algorithmic vs hand-annotated, all 4 comparisons -- 17:24:20-17:26:30", fontsize=13)
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_all_compare_window2.png"
fig.savefig(out, dpi=150)
print("->", out)

#!/usr/bin/env python3
"""Test the human-annotated good/bad segments (Chelten, 17:22:22-17:26:37) as a
validity mask applied to BOTH SCG and Polar, for the same window comparison."""
import os, sqlite3
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


# --- build the human good-mask function from the annotation db ---
con = sqlite3.connect("/tmp/annotation_chelten.db")
cur = con.cursor()
cur.execute("SELECT DISTINCT timestamp, label FROM events ORDER BY timestamp")
rows = cur.fetchall()
ev_ts = np.array([r[0] for r in rows], dtype="int64")
ev_lab = [r[1].lower() for r in rows]


def human_good(ts):
    """True where ts falls in a human-labeled 'good' interval."""
    good = np.zeros(len(ts), bool)
    for i in range(len(ev_ts)):
        t0 = ev_ts[i]
        t1 = ev_ts[i + 1] if i + 1 < len(ev_ts) else t0
        is_good_start = "good" in ev_lab[i] and "start" in ev_lab[i]
        if is_good_start:
            good |= (ts >= t0) & (ts < t1)
    return good


def load_coral_full(path, maskpath=None):
    k = pl.read_csv(path)
    ts = k["ts"].to_numpy().astype("datetime64[us]").astype("int64")
    bpm = k["bpm"].to_numpy().astype(float); sqi = k["sqi"].to_numpy().astype(float)
    algo_valid = sqi >= SQI_THR
    inband = (bpm >= ATTRACTOR_BAND[0]) & (bpm <= ATTRACTOR_BAND[1])
    algo_valid &= ~(inband & (sqi < BAND_THR))
    hopdt = np.median(np.diff(ts)) / 1e6 if len(ts) > 2 else 0.25
    w = max(5, int(round(90.0 / max(hopdt, 1e-3))) | 1)
    algo_valid &= rolling_ptp(bpm, w) > 2.0
    if maskpath and os.path.exists(maskpath):
        m = pl.read_parquet(maskpath)
        for a, b in zip(m["mask_start"].to_numpy().astype("datetime64[us]").astype("int64"),
                         m["mask_end"].to_numpy().astype("datetime64[us]").astype("int64")):
            algo_valid &= ~((ts >= a) & (ts <= b))
    algo_valid &= ~frozen_runs_mask(bpm, min_hops=8)
    hgood = human_good(ts)
    return ts, bpm, sqi, algo_valid, hgood


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


scg_ts, scg_bpm, scg_sqi, scg_algo, scg_hg = load_coral_full(
    f"{HERE}/coral_scg/Chelten/out.csv", f"{HERE}/coral_scg/Chelten/mask.parquet")
pts, pbpm, psqi, p_algo, p_hg = load_coral_full(
    f"{HERE}/coral_ecg/Chelten_ecg_polar_maskfix/out.csv",
    f"{HERE}/coral_ecg/Chelten_ecg_polar_maskfix/mask.parquet")

t0 = np.datetime64("2026-06-26T17:22:30").astype("datetime64[us]").astype(int)
t1 = np.datetime64("2026-06-26T17:26:30").astype("datetime64[us]").astype(int)

configs = {
    "human annotation ONLY": (scg_hg, p_hg),
    "algorithmic filters ONLY (current)": (scg_algo, p_algo),
    "human AND algorithmic (both must pass)": (scg_hg & scg_algo, p_hg & p_algo),
}

print(f"{'config':40s}  n     MAE    bias   within10   r")
results = {}
for label, (sv, pv) in configs.items():
    xa, yb, ta = align(scg_ts, scg_bpm, sv, pts, pbpm, pv)
    inwin = (ta >= t0) & (ta <= t1)
    R = stats(xa[inwin], yb[inwin])
    results[label] = (xa[inwin], yb[inwin], R)
    print(f"{label:40s}  {R['n']:4d}  {R['mae']:.2f}  {R['bias']:+.2f}  {100*R['within10']:.1f}%    {R['r']:.3f}")

# ---- figure: time series with human good/bad shading + scatter for each config ----
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1.0], hspace=0.35, wspace=0.3)
axT = fig.add_subplot(gs[0, :])

pad = 20_000_000
for ts_, bpm_, v_, color, label in [(scg_ts, scg_bpm, scg_algo, "#d62728", "SCG CORAL HR"),
                                      (pts, pbpm, p_algo, "#2ca02c", "Polar CORAL HR")]:
    m = (ts_ >= t0 - pad) & (ts_ <= t1 + pad)
    x = dn(ts_[m]); y = bpm_[m]
    axT.plot(x, y, color=color, lw=1.6, alpha=0.85, label=label, zorder=3)

# shade human bad intervals
i = 0
while i < len(ev_ts):
    if "bad" in ev_lab[i]:
        s = ev_ts[i]
        e = ev_ts[i+1] if i+1 < len(ev_ts) else s
        axT.axvspan(dn(s)[0], dn(e)[0], color="red", alpha=0.12, zorder=0)
    i += 1
axT.set_xlim(dn(t0 - pad)[0], dn(t1 + pad)[0]); axT.set_ylim(40, 150)
axT.xaxis.set_major_formatter(DateFormatter("%H:%M:%S", tz=UTC))
axT.set_ylabel("HR (bpm)"); axT.legend(loc="upper right"); axT.grid(True, alpha=0.3)
axT.set_title("red shading = human-labeled BAD segments (both signals) -- 17:22:30-17:26:30")

for ax_idx, (label, (xa, yb, R)) in enumerate(results.items()):
    ax = fig.add_subplot(gs[1, ax_idx])
    ax.scatter(yb, xa, s=18, alpha=0.6, color="#1f77b4", edgecolor="none")
    lim = [60, 120]; ax.plot(lim, lim, "k-", lw=1, alpha=0.6)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("Polar CORAL HR"); ax.set_ylabel("SCG CORAL HR")
    ax.set_title(label, fontsize=9.5)
    ax.text(0.03, 0.97, f"MAE {R['mae']:.2f}  bias {R['bias']:+.2f}\n"
                          f"≤10bpm {100*R['within10']:.1f}%  r={R['r']:.3f}\nn={R['n']}",
            transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="white", alpha=0.9))
    ax.grid(True, alpha=0.3)

fig.suptitle("[Chelten] does the human annotation beat the algorithmic filters? 17:22:30-17:26:30", fontsize=13)
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_human_annotation_test.png"
fig.savefig(out, dpi=150)
print("->", out)

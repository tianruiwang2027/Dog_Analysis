#!/usr/bin/env python3
"""17:26:30 -> end of annotation. All 4 comparisons, with lag-scan applied
independently to any pairing that spans SCG-side vs ECG-side timestamps
(the two independent recorders), since we've established a ~+7s clock offset:
  1. CORAL SCG   vs CORAL ECG   (own lag scan)
  2. Hand SCG    vs Hand ECG    (established best lag = +7.0s)
  3. CORAL SCG   vs Hand SCG    (same recorder/clock -> no lag needed)
  4. CORAL ECG   vs Hand ECG    (same recorder/clock -> no lag needed)
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


def load_peaks_and_good(path):
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = '' ORDER BY timestamp")
    pk = np.array([r[0] for r in cur.fetchall()], dtype="int64")
    cur.execute("SELECT DISTINCT timestamp, label FROM events WHERE label != '' ORDER BY timestamp")
    rows = cur.fetchall()
    ev_ts = np.array([r[0] for r in rows], dtype="int64")
    ev_lab = [r[1].lower() for r in rows]
    return pk, ev_ts, ev_lab


def good_intervals(ev_ts, ev_lab, span_end):
    ivs = []; state = "bad"; cur_start = None
    for t, lab in zip(ev_ts, ev_lab):
        if "good" in lab and "start" in lab:
            if state != "good": cur_start = t; state = "good"
        elif "bad" in lab and "start" in lab:
            if state == "good": ivs.append((cur_start, t)); state = "bad"
        elif "bad" in lab and "end" in lab:
            if state != "good": cur_start = t; state = "good"
        elif "good" in lab and "end" in lab:
            if state == "good": ivs.append((cur_start, t)); state = "bad"
        elif "stop" in lab:
            if state == "good": ivs.append((cur_start, t)); state = "bad"
    if state == "good": ivs.append((cur_start, span_end))
    return ivs


def good_fn(ev_ts, ev_lab, span_end):
    ivs = good_intervals(ev_ts, ev_lab, span_end)
    def good(ts):
        g = np.zeros(len(ts), bool)
        for t0, t1 in ivs: g |= (ts >= t0) & (ts < t1)
        return g
    return good, ivs


def beat_hr(peaks):
    rr = np.diff(peaks) / 1e6; hr = 60.0 / rr
    tmid = peaks[:-1] + np.diff(peaks) // 2
    return tmid, hr


def smooth(tmid, hr, win_s=SMOOTH_S):
    win_us = int(win_s * 1e6)
    out = np.empty(len(tmid))
    for i, t in enumerate(tmid):
        sel = (tmid >= t - win_us) & (tmid <= t + win_us)
        out[i] = hr[sel].mean()
    return out


def align(tsA, hrA, vA, tsB, hrB, vB, win_us=2_000_000, lag_us=0):
    bts, bhr = tsB[vB], hrB[vB]
    A = np.where(vA)[0]; xa, yb, ta = [], [], []
    for i in A:
        t = tsA[i] + lag_us; sel = (bts >= t - win_us) & (bts <= t + win_us)
        if sel.sum() >= 1:
            xa.append(hrA[i]); yb.append(bhr[sel].mean()); ta.append(tsA[i])
    return np.array(xa), np.array(yb), np.array(ta, dtype="int64")


def stats(a, b):
    d = a - b
    return dict(n=int(len(a)), bias=float(d.mean()), mae=float(np.abs(d).mean()),
                within10=float(np.mean(np.abs(d) <= 10)),
                r=float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan"))


t0 = np.datetime64("2026-06-26T17:26:30").astype("datetime64[us]").astype(int)

# ---- CORAL algorithmic series ----
scg_ts, scg_bpm, scg_v = load_coral(f"{HERE}/coral_scg/Chelten/out.csv", f"{HERE}/coral_scg/Chelten/mask.parquet")
ecg_ts, ecg_bpm, ecg_v = load_coral(f"{HERE}/coral_ecg/Chelten_ecg_polar_maskfix/out.csv",
                                     f"{HERE}/coral_ecg/Chelten_ecg_polar_maskfix/mask.parquet")

# ---- hand-annotated peaks (from the 17:22:22-18:00:xx annotation files) ----
h_ecg_pk, h_ecg_ev_ts, h_ecg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_ecg2.db")
h_scg_pk, h_scg_ev_ts, h_scg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_scg2.db")
span_end = max(h_ecg_pk.max(), h_scg_pk.max())
h_ecg_good, h_ecg_ivs = good_fn(h_ecg_ev_ts, h_ecg_ev_lab, span_end)
h_scg_good, h_scg_ivs = good_fn(h_scg_ev_ts, h_scg_ev_lab, span_end)
t1 = span_end

h_ecg_tmid, h_ecg_hr_raw = beat_hr(h_ecg_pk)
h_scg_tmid, h_scg_hr_raw = beat_hr(h_scg_pk)
h_ecg_hr = smooth(h_ecg_tmid, h_ecg_hr_raw)
h_scg_hr = smooth(h_scg_tmid, h_scg_hr_raw)
h_ecg_v = h_ecg_good(h_ecg_tmid)
h_scg_v = h_scg_good(h_scg_tmid)


def stats_in_window(xa, yb, ta):
    inw = (ta >= t0) & (ta <= t1)
    return xa[inw], yb[inw], stats(xa[inw], yb[inw])


def best_lag_scan(tsA, hrA, vA, tsB, hrB, vB, lags):
    rs = []
    for lag_s in lags:
        xa, yb, ta = align(tsA, hrA, vA, tsB, hrB, vB, lag_us=int(lag_s * 1e6))
        _, _, R = stats_in_window(xa, yb, ta)
        rs.append(R["r"])
    rs = np.array(rs)
    return lags[np.nanargmax(rs)], rs


lags = np.arange(-5, 20.01, 0.5)

# 1. CORAL SCG vs CORAL ECG -- own lag scan
best_lag_coral, rs_coral = best_lag_scan(scg_ts, scg_bpm, scg_v, ecg_ts, ecg_bpm, ecg_v, lags)
xa, yb, ta = align(scg_ts, scg_bpm, scg_v, ecg_ts, ecg_bpm, ecg_v, lag_us=0)
xa0, yb0, R0 = stats_in_window(xa, yb, ta)
xa, yb, ta = align(scg_ts, scg_bpm, scg_v, ecg_ts, ecg_bpm, ecg_v, lag_us=int(best_lag_coral * 1e6))
xaL, ybL, RL = stats_in_window(xa, yb, ta)
print(f"CORAL SCG vs CORAL ECG: best_lag={best_lag_coral:+.1f}s | lag=0 {R0} | lag=best {RL}")
results_lag0 = {"CORAL SCG vs CORAL ECG": (xa0, yb0, R0)}
results_bestlag = {"CORAL SCG vs CORAL ECG": (xaL, ybL, RL, best_lag_coral)}

# 2. Hand SCG vs Hand ECG -- established lag
best_lag_hand = 7.0
xa, yb, ta = align(h_scg_tmid, h_scg_hr, h_scg_v, h_ecg_tmid, h_ecg_hr, h_ecg_v, lag_us=0)
xa0, yb0, R0 = stats_in_window(xa, yb, ta)
xa, yb, ta = align(h_scg_tmid, h_scg_hr, h_scg_v, h_ecg_tmid, h_ecg_hr, h_ecg_v, lag_us=int(best_lag_hand * 1e6))
xaL, ybL, RL = stats_in_window(xa, yb, ta)
print(f"Hand SCG vs Hand ECG: lag=0 {R0} | lag={best_lag_hand:+.1f} {RL}")
results_lag0["Hand SCG vs Hand ECG"] = (xa0, yb0, R0)
results_bestlag["Hand SCG vs Hand ECG"] = (xaL, ybL, RL, best_lag_hand)

# 3. CORAL SCG vs Hand SCG -- same clock, no lag
xa, yb, ta = align(scg_ts, scg_bpm, scg_v, h_scg_tmid, h_scg_hr, h_scg_v, lag_us=0)
xa0, yb0, R0 = stats_in_window(xa, yb, ta)
print(f"CORAL SCG vs Hand SCG: {R0}")
results_lag0["CORAL SCG vs Hand SCG"] = (xa0, yb0, R0)
results_bestlag["CORAL SCG vs Hand SCG"] = (xa0, yb0, R0, 0.0)

# 4. CORAL ECG vs Hand ECG -- same clock, no lag
xa, yb, ta = align(ecg_ts, ecg_bpm, ecg_v, h_ecg_tmid, h_ecg_hr, h_ecg_v, lag_us=0)
xa0, yb0, R0 = stats_in_window(xa, yb, ta)
print(f"CORAL ECG vs Hand ECG: {R0}")
results_lag0["CORAL ECG vs Hand ECG"] = (xa0, yb0, R0)
results_bestlag["CORAL ECG vs Hand ECG"] = (xa0, yb0, R0, 0.0)

print(f"\n{'comparison':28s}  {'lag':>6s}  n     MAE    bias    within10   r")
for label in results_bestlag:
    xa, yb, R, lag = results_bestlag[label]
    print(f"{label:28s}  {lag:+5.1f}s {R['n']:4d}  {R['mae']:6.2f}  {R['bias']:+6.2f}  {100*R['within10']:6.1f}%   {R['r']:.3f}")

# ---- figure ----
fig = plt.figure(figsize=(17, 11))
gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 1.0], hspace=0.45, wspace=0.35)
axT = fig.add_subplot(gs[0, :])
mS = (scg_ts >= t0) & (scg_ts <= t1)
mE = (ecg_ts >= t0) & (ecg_ts <= t1)
axT.plot(dn(scg_ts[mS]), scg_bpm[mS], color="0.88", lw=0.8, zorder=1)
axT.plot(dn(ecg_ts[mE]), ecg_bpm[mE], color="0.88", lw=0.8, zorder=1)
SvC = scg_v & mS; EvC = ecg_v & mE
axT.plot(dn(scg_ts[SvC]), scg_bpm[SvC], color="#d62728", lw=1.2, alpha=0.85, label="CORAL SCG (filtered)", zorder=3)
axT.plot(dn(ecg_ts[EvC]), ecg_bpm[EvC], color="#2ca02c", lw=1.2, alpha=0.85, label="CORAL Polar/ECG (filtered)", zorder=3)
mHS = (h_scg_tmid >= t0) & (h_scg_tmid <= t1) & h_scg_v
mHE = (h_ecg_tmid >= t0) & (h_ecg_tmid <= t1) & h_ecg_v
axT.scatter(dn(h_scg_tmid[mHS]), h_scg_hr[mHS], s=6, color="#8c1515", marker="D", zorder=4, label="Hand SCG (smoothed, good)")
axT.scatter(dn(h_ecg_tmid[mHE]), h_ecg_hr[mHE], s=6, color="#145214", marker="D", zorder=4, label="Hand ECG (smoothed, good)")
axT.set_xlim(dn(t0)[0], dn(t1)[0]); axT.set_ylim(40, 130)
axT.xaxis.set_major_formatter(DateFormatter("%H:%M:%S", tz=UTC))
axT.set_ylabel("HR (bpm)"); axT.legend(loc="upper right", ncol=2, fontsize=9)
axT.grid(True, alpha=0.3)
axT.set_title("CORAL algorithmic (lines) vs hand-annotated (diamonds), NO lag correction shown -- 17:26:30-end")

colors = ["#1f77b4", "#9467bd", "#ff7f0e", "#17becf"]
for i, label in enumerate(results_bestlag):
    ax = fig.add_subplot(gs[1, i])
    xa, yb, R, lag = results_bestlag[label]
    ax.scatter(yb, xa, s=14, alpha=0.5, color=colors[i], edgecolor="none")
    lim = [40, 130]; ax.plot(lim, lim, "k-", lw=1, alpha=0.6)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    parts = label.split(" vs ")
    ax.set_xlabel(parts[1]); ax.set_ylabel(parts[0])
    lagtag = f" (lag +{lag:.1f}s)" if lag else " (no lag)"
    ax.set_title(label + lagtag, fontsize=9.5)
    ax.text(0.03, 0.97, f"MAE {R['mae']:.2f}  bias {R['bias']:+.2f}\n"
                          f"≤10bpm {100*R['within10']:.1f}%  r={R['r']:.3f}\nn={R['n']}",
            transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="white", alpha=0.9))
    ax.grid(True, alpha=0.3)

fig.suptitle("[Chelten] CORAL algorithmic vs hand-annotated, all 4 comparisons (lag-corrected) -- 17:26:30-end", fontsize=13)
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_all_compare_from172630.png"
fig.savefig(out, dpi=150)
print("->", out)

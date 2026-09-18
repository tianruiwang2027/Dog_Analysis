#!/usr/bin/env python3
import sqlite3
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    ivs = []; state="bad"; cur_start=None
    for t, lab in zip(ev_ts, ev_lab):
        if "good" in lab and "start" in lab:
            if state != "good": cur_start=t; state="good"
        elif "bad" in lab and "start" in lab:
            if state=="good": ivs.append((cur_start,t)); state="bad"
        elif "bad" in lab and "end" in lab:
            if state != "good": cur_start=t; state="good"
        elif "good" in lab and "end" in lab:
            if state=="good": ivs.append((cur_start,t)); state="bad"
        elif "stop" in lab:
            if state=="good": ivs.append((cur_start,t)); state="bad"
    if state=="good": ivs.append((cur_start, span_end))
    return ivs

def good_fn(ev_ts, ev_lab, span_end):
    ivs = good_intervals(ev_ts, ev_lab, span_end)
    def good(ts):
        g = np.zeros(len(ts), bool)
        for t0,t1 in ivs: g |= (ts>=t0)&(ts<t1)
        return g
    return good, ivs

ecg_pk, ecg_ev_ts, ecg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_ecg2.db")
scg_pk, scg_ev_ts, scg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_scg2.db")
span_end = max(ecg_pk.max(), scg_pk.max())
ecg_good, ecg_ivs = good_fn(ecg_ev_ts, ecg_ev_lab, span_end)
scg_good, scg_ivs = good_fn(scg_ev_ts, scg_ev_lab, span_end)

def beat_hr(peaks):
    rr = np.diff(peaks)/1e6; hr = 60.0/rr
    tmid = peaks[:-1] + np.diff(peaks)//2
    return tmid, hr

def smooth(tmid, hr, win_s=3.0):
    win_us = int(win_s*1e6)
    out = np.empty(len(tmid))
    for i,t in enumerate(tmid):
        sel = (tmid>=t-win_us)&(tmid<=t+win_us)
        out[i] = hr[sel].mean()
    return out

ecg_tmid, ecg_hr_raw = beat_hr(ecg_pk)
scg_tmid, scg_hr_raw = beat_hr(scg_pk)
ecg_hr = smooth(ecg_tmid, ecg_hr_raw)
scg_hr = smooth(scg_tmid, scg_hr_raw)
ecg_v = ecg_good(ecg_tmid)
scg_v = scg_good(scg_tmid)

def align(tsA, hrA, vA, tsB, hrB, vB, win_us=2_000_000, lag_us=0):
    bts, bhr = tsB[vB], hrB[vB]
    A = np.where(vA)[0]; xa, yb = [], []
    for i in A:
        t = tsA[i] + lag_us; sel = (bts>=t-win_us)&(bts<=t+win_us)
        if sel.sum()>=1:
            xa.append(hrA[i]); yb.append(bhr[sel].mean())
    return np.array(xa), np.array(yb)

def stats(a,b):
    d = a-b
    return dict(n=len(a), bias=d.mean(), mae=np.abs(d).mean(), r=np.corrcoef(a,b)[0,1] if len(a)>2 else float('nan'))

lags = np.arange(-5,20.01,0.5)
rs, maes = [], []
for lag_s in lags:
    xa,yb = align(scg_tmid, scg_hr, scg_v, ecg_tmid, ecg_hr, ecg_v, lag_us=int(lag_s*1e6))
    R = stats(xa,yb)
    rs.append(R['r']); maes.append(R['mae'])
rs = np.array(rs); maes = np.array(maes)
best_lag = lags[np.argmax(rs)]

xa0,yb0 = align(scg_tmid, scg_hr, scg_v, ecg_tmid, ecg_hr, ecg_v, lag_us=0)
R0 = stats(xa0,yb0)
xaL,ybL = align(scg_tmid, scg_hr, scg_v, ecg_tmid, ecg_hr, ecg_v, lag_us=int(best_lag*1e6))
RL = stats(xaL,ybL)

fig, axes = plt.subplots(1,3, figsize=(18,5.5))
ax = axes[0]
ax.plot(lags, rs, "-o", color="#1f77b4", ms=3)
ax.axvline(best_lag, color="red", ls="--", lw=1, label=f"best lag = +{best_lag:.1f}s")
ax.axvline(0, color="0.6", ls=":", lw=1)
ax.set_xlabel("lag applied to SCG timestamps (s)  [positive = shift SCG later]")
ax.set_ylabel("r (smoothed HR)")
ax.set_title("Correlation vs assumed clock offset")
ax.legend(); ax.grid(True, alpha=0.3)

lim = [40,150]
ax = axes[1]
ax.scatter(yb0, xa0, s=8, alpha=0.35, color="#1f77b4")
ax.plot(lim,lim,"k-",lw=1,alpha=0.6)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
ax.set_xlabel("Polar hand-beat HR"); ax.set_ylabel("SCG hand-beat HR")
ax.set_title(f"lag=0s: r={R0['r']:.3f} MAE={R0['mae']:.2f} n={R0['n']}")
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.scatter(ybL, xaL, s=8, alpha=0.35, color="#2ca02c")
ax.plot(lim,lim,"k-",lw=1,alpha=0.6)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
ax.set_xlabel("Polar hand-beat HR"); ax.set_ylabel("SCG hand-beat HR")
ax.set_title(f"lag=+{best_lag:.1f}s: r={RL['r']:.3f} MAE={RL['mae']:.2f} n={RL['n']}")
ax.grid(True, alpha=0.3)

fig.suptitle("[Chelten] SCG timestamps run ~7-8s AHEAD of Polar/ECG timestamps for the same physical heartbeat", fontsize=13)
fig.tight_layout()
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_lag_explain.png"
fig.savefig(out, dpi=150)
print(f"best lag = {best_lag:.2f}s, r0={R0['r']:.3f}, r_best={RL['r']:.3f}")
print("->", out)

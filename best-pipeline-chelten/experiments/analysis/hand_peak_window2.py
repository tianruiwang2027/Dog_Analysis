#!/usr/bin/env python3
"""Hand-annotated-peak SCG vs Polar comparison restricted to 17:24:20-17:26:30,
the sub-window where the SCG/ECG hand-click beat counts already agree closely
(ratio ~0.9-1.0 per segment), to test whether the earlier poor r was driven by
the noisier early segments."""
import sqlite3
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import datetime

UTC = datetime.timezone.utc
SMOOTH_S = 3.0


def dn(us):
    return mdates.date2num([datetime.datetime.fromtimestamp(t/1e6, tz=UTC) for t in np.atleast_1d(us)])


con = sqlite3.connect("/tmp/annotation_chelten_ecg.db")
cur = con.cursor()
cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = '' ORDER BY timestamp")
ecg_pk = np.array([r[0] for r in cur.fetchall()], dtype="int64")

con2 = sqlite3.connect("/tmp/annotation_chelten_scg.db")
cur2 = con2.cursor()
cur2.execute("SELECT DISTINCT timestamp FROM events WHERE label = '' ORDER BY timestamp")
scg_pk = np.array([r[0] for r in cur2.fetchall()], dtype="int64")
cur2.execute("SELECT DISTINCT timestamp, label FROM events WHERE label != '' ORDER BY timestamp")
scg_events = cur2.fetchall()
scg_ev_ts = np.array([r[0] for r in scg_events], dtype="int64")
scg_ev_lab = [r[1].lower() for r in scg_events]


def scg_good(ts):
    good = np.zeros(len(ts), bool)
    for i in range(len(scg_ev_ts)):
        if "good" in scg_ev_lab[i] and "start" in scg_ev_lab[i]:
            t0 = scg_ev_ts[i]; t1 = scg_ev_ts[i + 1] if i + 1 < len(scg_ev_ts) else t0
            good |= (ts >= t0) & (ts < t1)
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


ecg_tmid, ecg_hr_raw = beat_hr(ecg_pk)
scg_tmid, scg_hr_raw = beat_hr(scg_pk)
ecg_hr_sm = smooth(ecg_tmid, ecg_hr_raw)
scg_hr_sm = smooth(scg_tmid, scg_hr_raw)

ecg_good_mask = (ecg_tmid >= ecg_pk.min()) & (ecg_tmid <= ecg_pk.max())
scg_good_mask = scg_good(scg_tmid)

t0 = np.datetime64("2026-06-26T17:24:20").astype("datetime64[us]").astype(int)
t1 = np.datetime64("2026-06-26T17:26:30").astype("datetime64[us]").astype(int)


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


# raw
xa_r, yb_r, ta_r = align(scg_tmid, scg_hr_raw, scg_good_mask, ecg_tmid, ecg_hr_raw, ecg_good_mask)
inwin_r = (ta_r >= t0) & (ta_r <= t1)
R_raw = stats(xa_r[inwin_r], yb_r[inwin_r])
print("RAW beat-to-beat, 17:24:20-17:26:30:", R_raw)

# smoothed
xa_s, yb_s, ta_s = align(scg_tmid, scg_hr_sm, scg_good_mask, ecg_tmid, ecg_hr_sm, ecg_good_mask)
inwin_s = (ta_s >= t0) & (ta_s <= t1)
R_sm = stats(xa_s[inwin_s], yb_s[inwin_s])
print("3s-SMOOTHED, 17:24:20-17:26:30:", R_sm)

# peak count check in this window, per scg-good segment
print("\nSCG-good segments overlapping this window:")
i = 0
ivs = []
for i in range(len(scg_ev_ts)):
    if "good" in scg_ev_lab[i] and "start" in scg_ev_lab[i]:
        s = scg_ev_ts[i]; e = scg_ev_ts[i+1] if i+1 < len(scg_ev_ts) else s
        if e >= t0 and s <= t1:
            ivs.append((max(s,t0), min(e,t1)))
tot_s = tot_e = 0
for s,e in ivs:
    ns = ((scg_pk>=s)&(scg_pk<=e)).sum(); ne = ((ecg_pk>=s)&(ecg_pk<=e)).sum()
    tot_s += ns; tot_e += ne
    print(f"  dur={(e-s)/1e6:5.1f}s n_scg={ns:3d} n_ecg={ne:3d} ratio={ns/ne if ne else float('nan'):.3f}")
print(f"  TOTAL: n_scg={tot_s} n_ecg={tot_e} ratio={tot_s/tot_e:.3f}")

# ---- figure ----
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0], hspace=0.35, wspace=0.3)
axT = fig.add_subplot(gs[0, :])
pad = 5_000_000
mE = (ecg_tmid >= t0 - pad) & (ecg_tmid <= t1 + pad)
mS = (scg_tmid >= t0 - pad) & (scg_tmid <= t1 + pad)
axT.plot(dn(ecg_tmid[mE]), ecg_hr_sm[mE], color="0.8", lw=1, zorder=1)
axT.plot(dn(scg_tmid[mS]), scg_hr_sm[mS], color="0.85", lw=1, zorder=1)
Ev = ecg_good_mask & mE; Sv = scg_good_mask & mS
axT.scatter(dn(ecg_tmid[Ev]), ecg_hr_sm[Ev], s=10, color="#2ca02c", zorder=3, label="Polar (joint-good, smoothed)")
axT.scatter(dn(scg_tmid[Sv]), scg_hr_sm[Sv], s=10, color="#d62728", zorder=3, label="SCG (joint-good, smoothed)")
axT.set_xlim(dn(t0 - pad)[0], dn(t1 + pad)[0]); axT.set_ylim(40, 130)
axT.xaxis.set_major_formatter(DateFormatter("%H:%M:%S", tz=UTC))
axT.set_ylabel("HR (bpm), 3s-smoothed hand beats"); axT.legend(loc="upper right", fontsize=9)
axT.grid(True, alpha=0.3)
axT.set_title("Hand-annotated beats, 3s-smoothed HR, joint-good time -- 17:24:20-17:26:30")

axS1 = fig.add_subplot(gs[1, 0])
axS1.scatter(yb_r[inwin_r], xa_r[inwin_r], s=18, alpha=0.6, color="#1f77b4", edgecolor="none")
lim = [40, 130]; axS1.plot(lim, lim, "k-", lw=1, alpha=0.6)
axS1.set_xlim(lim); axS1.set_ylim(lim); axS1.set_aspect("equal")
axS1.set_xlabel("Polar hand-beat HR (raw)"); axS1.set_ylabel("SCG hand-beat HR (raw)")
axS1.set_title("Raw beat-to-beat")
axS1.text(0.03, 0.97, f"MAE {R_raw['mae']:.2f}  bias {R_raw['bias']:+.2f}\n"
                       f"≤10bpm {100*R_raw['within10']:.1f}%  r={R_raw['r']:.3f}\nn={R_raw['n']}",
          transform=axS1.transAxes, va="top", fontsize=10,
          bbox=dict(boxstyle="round", fc="white", alpha=0.9))
axS1.grid(True, alpha=0.3)

axS2 = fig.add_subplot(gs[1, 1])
axS2.scatter(yb_s[inwin_s], xa_s[inwin_s], s=18, alpha=0.6, color="#ff7f0e", edgecolor="none")
axS2.plot(lim, lim, "k-", lw=1, alpha=0.6)
axS2.set_xlim(lim); axS2.set_ylim(lim); axS2.set_aspect("equal")
axS2.set_xlabel("Polar hand-beat HR (3s-smoothed)"); axS2.set_ylabel("SCG hand-beat HR (3s-smoothed)")
axS2.set_title("3s-smoothed")
axS2.text(0.03, 0.97, f"MAE {R_sm['mae']:.2f}  bias {R_sm['bias']:+.2f}\n"
                       f"≤10bpm {100*R_sm['within10']:.1f}%  r={R_sm['r']:.3f}\nn={R_sm['n']}",
          transform=axS2.transAxes, va="top", fontsize=10,
          bbox=dict(boxstyle="round", fc="white", alpha=0.9))
axS2.grid(True, alpha=0.3)

fig.suptitle("[Chelten] SCG vs Polar HR, hand-annotated beats, joint-good -- 17:24:20-17:26:30", fontsize=13)
fig.tight_layout()
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_hand_peak_window2.png"
fig.savefig(out, dpi=150)
print("->", out)

#!/usr/bin/env python3
"""Compare SCG vs Polar ECG heart rate computed DIRECTLY from hand-annotated
beat-by-beat peaks (not CORAL's own bpm output), restricted to jointly-good
time (SCG hand-labeled good AND within the ECG annotated span), for
17:22:30-17:26:30."""
import sqlite3
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import datetime

UTC = datetime.timezone.utc


def dn(us):
    return mdates.date2num([datetime.datetime.fromtimestamp(t/1e6, tz=UTC) for t in np.atleast_1d(us)])


# ---- load ECG hand peaks (beat-by-beat, continuous annotated span) ----
con = sqlite3.connect("/tmp/annotation_chelten_ecg.db")
cur = con.cursor()
cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = '' ORDER BY timestamp")
ecg_pk = np.array([r[0] for r in cur.fetchall()], dtype="int64")

# ---- load SCG hand peaks + good/bad segment markers ----
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
            t0 = scg_ev_ts[i]
            t1 = scg_ev_ts[i + 1] if i + 1 < len(scg_ev_ts) else t0
            good |= (ts >= t0) & (ts < t1)
    return good


def beat_hr(peaks):
    """RR -> instantaneous HR at the midpoint of each beat interval."""
    rr = np.diff(peaks) / 1e6
    hr = 60.0 / rr
    tmid = peaks[:-1] + np.diff(peaks) // 2
    return tmid, hr


ecg_tmid, ecg_hr = beat_hr(ecg_pk)
scg_tmid, scg_hr = beat_hr(scg_pk)

ecg_span = (ecg_pk.min(), ecg_pk.max())
ecg_good_mask = (ecg_tmid >= ecg_span[0]) & (ecg_tmid <= ecg_span[1])  # continuous, no internal bad marks
scg_good_mask = scg_good(scg_tmid)

t0 = np.datetime64("2026-06-26T17:22:30").astype("datetime64[us]").astype(int)
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


# SCG as the "A" series (as in all prior scatters: SCG on y-axis)
xa, yb, ta = align(scg_tmid, scg_hr, scg_good_mask, ecg_tmid, ecg_hr, ecg_good_mask)
inwin = (ta >= t0) & (ta <= t1)
R = stats(xa[inwin], yb[inwin])
print("hand-peak-derived HR, joint-good, window stats:", R)

Rall = stats(xa, yb)
print("hand-peak-derived HR, joint-good, full annotated span:", Rall)

# ---- figure ----
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 1.0], hspace=0.35)
axT = fig.add_subplot(gs[0])

pad = 5_000_000
mE = (ecg_tmid >= t0 - pad) & (ecg_tmid <= t1 + pad)
mS = (scg_tmid >= t0 - pad) & (scg_tmid <= t1 + pad)
axT.plot(dn(ecg_tmid[mE]), ecg_hr[mE], color="0.75", lw=1, zorder=1, label="Polar (all hand beats)")
axT.plot(dn(scg_tmid[mS]), scg_hr[mS], color="0.85", lw=1, zorder=1, label="SCG (all hand beats)")
gm = mE & ecg_good_mask[:len(mE)] if len(ecg_good_mask) == len(mE) else None
# valid (joint-good) overlay
Ev = ecg_good_mask & (ecg_tmid >= t0 - pad) & (ecg_tmid <= t1 + pad)
Sv = scg_good_mask & (scg_tmid >= t0 - pad) & (scg_tmid <= t1 + pad)
axT.scatter(dn(ecg_tmid[Ev]), ecg_hr[Ev], s=10, color="#2ca02c", zorder=3, label="Polar (joint-good)")
axT.scatter(dn(scg_tmid[Sv]), scg_hr[Sv], s=10, color="#d62728", zorder=3, label="SCG (joint-good)")
axT.set_xlim(dn(t0 - pad)[0], dn(t1 + pad)[0]); axT.set_ylim(30, 160)
axT.xaxis.set_major_formatter(DateFormatter("%H:%M:%S", tz=UTC))
axT.set_ylabel("beat-to-beat HR (bpm)"); axT.legend(loc="upper right", ncol=2, fontsize=9)
axT.grid(True, alpha=0.3)
axT.set_title("Hand-annotated beat-to-beat HR (RR-derived), colored/faint = joint-good vs not")

axS = fig.add_subplot(gs[1])
axS.scatter(yb[inwin], xa[inwin], s=18, alpha=0.6, color="#1f77b4", edgecolor="none")
lim = [40, 150]; axS.plot(lim, lim, "k-", lw=1, alpha=0.6)
axS.set_xlim(lim); axS.set_ylim(lim); axS.set_aspect("equal")
axS.set_xlabel("Polar hand-beat HR"); axS.set_ylabel("SCG hand-beat HR")
axS.text(0.03, 0.97, f"MAE {R['mae']:.2f}  bias {R['bias']:+.2f}\n"
                       f"≤10bpm {100*R['within10']:.1f}%  r={R['r']:.3f}\nn={R['n']}",
         transform=axS.transAxes, va="top", fontsize=10,
         bbox=dict(boxstyle="round", fc="white", alpha=0.9))
axS.grid(True, alpha=0.3)

fig.suptitle("[Chelten] SCG vs Polar HR from HAND-ANNOTATED beats only, joint-good time -- 17:22:30-17:26:30", fontsize=12)
fig.tight_layout()
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_hand_peak_test.png"
fig.savefig(out, dpi=150)
print("->", out)

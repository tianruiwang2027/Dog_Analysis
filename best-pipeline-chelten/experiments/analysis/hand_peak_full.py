#!/usr/bin/env python3
"""Hand-annotated-peak SCG vs Polar/ECG comparison over the FULL extended
annotation span (~17:22:22-18:00:00), restricted to time where BOTH signals
are hand-labeled good (their own independent good/bad segment markers)."""
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


ecg_pk, ecg_ev_ts, ecg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_ecg2.db")
scg_pk, scg_ev_ts, scg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_scg2.db")


def good_intervals(ev_ts, ev_lab, span_end):
    """State machine handling BOTH annotation conventions seen in these files:
    (a) alternating 'Good Data Start' / 'Bad Data Start' with no explicit end
        markers (each Start closes the previous run) -- the SCG style;
    (b) a single 'Good Data Start' followed by isolated 'Bad Data Start' /
        'Bad Data End' brackets, where good resumes implicitly at Bad End
        with no repeated 'Good Data Start' -- the ECG style.
    """
    ivs = []
    state = "bad"
    cur_start = None
    for t, lab in zip(ev_ts, ev_lab):
        if "good" in lab and "start" in lab:
            if state != "good":
                cur_start = t; state = "good"
        elif "bad" in lab and "start" in lab:
            if state == "good":
                ivs.append((cur_start, t)); state = "bad"
        elif "bad" in lab and "end" in lab:
            if state != "good":
                cur_start = t; state = "good"
        elif "good" in lab and "end" in lab:
            if state == "good":
                ivs.append((cur_start, t)); state = "bad"
        elif "stop" in lab:
            if state == "good":
                ivs.append((cur_start, t)); state = "bad"
    if state == "good":
        ivs.append((cur_start, span_end))
    return ivs


def good_fn(ev_ts, ev_lab, span_end):
    ivs = good_intervals(ev_ts, ev_lab, span_end)
    def good(ts):
        g = np.zeros(len(ts), bool)
        for t0, t1 in ivs:
            g |= (ts >= t0) & (ts < t1)
        return g
    return good, ivs


span_end = max(ecg_pk.max(), scg_pk.max())
ecg_good, ecg_ivs = good_fn(ecg_ev_ts, ecg_ev_lab, span_end)
scg_good, scg_ivs = good_fn(scg_ev_ts, scg_ev_lab, span_end)


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

ecg_v = ecg_good(ecg_tmid)
scg_v = scg_good(scg_tmid)

t0 = np.datetime64("2026-06-26T17:22:22").astype("datetime64[us]").astype(int)
t1 = np.datetime64("2026-06-26T18:00:00").astype("datetime64[us]").astype(int)


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
xa_r, yb_r, ta_r = align(scg_tmid, scg_hr_raw, scg_v, ecg_tmid, ecg_hr_raw, ecg_v)
inw_r = (ta_r >= t0) & (ta_r <= t1)
R_raw = stats(xa_r[inw_r], yb_r[inw_r])
print("RAW beat-to-beat, full span joint-good:", R_raw)

# smoothed
xa_s, yb_s, ta_s = align(scg_tmid, scg_hr_sm, scg_v, ecg_tmid, ecg_hr_sm, ecg_v)
inw_s = (ta_s >= t0) & (ta_s <= t1)
R_sm = stats(xa_s[inw_s], yb_s[inw_s])
print("3s-SMOOTHED, full span joint-good:", R_sm)

# how much time is jointly good?
tgrid = np.arange(t0, t1, 1_000_000)
jg = ecg_good(tgrid) & scg_good(tgrid)
print(f"\njoint-good coverage: {jg.mean()*100:.1f}% of {( t1-t0)/1e6/60:.1f} min span")
print(f"total SCG peaks in span: {len(scg_pk)}, ECG peaks in span: {len(ecg_pk)}")
print(f"ECG good intervals: {len(ecg_ivs)}, SCG good intervals: {len(scg_ivs)}")

# ---- figure ----
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.3)
axT = fig.add_subplot(gs[0, :])
mE = (ecg_tmid >= t0) & (ecg_tmid <= t1)
mS = (scg_tmid >= t0) & (scg_tmid <= t1)
axT.plot(dn(ecg_tmid[mE]), ecg_hr_sm[mE], color="0.85", lw=0.8, zorder=1)
axT.plot(dn(scg_tmid[mS]), scg_hr_sm[mS], color="0.85", lw=0.8, zorder=1)
Ev = ecg_v & mE; Sv = scg_v & mS
axT.scatter(dn(ecg_tmid[Ev]), ecg_hr_sm[Ev], s=6, color="#2ca02c", zorder=3, label="Polar (joint-good, smoothed)")
axT.scatter(dn(scg_tmid[Sv]), scg_hr_sm[Sv], s=6, color="#d62728", zorder=3, label="SCG (joint-good, smoothed)")
axT.set_xlim(dn(t0)[0], dn(t1)[0]); axT.set_ylim(40, 160)
axT.xaxis.set_major_formatter(DateFormatter("%H:%M:%S", tz=UTC))
axT.set_ylabel("HR (bpm), 3s-smoothed hand beats"); axT.legend(loc="upper right", fontsize=9)
axT.grid(True, alpha=0.3)
axT.set_title("Hand-annotated beats, 3s-smoothed HR, joint-good time -- 17:22:22-18:00:00")

axS1 = fig.add_subplot(gs[1, 0])
axS1.scatter(yb_r[inw_r], xa_r[inw_r], s=10, alpha=0.4, color="#1f77b4", edgecolor="none")
lim = [40, 160]; axS1.plot(lim, lim, "k-", lw=1, alpha=0.6)
axS1.set_xlim(lim); axS1.set_ylim(lim); axS1.set_aspect("equal")
axS1.set_xlabel("Polar hand-beat HR (raw)"); axS1.set_ylabel("SCG hand-beat HR (raw)")
axS1.set_title("Raw beat-to-beat")
axS1.text(0.03, 0.97, f"MAE {R_raw['mae']:.2f}  bias {R_raw['bias']:+.2f}\n"
                       f"≤10bpm {100*R_raw['within10']:.1f}%  r={R_raw['r']:.3f}\nn={R_raw['n']}",
          transform=axS1.transAxes, va="top", fontsize=10,
          bbox=dict(boxstyle="round", fc="white", alpha=0.9))
axS1.grid(True, alpha=0.3)

axS2 = fig.add_subplot(gs[1, 1])
axS2.scatter(yb_s[inw_s], xa_s[inw_s], s=10, alpha=0.4, color="#ff7f0e", edgecolor="none")
axS2.plot(lim, lim, "k-", lw=1, alpha=0.6)
axS2.set_xlim(lim); axS2.set_ylim(lim); axS2.set_aspect("equal")
axS2.set_xlabel("Polar hand-beat HR (3s-smoothed)"); axS2.set_ylabel("SCG hand-beat HR (3s-smoothed)")
axS2.set_title("3s-smoothed")
axS2.text(0.03, 0.97, f"MAE {R_sm['mae']:.2f}  bias {R_sm['bias']:+.2f}\n"
                       f"≤10bpm {100*R_sm['within10']:.1f}%  r={R_sm['r']:.3f}\nn={R_sm['n']}",
          transform=axS2.transAxes, va="top", fontsize=10,
          bbox=dict(boxstyle="round", fc="white", alpha=0.9))
axS2.grid(True, alpha=0.3)

fig.suptitle("[Chelten] SCG vs Polar HR, hand-annotated beats, joint-good -- 17:22:22-18:00:00", fontsize=13)
fig.tight_layout()
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_hand_peak_full.png"
fig.savefig(out, dpi=150)
print("->", out)

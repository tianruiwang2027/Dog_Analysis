#!/usr/bin/env python3
"""Run/re-use the existing Pan-Tompkins-style QRS detector (analysis/ecg_hr.py's
detect_qrs + clean_hr) on the raw Polar ECG signal for Chelten, then compare
its beat-derived HR against: (a) the hand-clicked ECG beats (validates the
detector), (b) CORAL's own ECG track, and (c) both SCG series (CORAL SCG and
hand-clicked SCG), with the same clock-offset lag correction established
earlier applied only to ECG-vs-SCG pairs. Restricted to 17:26:30-end, and only
where BOTH channels are hand-labeled good."""
import os, sys, glob, sqlite3
import numpy as np, polars as pl
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from scipy import signal as sg
from scipy.ndimage import median_filter, maximum_filter1d, minimum_filter1d, uniform_filter1d
import datetime

HERE = "/tmp/dog-test-ecg/dog-test-ecg-code/analysis"
HIVE = "/tmp/dog-test-ecg/dog-test-ecg-code/hive"
UTC = datetime.timezone.utc
SMOOTH_S = 3.0
SQI_THR = 0.10
ATTRACTOR_BAND = (95, 125)
BAND_THR = 0.50
LAG_CORAL = 8.0
LAG_HAND = 7.0

FC_LO, FC_HI = 10.0, 30.0
SEARCH_LO, SEARCH_HI = 50, 200
REFRACT_S = 0.28


def dn(us):
    return mdates.date2num([datetime.datetime.fromtimestamp(t/1e6, tz=UTC) for t in np.atleast_1d(us)])


def detect_qrs(x, fs):
    b = sg.butter(2, [FC_LO/(fs/2), FC_HI/(fs/2)], btype="band")
    xf = sg.filtfilt(*b, x - np.mean(x))
    deriv = np.convolve(xf, np.array([1,2,0,-2,-1])*(fs/8.0), mode="same")
    sqd = deriv**2
    mwi = uniform_filter1d(sqd, max(1, int(0.05*fs)))
    floor = median_filter(mwi, int(2*fs)|1, mode="nearest") + 1e-12
    pk, _ = sg.find_peaks(mwi, distance=int(fs*REFRACT_S), height=None)
    snr = mwi[pk] / floor[pk]
    pk = pk[snr > 4.0]
    win = int(0.06*fs)
    R = []
    for p in pk:
        a, bnd = max(0, p-win), min(len(xf), p+win)
        R.append(a + int(np.argmax(np.abs(xf[a:bnd]))))
    return np.unique(R), xf


def clean_hr(tpk):
    if len(tpk) < 4:
        return np.array([]), np.array([])
    rr = np.diff(tpk)/1e6
    hr = 60.0/rr
    tmid = tpk[:-1] + np.diff(tpk)//2
    keep = (hr >= SEARCH_LO) & (hr <= SEARCH_HI)
    med = median_filter(hr, 7, mode="nearest")
    mad = median_filter(np.abs(hr-med), 7, mode="nearest") + 1e-6
    keep &= np.abs(hr-med) <= 5.0*1.4826*mad
    return tmid[keep], hr[keep]


print("running Pan-Tompkins on raw Polar ECG (Chelten)...")
p = glob.glob(f"{HIVE}/username=ecg_polar/device=Chelten/stream=0/date=*/data_0.parquet")[0]
df = pl.read_parquet(p, columns=["ts", "c1"])
ts_raw = df["ts"].to_numpy().astype("int64")
x_raw = df["c1"].to_numpy().astype(float)
fs = float(1e6 / np.median(np.diff(ts_raw)))
print(f"  loaded {len(ts_raw)} samples @ fs={fs:.2f}Hz, span={(ts_raw[-1]-ts_raw[0])/1e6/60:.1f} min")

R_idx, xf = detect_qrs(x_raw, fs)
R_ts = ts_raw[R_idx]
print(f"  detected {len(R_ts)} R-peaks")
pt_tmid, pt_hr_raw = clean_hr(R_ts)
print(f"  after clean_hr (Hampel): {len(pt_tmid)} beat-HR points")


def smooth(tmid, hr, win_s=SMOOTH_S):
    win_us = int(win_s*1e6)
    out = np.empty(len(tmid))
    for i, t in enumerate(tmid):
        sel = (tmid >= t-win_us) & (tmid <= t+win_us)
        out[i] = hr[sel].mean()
    return out


pt_hr = smooth(pt_tmid, pt_hr_raw)

# ---- CORAL loaders ----
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
    hopdt = np.median(np.diff(ts))/1e6 if len(ts) > 2 else 0.25
    w = max(5, int(round(90.0/max(hopdt, 1e-3))) | 1)
    valid &= rolling_ptp(bpm, w) > 2.0
    if maskpath and os.path.exists(maskpath):
        m = pl.read_parquet(maskpath)
        for a, b in zip(m["mask_start"].to_numpy().astype("datetime64[us]").astype("int64"),
                         m["mask_end"].to_numpy().astype("datetime64[us]").astype("int64")):
            valid &= ~((ts >= a) & (ts <= b))
    valid &= ~frozen_runs_mask(bpm, min_hops=8)
    return ts, bpm, valid

scg_ts, scg_bpm, scg_v = load_coral(f"{HERE}/coral_scg/Chelten/out.csv", f"{HERE}/coral_scg/Chelten/mask.parquet")
ecg_ts, ecg_bpm, ecg_v = load_coral(f"{HERE}/coral_ecg/Chelten_ecg_polar_maskfix/out.csv",
                                     f"{HERE}/coral_ecg/Chelten_ecg_polar_maskfix/mask.parquet")

# ---- hand annotations ----
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

def beat_hr(peaks):
    rr = np.diff(peaks)/1e6; hr = 60.0/rr
    tmid = peaks[:-1] + np.diff(peaks)//2
    return tmid, hr

h_ecg_pk, h_ecg_ev_ts, h_ecg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_ecg2.db")
h_scg_pk, h_scg_ev_ts, h_scg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_scg2.db")
span_end = max(h_ecg_pk.max(), h_scg_pk.max())
h_ecg_good, _ = good_fn(h_ecg_ev_ts, h_ecg_ev_lab, span_end)
h_scg_good, _ = good_fn(h_scg_ev_ts, h_scg_ev_lab, span_end)

h_ecg_tmid, h_ecg_hr_raw = beat_hr(h_ecg_pk)
h_scg_tmid, h_scg_hr_raw = beat_hr(h_scg_pk)
h_ecg_hr = smooth(h_ecg_tmid, h_ecg_hr_raw)
h_scg_hr = smooth(h_scg_tmid, h_scg_hr_raw)
h_ecg_v = h_ecg_good(h_ecg_tmid)
h_scg_v = h_scg_good(h_scg_tmid)

pt_v = h_ecg_good(pt_tmid)   # restrict Pan-Tompkins ECG to hand-labeled-good ECG time too

t0 = np.datetime64("2026-06-26T17:26:30").astype("datetime64[us]").astype(int)
t1 = span_end


def align(tsA, hrA, vA, tsB, hrB, vB, win_us=2_000_000, lag_us=0):
    bts, bhr = tsB[vB], hrB[vB]
    A = np.where(vA)[0]; xa, yb, ta = [], [], []
    for i in A:
        t = tsA[i] + lag_us; sel = (bts>=t-win_us)&(bts<=t+win_us)
        if sel.sum() >= 1:
            xa.append(hrA[i]); yb.append(bhr[sel].mean()); ta.append(tsA[i])
    return np.array(xa), np.array(yb), np.array(ta, dtype="int64")

def stats(a, b):
    d = a-b
    return dict(n=int(len(a)), bias=float(d.mean()), mae=float(np.abs(d).mean()),
                within10=float(np.mean(np.abs(d)<=10)), r=float(np.corrcoef(a,b)[0,1]) if len(a)>2 else float("nan"))

def in_window(xa, yb, ta):
    inw = (ta>=t0)&(ta<=t1)
    return xa[inw], yb[inw], stats(xa[inw], yb[inw])

results = {}
# A. Pan-Tompkins ECG vs Hand ECG (no lag, same clock)
xa,yb,ta = align(pt_tmid, pt_hr, pt_v, h_ecg_tmid, h_ecg_hr, h_ecg_v, lag_us=0)
results["Pan-Tompkins ECG vs Hand ECG"] = (*in_window(xa,yb,ta), 0.0)

# B. Pan-Tompkins ECG vs CORAL ECG (no lag, same clock)
xa,yb,ta = align(pt_tmid, pt_hr, pt_v, ecg_ts, ecg_bpm, ecg_v, lag_us=0)
results["Pan-Tompkins ECG vs CORAL ECG"] = (*in_window(xa,yb,ta), 0.0)

# C. Pan-Tompkins ECG vs Hand SCG (lag: shift SCG later by 7s == shift ECG-side earlier;
#    our align() shifts tsA later, so put SCG as A and ECG as B like before, OR
#    equivalently shift PT-ECG EARLIER by same amount -> use tsA=PT with lag=-7)
xa,yb,ta = align(h_scg_tmid, h_scg_hr, h_scg_v, pt_tmid, pt_hr, pt_v, lag_us=int(LAG_HAND*1e6))
results["Hand SCG vs Pan-Tompkins ECG"] = (*in_window(xa,yb,ta), LAG_HAND)

# D. CORAL SCG vs Pan-Tompkins ECG
xa,yb,ta = align(scg_ts, scg_bpm, scg_v, pt_tmid, pt_hr, pt_v, lag_us=int(LAG_CORAL*1e6))
results["CORAL SCG vs Pan-Tompkins ECG"] = (*in_window(xa,yb,ta), LAG_CORAL)

print(f"\n{'comparison':32s} {'lag':>6s}  n     MAE    bias    within10   r")
for label,(xa,yb,R,lag) in results.items():
    print(f"{label:32s} {lag:+5.1f}s {R['n']:4d}  {R['mae']:6.2f}  {R['bias']:+6.2f}  {100*R['within10']:6.1f}%   {R['r']:.3f}")

# ---- figure ----
fig = plt.figure(figsize=(17, 11))
gs = fig.add_gridspec(2, 4, height_ratios=[1.2,1.0], hspace=0.45, wspace=0.35)
axT = fig.add_subplot(gs[0,:])
mE = (ecg_ts>=t0)&(ecg_ts<=t1)
mS = (scg_ts>=t0)&(scg_ts<=t1)
axT.plot(dn(ecg_ts[mE]), ecg_bpm[mE], color="0.85", lw=0.6, zorder=1)
axT.plot(dn(scg_ts[mS]), scg_bpm[mS], color="0.85", lw=0.6, zorder=1)
axT.plot(dn(ecg_ts[ecg_v&mE]), ecg_bpm[ecg_v&mE], color="#2ca02c", lw=1.0, alpha=0.7, label="CORAL ECG", zorder=2)
axT.plot(dn(scg_ts[scg_v&mS]), scg_bpm[scg_v&mS], color="#d62728", lw=1.0, alpha=0.7, label="CORAL SCG", zorder=2)
mPT = (pt_tmid>=t0)&(pt_tmid<=t1)&pt_v
axT.scatter(dn(pt_tmid[mPT]), pt_hr[mPT], s=5, color="#000000", zorder=4, label="Pan-Tompkins ECG (good)")
mHE = (h_ecg_tmid>=t0)&(h_ecg_tmid<=t1)&h_ecg_v
axT.scatter(dn(h_ecg_tmid[mHE]), h_ecg_hr[mHE], s=5, color="#145214", marker="D", zorder=3, label="Hand ECG (good)")
axT.set_xlim(dn(t0)[0], dn(t1)[0]); axT.set_ylim(40,130)
axT.xaxis.set_major_formatter(DateFormatter("%H:%M:%S", tz=UTC))
axT.set_ylabel("HR (bpm)"); axT.legend(loc="upper right", ncol=2, fontsize=8.5)
axT.grid(True, alpha=0.3)
axT.set_title("Pan-Tompkins ECG (black dots) vs CORAL ECG/SCG and Hand ECG -- 17:26:30-end (no lag shift shown)")

colors = ["#e377c2","#8c564b","#bcbd22","#7f7f7f"]
for i,(label,(xa,yb,R,lag)) in enumerate(results.items()):
    ax = fig.add_subplot(gs[1,i])
    ax.scatter(yb, xa, s=14, alpha=0.5, color=colors[i], edgecolor="none")
    lim=[40,130]; ax.plot(lim,lim,"k-",lw=1,alpha=0.6)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    parts = label.split(" vs ")
    ax.set_xlabel(parts[1]); ax.set_ylabel(parts[0])
    lagtag = f" (lag +{lag:.1f}s)" if lag else " (no lag)"
    ax.set_title(label+lagtag, fontsize=9)
    ax.text(0.03,0.97, f"MAE {R['mae']:.2f}  bias {R['bias']:+.2f}\n≤10bpm {100*R['within10']:.1f}%  r={R['r']:.3f}\nn={R['n']}",
            transform=ax.transAxes, va="top", fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.9))
    ax.grid(True, alpha=0.3)

fig.suptitle("[Chelten] Pan-Tompkins ECG detector vs Hand/CORAL, joint-good time -- 17:26:30-end", fontsize=13)
out = f"{HERE}/../figs/chelten_pantompkins_compare.png".replace("../","")
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_pantompkins_compare.png"
fig.savefig(out, dpi=150)
print("->", out)

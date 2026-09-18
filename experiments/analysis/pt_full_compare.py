#!/usr/bin/env python3
"""Build a Pan-Tompkins-STYLE simple beat detector for BOTH channels (same
bandpass->derivative->square->moving-window-integrate->adaptive-threshold
structure; only the frequency band changes to match each signal's beat
morphology: 10-30Hz QRS band for ECG, 40-150Hz heart-sound band for SCG -- the
same band scg_hr.py already established), then compare all 4 series pairwise:
  Hand ECG, Hand SCG, Pan-Tompkins ECG, Pan-Tompkins SCG
restricted to hand-labeled-good time, 17:26:30 to end of annotation, with the
established SCG<->ECG recorder-clock lag applied to any cross-channel pair."""
import os, glob, sqlite3
import numpy as np, polars as pl
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from scipy import signal as sg
from scipy.ndimage import median_filter, uniform_filter1d
import datetime

HERE = "/tmp/dog-test-ecg/dog-test-ecg-code/analysis"
HIVE = "/tmp/dog-test-ecg/dog-test-ecg-code/hive"
UTC = datetime.timezone.utc
SMOOTH_S = 3.0
SEARCH_LO, SEARCH_HI = 50, 200
REFRACT_S = 0.28


def dn(us):
    return mdates.date2num([datetime.datetime.fromtimestamp(t/1e6, tz=UTC) for t in np.atleast_1d(us)])


def detect_beats(x, fs, fc_lo, fc_hi, snr_thr=4.0):
    """Same simple structure for both channels; only the band differs."""
    b = sg.butter(2, [fc_lo/(fs/2), fc_hi/(fs/2)], btype="band")
    xf = sg.filtfilt(*b, x - np.mean(x))
    deriv = np.convolve(xf, np.array([1,2,0,-2,-1])*(fs/8.0), mode="same")
    sqd = deriv**2
    mwi = uniform_filter1d(sqd, max(1, int(0.05*fs)))
    floor = median_filter(mwi, int(2*fs)|1, mode="nearest") + 1e-12
    pk, _ = sg.find_peaks(mwi, distance=int(fs*REFRACT_S), height=None)
    snr = mwi[pk] / floor[pk]
    pk = pk[snr > snr_thr]
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


def smooth(tmid, hr, win_s=SMOOTH_S):
    win_us = int(win_s*1e6)
    out = np.empty(len(tmid))
    for i, t in enumerate(tmid):
        sel = (tmid >= t-win_us) & (tmid <= t+win_us)
        out[i] = hr[sel].mean()
    return out


print("--- ECG: Pan-Tompkins (10-30Hz QRS band) ---")
p_ecg = glob.glob(f"{HIVE}/username=ecg_polar/device=Chelten/stream=0/date=*/data_0.parquet")[0]
df = pl.read_parquet(p_ecg, columns=["ts", "c1"])
ts_e = df["ts"].to_numpy().astype("int64"); x_e = df["c1"].to_numpy().astype(float)
fs_e = float(1e6/np.median(np.diff(ts_e)))
R_e, _ = detect_beats(x_e, fs_e, 10.0, 30.0, snr_thr=4.0)
pt_ecg_tmid, pt_ecg_hr_raw = clean_hr(ts_e[R_e])
print(f"  n R-peaks={len(R_e)}  n beat-HR (cleaned)={len(pt_ecg_tmid)}")

print("--- SCG: Pan-Tompkins-style (40-150Hz heart-sound band) ---")
p_scg = glob.glob(f"{HIVE}/username=scg_mwd/device=Chelten/stream=45/date=*/data_0.parquet")[0]
df = pl.read_parquet(p_scg, columns=["ts", "c1"])
ts_s = df["ts"].to_numpy().astype("int64"); x_s = df["c1"].to_numpy().astype(float)
fs_s = float(1e6/np.median(np.diff(ts_s)))
R_s, _ = detect_beats(x_s, fs_s, 40.0, 150.0, snr_thr=4.0)
pt_scg_tmid, pt_scg_hr_raw = clean_hr(ts_s[R_s])
print(f"  n peaks={len(R_s)}  n beat-HR (cleaned)={len(pt_scg_tmid)}")

pt_ecg_hr = smooth(pt_ecg_tmid, pt_ecg_hr_raw)
pt_scg_hr = smooth(pt_scg_tmid, pt_scg_hr_raw)

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
pt_ecg_v = h_ecg_good(pt_ecg_tmid)
pt_scg_v = h_scg_good(pt_scg_tmid)

t0 = np.datetime64("2026-06-26T17:26:30").astype("datetime64[us]").astype(int)
t1 = span_end

SERIES = {
    "Hand ECG": (h_ecg_tmid, h_ecg_hr, h_ecg_v, "ecg"),
    "Hand SCG": (h_scg_tmid, h_scg_hr, h_scg_v, "scg"),
    "PT ECG": (pt_ecg_tmid, pt_ecg_hr, pt_ecg_v, "ecg"),
    "PT SCG": (pt_scg_tmid, pt_scg_hr, pt_scg_v, "scg"),
}


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

def best_lag(nameA, nameB, lags):
    tsA,hrA,vA,sideA = SERIES[nameA]; tsB,hrB,vB,sideB = SERIES[nameB]
    if sideA == sideB:
        xa,yb,ta = align(tsA,hrA,vA,tsB,hrB,vB, lag_us=0)
        return 0.0, in_window(xa,yb,ta)
    best=(0.0,-2); best_r=None
    for lag_s in lags:
        xa,yb,ta = align(tsA,hrA,vA,tsB,hrB,vB, lag_us=int(lag_s*1e6))
        _,_,R = in_window(xa,yb,ta)
        if R['r'] is not None and not np.isnan(R['r']) and R['r']>best[1]:
            best=(lag_s,R['r'])
    lag_s = best[0]
    xa,yb,ta = align(tsA,hrA,vA,tsB,hrB,vB, lag_us=int(lag_s*1e6))
    return lag_s, in_window(xa,yb,ta)

pairs = [("Hand ECG","Hand SCG"), ("PT ECG","PT SCG"), ("Hand ECG","PT ECG"),
         ("Hand SCG","PT SCG"), ("PT ECG","Hand SCG"), ("Hand ECG","PT SCG")]

lags = np.arange(-20, 20.01, 0.5)
results = {}
print(f"\n{'A vs B':22s} {'lag':>6s}  n     MAE    bias    within10   r")
for a,b in pairs:
    lag, (xa,yb,R) = best_lag(a,b,lags)
    results[f"{a} vs {b}"] = (xa,yb,R,lag)
    print(f"{a+' vs '+b:22s} {lag:+5.1f}s {R['n']:4d}  {R['mae']:6.2f}  {R['bias']:+6.2f}  {100*R['within10']:6.1f}%   {R['r']:.3f}")

import pickle
with open("/tmp/pt_full_results.pkl","wb") as f:
    pickle.dump(dict(results=results, t0=t0, t1=t1), f)
print("saved results pickle")

#!/usr/bin/env python3
"""Template-matching (matched-filter / normalized cross-correlation) beat
detector for BOTH channels, built from hand-clicked beats as the template:
  1. bandpass the raw signal (same bands as before: 10-30Hz ECG, 40-150Hz SCG)
  2. average a window around several hand-clicked peaks (from calm, good
     stretches) into a template waveform
  3. slide the (zero-mean, unit-norm) template across the whole bandpassed
     signal computing normalized cross-correlation (NCC) at every sample
  4. pick peaks in the NCC trace above a threshold, with a canine refractory
  5. same clean_hr (Hampel) + smoothing + good-data restriction as before,
  then compare against hand-clicked HR (all 6 pairwise, same as before)."""
import glob, sqlite3
import numpy as np, polars as pl
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sg
from scipy.ndimage import median_filter
import datetime, pickle

HIVE = "/tmp/dog-test-ecg/dog-test-ecg-code/hive"
UTC = datetime.timezone.utc
SMOOTH_S = 3.0
SEARCH_LO, SEARCH_HI = 50, 200
REFRACT_S = 0.28


def bandpass(x, fs, lo, hi):
    b = sg.butter(2, [lo/(fs/2), hi/(fs/2)], btype="band")
    return sg.filtfilt(*b, x - np.mean(x))


def build_template(xf, ts, peak_times, half_win_s, fs, max_n=60):
    """Average a fixed window around a subset of peak times into one template."""
    half = int(half_win_s * fs)
    use = peak_times[:max_n] if len(peak_times) > max_n else peak_times
    snips = []
    for pt in use:
        idx = np.searchsorted(ts, pt)
        a, b = idx - half, idx + half
        if a >= 0 and b < len(xf):
            snips.append(xf[a:b])
    snips = np.array(snips)
    tmpl = snips.mean(axis=0)
    return tmpl


def ncc_peaks(xf, tmpl, fs, thr=0.5):
    """Sliding normalized cross-correlation of xf against tmpl; return peak sample indices."""
    L = len(tmpl)
    t = tmpl - tmpl.mean()
    t = t / np.linalg.norm(t)
    num = sg.correlate(xf, t, mode="valid", method="fft")   # length N-L+1, aligned to window START
    S1 = np.cumsum(np.insert(xf, 0, 0.0))
    S2 = np.cumsum(np.insert(xf**2, 0, 0.0))
    wsum = S1[L:] - S1[:-L]
    wsumsq = S2[L:] - S2[:-L]
    wmean = wsum / L
    wvar = np.maximum(wsumsq - L * wmean**2, 1e-9)
    denom = np.sqrt(wvar)
    ncc = num / denom
    # peak of NCC corresponds to window START aligned; true peak sits at window start + argmax(|t|) offset
    center_off = int(np.argmax(np.abs(tmpl)))
    pk, _ = sg.find_peaks(ncc, height=thr, distance=int(fs * REFRACT_S))
    return pk + center_off, ncc


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

# ---- ECG: build template from hand-clicked peaks, then NCC across whole signal ----
print("--- ECG template matching ---")
p_ecg = glob.glob(f"{HIVE}/username=ecg_polar/device=Chelten/stream=0/date=*/data_0.parquet")[0]
df = pl.read_parquet(p_ecg, columns=["ts","c1"])
ts_e = df["ts"].to_numpy().astype("int64"); x_e = df["c1"].to_numpy().astype(float)
fs_e = float(1e6/np.median(np.diff(ts_e)))
xf_e = bandpass(x_e, fs_e, 10.0, 30.0)
seed_ecg = h_ecg_pk[(h_ecg_pk>=np.datetime64("2026-06-26T17:27:00").astype("datetime64[us]").astype(int)) &
                     (h_ecg_pk<=np.datetime64("2026-06-26T17:28:00").astype("datetime64[us]").astype(int))]
tmpl_e = build_template(xf_e, ts_e, seed_ecg, 0.08, fs_e, max_n=60)
print(f"  template from {len(seed_ecg)} hand-clicked ECG peaks, len={len(tmpl_e)} samples ({len(tmpl_e)/fs_e*1000:.0f}ms)")
pk_e, ncc_e = ncc_peaks(xf_e, tmpl_e, fs_e, thr=0.82)
R_ts_e = ts_e[pk_e]
print(f"  n peaks detected={len(R_ts_e)}")
tm_ecg_tmid, tm_ecg_hr_raw = clean_hr(R_ts_e)
print(f"  n beat-HR (cleaned)={len(tm_ecg_tmid)}")

# ---- SCG: same ----
print("--- SCG template matching ---")
p_scg = glob.glob(f"{HIVE}/username=scg_mwd/device=Chelten/stream=45/date=*/data_0.parquet")[0]
df = pl.read_parquet(p_scg, columns=["ts","c1"])
ts_s = df["ts"].to_numpy().astype("int64"); x_s = df["c1"].to_numpy().astype(float)
fs_s = float(1e6/np.median(np.diff(ts_s)))
xf_s = bandpass(x_s, fs_s, 40.0, 150.0)
seed_scg = h_scg_pk[(h_scg_pk>=np.datetime64("2026-06-26T17:27:32").astype("datetime64[us]").astype(int)) &
                     (h_scg_pk<=np.datetime64("2026-06-26T17:31:06").astype("datetime64[us]").astype(int))]
tmpl_s = build_template(xf_s, ts_s, seed_scg, 0.12, fs_s, max_n=60)
print(f"  template from {len(seed_scg)} hand-clicked SCG peaks, len={len(tmpl_s)} samples ({len(tmpl_s)/fs_s*1000:.0f}ms)")
pk_s, ncc_s = ncc_peaks(xf_s, tmpl_s, fs_s, thr=0.33)
R_ts_s = ts_s[pk_s]
print(f"  n peaks detected={len(R_ts_s)}")
tm_scg_tmid, tm_scg_hr_raw = clean_hr(R_ts_s)
print(f"  n beat-HR (cleaned)={len(tm_scg_tmid)}")

tm_ecg_hr = smooth(tm_ecg_tmid, tm_ecg_hr_raw)
tm_scg_hr = smooth(tm_scg_tmid, tm_scg_hr_raw)

h_ecg_tmid, h_ecg_hr_raw = beat_hr(h_ecg_pk)
h_scg_tmid, h_scg_hr_raw = beat_hr(h_scg_pk)
h_ecg_hr = smooth(h_ecg_tmid, h_ecg_hr_raw)
h_scg_hr = smooth(h_scg_tmid, h_scg_hr_raw)

h_ecg_v = h_ecg_good(h_ecg_tmid)
h_scg_v = h_scg_good(h_scg_tmid)
tm_ecg_v = h_ecg_good(tm_ecg_tmid)
tm_scg_v = h_scg_good(tm_scg_tmid)

t0 = np.datetime64("2026-06-26T17:26:30").astype("datetime64[us]").astype(int)
t1 = span_end

SERIES = {
    "Hand ECG": (h_ecg_tmid, h_ecg_hr, h_ecg_v, "ecg"),
    "Hand SCG": (h_scg_tmid, h_scg_hr, h_scg_v, "scg"),
    "TM ECG": (tm_ecg_tmid, tm_ecg_hr, tm_ecg_v, "ecg"),
    "TM SCG": (tm_scg_tmid, tm_scg_hr, tm_scg_v, "scg"),
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
    best=(0.0,-2)
    for lag_s in lags:
        xa,yb,ta = align(tsA,hrA,vA,tsB,hrB,vB, lag_us=int(lag_s*1e6))
        _,_,R = in_window(xa,yb,ta)
        if not np.isnan(R['r']) and R['r']>best[1]:
            best=(lag_s,R['r'])
    lag_s = best[0]
    xa,yb,ta = align(tsA,hrA,vA,tsB,hrB,vB, lag_us=int(lag_s*1e6))
    return lag_s, in_window(xa,yb,ta)

pairs = [("Hand ECG","Hand SCG"), ("TM ECG","TM SCG"), ("Hand ECG","TM ECG"),
         ("Hand SCG","TM SCG"), ("TM ECG","Hand SCG"), ("Hand ECG","TM SCG")]

lags = np.arange(-20, 20.01, 0.5)
results = {}
print(f"\n{'A vs B':22s} {'lag':>6s}  n     MAE    bias    within10   r")
for a,b in pairs:
    lag, (xa,yb,R) = best_lag(a,b,lags)
    results[f"{a} vs {b}"] = (xa,yb,R,lag)
    print(f"{a+' vs '+b:22s} {lag:+5.1f}s {R['n']:4d}  {R['mae']:6.2f}  {R['bias']:+6.2f}  {100*R['within10']:6.1f}%   {R['r']:.3f}")

with open("/tmp/tm_full_results.pkl","wb") as f:
    pickle.dump(dict(results=results, t0=t0, t1=t1), f)
print("saved results pickle")

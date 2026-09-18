#!/usr/bin/env python3
"""User-specified Shannon-energy envelope pipeline:
  1. bandpass 10-100Hz
  2. Shannon energy:  SE(n) = -xn(n)^2 * log(xn(n)^2 + eps),  xn = x / max(|x|)
     (+ short moving-average to get the classic "average Shannon energy")
  3. Gaussian low-pass smoothing of the SE envelope (no ringing, unlike a
     sharp-cutoff Butterworth)
  4. "monotone" sharpening: normalize by the 99.5th percentile (q995) of the
     envelope, then exponentiate -> emphasizes true peaks, suppresses noise
     floor, without introducing new zero-crossings/ringing
  5. peak-picking on the sharpened envelope with a short refractory (must be
     shorter than the S1-S2 gap so BOTH can be found), then a cardiac-cycle
     state machine: pair each peak with the next one 180+-20ms later as its
     S2; after S2 there must be >=150ms of refractory before the next S1.
     RR / HR is computed beat-to-beat from accepted S1 times.
For ECG there's no S1/S2 heart-sound doublet, so the same front end
(bandpass->Shannon energy->Gaussian smooth->sharpen) is used but with plain
single-peak-per-refractory picking (no doublet pairing) -- noted honestly."""
import glob, sqlite3
import numpy as np, polars as pl
from scipy import signal as sg
from scipy.ndimage import gaussian_filter1d, uniform_filter1d, median_filter
import datetime, pickle

HIVE = "/tmp/dog-test-ecg/dog-test-ecg-code/hive"
SMOOTH_S = 3.0
SEARCH_LO, SEARCH_HI = 50, 200


def bandpass(x, fs, lo, hi):
    hi = min(hi, 0.45*fs)
    b = sg.butter(2, [lo/(fs/2), hi/(fs/2)], btype="band")
    return sg.filtfilt(*b, x - np.mean(x))


def shannon_energy(xf, fs, avg_win_s=0.02, gauss_sigma_s=0.01):
    xn = xf / (np.max(np.abs(xf)) + 1e-12)
    se = -(xn**2) * np.log(xn**2 + 1e-9)
    se_avg = uniform_filter1d(se, max(1, int(avg_win_s*fs)))
    se_smooth = gaussian_filter1d(se_avg, sigma=max(1, gauss_sigma_s*fs))
    q995 = np.percentile(se_smooth, 99.5)
    sharp = np.exp(se_smooth / max(q995, 1e-12)) - 1.0
    return sharp


def pick_doublets(ts, env, fs, refract_pk_s=0.08, s1s2_lo=0.16, s1s2_hi=0.20, post_s2_refract_s=0.15,
                   height_pctile=80):
    """Find candidate peaks (short refractory so S1 and S2 both survive), then
    run the cardiac-cycle state machine: S1 -> S2 within [s1s2_lo,s1s2_hi] ->
    refractory >= post_s2_refract_s before next S1."""
    thr = np.percentile(env, height_pctile)
    cand, _ = sg.find_peaks(env, height=thr, distance=max(1, int(refract_pk_s*fs)))
    cand_t = ts[cand]
    s1_times = []
    i = 0
    n = len(cand_t)
    while i < n - 1:
        t1 = cand_t[i]
        # find candidate S2 within [t1+s1s2_lo, t1+s1s2_hi] (in us)
        lo = t1 + int(s1s2_lo*1e6); hi = t1 + int(s1s2_hi*1e6)
        j = i + 1
        found = -1
        while j < n and cand_t[j] <= hi:
            if cand_t[j] >= lo:
                found = j; break
            j += 1
        if found >= 0:
            s1_times.append(t1)
            # refractory: skip all candidates within post_s2_refract_s after S2
            s2_t = cand_t[found]
            k = found + 1
            while k < n and cand_t[k] < s2_t + int(post_s2_refract_s*1e6):
                k += 1
            i = k
        else:
            i += 1
    return np.array(s1_times, dtype="int64"), cand_t


def pick_singles(ts, env, fs, refract_s=0.28, height_pctile=80):
    thr = np.percentile(env, height_pctile)
    cand, _ = sg.find_peaks(env, height=thr, distance=max(1, int(refract_s*fs)))
    return ts[cand]


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

print("--- SCG: Shannon-energy envelope + S1/S2 doublet pairing ---")
p_scg = glob.glob(f"{HIVE}/username=scg_mwd/device=Chelten/stream=45/date=*/data_0.parquet")[0]
df = pl.read_parquet(p_scg, columns=["ts","c1"])
ts_s = df["ts"].to_numpy().astype("int64"); x_s = df["c1"].to_numpy().astype(float)
fs_s = float(1e6/np.median(np.diff(ts_s)))
xf_s = bandpass(x_s, fs_s, 10.0, 100.0)
env_s = shannon_energy(xf_s, fs_s)
s1_scg, cand_scg = pick_doublets(ts_s, env_s, fs_s, height_pctile=85)
print(f"  candidate peaks={len(cand_scg)}  accepted S1 (cycles)={len(s1_scg)}")
sh_scg_tmid, sh_scg_hr_raw = clean_hr(s1_scg)
print(f"  n beat-HR (cleaned)={len(sh_scg_tmid)}")

print("--- ECG: same envelope front-end, single-peak-per-refractory (no doublet) ---")
p_ecg = glob.glob(f"{HIVE}/username=ecg_polar/device=Chelten/stream=0/date=*/data_0.parquet")[0]
df = pl.read_parquet(p_ecg, columns=["ts","c1"])
ts_e = df["ts"].to_numpy().astype("int64"); x_e = df["c1"].to_numpy().astype(float)
fs_e = float(1e6/np.median(np.diff(ts_e)))
xf_e = bandpass(x_e, fs_e, 10.0, 100.0)
env_e = shannon_energy(xf_e, fs_e)
R_e = pick_singles(ts_e, env_e, fs_e, refract_s=0.28, height_pctile=85)
print(f"  n peaks={len(R_e)}")
sh_ecg_tmid, sh_ecg_hr_raw = clean_hr(R_e)
print(f"  n beat-HR (cleaned)={len(sh_ecg_tmid)}")

sh_scg_hr = smooth(sh_scg_tmid, sh_scg_hr_raw)
sh_ecg_hr = smooth(sh_ecg_tmid, sh_ecg_hr_raw)

h_ecg_tmid, h_ecg_hr_raw = beat_hr(h_ecg_pk)
h_scg_tmid, h_scg_hr_raw = beat_hr(h_scg_pk)
h_ecg_hr = smooth(h_ecg_tmid, h_ecg_hr_raw)
h_scg_hr = smooth(h_scg_tmid, h_scg_hr_raw)

h_ecg_v = h_ecg_good(h_ecg_tmid)
h_scg_v = h_scg_good(h_scg_tmid)
sh_ecg_v = h_ecg_good(sh_ecg_tmid)
sh_scg_v = h_scg_good(sh_scg_tmid)

t0 = np.datetime64("2026-06-26T17:26:30").astype("datetime64[us]").astype(int)
t1 = span_end

SERIES = {
    "Hand ECG": (h_ecg_tmid, h_ecg_hr, h_ecg_v, "ecg"),
    "Hand SCG": (h_scg_tmid, h_scg_hr, h_scg_v, "scg"),
    "Shannon ECG": (sh_ecg_tmid, sh_ecg_hr, sh_ecg_v, "ecg"),
    "Shannon SCG": (sh_scg_tmid, sh_scg_hr, sh_scg_v, "scg"),
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

pairs = [("Hand ECG","Hand SCG"), ("Shannon ECG","Shannon SCG"), ("Hand ECG","Shannon ECG"),
         ("Hand SCG","Shannon SCG"), ("Shannon ECG","Hand SCG"), ("Hand ECG","Shannon SCG")]

lags = np.arange(-20, 20.01, 0.5)
results = {}
print(f"\n{'A vs B':26s} {'lag':>6s}  n     MAE    bias    within10   r")
for a,b in pairs:
    lag, (xa,yb,R) = best_lag(a,b,lags)
    results[f"{a} vs {b}"] = (xa,yb,R,lag)
    print(f"{a+' vs '+b:26s} {lag:+5.1f}s {R['n']:4d}  {R['mae']:6.2f}  {R['bias']:+6.2f}  {100*R['within10']:6.1f}%   {R['r']:.3f}")

with open("/tmp/shannon_full_results.pkl","wb") as f:
    pickle.dump(dict(results=results, t0=t0, t1=t1), f)
print("saved results pickle")

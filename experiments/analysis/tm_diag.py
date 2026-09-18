#!/usr/bin/env python3
import glob, sqlite3
import numpy as np, polars as pl
from scipy import signal as sg
import datetime

HIVE = "/tmp/dog-test-ecg/dog-test-ecg-code/hive"
REFRACT_S = 0.28

def bandpass(x, fs, lo, hi):
    b = sg.butter(2, [lo/(fs/2), hi/(fs/2)], btype="band")
    return sg.filtfilt(*b, x - np.mean(x))

def build_template(xf, ts, peak_times, half_win_s, fs, max_n=60):
    half = int(half_win_s * fs)
    use = peak_times[:max_n] if len(peak_times) > max_n else peak_times
    snips = []
    for pt in use:
        idx = np.searchsorted(ts, pt)
        a, b = idx - half, idx + half
        if a >= 0 and b < len(xf):
            snips.append(xf[a:b])
    return np.array(snips).mean(axis=0)

def compute_ncc(xf, tmpl, fs):
    L = len(tmpl)
    t = tmpl - tmpl.mean(); t = t/np.linalg.norm(t)
    num = sg.correlate(xf, t, mode="valid", method="fft")
    S1 = np.cumsum(np.insert(xf,0,0.0)); S2 = np.cumsum(np.insert(xf**2,0,0.0))
    wsum = S1[L:]-S1[:-L]; wsumsq = S2[L:]-S2[:-L]
    wmean = wsum/L
    wvar = np.maximum(wsumsq - L*wmean**2, 1e-9)
    return num/np.sqrt(wvar)

def load_peaks(path):
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = '' ORDER BY timestamp")
    return np.array([r[0] for r in cur.fetchall()], dtype="int64")

h_ecg_pk = load_peaks("/tmp/annotation_chelten_ecg2.db")
h_scg_pk = load_peaks("/tmp/annotation_chelten_scg2.db")

print("=== ECG ===")
p = glob.glob(f"{HIVE}/username=ecg_polar/device=Chelten/stream=0/date=*/data_0.parquet")[0]
df = pl.read_parquet(p, columns=["ts","c1"])
ts_e = df["ts"].to_numpy().astype("int64"); x_e = df["c1"].to_numpy().astype(float)
fs_e = float(1e6/np.median(np.diff(ts_e)))
xf_e = bandpass(x_e, fs_e, 10.0, 30.0)
seed_ecg = h_ecg_pk[(h_ecg_pk>=np.datetime64("2026-06-26T17:27:00").astype("datetime64[us]").astype(int)) &
                     (h_ecg_pk<=np.datetime64("2026-06-26T17:28:00").astype("datetime64[us]").astype(int))]
tmpl_e = build_template(xf_e, ts_e, seed_ecg, 0.08, fs_e, max_n=60)
ncc_e = compute_ncc(xf_e, tmpl_e, fs_e)
np.save("/tmp/ncc_e.npy", ncc_e.astype(np.float32))
print("ncc_e percentiles 50/90/95/99/99.9:", np.percentile(ncc_e, [50,90,95,99,99.9]))
for thr in [0.5,0.6,0.7,0.75,0.8,0.85,0.9]:
    pk,_ = sg.find_peaks(ncc_e, height=thr, distance=int(fs_e*REFRACT_S))
    print(f"  thr={thr}: n_peaks={len(pk)}  rate={len(pk)/((ts_e[-1]-ts_e[0])/1e6)*60:.1f}bpm")

print("\n=== SCG ===")
p = glob.glob(f"{HIVE}/username=scg_mwd/device=Chelten/stream=45/date=*/data_0.parquet")[0]
df = pl.read_parquet(p, columns=["ts","c1"])
ts_s = df["ts"].to_numpy().astype("int64"); x_s = df["c1"].to_numpy().astype(float)
fs_s = float(1e6/np.median(np.diff(ts_s)))
xf_s = bandpass(x_s, fs_s, 40.0, 150.0)
seed_scg = h_scg_pk[(h_scg_pk>=np.datetime64("2026-06-26T17:27:32").astype("datetime64[us]").astype(int)) &
                     (h_scg_pk<=np.datetime64("2026-06-26T17:31:06").astype("datetime64[us]").astype(int))]
tmpl_s = build_template(xf_s, ts_s, seed_scg, 0.12, fs_s, max_n=60)
ncc_s = compute_ncc(xf_s, tmpl_s, fs_s)
np.save("/tmp/ncc_s.npy", ncc_s.astype(np.float32))
print("ncc_s percentiles 50/90/95/99/99.9:", np.percentile(ncc_s, [50,90,95,99,99.9]))
for thr in [0.2,0.3,0.35,0.4,0.45,0.5,0.55,0.6]:
    pk,_ = sg.find_peaks(ncc_s, height=thr, distance=int(fs_s*REFRACT_S))
    print(f"  thr={thr}: n_peaks={len(pk)}  rate={len(pk)/((ts_s[-1]-ts_s[0])/1e6)*60:.1f}bpm")

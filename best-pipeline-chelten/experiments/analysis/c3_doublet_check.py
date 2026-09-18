import glob, sqlite3, datetime, pickle
import numpy as np, polars as pl
from scipy import signal as sg
from scipy.ndimage import gaussian_filter1d, uniform_filter1d, percentile_filter

FSD = 200.0

def bandpass(x, fs, lo, hi):
    hi = min(hi, 0.45*fs)
    b = sg.butter(2, [lo/(fs/2), hi/(fs/2)], btype="band")
    return sg.filtfilt(*b, x - np.mean(x))

def shannon_envelope_decimated(xf, ts, fs, avg_win_s=0.02, gauss_sigma_s=0.01):
    xn = xf / (np.max(np.abs(xf)) + 1e-12)
    se = -(xn**2) * np.log(xn**2 + 1e-9)
    se_avg = uniform_filter1d(se, max(1, int(avg_win_s*fs)))
    step = max(1, int(round(fs/FSD)))
    se_d = se_avg[::step]; ts_d = ts[::step]; fsd = fs/step
    se_smooth = gaussian_filter1d(se_d, sigma=max(1, gauss_sigma_s*fsd))
    return ts_d, se_smooth, fsd

def sharpen_local(env, fsd, q995_win_s=8.0, q=99.5):
    win = max(3, int(q995_win_s*fsd)) | 1
    local_q995 = percentile_filter(env, q, size=win, mode="nearest")
    return np.exp(env / np.maximum(local_q995, 1e-9)) - 1.0, local_q995

def pick_doublets(ts, env, fsd, refract_pk_s=0.08, s1s2_lo=0.16, s1s2_hi=0.28, post_s2_refract_s=0.15, thr=0.3):
    cand, _ = sg.find_peaks(env, height=thr, distance=max(1, int(refract_pk_s*fsd)))
    cand_t = ts[cand]
    s1_times = []; s2_times = []
    i = 0; n = len(cand_t)
    while i < n - 1:
        t1 = cand_t[i]
        lo = t1 + int(s1s2_lo*1e6); hi = t1 + int(s1s2_hi*1e6)
        j = i + 1; found = -1
        while j < n and cand_t[j] <= hi:
            if cand_t[j] >= lo:
                found = j; break
            j += 1
        if found >= 0:
            s1_times.append(t1); s2_times.append(cand_t[found])
            s2_t = cand_t[found]
            k = found + 1
            while k < n and cand_t[k] < s2_t + int(post_s2_refract_s*1e6):
                k += 1
            i = k
        else:
            i += 1
    return np.array(s1_times, dtype="int64"), np.array(s2_times, dtype="int64"), cand_t

p = glob.glob("/tmp/dog-test-ecg/dog-test-ecg-code/hive/username=scg_mwd/device=Chelten/stream=45/date=*/data_0.parquet")[0]
df = pl.read_parquet(p, columns=["ts","c1","c2","c3"])
ts = df["ts"].to_numpy().astype("int64")

def to_us(dt):
    return int(dt.replace(tzinfo=datetime.timezone.utc).timestamp()*1e6)

t0 = to_us(datetime.datetime(2026,6,26,17,55,50))
t1 = to_us(datetime.datetime(2026,6,26,17,56,20))
a = np.searchsorted(ts, t0); b = np.searchsorted(ts, t1)
ts_w = ts[a:b]
fs = 1e6 / np.median(np.diff(ts_w[:5000]))
print("fs=", fs, "n=", len(ts_w))

results = {}
for ch in ["c1","c2","c3"]:
    x = df[ch].to_numpy().astype(float)[a:b]
    xf = bandpass(x, fs, 10, 100)
    ts_d, env, fsd = shannon_envelope_decimated(xf, ts_w, fs)
    sharp, q995 = sharpen_local(env, fsd)
    s1, s2, cand = pick_doublets(ts_d, sharp, fsd, thr=0.3, s1s2_lo=0.14, s1s2_hi=0.28)
    results[ch] = dict(x=x, xf=xf, ts_d=ts_d, env=env, sharp=sharp, s1=s1, s2=s2, cand=cand)
    print(f"\n=== {ch} ===  n doublets: {len(s1)}")
    for a1,a2 in zip(s1,s2):
        dt1 = datetime.datetime.utcfromtimestamp(a1/1e6)
        gap_ms = (a2-a1)/1000
        print(f"  S1={dt1}  gap={gap_ms:.1f}ms")

with open("/tmp/c3_doublet_check_results.pkl","wb") as f:
    pickle.dump(dict(ts_w=ts_w, fs=fs, results=results), f)

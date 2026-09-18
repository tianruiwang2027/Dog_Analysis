import glob, pickle, datetime
import numpy as np, polars as pl
from scipy import signal as sg
from scipy.ndimage import gaussian_filter1d, uniform_filter1d, percentile_filter

# ---------- Step 1: extend the Shannon-envelope computation from the previously
# hand-click-restricted window (17:26:30-18:00:08, ~34min) to the FULL overlap between
# the ecg_polar and scg_mwd recordings, so the whole recording -- not just the
# hand-labeled middle -- gets run through the SCG pipeline. Same functions, same
# parameters (bandpass 10-100Hz, avg_win_s=0.02, gauss_sigma_s=0.01, q995_win_s=8.0)
# as shannon_pipeline_lowthr.py / shannon_pipeline_restricted.py -- nothing about the
# SCG-side signal processing changes, only the time window it's applied to. ----------

HIVE = "/tmp/dog-test-ecg/dog-test-ecg-code/hive"
FSD = 200.0
PAD_S = 30.0
UTC = datetime.timezone.utc

def bandpass(x, fs, lo, hi):
    hi = min(hi, 0.45*fs)
    b = sg.butter(2, [lo/(fs/2), hi/(fs/2)], btype="band")
    return sg.filtfilt(*b, x - np.mean(x))

def load_full(glob_pat):
    p = glob.glob(glob_pat)[0]
    df = pl.read_parquet(p, columns=["ts", "c1"])
    ts = df["ts"].to_numpy().astype("int64")
    x = df["c1"].to_numpy().astype(float)
    return ts, x

def load_window(glob_pat, t0, t1, pad_s=PAD_S):
    p = glob.glob(glob_pat)[0]
    df = pl.read_parquet(p, columns=["ts", "c1"])
    ts = df["ts"].to_numpy().astype("int64")
    x = df["c1"].to_numpy().astype(float)
    a = np.searchsorted(ts, t0 - int(pad_s*1e6))
    b = np.searchsorted(ts, t1 + int(pad_s*1e6))
    return ts[a:b], x[a:b]

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

# ---- determine the full overlap between ecg_polar and scg_mwd ----
ecg_pat = f"{HIVE}/username=ecg_polar/device=Chelten/stream=0/date=*/data_0.parquet"
scg_pat = f"{HIVE}/username=scg_mwd/device=Chelten/stream=45/date=*/data_0.parquet"
ts_e_full, _ = load_full(ecg_pat)
ts_s_full, _ = load_full(scg_pat)
ov_lo = max(ts_e_full.min(), ts_s_full.min())
ov_hi = min(ts_e_full.max(), ts_s_full.max())
print(f"ecg_polar span: {datetime.datetime.fromtimestamp(ts_e_full.min()/1e6,tz=UTC)} -> {datetime.datetime.fromtimestamp(ts_e_full.max()/1e6,tz=UTC)}")
print(f"scg_mwd span:   {datetime.datetime.fromtimestamp(ts_s_full.min()/1e6,tz=UTC)} -> {datetime.datetime.fromtimestamp(ts_s_full.max()/1e6,tz=UTC)}")
print(f"overlap: {datetime.datetime.fromtimestamp(ov_lo/1e6,tz=UTC)} -> {datetime.datetime.fromtimestamp(ov_hi/1e6,tz=UTC)}  ({(ov_hi-ov_lo)/1e6/60:.2f} min)")

RESTRICT_PREV = (1782494790000000, 1782496808699088)
print(f"previously covered (hand-click-restricted) window: {datetime.datetime.fromtimestamp(RESTRICT_PREV[0]/1e6,tz=UTC)} -> {datetime.datetime.fromtimestamp(RESTRICT_PREV[1]/1e6,tz=UTC)}")

t0, t1 = int(ov_lo), int(ov_hi)

print("\n--- SCG, full overlap window ---")
ts_s, x_s = load_window(scg_pat, t0, t1)
fs_s = float(1e6/np.median(np.diff(ts_s)))
print(f"  loaded {len(ts_s)} samples @ {fs_s:.1f}Hz")
xf_s = bandpass(x_s, fs_s, 10.0, 100.0)
ts_sd, env_sd, fsd_s = shannon_envelope_decimated(xf_s, ts_s, fs_s)
sharp_s, q995_s = sharpen_local(env_sd, fsd_s, q995_win_s=8.0)
print(f"  ts_sd span: {(ts_sd.max()-ts_sd.min())/1e6/60:.2f} min, {len(ts_sd)} samples @ {fsd_s:.1f}Hz")

with open("/tmp/extended_full_envelope.pkl","wb") as f:
    pickle.dump(dict(ts_sd=ts_sd, env_sd=env_sd, sharp_s=sharp_s, fsd_s=fsd_s, t0=t0, t1=t1,
                      RESTRICT_PREV=RESTRICT_PREV), f)
print("saved /tmp/extended_full_envelope.pkl")

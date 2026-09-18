import glob, pickle, datetime
import numpy as np, polars as pl
from scipy import signal as sg
from scipy.ndimage import gaussian_filter1d, uniform_filter1d, percentile_filter

# ---------- Dasty cross-dog generalization check, SCG side. Same signal-processing
# pipeline as Chelten (bandpass 10-100Hz -> Shannon envelope, decimated to 200Hz -> local
# sharpening), applied to Dasty's own SCG channel over the ECG/SCG overlap window. Axis
# selection follows the SAME convention the existing CORAL run already established for
# this dog (AXIS = {"Dasty": "c2"} in coral_scg.py) rather than "c1"/"mag" used for
# Chelten -- Dasty's raw scg_mwd stream has 3 separate axes (c1,c2,c3), unlike Chelten's
# single pre-selected channel. Uses the recording's OWN self-normalizing global max (the
# pipeline's designed-in device-scale invariance), NOT the Chelten-specific fixed
# reference used to patch the within-Chelten segment-boundary issue -- there is no other
# Dasty window to stay consistent with, and self-normalization is what lets the same CNN
# operate across different sensor units/gains in the first place. ----------

HIVE = "/tmp/dog-test-ecg/dog-test-ecg-code/hive"
FSD = 200.0
PAD_S = 30.0
UTC = datetime.timezone.utc
SCG_AXIS = "c2"  # per coral_scg.py's AXIS dict for Dasty

def bandpass(x, fs, lo, hi):
    hi = min(hi, 0.45*fs)
    b = sg.butter(2, [lo/(fs/2), hi/(fs/2)], btype="band")
    return sg.filtfilt(*b, x - np.mean(x))

def load_window(glob_pat, t0, t1, col, pad_s=PAD_S):
    p = glob.glob(glob_pat)[0]
    df = pl.read_parquet(p, columns=["ts", col])
    ts = df["ts"].to_numpy().astype("int64")
    x = df[col].to_numpy().astype(float)
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

ecg_pat = f"{HIVE}/username=ecg_biopac/device=Dasty/stream=0/date=*/data_0.parquet"
scg_pat = f"{HIVE}/username=scg_mwd/device=Dasty/stream=45/date=*/data_0.parquet"

def full_span(pat, col="ts"):
    p = glob.glob(pat)[0]
    df = pl.read_parquet(p, columns=["ts"])
    ts = df["ts"].to_numpy().astype("int64")
    return ts.min(), ts.max()

e_lo, e_hi = full_span(ecg_pat)
s_lo, s_hi = full_span(scg_pat)
ov_lo, ov_hi = max(e_lo, s_lo), min(e_hi, s_hi)
print(f"ECG-biopac span: {datetime.datetime.fromtimestamp(e_lo/1e6,tz=UTC)} -> {datetime.datetime.fromtimestamp(e_hi/1e6,tz=UTC)}")
print(f"SCG-mwd span:    {datetime.datetime.fromtimestamp(s_lo/1e6,tz=UTC)} -> {datetime.datetime.fromtimestamp(s_hi/1e6,tz=UTC)}")
print(f"overlap: {datetime.datetime.fromtimestamp(ov_lo/1e6,tz=UTC)} -> {datetime.datetime.fromtimestamp(ov_hi/1e6,tz=UTC)}  ({(ov_hi-ov_lo)/1e6/60:.2f} min)")

t0, t1 = int(ov_lo), int(ov_hi)
ts_s, x_s = load_window(scg_pat, t0, t1, SCG_AXIS)
fs_s = float(1e6/np.median(np.diff(ts_s)))
print(f"\nSCG (axis={SCG_AXIS}): loaded {len(ts_s)} samples @ {fs_s:.1f}Hz")

xf_s = bandpass(x_s, fs_s, 10.0, 100.0)
print(f"max|xf_s|={np.max(np.abs(xf_s)):.1f}  p99.9={np.percentile(np.abs(xf_s),99.9):.1f}  p99={np.percentile(np.abs(xf_s),99):.1f}  "
      f"(large gap between max and p99.9 would flag an outlier-spike risk)")

ts_sd, env_sd, fsd_s = shannon_envelope_decimated(xf_s, ts_s, fs_s)
sharp_s, q995_s = sharpen_local(env_sd, fsd_s, q995_win_s=8.0)
print(f"ts_sd span: {(ts_sd.max()-ts_sd.min())/1e6/60:.2f} min, {len(ts_sd)} samples @ {fsd_s:.1f}Hz")

with open("/tmp/dasty_scg_envelope.pkl","wb") as f:
    pickle.dump(dict(ts_sd=ts_sd, env_sd=env_sd, sharp_s=sharp_s, fsd_s=fsd_s, t0=t0, t1=t1), f)
print("saved /tmp/dasty_scg_envelope.pkl")

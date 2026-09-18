import glob, pickle, datetime, sys
import numpy as np, polars as pl
from scipy import signal as sg
from scipy.ndimage import gaussian_filter1d, uniform_filter1d, percentile_filter

# ---------- IMPORTANT CORRECTION vs the first attempt: loading the full 86-minute
# overlap in one shot and computing shannon_envelope_decimated's GLOBAL max-abs
# normalization over that whole span pulled in a much larger transient elsewhere in
# the recording (a big amplitude spike right around 17:22:20, just before the hand
# clicks even start -- almost certainly sensor handling/placement, plus another one
# right at the very end ~18:30:03, likely sensor removal). That single global max
# rescaled the envelope DOWN by ~8.5x even inside the already-validated 17:26-18:00
# window, which the main CNN was never trained to see (it was trained on envelope
# snippets normalized against only the original 34-minute window's own max) -- so
# confirmations collapsed there for a reason that had nothing to do with the new
# data itself. Fix: process each segment as its OWN independently-normalized window
# (padding for filter settle), exactly like the original restricted-window script did
# -- this keeps each segment's envelope scale consistent with what the CNN/S2 models
# were actually trained/calibrated against. ----------

HIVE = "/tmp/dog-test-ecg/dog-test-ecg-code/hive"
FSD = 200.0
PAD_S = 30.0
UTC = datetime.timezone.utc

def bandpass(x, fs, lo, hi):
    hi = min(hi, 0.45*fs)
    b = sg.butter(2, [lo/(fs/2), hi/(fs/2)], btype="band")
    return sg.filtfilt(*b, x - np.mean(x))

def load_window(glob_pat, t0, t1, pad_s=PAD_S):
    p = glob.glob(glob_pat)[0]
    df = pl.read_parquet(p, columns=["ts", "c1"])
    ts = df["ts"].to_numpy().astype("int64")
    x = df["c1"].to_numpy().astype(float)
    a = np.searchsorted(ts, t0 - int(pad_s*1e6))
    b = np.searchsorted(ts, t1 + int(pad_s*1e6))
    return ts[a:b], x[a:b]

REF_MAX = 4862.704192475711  # exact max|xf_s| of the ORIGINAL validated 17:26-18:00 window --
# the scale the main CNN/S2Net were actually calibrated against (via their fixed learned
# scale_main/scale_s2 constants). Reusing this FIXED reference for every new segment (instead
# of each segment computing its own global max) keeps the envelope's absolute amplitude scale
# consistent with what the models were trained to see, rather than being redefined by whatever
# one-off handling/motion spike happens to be the biggest sample in that particular segment
# (confirmed as the cause of a spurious ~8.5x envelope shrinkage when this was first tried as
# one monolithic 86-minute load with its own global max).

def shannon_envelope_decimated(xf, ts, fs, avg_win_s=0.02, gauss_sigma_s=0.01, ref_max=None):
    denom = ref_max if ref_max is not None else np.max(np.abs(xf))
    xn = xf / (denom + 1e-12)
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

SEG = sys.argv[1]  # "before" or "after"
scg_pat = f"{HIVE}/username=scg_mwd/device=Chelten/stream=45/date=*/data_0.parquet"

OV_LO, OV_HI = 1782493462462787, 1782498585501961  # ecg_polar full span (== overlap since scg_mwd is wider)
RESTRICT_PREV = (1782494790000000, 1782496808699088)

if SEG == "before":
    t0, t1 = OV_LO, RESTRICT_PREV[0]
elif SEG == "after":
    t0, t1 = RESTRICT_PREV[1], OV_HI
else:
    raise ValueError(SEG)

print(f"segment={SEG}: {datetime.datetime.fromtimestamp(t0/1e6,tz=UTC)} -> {datetime.datetime.fromtimestamp(t1/1e6,tz=UTC)}  ({(t1-t0)/1e6/60:.2f} min)")

ts_s, x_s = load_window(scg_pat, t0, t1)
fs_s = float(1e6/np.median(np.diff(ts_s)))
print(f"  loaded {len(ts_s)} samples @ {fs_s:.1f}Hz")
xf_s = bandpass(x_s, fs_s, 10.0, 100.0)
print(f"  max|xf_s| in this segment's own window: {np.max(np.abs(xf_s)):.1f}  (REF_MAX used: {REF_MAX:.1f})")
ts_sd, env_sd, fsd_s = shannon_envelope_decimated(xf_s, ts_s, fs_s, ref_max=REF_MAX)
sharp_s, q995_s = sharpen_local(env_sd, fsd_s, q995_win_s=8.0)

with open(f"/tmp/extended_seg_{SEG}_envelope.pkl","wb") as f:
    pickle.dump(dict(ts_sd=ts_sd, env_sd=env_sd, sharp_s=sharp_s, fsd_s=fsd_s, t0=t0, t1=t1), f)
print(f"saved /tmp/extended_seg_{SEG}_envelope.pkl")

#!/usr/bin/env python3
import polars as pl, glob, numpy as np
from scipy import signal as sg
from scipy.ndimage import uniform_filter1d, gaussian_filter1d, percentile_filter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import datetime
UTC = datetime.timezone.utc

def dn(us):
    return mdates.date2num([datetime.datetime.fromtimestamp(t/1e6, tz=UTC) for t in np.atleast_1d(us)])

p = glob.glob("/tmp/dog-test-ecg/dog-test-ecg-code/hive/username=scg_mwd/device=Chelten/stream=45/date=*/data_0.parquet")[0]
df = pl.read_parquet(p, columns=["ts","c1","c2","c3"])
ts = df["ts"].to_numpy().astype("int64")
w0 = np.datetime64("2026-06-26T17:56:00").astype("datetime64[us]").astype(int)
w1 = np.datetime64("2026-06-26T17:56:10").astype("datetime64[us]").astype(int)
PAD = 5_000_000
a = np.searchsorted(ts, w0-PAD); b = np.searchsorted(ts, w1+PAD)
ts_w = ts[a:b]
fs = float(1e6/np.median(np.diff(ts_w)))

def bandpass(x, fs, lo=10.0, hi=100.0):
    hi = min(hi, 0.45*fs)
    bcoef = sg.butter(2, [lo/(fs/2), hi/(fs/2)], btype="band")
    return sg.filtfilt(*bcoef, x - np.mean(x))

def shannon_env(xf, fs, avg_win_s=0.02, gauss_sigma_s=0.01, FSD=200.0):
    xn = xf/(np.max(np.abs(xf))+1e-12); se=-(xn**2)*np.log(xn**2+1e-9)
    se_avg = uniform_filter1d(se, max(1,int(avg_win_s*fs)))
    step=max(1,int(round(fs/FSD))); se_d=se_avg[::step]; fsd=fs/step
    return gaussian_filter1d(se_d, sigma=max(1,gauss_sigma_s*fsd)), fsd, step

def sharpen_local(env, fsd, q995_win_s=8.0, q=99.5):
    win=max(3,int(q995_win_s*fsd))|1
    local_q995 = percentile_filter(env, q, size=win, mode="nearest")
    return np.exp(env/np.maximum(local_q995,1e-9))-1.0

raw = {}
xf = {}
env = {}
sharp = {}
ts_d = None
for col in ["c1","c2","c3"]:
    x = df[col].to_numpy().astype(float)[a:b]
    raw[col] = x
    xfc = bandpass(x, fs)
    xf[col] = xfc
    e, fsd, step = shannon_env(xfc, fs)
    env[col] = e
    sharp[col] = sharpen_local(e, fsd)
    if ts_d is None: ts_d = ts_w[::step]

# vector magnitude of bandpassed signal (axis-fusion candidate)
mag = np.sqrt(xf["c1"]**2 + xf["c2"]**2 + xf["c3"]**2)
e_mag, fsd_mag, step_mag = shannon_env(mag, fs)
sharp_mag = sharpen_local(e_mag, fsd_mag)

mR = (ts_w>=w0)&(ts_w<=w1)
mD = (ts_d>=w0)&(ts_d<=w1)

fig, axs = plt.subplots(8, 1, figsize=(16, 20), sharex=True,
                         gridspec_kw={"height_ratios":[1,1,1,1,1,1,1,1]})
colors = {"c1":"#1f77b4","c2":"#ff7f0e","c3":"#2ca02c"}
for i, col in enumerate(["c1","c2","c3"]):
    axs[i].plot(dn(ts_w[mR]), raw[col][mR], color=colors[col], lw=0.6)
    axs[i].set_ylabel(f"raw {col}"); axs[i].set_title(f"Raw {col}")
    axs[i].grid(True, alpha=0.3)
for i, col in enumerate(["c1","c2","c3"]):
    axs[3+i].plot(dn(ts_w[mR]), xf[col][mR], color=colors[col], lw=0.7)
    axs[3+i].set_ylabel(f"bp {col}"); axs[3+i].set_title(f"Bandpassed 10-100Hz {col}")
    axs[3+i].grid(True, alpha=0.3)
axs[6].plot(dn(ts_d[mD]), sharp["c1"][mD], color=colors["c1"], lw=1, label="c1", alpha=0.8)
axs[6].plot(dn(ts_d[mD]), sharp["c2"][mD], color=colors["c2"], lw=1, label="c2", alpha=0.8)
axs[6].plot(dn(ts_d[mD]), sharp["c3"][mD], color=colors["c3"], lw=1, label="c3", alpha=0.8)
axs[6].axhline(0.3, color="0.3", ls="--", lw=0.8)
axs[6].set_ylabel("sharpened env"); axs[6].set_title("Shannon-sharpened envelope, all 3 axes overlaid")
axs[6].legend(fontsize=8); axs[6].grid(True, alpha=0.3)
axs[7].plot(dn(ts_d[mD]), sharp_mag[mD], color="k", lw=1.2)
axs[7].axhline(0.3, color="0.3", ls="--", lw=0.8)
axs[7].set_ylabel("sharpened env"); axs[7].set_title("Shannon-sharpened envelope of VECTOR MAGNITUDE sqrt(c1^2+c2^2+c3^2)")
axs[7].xaxis.set_major_formatter(DateFormatter("%H:%M:%S", tz=UTC))
axs[7].grid(True, alpha=0.3)

fig.suptitle("[Chelten] SCG 3-axis comparison -- 17:56:00-17:56:10", fontsize=13)
fig.tight_layout()
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_axes_compare.png"
fig.savefig(out, dpi=140)
print("->", out)

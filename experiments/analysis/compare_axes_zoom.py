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
w0 = np.datetime64("2026-06-26T17:56:03").astype("datetime64[us]").astype(int)
w1 = np.datetime64("2026-06-26T17:56:07").astype("datetime64[us]").astype(int)
PAD = 3_000_000
a = np.searchsorted(ts, w0-PAD); b = np.searchsorted(ts, w1+PAD)
ts_w = ts[a:b]
fs = float(1e6/np.median(np.diff(ts_w)))

def bandpass(x, fs, lo=10.0, hi=100.0):
    hi = min(hi, 0.45*fs)
    bcoef = sg.butter(2, [lo/(fs/2), hi/(fs/2)], btype="band")
    return sg.filtfilt(*bcoef, x - np.mean(x))

raw = {}; xf = {}
for col in ["c1","c2","c3"]:
    x = df[col].to_numpy().astype(float)[a:b]
    raw[col] = x
    xf[col] = bandpass(x, fs)

mR = (ts_w>=w0)&(ts_w<=w1)

fig, axs = plt.subplots(6, 1, figsize=(16, 14), sharex=True)
colors = {"c1":"#1f77b4","c2":"#ff7f0e","c3":"#2ca02c"}
for i, col in enumerate(["c1","c2","c3"]):
    axs[i].plot(dn(ts_w[mR]), raw[col][mR], color=colors[col], lw=0.8)
    axs[i].set_ylabel(f"raw {col}"); axs[i].set_title(f"Raw {col} (zoomed)")
    axs[i].grid(True, alpha=0.3)
for i, col in enumerate(["c1","c2","c3"]):
    axs[3+i].plot(dn(ts_w[mR]), xf[col][mR], color=colors[col], lw=0.8)
    axs[3+i].set_ylabel(f"bp {col}"); axs[3+i].set_title(f"Bandpassed 10-100Hz {col} (zoomed)")
    axs[3+i].grid(True, alpha=0.3)
axs[5].xaxis.set_major_formatter(DateFormatter("%H:%M:%S", tz=UTC))
fig.suptitle("[Chelten] SCG 3-axis ZOOM -- 17:56:03-17:56:07", fontsize=13)
fig.tight_layout()
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_axes_compare_zoom.png"
fig.savefig(out, dpi=140)
print("->", out)

# save arrays for further spectral analysis
import pickle
with open("/tmp/axes_zoom_data.pkl","wb") as f:
    pickle.dump(dict(ts_w=ts_w, raw=raw, xf=xf, fs=fs, w0=w0, w1=w1), f)

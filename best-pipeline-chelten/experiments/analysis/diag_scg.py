#!/usr/bin/env python3
"""Is there a cardiac signal in the SCG at all? Beat-triggered average (BTA) of
the accelerometer locked to the ECG R-peaks: if a repeatable complex emerges, the
SCG carries the heartbeat (a tracking-config problem); if it's flat, the sensor's
SCG doesn't carry trackable beats (a sensor/placement limitation). Also shows raw
axes with R-peak markers, per-axis envelope autocorrelation at the true RR, and a
spectrogram. Reads short windows only.

Usage: diag_scg.py <Dog> <ecg_modality> <local 'YYYY-MM-DD HH:MM:SS'> [dur_s]
"""
import sys, os, glob
import numpy as np, polars as pl
from scipy import signal as sg
import datetime
from zoneinfo import ZoneInfo
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
HIVE = os.path.join(os.path.dirname(HERE), "hive")
FIGS = os.path.join(os.path.dirname(HERE), "figs")
TZ = ZoneInfo("America/Chicago")

def to_us(s): return int(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ).timestamp()*1e6)
def part(m,d,st="0"): return glob.glob(f"{HIVE}/username={m}/device={d}/stream={st}/date=*/data_0.parquet")[0]

dog, ecgmod, win = sys.argv[1], sys.argv[2], sys.argv[3]
dur = int(sys.argv[4]) if len(sys.argv) > 4 else 60
t0 = to_us(win); t1 = t0 + dur*1_000_000

# SCG 3 axes
sd = pl.read_parquet(part("scg_mwd", dog, "45"), columns=["ts","c1","c2","c3"])
sts = sd["ts"].to_numpy().astype("int64"); m = (sts>=t0)&(sts<t1); sts = sts[m]
fs = 1e6/np.median(np.diff(sts))
ax = {"x": sd["c1"].to_numpy()[m].astype(float), "y": sd["c2"].to_numpy()[m].astype(float),
      "z": sd["c3"].to_numpy()[m].astype(float)}
ax["mag"] = np.sqrt(sum((v-v.mean())**2 for v in ax.values()))

# ECG R-peaks in window (from ecg_hr output beat times)
ehr = pl.read_parquet(os.path.join(HERE,"ecg_hr",f"{dog}_{ecgmod}.parquet"))
rts = ehr["ts"].to_numpy().astype("int64"); rts = rts[(rts>=t0)&(rts<t1)]
rr = np.median(np.diff(rts))/1e6 if len(rts)>2 else np.nan
print(f"{dog} {win}+{dur}s  SCG fs={fs:.0f}Hz n={len(sts):,}  ECG beats={len(rts)} medRR={rr:.3f}s ({60/rr:.0f}bpm)")

def bp(x, lo, hi):
    b = sg.butter(4,[lo/(fs/2),hi/(fs/2)],btype="band"); return sg.filtfilt(*b, x-x.mean())

# Beat-triggered average over +-0.30 s, on bandpassed signal + its envelope
half = int(0.30*fs); lag = np.arange(-half, half)/fs*1000
def bta(sig):
    segs = [sig[i-half:i+half] for t in rts for i in [int(np.searchsorted(sts, t))] if half<=i<len(sig)-half]
    A = np.array(segs); return A.mean(0), A.std(0), len(A)

fig, axs = plt.subplots(3, 2, figsize=(16, 12))
# (0,0) raw axes 8s + R markers
tt = (sts - sts[0])/1e6; show = tt < 8
for k in ["x","y","z"]:
    axs[0,0].plot(tt[show], ax[k][show], lw=0.5, label=k)
for t in rts:
    if (t-sts[0])/1e6 < 8: axs[0,0].axvline((t-sts[0])/1e6, color="r", lw=0.6, alpha=0.5)
axs[0,0].legend(loc="upper right"); axs[0,0].set_title(f"{dog} raw accel (8s) + ECG R-peaks (red)"); axs[0,0].set_xlabel("s")

# (0,1) bandpassed best axis + envelope (8s) + R markers
best = max(ax, key=lambda k: np.var(bp(ax[k],8,40)))
xf = bp(ax[best],8,40); env=np.abs(sg.hilbert(xf))
axs[0,1].plot(tt[show], xf[show], lw=0.6, color="0.3", label=f"{best} bp8-40")
axs[0,1].plot(tt[show], env[show], lw=0.8, color="orange", label="envelope")
for t in rts:
    if (t-sts[0])/1e6 < 8: axs[0,1].axvline((t-sts[0])/1e6, color="r", lw=0.6, alpha=0.5)
axs[0,1].legend(loc="upper right"); axs[0,1].set_title(f"bandpassed {best}-axis + envelope + R-peaks"); axs[0,1].set_xlabel("s")

# (1,0) BTA of bandpassed each axis
for k in ["x","y","z","mag"]:
    mu,sd_,nA = bta(bp(ax[k],8,40)); axs[1,0].plot(lag, mu, label=f"{k} (n={nA})")
axs[1,0].axvline(0,color="r",lw=1); axs[1,0].legend(); axs[1,0].set_xlabel("ms from R-peak")
axs[1,0].set_title("Beat-triggered average — bandpassed accel (flat = no cardiac lock)")

# (1,1) BTA of envelope (energy) each axis
for k in ["x","y","z","mag"]:
    mu,sd_,nA = bta(np.abs(sg.hilbert(bp(ax[k],8,40)))); axs[1,1].plot(lag, mu, label=k)
axs[1,1].axvline(0,color="r",lw=1); axs[1,1].legend(); axs[1,1].set_xlabel("ms from R-peak")
axs[1,1].set_title("Beat-triggered average — accel ENVELOPE (SCG energy)")

# (2,0) envelope autocorr of best axis: peak near true RR?
e = np.abs(sg.hilbert(xf)); e=e-e.mean()
n=1<<int(np.ceil(np.log2(2*len(e)))); f=np.fft.rfft(e,n); acf=np.fft.irfft(f*np.conj(f),n)[:len(e)]
acf/=acf[0]; lags=np.arange(len(acf))/fs
sel=lags<2.0
axs[2,0].plot(lags[sel], acf[sel])
if np.isfinite(rr): axs[2,0].axvline(rr,color="r",ls="--",label=f"ECG RR={rr:.3f}s")
axs[2,0].legend(); axs[2,0].set_xlabel("lag (s)"); axs[2,0].set_title(f"{best}-axis envelope autocorrelation")

# (2,1) spectrogram of best axis (broad band)
f_,t_,S = sg.spectrogram(ax[best]-ax[best].mean(), fs=fs, nperseg=int(2*fs), noverlap=int(1.5*fs))
axs[2,1].pcolormesh(t_, f_[f_<80], 10*np.log10(S[f_<80]+1e-12), shading="auto")
axs[2,1].set_ylim(0,80); axs[2,1].set_xlabel("s"); axs[2,1].set_ylabel("Hz")
axs[2,1].set_title(f"{best}-axis spectrogram (0-80 Hz)")

fig.suptitle(f"{dog} SCG cardiac-content diagnostic @ {win} (+{dur}s) — ECG {60/rr:.0f} bpm reference", fontsize=14)
fig.tight_layout(rect=[0,0,1,0.98])
out=os.path.join(FIGS, f"diag_scg_{dog}.png"); fig.savefig(out, dpi=140); print("->",out)

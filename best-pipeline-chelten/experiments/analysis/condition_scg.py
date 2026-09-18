#!/usr/bin/env python3
"""Intentional SCG conditioning, visually inspected against the ECG beats.
Goal: turn the raw accelerometer into a clean ONE-PULSE-PER-BEAT signal (the S1
heart sound) so rate extraction is unambiguous. Compares envelope methods
(Hilbert | Shannon energy | RMS) of the 40-150 Hz heart-sound band, overlaid on
ECG R-peaks, plus a zoom on a few beats and a per-method 'pulses-per-beat' count.

Usage: condition_scg.py <Dog> <ecg_modality> <start 'YYYY-MM-DD HH:MM:SS'> [dur_s]
"""
import sys, os, glob
import numpy as np, polars as pl
from scipy import signal as sg
from scipy.ndimage import uniform_filter1d
import datetime
from zoneinfo import ZoneInfo
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
HIVE = os.path.join(os.path.dirname(HERE), "hive")
FIGS = os.path.join(os.path.dirname(HERE), "figs")
TZ = ZoneInfo("America/Chicago")
def to_us(s): return int(datetime.datetime.strptime(s,"%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ).timestamp()*1e6)
def part(m,d,st="0"): return glob.glob(f"{HIVE}/username={m}/device={d}/stream={st}/date=*/data_0.parquet")[0]

dog, ecgmod, start = sys.argv[1], sys.argv[2], sys.argv[3]
dur = int(sys.argv[4]) if len(sys.argv) > 4 else 60
t0 = to_us(start); t1 = t0 + dur*1_000_000
AXIS = {"Dasty":"c2","Chelten":"c1","Chuck":"c1"}.get(dog,"c2")

sd = pl.read_parquet(part("scg_mwd",dog,"45"), columns=["ts",AXIS])
sts = sd["ts"].to_numpy().astype("int64"); m=(sts>=t0)&(sts<t1); sts=sts[m]
x = sd[AXIS].to_numpy()[m].astype(float); fs = 1e6/np.median(np.diff(sts))
ehr = pl.read_parquet(os.path.join(HERE,"ecg_hr",f"{dog}_{ecgmod}.parquet"))
rts = ehr["ts"].to_numpy().astype("int64"); rts = rts[(rts>=t0)&(rts<t1)]
rr = np.median(np.diff(rts))/1e6
print(f"{dog} axis={AXIS} {start}+{dur}s fs={fs:.0f} ECG {len(rts)} beats medRR={rr:.3f}s ({60/rr:.0f}bpm)")

def bandpass(x, lo, hi): b=sg.butter(4,[lo/(fs/2),hi/(fs/2)],btype="band"); return sg.filtfilt(*b, x-x.mean())
xf = bandpass(x, 40, 150)

def smooth(v, ms): return uniform_filter1d(v, max(1,int(ms/1000*fs)))
def norm(v): return (v - np.percentile(v,5)) / (np.percentile(v,99) - np.percentile(v,5) + 1e-9)

# --- envelope methods of the heart-sound band ---
env_hil = smooth(np.abs(sg.hilbert(xf)), 30)
xn = xf/ (np.abs(xf).max()+1e-9)
shannon = -(xn**2) * np.log(xn**2 + 1e-12)            # average Shannon energy (Liang 1997)
env_shan = smooth(shannon, 40)
env_rms = np.sqrt(smooth(xf**2, 40))
envs = {"Hilbert":env_hil, "Shannon":env_shan, "RMS":env_rms}

# S1 isolation: 0.28 s refractory (> the ~170 ms S1->S2 gap) keeps one dominant peak/beat
REFRACT = 0.28
def detect_s1(env):
    e = norm(env)
    pk,_ = sg.find_peaks(e, distance=int(REFRACT*fs), prominence=0.10)
    return pk, len(pk)/max(1,len(rts))
stats = {k: detect_s1(v) for k,v in envs.items()}

def ecg_windowed(thops, win_us=2_000_000):
    bts = rts
    bhr = 60.0/ (np.diff(bts)/1e6); bmid = bts[:-1]+np.diff(bts)//2
    out=np.full(len(thops),np.nan)
    for i,t in enumerate(thops):
        sel=(bmid>=t-win_us)&(bmid<=t+win_us)
        if sel.sum()>=2: out[i]=bhr[sel].mean()
    return out

print(f"  {'method':<9}{'peaks':>6}{'/beat':>7}{'SCG-HR':>8}{'vs ECG MAE':>11}{'bias':>7}{'r':>6}")
for k,(pk,ppb) in stats.items():
    s1ts = sts[pk]; hr = 60.0/(np.diff(s1ts)/1e6); tmid = s1ts[:-1]+np.diff(s1ts)//2
    good = (hr>=50)&(hr<=200); hr,tmid = hr[good],tmid[good]
    y = ecg_windowed(tmid); v=np.isfinite(y)
    mae = np.abs(hr[v]-y[v]).mean() if v.sum()>5 else np.nan
    bias = (hr[v]-y[v]).mean() if v.sum()>5 else np.nan
    r = np.corrcoef(hr[v],y[v])[0,1] if v.sum()>5 else np.nan
    print(f"  {k:<9}{len(pk):>6}{ppb:>7.2f}{np.median(hr):>8.0f}{mae:>11.1f}{bias:>+7.1f}{r:>6.2f}")

# --------------- figure ---------------
fig, axs = plt.subplots(4, 1, figsize=(16, 12))
tt = (sts - sts[0])/1e6; SHOW = 8.0; s = tt < SHOW
rloc = (rts - sts[0])/1e6; rs = rloc[rloc < SHOW]
axs[0].plot(tt[s], x[s], lw=0.5, color="0.4"); axs[0].set_title(f"{dog} raw accel {AXIS} (8s)  — gravity DC + motion")
axs[1].plot(tt[s], xf[s], lw=0.5, color="navy"); axs[1].set_title("bandpass 40-150 Hz (S1/S2 heart-sound bursts)")
for r in rs: axs[1].axvline(r, color="r", lw=0.6, alpha=0.5)
for k,c in [("Hilbert","green"),("Shannon","darkorange"),("RMS","purple")]:
    axs[2].plot(tt[s], norm(envs[k])[s], lw=1.0, color=c, label=f"{k} ({stats[k][1]:.2f}/beat)")
for r in rs: axs[2].axvline(r, color="r", lw=0.7, alpha=0.6)
axs[2].legend(loc="upper right"); axs[2].set_title("conditioned envelopes (normalized) + ECG R-peaks (red) — want ONE peak/beat")
# zoom 3s on Shannon with detected peaks
z = tt < 3.0; pk,_ = stats["Shannon"]
pkz = pk[(tt[pk] < 3.0)]
axs[3].plot(tt[z], norm(env_shan)[z], color="darkorange", lw=1.2, label="Shannon env")
axs[3].plot(tt[pkz], norm(env_shan)[pkz], "kv", ms=8, label="detected S1?")
for r in rloc[rloc<3.0]: axs[3].axvline(r, color="r", lw=0.8, alpha=0.6)
axs[3].legend(loc="upper right"); axs[3].set_title("zoom (3s): Shannon envelope + detected peaks vs ECG R-peaks (red)")
for a in axs: a.set_xlabel("s"); a.grid(alpha=0.3)
fig.suptitle(f"{dog} SCG conditioning — ECG {60/rr:.0f} bpm reference", fontsize=14)
fig.tight_layout(rect=[0,0,1,0.98])
out = os.path.join(FIGS, f"condition_scg_{dog}.png"); fig.savefig(out, dpi=140); print("->", out)

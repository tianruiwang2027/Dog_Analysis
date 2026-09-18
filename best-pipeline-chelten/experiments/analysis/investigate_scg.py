#!/usr/bin/env python3
"""Rigorous test for beat-locked cardiac energy in the SCG, immune to RSA and to
SCG/ECG clock offset: beat-triggered average of the SCG envelope scanned over a
WIDE lag (-1.5..+1.5 s). A real SCG produces a sharp, prominent peak at some lag
(= clock offset + AO delay); pure noise/motion stays flat. Swept over axes x/y/z/mag
and several bands, on the SCG∩ECG window. Prominence = (peak-median)/MAD of the
lag curve. Reports the winner per dog and plots the best lag curve + its BTA.

Usage: investigate_scg.py <Dog> <ecg_modality> <start 'YYYY-MM-DD HH:MM:SS'> <dur_s>
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
BANDS = [(8, 40), (15, 60), (20, 80), (30, 120), (40, 150)]
AXES = ["x", "y", "z", "mag"]

def to_us(s): return int(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ).timestamp()*1e6)
def part(m,d,st="0"): return glob.glob(f"{HIVE}/username={m}/device={d}/stream={st}/date=*/data_0.parquet")[0]

dog, ecgmod, start, dur = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
t0 = to_us(start); t1 = t0 + dur*1_000_000

sd = pl.read_parquet(part("scg_mwd", dog, "45"), columns=["ts","c1","c2","c3"])
sts = sd["ts"].to_numpy().astype("int64"); m=(sts>=t0)&(sts<t1)
sts=sts[m]; fs=1e6/np.median(np.diff(sts))
raw={"x":sd["c1"].to_numpy()[m].astype(float),"y":sd["c2"].to_numpy()[m].astype(float),"z":sd["c3"].to_numpy()[m].astype(float)}
raw["mag"]=np.sqrt(sum((v-v.mean())**2 for v in raw.values()))

ehr=pl.read_parquet(os.path.join(HERE,"ecg_hr",f"{dog}_{ecgmod}.parquet"))
rts=ehr["ts"].to_numpy().astype("int64"); rts=rts[(rts>=t0+1_600_000)&(rts<t1-1_600_000)]
print(f"{dog} {start}+{dur}s SCG n={len(sts):,} fs={fs:.0f}  ECG beats(usable)={len(rts)}")
if len(rts)<20 or len(sts)<fs*30:
    print("insufficient data"); sys.exit(0)

LAG=1.5; half=int(LAG*fs); lags=np.arange(-half,half)/fs
def bp(x,lo,hi):
    b=sg.butter(4,[lo/(fs/2),hi/(fs/2)],btype="band"); return sg.filtfilt(*b,x-x.mean())

results=[]
ridx=np.searchsorted(sts, rts)
for (lo,hi) in BANDS:
    for a in AXES:
        env=np.abs(sg.hilbert(bp(raw[a],lo,hi)))
        env=(env-env.mean())/(env.std()+1e-9)
        acc=np.zeros(2*half); k=0
        for i in ridx:
            if half<=i<len(env)-half:
                acc+=env[i-half:i+half]; k+=1
        bta=acc/max(k,1)
        med=np.median(bta); mad=np.median(np.abs(bta-med))+1e-9
        prom=(bta.max()-med)/(1.4826*mad)
        peak_lag=lags[np.argmax(bta)]
        results.append(((lo,hi),a,prom,peak_lag,bta,k))

results.sort(key=lambda r:-r[2])
print("  band       axis  prominence  peak_lag(ms)")
for (b,a,prom,pl_,bta,k) in results[:8]:
    print(f"   {str(b):<9} {a:<4} {prom:8.2f}   {1000*pl_:+7.0f}")

# plot the top-3 BTA lag curves
fig,axs=plt.subplots(2,1,figsize=(16,9))
for (b,a,prom,pl_,bta,k) in results[:3]:
    axs[0].plot(lags*1000, bta, label=f"{a} {b} prom={prom:.1f} lag={1000*pl_:+.0f}ms (n={k})")
axs[0].axvline(0,color="r",lw=1,alpha=0.5); axs[0].legend()
axs[0].set_xlabel("lag from R-peak (ms)"); axs[0].set_ylabel("envelope BTA (z)")
axs[0].set_title(f"{dog} SCG beat-triggered envelope vs lag — sharp prominent peak = cardiac signal present")
# zoom best to +-400ms
b,a,prom,pl_,bta,k=results[0]
zoom=np.abs(lags)<0.4
axs[1].plot(lags[zoom]*1000, bta[zoom]); axs[1].axvline(1000*pl_,color="g",ls="--",label=f"peak {1000*pl_:+.0f}ms")
axs[1].axvline(0,color="r",lw=1,alpha=0.5); axs[1].legend()
axs[1].set_xlabel("lag (ms)"); axs[1].set_title(f"best: {a} {b} (zoom)  prominence={prom:.2f}")
fig.tight_layout(); out=os.path.join(FIGS,f"investigate_scg_{dog}.png"); fig.savefig(out,dpi=140); print("->",out)
print(f"\nVERDICT: best prominence={results[0][2]:.2f} ({'LIKELY cardiac' if results[0][2]>6 else 'WEAK/absent'})  "
      f"axis={results[0][1]} band={results[0][0]} lag={1000*results[0][3]:+.0f}ms")

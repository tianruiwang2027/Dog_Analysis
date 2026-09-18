#!/usr/bin/env python3
"""Parallel grid search for the best SCG->HR config, scored against the ECG
reference on the overlap window. Scores CORAL's HR against a WINDOW-AVERAGED ECG
HR (±2 s, matching CORAL's analysis window) so we measure real tracking, not the
RSA-timescale mismatch. Prints a ranked table.

Usage: tune_scg.py <Dog> <ecg_modality> <start 'YYYY-MM-DD HH:MM:SS'> <dur_s>
"""
import sys, os, glob, subprocess, tempfile, shutil
import numpy as np, polars as pl
from scipy import signal as sg
from concurrent.futures import ThreadPoolExecutor
import datetime
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
HIVE = os.path.join(os.path.dirname(HERE), "hive")
CORAL = shutil.which("coral-st") or "coral-st"
TZ = ZoneInfo("America/Chicago")
def to_us(s): return int(datetime.datetime.strptime(s,"%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ).timestamp()*1e6)
def part(m,d,st="0"): return glob.glob(f"{HIVE}/username={m}/device={d}/stream={st}/date=*/data_0.parquet")[0]

dog, ecgmod, start, dur = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
t0=to_us(start); t1=t0+dur*1_000_000
sd=pl.read_parquet(part("scg_mwd",dog,"45"),columns=["ts","c1","c2","c3"])
sts=sd["ts"].to_numpy().astype("int64"); m=(sts>=t0)&(sts<t1); sts=sts[m]
fs=1e6/np.median(np.diff(sts))
raw={"c1":sd["c1"].to_numpy()[m].astype(float),"c2":sd["c2"].to_numpy()[m].astype(float),"c3":sd["c3"].to_numpy()[m].astype(float)}
raw["mag"]=np.sqrt(sum((v-v.mean())**2 for v in [raw["c1"],raw["c2"],raw["c3"]]))
ehr=pl.read_parquet(os.path.join(HERE,"ecg_hr",f"{dog}_{ecgmod}.parquet"))
bts=ehr["ts"].to_numpy().astype("int64"); bhr=ehr["bpm"].to_numpy()
mm=(bts>=t0)&(bts<t1); bts,bhr=bts[mm],bhr[mm]
print(f"{dog} {start}+{dur}s SCG n={len(sts):,} fs={fs:.0f}  ECG beats={len(bts)} median={np.median(bhr):.0f}bpm")

def bp(x,lo,hi): b=sg.butter(4,[lo/(fs/2),hi/(fs/2)],btype="band"); return sg.filtfilt(*b,x-x.mean())
def envof(x,lo,hi):
    xf=bp(x,lo,hi); bl=sg.butter(4,12/(fs/2),btype="low"); return sg.filtfilt(*bl,np.abs(xf))

def ecg_windowed(thops, win_us=2_000_000):
    """mean ECG beat-HR within +-win of each hop (matches CORAL window timescale)."""
    out=np.full(len(thops),np.nan)
    for i,t in enumerate(thops):
        sel=(bts>=t-win_us)&(bts<=t+win_us)
        if sel.sum()>=2: out[i]=bhr[sel].mean()
    return out

CFGS=[]
for axis in ["c1","c2","c3"]:
    for band in [(40,150),(30,120),(50,180),(25,200)]:
        for afs,hop,wins in [("512","128","256,512,1024,1536"),("1024","256","512,1024,2048,3072")]:
            CFGS.append(dict(mode="raw",axis=axis,band=band,fc=band,afs=afs,hop=hop,wins=wins))

tmp=tempfile.mkdtemp(prefix="tune_")
def run(cfg):
    sig = raw[cfg["axis"]] if cfg["mode"]=="raw" else envof(raw[cfg["axis"]],*cfg["band"])
    tag=f"{cfg['mode']}_{cfg['axis']}_{cfg['band'][0]}-{cfg['band'][1]}"
    inp=os.path.join(tmp,tag+".parquet"); outp=os.path.join(tmp,tag+".csv")
    pl.DataFrame({"ts":sts,"c1":sig}).write_parquet(inp)
    cmd=[CORAL,"-i",inp,"-o",outp,"--fc-low",str(cfg["fc"][0]),"--fc-high",str(cfg["fc"][1]),
         "--analysis-fs",cfg["afs"],"--analysis-hop",cfg["hop"],"--analysis-windows",cfg["wins"],
         "--alpha","0.0","--beta","30","--gamma","3","--delta","10","--epsilon","0.005","--zeta","5","--eta","4",
         "--search-bpm-low","50","--search-bpm-high","200","--max-delta-pct-up","20","--max-delta-pct-down","20"]
    r=subprocess.run(cmd,capture_output=True,text=True)
    if r.returncode!=0: return (tag,None)
    k=pl.read_csv(outp)
    cts=k["ts"].to_numpy().astype("datetime64[us]").astype("int64"); cb=k["bpm"].to_numpy(); cq=k["sqi"].to_numpy()
    ok=cq>=0.10
    y=ecg_windowed(cts); v=ok&np.isfinite(y)
    if v.sum()<10: return (tag,None)
    d=cb[v]-y[v]
    return (tag,dict(n=int(v.sum()),mae=float(np.abs(d).mean()),bias=float(d.mean()),
                     w10=float(np.mean(np.abs(d)<=10)),r=float(np.corrcoef(cb[v],y[v])[0,1]),
                     med=float(np.median(cb[ok]))))

with ThreadPoolExecutor(max_workers=8) as ex:
    res=list(ex.map(run,CFGS))
res=[(t,s) for t,s in res if s]
res.sort(key=lambda x:(x[1]["mae"]))
print(f"\n{'config':<22}{'n':>6}{'MAE':>7}{'bias':>7}{'<=10':>7}{'r':>7}{'med':>6}")
for t,s in res:
    print(f"{t:<22}{s['n']:>6}{s['mae']:>7.1f}{s['bias']:>+7.1f}{100*s['w10']:>6.0f}%{s['r']:>7.2f}{s['med']:>6.0f}")
shutil.rmtree(tmp,ignore_errors=True)
print(f"\nBEST: {res[0][0]}  MAE={res[0][1]['mae']:.1f} r={res[0][1]['r']:.2f} (ECG median {np.median(bhr):.0f})")

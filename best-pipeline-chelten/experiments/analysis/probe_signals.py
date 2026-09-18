#!/usr/bin/env python3
"""Recon to configure the CORAL-SCG harness: which accel axis (or magnitude)
carries the cleanest canine heartbeat, and what vibration band to lock on.
Also gauges BioPac per-lead railing so we pick a clean ECG reference lead.
Reads short windows only (raw stays raw; this just informs config)."""
import os, glob, json
import numpy as np, polars as pl
from scipy import signal as sg
import datetime
from zoneinfo import ZoneInfo

HIVE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hive")
TZ = ZoneInfo("America/Chicago"); UTC = datetime.timezone.utc
SEARCH_LO, SEARCH_HI = 50, 200          # canine bpm

def part(modality, dog, stream="0", date="20260626"):
    return glob.glob(f"{HIVE}/username={modality}/device={dog}/stream={stream}/date={date}/data_0.parquet")[0]

def to_us(local_str):
    return int(datetime.datetime.strptime(local_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ).timestamp()*1e6)

def slice_df(p, t0_us, t1_us, cols):
    df = pl.read_parquet(p, columns=["ts"]+cols)
    ts = df["ts"].to_numpy().astype("int64")
    m = (ts >= t0_us) & (ts < t1_us)
    return ts[m], {c: df[c].to_numpy()[m].astype(float) for c in cols}

def periodicity(x, fs, band=(10,45)):
    """band-limited autocorr peak in the cardiac lag band -> (periodicity, bpm)."""
    b = sg.butter(4, [band[0]/(fs/2), band[1]/(fs/2)], btype="band")
    xf = sg.filtfilt(*b, x - x.mean())
    env = np.abs(sg.hilbert(xf))           # SCG energy envelope (beats modulate it)
    env = env - env.mean()
    lo, hi = int(fs*60/SEARCH_HI), int(fs*60/SEARCH_LO)
    n = 1 << int(np.ceil(np.log2(2*len(env))))
    f = np.fft.rfft(env, n); ac = np.fft.irfft(f*np.conj(f), n)[:len(env)]
    if ac[0] <= 0: return 0.0, np.nan
    seg = ac[lo:hi]; k = lo + int(np.argmax(seg))
    return float(seg.max()/ac[0]), 60.0*fs/k

print("==== SCG axis/band probe (2-min clean windows) ====")
for dog, win in [("Chelten", "2026-06-26 12:30:00"), ("Dasty", "2026-06-26 10:40:00")]:
    t0 = to_us(win); t1 = t0 + 120_000_000
    ts, ch = slice_df(part("scg_mwd", dog, stream="45"), t0, t1, ["c1","c2","c3"])
    if len(ts) < 1000:
        print(f"  {dog}: only {len(ts)} samples in window -> skip"); continue
    fs = 1e6/np.median(np.diff(ts))
    mag = np.sqrt(sum((ch[c]-ch[c].mean())**2 for c in ch))
    sigs = {"x":ch["c1"], "y":ch["c2"], "z":ch["c3"], "mag":mag}
    print(f"  {dog} @ {win} fs={fs:.0f}Hz n={len(ts):,}")
    for band in [(10,45),(8,40),(15,60),(20,80)]:
        res = {name: periodicity(s, fs, band) for name, s in sigs.items()}
        best = max(res, key=lambda k: res[k][0])
        cells = "  ".join(f"{name}:{res[name][0]:.2f}({res[name][1]:.0f})" for name in ["x","y","z","mag"])
        print(f"    band{band}: {cells}   -> best={best} ({res[best][0]:.2f}, {res[best][1]:.0f}bpm)")

print("\n==== BioPac lead railing (full record) + range ====")
NAMES = json.load(open(part("ecg_biopac","Chelten").replace("data_0.parquet","channels.json")))["channels"]
for dog in ["Dasty","Chelten","Chuck"]:
    p = part("ecg_biopac", dog)
    df = pl.read_parquet(p)
    print(f"  {dog}:")
    for c in [f"c{i}" for i in range(1,10)]:
        v = df[c].to_numpy()
        rail = float((np.abs(v) >= 4.999).mean())
        print(f"    {c} {NAMES[c]['name']:<16} rail={100*rail:5.1f}%  range[{v.min():.2f},{v.max():.2f}]")

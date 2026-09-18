#!/usr/bin/env python3
"""Quick raw-ECG QC: 6 s of several BioPac leads + Polar to see canine QRS
morphology and pick the cleanest detection lead. Raw, no filtering shown."""
import os, glob
import numpy as np, polars as pl
import datetime
from zoneinfo import ZoneInfo
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
TZ = ZoneInfo("America/Chicago")
HIVE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hive")

def to_us(s): return int(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ).timestamp()*1e6)
def part(m, d, st="0"): return glob.glob(f"{HIVE}/username={m}/device={d}/stream={st}/date=*/data_0.parquet")[0]
def sl(p, cols, t0, dur=6):
    df = pl.read_parquet(p, columns=["ts"]+cols); ts = df["ts"].to_numpy().astype("int64")
    m = (ts >= t0) & (ts < t0+dur*1_000_000)
    return (ts[m]-ts[m][0])/1e6, {c: df[c].to_numpy()[m].astype(float) for c in cols}

fig, axs = plt.subplots(4, 1, figsize=(16, 12))
t, ch = sl(part("ecg_biopac","Chelten"), ["c2","c3","c4"], to_us("2026-06-26 11:15:30"))
for c, name in [("c2","Lead2"),("c3","Chest"),("c4","Lead3")]:
    axs[0].plot(t, ch[c], lw=0.6, label=name)
axs[0].set_title("Chelten BioPac 11:15:30 (+6s) — leads"); axs[0].legend(loc="upper right")
for i,(c,name) in enumerate([("c2","Lead2"),("c3","Chest"),("c4","Lead3")]):
    axs[1].plot(t, ch[c], lw=0.6);
axs[1].plot(t, ch["c4"], lw=0.8, color="k"); axs[1].set_title("Chelten BioPac Lead3 (c4) alone")
tp, chp = sl(part("ecg_polar","Chelten"), ["c1"], to_us("2026-06-26 12:30:00"))
axs[2].plot(tp, chp["c1"], lw=0.7, color="g"); axs[2].set_title("Chelten Polar H10 12:30:00 (+6s)")
td, chd = sl(part("ecg_biopac","Dasty"), ["c4"], to_us("2026-06-26 10:40:00"))
axs[3].plot(td, chd["c4"], lw=0.7, color="purple"); axs[3].set_title("Dasty BioPac Lead3 (c4) 10:40:00 (+6s)")
for a in axs: a.set_xlabel("s"); a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(HIVE),"figs","diag_ecg.png"), dpi=150)
print("ok")

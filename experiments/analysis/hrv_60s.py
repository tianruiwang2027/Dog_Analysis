#!/usr/bin/env python3
"""Rolling 60 s HRV — from the RR intervals (RRi, Pan-Tompkins beats = ground
truth) vs from CORAL (its smoothed HR sampled AT the same beats, so the two are on
one identical beat grid and differ only by method, not sampling).

HRV (time domain, ms): SDNN = std of NN in the window; RMSSD = rms of successive
NN differences (the beat-to-beat / vagal metric). CORAL is a DP-smoothed RATE, so
the question this answers is: how much HRV does CORAL recover vs the true RRi?
(Expect SDNN partly tracked — it's the slow RSA envelope — but RMSSD strongly
suppressed, since CORAL smooths away beat-to-beat change.)

Per dog: RRi (black) + each tool's CORAL sampled at its beats (SCG red, BioPac
blue, Polar green). Windowed to the ECG-data envelope. Usage: hrv_60s.py <Dog>
"""
import sys, os, glob
import numpy as np, polars as pl
from scipy.ndimage import maximum_filter1d, minimum_filter1d
import datetime
from zoneinfo import ZoneInfo
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(os.path.dirname(HERE), "figs")
TZ = ZoneInfo("America/Chicago"); UTC = datetime.timezone.utc
SQI = 0.10
WIN_US = 60_000_000          # 60 s HRV window
MIN_BEATS = 12               # need enough NN to estimate HRV
COL = {"SCG (MWD)": "#d62728", "BioPac": "#1f77b4", "Polar": "#2ca02c"}


def dn(us):
    return mdates.date2num([datetime.datetime.fromtimestamp(t/1e6, tz=UTC).astimezone(TZ)
                            for t in np.atleast_1d(np.asarray(us, "int64"))])


def load_coral_valid(path, maskpath):
    """CORAL ts,bpm with the same validity gate as compare_all (SQI, anti-coast, mask)."""
    k = pl.read_csv(path)
    ts = k["ts"].to_numpy().astype("datetime64[us]").astype("int64")
    bpm = k["bpm"].to_numpy().astype(float); sqi = k["sqi"].to_numpy().astype(float)
    v = sqi >= SQI
    hop = np.median(np.diff(ts))/1e6 if len(ts) > 2 else 0.25
    w = max(5, int(round(90.0/max(hop, 1e-3))) | 1)
    v &= (maximum_filter1d(bpm, w, mode="nearest") - minimum_filter1d(bpm, w, mode="nearest")) > 2.0
    if maskpath and os.path.exists(maskpath):
        m = pl.read_parquet(maskpath)
        for a, b in zip(m["mask_start"].to_numpy().astype("int64"), m["mask_end"].to_numpy().astype("int64")):
            v &= ~((ts >= a) & (ts <= b))
    return ts[v], bpm[v]


def coral_at_beats(beat_ts, cts, cbpm, max_gap_us=2_000_000):
    """Interpolate CORAL bpm at each beat time; NaN where no valid CORAL within max_gap."""
    if len(cts) < 2:
        return np.full(len(beat_ts), np.nan)
    bpm = np.interp(beat_ts, cts, cbpm)
    idx = np.clip(np.searchsorted(cts, beat_ts), 1, len(cts)-1)
    near = np.minimum(np.abs(beat_ts - cts[idx-1]), np.abs(beat_ts - cts[idx]))
    bpm[near > max_gap_us] = np.nan
    return bpm


def rolling_hrv(t_us, nn_ms):
    """Centered 60 s rolling SDNN and RMSSD over a (possibly NaN) NN series on a
    beat grid. Returns arrays aligned to t_us; NaN where < MIN_BEATS valid in window."""
    n = len(t_us); sdnn = np.full(n, np.nan); rmssd = np.full(n, np.nan)
    lo = np.searchsorted(t_us, t_us - WIN_US//2, side="left")
    hi = np.searchsorted(t_us, t_us + WIN_US//2, side="right")
    for i in range(n):
        seg = nn_ms[lo[i]:hi[i]]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= MIN_BEATS:
            sdnn[i] = np.std(seg)
            d = np.diff(seg)
            rmssd[i] = np.sqrt(np.mean(d*d))
    return sdnn, rmssd


def seg_plot(ax, t_us, y, color, label, lw=1.8, gap_us=5_000_000):
    """Plot a line, broken where the beat gap exceeds gap_us (e.g. BioPac->Polar)."""
    x = dn(t_us); cut = np.where(np.diff(t_us) > gap_us)[0]
    first, start = True, 0
    for c in list(cut) + [len(t_us)-1]:
        sl = slice(start, c+1)
        if np.isfinite(y[sl]).sum() > 1:
            ax.plot(x[sl], y[sl], "-", color=color, lw=lw, label=(label if first else None))
            first = False
        start = c+1
    if first:    # nothing drawn yet -> ensure legend entry
        ax.plot([], [], "-", color=color, lw=lw, label=label)


def main():
    dog = sys.argv[1]
    # ground-truth beats per ECG modality -> NN(ms)
    beats = {}
    for mod, name in [("ecg_biopac", "BioPac"), ("ecg_polar", "Polar")]:
        f = os.path.join(HERE, "ecg_hr", f"{dog}_{mod}.parquet")
        if os.path.exists(f):
            e = pl.read_parquet(f)
            t = e["ts"].to_numpy().astype("int64"); bpm = e["bpm"].to_numpy().astype(float)
            o = np.argsort(t); beats[name] = (t[o], 60000.0/bpm[o])
    if not beats:
        sys.exit(f"no ECG beats for {dog}")
    # unified, time-sorted beat grid across modalities (RRi ground truth)
    bt = np.concatenate([beats[n][0] for n in beats])
    bn = np.concatenate([beats[n][1] for n in beats])
    o = np.argsort(bt); bt, bn = bt[o], bn[o]

    # CORAL tracks
    coral = {}
    sp = os.path.join(HERE, "coral_scg", dog, "out.csv")
    if os.path.exists(sp):
        coral["SCG (MWD)"] = load_coral_valid(sp, os.path.join(HERE, "coral_scg", dog, "mask.parquet"))
    for mod, name in [("ecg_biopac", "BioPac"), ("ecg_polar", "Polar")]:
        cp = os.path.join(HERE, "coral_ecg", f"{dog}_{mod}", "out.csv")
        if os.path.exists(cp):
            coral[name] = load_coral_valid(cp, os.path.join(HERE, "coral_ecg", f"{dog}_{mod}", "mask.parquet"))

    # HRV from RRi (truth) on the unified beat grid
    sdnn_rri, rmssd_rri = rolling_hrv(bt, bn)
    # HRV from CORAL sampled at the SAME beats
    coral_hrv = {}
    for name, (cts, cbpm) in coral.items():
        nn_c = 60000.0 / coral_at_beats(bt, cts, cbpm)
        coral_hrv[name] = rolling_hrv(bt, nn_c)

    WIN_LO, WIN_HI = int(bt.min())-60_000_000, int(bt.max())+60_000_000

    fig, (axR, axS) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    for ax, (metric, ridx) in [(axR, ("RMSSD", 1)), (axS, ("SDNN", 0))]:
        rri = (rmssd_rri if ridx == 1 else sdnn_rri)
        seg_plot(ax, bt, rri, "black", "RRi (Pan-Tompkins, ground truth)", lw=2.4)
        for name in coral_hrv:
            y = coral_hrv[name][ridx]
            seg_plot(ax, bt, y, COL.get(name.split()[0] if name != "SCG (MWD)" else "SCG (MWD)",
                                       COL.get(name, "0.5")), f"CORAL: {name}", lw=1.5)
        ax.set_ylabel(f"{metric} (ms)")
        ax.grid(True, alpha=0.3); ax.set_xlim(dn(WIN_LO)[0], dn(WIN_HI)[0])
        ax.set_ylim(0, None)
        # quantitative medians (over beats where both RRi and at least one CORAL exist)
        med = {"RRi": np.nanmedian(rri)}
        for name in coral_hrv:
            med[name] = np.nanmedian(coral_hrv[name][ridx])
        txt = "  ".join(f"{k.split()[0]} {v:.0f}" for k, v in med.items() if np.isfinite(v))
        ax.text(0.005, 0.97, f"median {metric} (ms): {txt}", transform=ax.transAxes,
                va="top", fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    axR.legend(loc="upper right", ncol=2, framealpha=0.95, fontsize=9)
    axR.set_title(f"[{dog}] HRV (60 s rolling) — RRi vs CORAL  ·  CORAL is a smoothed rate, so it "
                  f"recovers the slow RSA envelope (SDNN) but suppresses beat-to-beat HRV (RMSSD)")
    axS.xaxis.set_major_formatter(DateFormatter("%H:%M", tz=TZ))
    axS.set_xlabel("America/Chicago local time · 2026-06-26")
    fig.tight_layout()
    out = os.path.join(FIGS, f"hrv60_{dog}.png"); fig.savefig(out, dpi=200); print("->", out)
    print(f"  RRi    median RMSSD {np.nanmedian(rmssd_rri):.0f} ms  SDNN {np.nanmedian(sdnn_rri):.0f} ms")
    for name in coral_hrv:
        print(f"  CORAL {name:11} RMSSD {np.nanmedian(coral_hrv[name][1]):.0f}  SDNN {np.nanmedian(coral_hrv[name][0]):.0f}")


if __name__ == "__main__":
    main()

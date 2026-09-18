#!/usr/bin/env python3
"""Unified 3-tool HR comparison per dog: SCG (MWD), BioPac ECG, Polar H10 ECG.
Each ECG tool's HR comes from CORAL (smooth, DP-tracked) and is validated against
its own Pan-Tompkins RRi. SCG HR also from CORAL (same estimator -> fair). We
report, on every temporal overlap: bias / MAE / RMSE / %within tol / Pearson r,
plus per-tool coverage. Master figure: all tools' HR over time + pairwise scatter.

Usage: compare_all.py <Dog>
"""
import sys, os, glob, json
import numpy as np, polars as pl
from scipy.ndimage import maximum_filter1d, minimum_filter1d
import datetime
from zoneinfo import ZoneInfo
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.collections import LineCollection

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(os.path.dirname(HERE), "figs")
TZ = ZoneInfo("America/Chicago"); UTC = datetime.timezone.utc
SQI = 0.10
COL = {"SCG (MWD)": "#d62728", "BioPac CORAL": "#1f77b4", "Polar CORAL": "#2ca02c"}


def dn(us):
    return mdates.date2num([datetime.datetime.fromtimestamp(t/1e6, tz=UTC).astimezone(TZ)
                            for t in np.atleast_1d(np.asarray(us, dtype="int64"))])


def rolling_ptp(x, w):
    """Rolling peak-to-peak (max-min) over a window of w samples."""
    return maximum_filter1d(x, w, mode="nearest") - minimum_filter1d(x, w, mode="nearest")


def load_coral(path, maskpath=None):
    k = pl.read_csv(path)
    ts = k["ts"].to_numpy().astype("datetime64[us]").astype("int64")
    bpm = k["bpm"].to_numpy().astype(float); sqi = k["sqi"].to_numpy().astype(float)
    valid = sqi >= SQI
    # reject DP "coasting": a real HR always drifts/jitters with RSA, so a run held
    # within a tiny band (peak-to-peak < 2 bpm over 90 s) is CORAL holding a constant
    # value on no signal (e.g. Chelten's 157.5 bpm coast, flat for 44 min at high
    # SQI) -> not a valid estimate, otherwise it inflates coverage.
    hopdt = np.median(np.diff(ts)) / 1e6 if len(ts) > 2 else 0.25
    w = max(5, int(round(90.0 / max(hopdt, 1e-3))) | 1)
    valid &= rolling_ptp(bpm, w) > 2.0
    if maskpath and os.path.exists(maskpath):
        m = pl.read_parquet(maskpath)
        for a, b in zip(m["mask_start"].to_numpy().astype("int64"), m["mask_end"].to_numpy().astype("int64")):
            valid &= ~((ts >= a) & (ts <= b))
    return ts, bpm, sqi, valid


def align(tsA, hrA, vA, tsB, hrB, vB, win_us=2_000_000):
    """windowed-mean of B (valid) within +-win of each valid A sample -> paired (a,b)."""
    bts, bhr = tsB[vB], hrB[vB]
    if len(bts) < 5:
        return np.array([]), np.array([]), np.array([])
    A = np.where(vA)[0]; xa, yb, ta = [], [], []
    for i in A:
        t = tsA[i]; sel = (bts >= t-win_us) & (bts <= t+win_us)
        if sel.sum() >= 2:
            xa.append(hrA[i]); yb.append(bhr[sel].mean()); ta.append(t)
    return np.array(xa), np.array(yb), np.array(ta, dtype="int64")


def stats(a, b):
    d = a - b
    return dict(n=int(len(a)), bias=float(d.mean()), mae=float(np.abs(d).mean()),
                rmse=float(np.sqrt((d**2).mean())),
                within5=float(np.mean(np.abs(d) <= 5)), within10=float(np.mean(np.abs(d) <= 10)),
                r=float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan"),
                loa=[float(d.mean()-1.96*d.std()), float(d.mean()+1.96*d.std())])


def main():
    dog = sys.argv[1]
    tools = {}
    # SCG
    sp = os.path.join(HERE, "coral_scg", dog, "out.csv")
    if os.path.exists(sp):
        tools["SCG (MWD)"] = load_coral(sp, os.path.join(HERE, "coral_scg", dog, "mask.parquet"))
    # ECG CORAL
    for mod, name in [("ecg_biopac", "BioPac CORAL"), ("ecg_polar", "Polar CORAL")]:
        cp = os.path.join(HERE, "coral_ecg", f"{dog}_{mod}", "out.csv")
        if os.path.exists(cp):
            tools[name] = load_coral(cp, os.path.join(HERE, "coral_ecg", f"{dog}_{mod}", "mask.parquet"))
    # Pan-Tompkins (validation series)
    pt = {}
    for mod, name in [("ecg_biopac", "BioPac"), ("ecg_polar", "Polar")]:
        f = os.path.join(HERE, "ecg_hr", f"{dog}_{mod}.parquet")
        if os.path.exists(f):
            e = pl.read_parquet(f); pt[name] = (e["ts"].to_numpy().astype("int64"), e["bpm"].to_numpy())

    # ---- restrict to the real-data window (where the reference ECGs recorded) ----
    # SCG (MWD) spans hours of mostly-coasting; the meaningful comparison is the
    # window the ECGs cover. Use the ECG beat envelope (+-60 s pad); fall back to the
    # union of valid spans if no ECG beats exist.
    if pt:
        bt = np.concatenate([pt[n][0] for n in pt])
        WIN_LO, WIN_HI = int(bt.min()) - 60_000_000, int(bt.max()) + 60_000_000
    else:
        allv = np.concatenate([tools[n][0][tools[n][3]] for n in tools]) if tools else np.array([0, 1], "int64")
        WIN_LO, WIN_HI = int(allv.min()), int(allv.max())
    for n in list(tools):                          # clip validity to the window
        ts, bpm, sqi, v = tools[n]
        tools[n] = (ts, bpm, sqi, v & (ts >= WIN_LO) & (ts <= WIN_HI))

    # ---- pairwise agreement ----
    names = list(tools)
    result = {"coverage": {}, "pairs": {}, "validation": {}, "window_us": [WIN_LO, WIN_HI]}
    for n in names:
        ts, bpm, sqi, v = tools[n]
        inwin = (ts >= WIN_LO) & (ts <= WIN_HI)
        hopdt = np.median(np.diff(ts))/1e6
        result["coverage"][n] = dict(valid_frac=float(v.sum()/max(1, inwin.sum())),
                                     valid_min=float(v.sum()*hopdt/60),
                                     span_min=float((WIN_HI-WIN_LO)/1e6/60))
    cross = [("SCG (MWD)", "BioPac CORAL"), ("SCG (MWD)", "Polar CORAL"), ("BioPac CORAL", "Polar CORAL")]
    def series(n): ts, bpm, sqi, v = tools[n]; return ts, bpm, v
    for A, B in cross:
        if A in tools and B in tools:
            tA, hA, vA = series(A); tB, hB, vB = series(B)
            xa, yb, _ = align(tA, hA, vA, tB, hB, vB)
            if len(xa) >= 10:
                result["pairs"][f"{A} vs {B}"] = stats(xa, yb)
    # validation: ECG CORAL vs its Pan-Tompkins
    for mod, cname, pname in [("ecg_biopac", "BioPac CORAL", "BioPac"), ("ecg_polar", "Polar CORAL", "Polar")]:
        ap = os.path.join(HERE, "coral_ecg", f"{dog}_{mod}", "agree.json")
        if os.path.exists(ap):
            result["validation"][f"{cname} vs {pname} Pan-Tompkins"] = json.load(open(ap))

    json.dump(result, open(os.path.join(HERE, f"compare3_{dog}.json"), "w"), indent=2)

    # ---- figure ----
    npair = max(1, len(result["pairs"]))
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, max(2, npair), height_ratios=[1.5, 1.0], hspace=0.3, wspace=0.28)
    axT = fig.add_subplot(gs[0, :])
    for n in names:
        ts, bpm, sqi, v = tools[n]
        xs = dn(ts); pts = np.column_stack([xs, bpm]); segs = np.stack([pts[:-1], pts[1:]], 1)
        a = np.where(v[:-1], 0.85, 0.0)
        c = np.array(plt.matplotlib.colors.to_rgba(COL[n]))[None, :].repeat(len(segs), 0); c[:, 3] = a
        axT.add_collection(LineCollection(segs, colors=c, linewidths=1.6, zorder=4, label=n))
    for name, (ts, hr) in pt.items():
        axT.plot(dn(ts), hr, ".", ms=1.5, color="0.35", alpha=0.30, zorder=2)
    axT.set_ylim(40, 210); axT.set_ylabel("Heart rate (bpm)")
    axT.xaxis.set_major_formatter(DateFormatter("%H:%M", tz=TZ)); axT.grid(True, alpha=0.3)
    axT.set_xlim(dn(WIN_LO)[0], dn(WIN_HI)[0])
    axT.legend(loc="upper right", ncol=3, framealpha=0.95)
    cov = " · ".join(f"{n.split()[0]} {100*result['coverage'][n]['valid_frac']:.0f}%" for n in names)
    axT.set_title(f"[{dog}] heart rate across all tools (CORAL; grey dots = Pan-Tompkins beats)\ncoverage: {cov}")
    axT.set_xlabel("America/Chicago local time · 2026-06-26")
    # pairwise scatter
    j = 0
    for A, B in cross:
        key = f"{A} vs {B}"
        if key not in result["pairs"]:
            continue
        ax = fig.add_subplot(gs[1, j]); j += 1
        tA, hA, vA = series(A); tB, hB, vB = series(B)
        xa, yb, _ = align(tA, hA, vA, tB, hB, vB)
        ax.scatter(yb, xa, s=5, alpha=0.3, color=COL.get(A, "0.3"), edgecolor="none")
        lim = [40, 210]; ax.plot(lim, lim, "k-", lw=1, alpha=0.6)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_xlabel(f"{B} HR"); ax.set_ylabel(f"{A} HR")
        R = result["pairs"][key]
        ax.set_title(f"{A.split()[0]} vs {B.split()[0]}", fontsize=10)
        ax.text(0.03, 0.97, f"MAE {R['mae']:.1f} bias {R['bias']:+.1f}\n≤10bpm {100*R['within10']:.0f}% r={R['r']:.2f}\nn={R['n']}",
                transform=ax.transAxes, va="top", fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.85))
        ax.grid(True, alpha=0.3)
    out = os.path.join(FIGS, f"compare3_{dog}.png"); fig.savefig(out, dpi=200); print("->", out)
    for k, R in result["pairs"].items():
        print(f"  {k}: MAE {R['mae']:.1f} bias {R['bias']:+.1f} ≤10 {100*R['within10']:.0f}% r={R['r']:.2f} n={R['n']}")
    for k, R in result["validation"].items():
        print(f"  [val] {k}: MAE {R['mae']:.1f} ≤5 {100*R['within5']:.0f}% r={R['r']:.2f}")


if __name__ == "__main__":
    main()

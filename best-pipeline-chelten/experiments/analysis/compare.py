#!/usr/bin/env python3
"""Time-aligned SCG-CORAL HR vs ECG-reference HR, per dog. Quantifies agreement
(bias, MAE, RMSE, %within tolerance, Bland-Altman LoA, harmonic-lock fraction)
ONLY where both are valid: SCG hop SQI>=thr and unmasked, ECG inside a continuous
run of beats (no lead-off). Also quantifies coverage: how much of the SCG session
yields a usable HR, and how much is comparable to ECG. Emits a per-dog figure +
analysis/compare_<dog>.json. Raw stays raw; HR series are genuine derivations.

Usage: compare.py <Dog>
"""
import sys, os, glob, json
import numpy as np, polars as pl
import datetime
from zoneinfo import ZoneInfo
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.collections import LineCollection

HERE = os.path.dirname(os.path.abspath(__file__))
HIVE = os.path.join(os.path.dirname(HERE), "hive")
FIGS = os.path.join(os.path.dirname(HERE), "figs")
TZ = ZoneInfo("America/Chicago"); UTC = datetime.timezone.utc
SQI_THR = 0.10
TOL_BPM5, TOL_BPM10 = 5.0, 10.0
REF_LABEL = {"ecg_biopac": "BioPac ECG", "ecg_polar": "Polar H10 ECG"}
REF_COLOR = {"ecg_biopac": "#1f77b4", "ecg_polar": "#2ca02c"}


def dn(us):
    return mdates.date2num([datetime.datetime.fromtimestamp(t / 1e6, tz=UTC).astimezone(TZ)
                            for t in np.atleast_1d(np.asarray(us, dtype="int64"))])


def load_coral(dog):
    d = os.path.join(HERE, "coral_scg", dog)
    k = pl.read_csv(os.path.join(d, "out.csv"))
    ts = k["ts"].to_numpy().astype("datetime64[us]").astype("int64")
    bpm = k["bpm"].to_numpy().astype(float); sqi = k["sqi"].to_numpy().astype(float)
    mrows = []
    mp = os.path.join(d, "mask.parquet")
    if os.path.exists(mp):
        m = pl.read_parquet(mp)
        mrows = list(zip(m["mask_start"].to_numpy().astype("int64"),
                         m["mask_end"].to_numpy().astype("int64")))
    return ts, bpm, sqi, mrows


def ecg_at(t, tmid, bpm, max_gap_us=2_000_000):
    """ECG HR interpolated onto times t, NaN where t falls in a >max_gap beat gap."""
    out = np.full(len(t), np.nan)
    if len(tmid) < 2:
        return out
    idx = np.searchsorted(tmid, t)
    v = (idx > 0) & (idx < len(tmid))
    i = idx[v]; lo = tmid[i - 1]; hi = tmid[i]; gap = (hi - lo)
    frac = (t[v] - lo) / np.maximum(gap, 1)
    val = bpm[i - 1] * (1 - frac) + bpm[i] * frac
    val[gap > max_gap_us] = np.nan
    out[v] = val
    return out


def refs_for(dog):
    out = []
    for modality in ("ecg_biopac", "ecg_polar"):
        f = os.path.join(HERE, "ecg_hr", f"{dog}_{modality}.parquet")
        if os.path.exists(f):
            df = pl.read_parquet(f)
            j = json.load(open(f.replace(".parquet", ".json")))
            out.append((modality, df["ts"].to_numpy().astype("int64"),
                        df["bpm"].to_numpy().astype(float), j.get("leadoff_intervals_us", [])))
    return out


def main():
    dog = sys.argv[1]
    cts, cbpm, csqi, cmask = load_coral(dog)
    span_s = (cts[-1] - cts[0]) / 1e6
    in_mask = np.zeros(len(cts), bool)
    for s, e in cmask:
        in_mask |= (cts >= s) & (cts <= e)
    scg_valid = (csqi >= SQI_THR) & ~in_mask
    refs = refs_for(dog)

    # ---- coverage of the SCG session ----
    hopdt = np.median(np.diff(cts)) / 1e6
    scg_valid_s = scg_valid.sum() * hopdt
    cov = {"scg_span_min": span_s / 60, "scg_valid_min": scg_valid_s / 60,
           "scg_valid_frac": float(scg_valid.mean()),
           "sqi_thr": SQI_THR, "comparisons": {}}

    # ---- per-ref agreement ----
    results = {}
    for modality, rts, rbpm, leadoff in refs:
        w0, w1 = max(cts[0], rts[0]), min(cts[-1], rts[-1])
        sel = scg_valid & (cts >= w0) & (cts <= w1)
        x = cbpm[sel]                          # SCG-CORAL HR
        y = ecg_at(cts[sel], rts, rbpm)        # ECG ref HR at the same hop times
        ok = np.isfinite(y)
        x, y, tsel = x[ok], y[ok], cts[sel][ok]
        if len(x) < 10:
            results[modality] = {"n": int(len(x)), "note": "insufficient overlap"}
            cov["comparisons"][modality] = results[modality]; continue
        diff = x - y
        within5 = float(np.mean(np.abs(diff) <= TOL_BPM5))
        within10 = float(np.mean(np.abs(diff) <= TOL_BPM10))
        harm_double = float(np.mean(np.abs(x - 2 * y) <= 0.10 * 2 * y))
        harm_half = float(np.mean(np.abs(x - 0.5 * y) <= 0.10 * 0.5 * y))
        r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")
        comparable_s = len(x) * hopdt
        results[modality] = {
            "n": int(len(x)), "window_local": [str(dn(w0)), str(dn(w1))],
            "comparable_min": comparable_s / 60,
            "bias_bpm": float(diff.mean()), "mae_bpm": float(np.abs(diff).mean()),
            "rmse_bpm": float(np.sqrt((diff ** 2).mean())),
            "loa_bpm": [float(diff.mean() - 1.96 * diff.std()), float(diff.mean() + 1.96 * diff.std())],
            "within_5bpm": within5, "within_10bpm": within10,
            "pearson_r": r, "harmonic_double_frac": harm_double, "harmonic_half_frac": harm_half,
            "ecg_hr_median": float(np.median(y)), "scg_hr_median": float(np.median(x)),
            "_pairs": (tsel, x, y)}
        cov["comparisons"][modality] = {k: v for k, v in results[modality].items() if k != "_pairs"}

    json.dump(cov, open(os.path.join(HERE, f"compare_{dog}.json"), "w"), indent=2, default=str)

    # --------------------------------- figure ---------------------------------
    nref = max(1, len(refs))
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, max(2, nref), height_ratios=[1.5, 1.0], hspace=0.28, wspace=0.26)

    # row 1: HR over time
    axT = fig.add_subplot(gs[0, :])
    x = dn(cts)
    pts = np.column_stack([x, cbpm]); segs = np.stack([pts[:-1], pts[1:]], axis=1)
    a = np.where(scg_valid[:-1], 0.25 + 0.75 * np.clip(csqi[:-1] / 0.25, 0, 1), 0.0)
    colors = np.zeros((len(segs), 4)); colors[:, 0] = 0.84; colors[:, 3] = a   # red, alpha∝SQI
    axT.add_collection(LineCollection(segs, colors=colors, linewidths=1.6, zorder=4,
                                      label="SCG-CORAL HR (opacity ∝ SQI)"))
    for s, e in cmask:
        axT.axvspan(dn(s)[0], dn(e)[0], color="0.6", alpha=0.30, zorder=1)
    for modality, rts, rbpm, leadoff in refs:
        axT.plot(dn(rts), rbpm, ".", ms=2.6, color=REF_COLOR[modality], alpha=0.7, zorder=5,
                 label=f"{REF_LABEL[modality]} R-peak HR")
        for s, e in leadoff:
            axT.axvspan(dn(s)[0], dn(e)[0], color=REF_COLOR[modality], alpha=0.06, zorder=0)
    axT.set_ylim(40, 210); axT.set_ylabel("Heart rate (bpm)")
    axT.xaxis.set_major_formatter(DateFormatter("%H:%M", tz=TZ))
    axT.set_xlim(x[0], x[-1]); axT.grid(True, alpha=0.3)
    axT.legend(loc="upper right", markerscale=3, framealpha=0.95, fontsize=9)
    axT.set_title(f"[{dog}] SCG-CORAL heart rate vs ECG reference — time-aligned "
                  f"(grey = SCG excluded; tinted = ECG lead-off)", fontsize=12)
    axT.set_xlabel(f"America/Chicago local time · 2026-06-26 · SCG valid {100*scg_valid.mean():.0f}% "
                   f"of {span_s/60:.0f} min")

    # row 2: agreement per ref (Bland-Altman style scatter, colored by which is bigger)
    for j, (modality, rts, rbpm, leadoff) in enumerate(refs):
        ax = fig.add_subplot(gs[1, j])
        R = results[modality]
        if "_pairs" not in R:
            ax.text(0.5, 0.5, f"{REF_LABEL[modality]}\nno overlap", ha="center", va="center")
            ax.axis("off"); continue
        _, xx, yy = R["_pairs"]
        ax.scatter(yy, xx, s=4, alpha=0.25, color=REF_COLOR[modality], edgecolor="none")
        lim = [40, 210]
        ax.plot(lim, lim, "k-", lw=1, alpha=0.6)
        ax.plot(lim, [2 * l for l in lim], "k:", lw=0.8, alpha=0.4)       # 2x harmonic
        ax.plot(lim, [0.5 * l for l in lim], "k:", lw=0.8, alpha=0.4)     # 0.5x harmonic
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_xlabel(f"{REF_LABEL[modality]} HR (bpm)"); ax.set_ylabel("SCG-CORAL HR (bpm)")
        ax.set_title(f"{REF_LABEL[modality]}  ({R['comparable_min']:.0f} min, n={R['n']:,})", fontsize=10)
        txt = (f"bias {R['bias_bpm']:+.1f}  MAE {R['mae_bpm']:.1f}  RMSE {R['rmse_bpm']:.1f} bpm\n"
               f"≤5bpm {100*R['within_5bpm']:.0f}%  ≤10bpm {100*R['within_10bpm']:.0f}%  r={R['pearson_r']:.2f}\n"
               f"2× lock {100*R['harmonic_double_frac']:.0f}%  ½× {100*R['harmonic_half_frac']:.0f}%")
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85))
        ax.grid(True, alpha=0.3)

    fig.suptitle("")
    out = os.path.join(FIGS, f"compare_{dog}.png")
    fig.savefig(out, dpi=200, bbox_inches=None); print("->", out)
    # console summary
    print(f"[{dog}] SCG valid {100*scg_valid.mean():.0f}% of {span_s/60:.0f}min")
    for modality, *_ in refs:
        R = results[modality]
        if "_pairs" in R:
            print(f"  vs {REF_LABEL[modality]}: n={R['n']:,} {R['comparable_min']:.0f}min  "
                  f"bias {R['bias_bpm']:+.1f} MAE {R['mae_bpm']:.1f} RMSE {R['rmse_bpm']:.1f}  "
                  f"≤10bpm {100*R['within_10bpm']:.0f}%  r={R['pearson_r']:.2f}  "
                  f"2×lock {100*R['harmonic_double_frac']:.0f}%")


if __name__ == "__main__":
    main()

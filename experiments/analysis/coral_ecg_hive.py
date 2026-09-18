#!/usr/bin/env python3
"""CORAL HR on an ECG signal (BioPac Lead3 or Polar H10), with the correloform as
the visual proof that the CORAL-locked rate coincides with the Pan-Tompkins R-peak
beats. Renders: coral-st's solved correloform (heatmap) + CORAL HR (opacity in SQI)
+ Pan-Tompkins RRi HR (from ecg_hr.py) overlaid. Quantifies CORAL-vs-Pan-Tompkins
agreement and saves it. Lead-off intervals from ecg_hr gate both.

Usage: coral_ecg_hive.py <Dog> <ecg_biopac|ecg_polar>
"""
import sys, os, glob, json, shutil, subprocess
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
CORAL = shutil.which("coral-st") or "coral-st"
TZ = ZoneInfo("America/Chicago"); UTC = datetime.timezone.utc
LEAD = {"ecg_biopac": "c4", "ecg_polar": "c1"}
SEARCH_LO, SEARCH_HI = 50, 200
FC_LO, FC_HI = 8.0, 40.0                      # QRS band
A_FS, A_HOP, A_WINS = "512", "128", "256,512,1024,1536"


def dn(us):
    return mdates.date2num([datetime.datetime.fromtimestamp(t/1e6, tz=UTC).astimezone(TZ)
                            for t in np.atleast_1d(np.asarray(us, dtype="int64"))])


def main():
    dog, mod = sys.argv[1], sys.argv[2]
    lead = LEAD[mod]
    p = glob.glob(f"{HIVE}/username={mod}/device={dog}/stream=0/date=*/data_0.parquet")[0]
    df = pl.read_parquet(p, columns=["ts", lead])
    ts = df["ts"].to_numpy().astype("int64"); x = df[lead].to_numpy().astype(float)
    outdir = os.path.join(HERE, "coral_ecg", f"{dog}_{mod}"); os.makedirs(outdir, exist_ok=True)
    inp = os.path.join(outdir, "ecg_in.parquet")
    pl.DataFrame({"ts": ts, "c1": x}).write_parquet(inp)

    # lead-off mask from the Pan-Tompkins stage
    ej = os.path.join(HERE, "ecg_hr", f"{dog}_{mod}.json")
    leadoff = json.load(open(ej)).get("leadoff_intervals_us", []) if os.path.exists(ej) else []
    maskp = os.path.join(outdir, "mask.parquet")
    if leadoff:
        pl.DataFrame({"mask_start": pl.Series([a for a, _ in leadoff]).cast(pl.Datetime("us")),
                      "mask_end": pl.Series([b for _, b in leadoff]).cast(pl.Datetime("us"))}).write_parquet(maskp)

    cmd = [CORAL, "-i", inp, "-o", os.path.join(outdir, "out.csv"),
           "--fc-low", str(FC_LO), "--fc-high", str(FC_HI),
           "--analysis-fs", A_FS, "--analysis-hop", A_HOP, "--analysis-windows", A_WINS,
           "--alpha", "0.0", "--beta", "30", "--gamma", "3", "--delta", "10",
           "--epsilon", "0.005", "--zeta", "5", "--eta", "4",
           "--search-bpm-low", str(SEARCH_LO), "--search-bpm-high", str(SEARCH_HI),
           "--max-delta-pct-up", "20", "--max-delta-pct-down", "20",
           "--correloform-png-path", os.path.join(outdir, "cf.png"),
           "--estimates-png-path", os.path.join(outdir, "est.png")]
    if leadoff:
        cmd += ["--mask", maskp]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("CORAL FAILED:\n", r.stderr[-1500:]); raise SystemExit(1)

    k = pl.read_csv(os.path.join(outdir, "out.csv"))
    cts = k["ts"].to_numpy().astype("datetime64[us]").astype("int64")
    cb = k["bpm"].to_numpy().astype(float); cq = k["sqi"].to_numpy().astype(float)

    # Pan-Tompkins beats
    e = pl.read_parquet(os.path.join(HERE, "ecg_hr", f"{dog}_{mod}.parquet"))
    ets = e["ts"].to_numpy().astype("int64"); ehr = e["bpm"].to_numpy()
    emid = ets; ebh = ehr                  # stored Pan-Tompkins beat HR (already cleaned)

    # agreement CORAL vs Pan-Tompkins (windowed ECG at coral hops, SQI-gated)
    ok = cq >= 0.10
    yy = np.full(len(cts), np.nan)
    for i, t in enumerate(cts):
        m = (emid >= t-2_000_000) & (emid <= t+2_000_000)
        if m.sum() >= 2: yy[i] = ebh[m].mean()
    v = ok & np.isfinite(yy); d = cb[v]-yy[v]
    agree = dict(n=int(v.sum()), mae=float(np.abs(d).mean()), bias=float(d.mean()),
                 within5=float(np.mean(np.abs(d) <= 5)), within10=float(np.mean(np.abs(d) <= 10)),
                 r=float(np.corrcoef(cb[v], yy[v])[0, 1]) if v.sum() > 2 else float("nan"),
                 coral_valid_frac=float(ok.mean()))
    json.dump(agree, open(os.path.join(outdir, "agree.json"), "w"), indent=2)

    # figure: correloform + CORAL HR + Pan-Tompkins HR
    fig, ax = plt.subplots(figsize=(16, 9))
    x0, x1 = dn(cts[0])[0], dn(cts[-1])[0]
    if os.path.exists(os.path.join(outdir, "cf.png")):
        img = plt.imread(os.path.join(outdir, "cf.png")); img = img[:, :, :3] if img.ndim == 3 else img
        ax.imshow(img**0.45, aspect="auto", origin="upper", extent=[x0, x1, SEARCH_LO, SEARCH_HI],
                  zorder=0, interpolation="bilinear")
    xs = dn(cts); pts = np.column_stack([xs, cb]); segs = np.stack([pts[:-1], pts[1:]], 1)
    a = np.where(ok[:-1], 0.3+0.7*np.clip(cq[:-1]/0.25, 0, 1), 0.0)
    cols = np.zeros((len(segs), 4)); cols[:, 0] = 1.0; cols[:, 3] = a
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=2.0, zorder=5, label="CORAL HR (opacity∝SQI)"))
    ax.plot(dn(emid), ebh, ".", ms=2.2, color="#11ddff", alpha=0.55, zorder=4, label="Pan-Tompkins R-peak HR")
    ax.set_ylim(SEARCH_LO, SEARCH_HI); ax.set_xlim(x0, x1)
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M", tz=TZ)); ax.set_ylabel("Heart rate (bpm)")
    ax.legend(loc="upper right", markerscale=3, framealpha=0.9)
    ax.set_title(f"[{dog}] {mod} — CORAL correloform + CORAL HR vs Pan-Tompkins RRi\n"
                 f"agreement: MAE {agree['mae']:.1f}, bias {agree['bias']:+.1f} bpm, "
                 f"≤5bpm {100*agree['within5']:.0f}%, r={agree['r']:.2f}, CORAL valid {100*agree['coral_valid_frac']:.0f}%")
    ax.set_xlabel(f"America/Chicago time · CORAL lock vs detected beats (n={agree['n']})")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, f"coral_ecg_{dog}_{mod}.png"), dpi=150)
    print(f"[{dog}/{mod}] CORAL-vs-PanTompkins: MAE {agree['mae']:.1f} bias {agree['bias']:+.1f} "
          f"≤5 {100*agree['within5']:.0f}% r={agree['r']:.2f} (n={agree['n']}) valid {100*agree['coral_valid_frac']:.0f}%")
    print("  ->", os.path.join(FIGS, f"coral_ecg_{dog}_{mod}.png"))


if __name__ == "__main__":
    main()

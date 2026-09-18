#!/usr/bin/env python3
"""One 16x12 summary per dog (3 rows x 2 cols):
  row 1 (full width): HR over time — SCG/BioPac/Polar CORAL + Pan-Tompkins beats
  row 2 (full width): HRV (RMSSD, 60 s) over time — RRi on the left axis, CORAL on a
                      twin right axis (CORAL HRV is ~5 ms vs RRi ~200 ms, so a shared
                      axis would flatten it)
  row 3: HRV scatter RRi vs CORAL — RMSSD (left) and SDNN (right), sources overlaid

Reuses the validity gate / HRV helpers from compare_all + hrv_60s + hrv_scatter.
Usage: combined_hrv_figure.py <Dog>
"""
import sys, os, glob, datetime
import numpy as np, polars as pl
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import matplotlib.patheffects as pe
from scipy.ndimage import median_filter, uniform_filter1d

HALO = [pe.Stroke(linewidth=3.6, foreground="white", alpha=0.75), pe.Normal()]  # keep colored lines legible over the RRi cloud

import compare_all as C
import hrv_60s as H
import hrv_scatter as SC

COL = {"SCG": "#d62728", "BioPac": "#1f77b4", "Polar": "#2ca02c"}
HRV_TOOLS = [("SCG (MWD)", "SCG"), ("BioPac", "BioPac"), ("Polar", "Polar")]


def seg(ax, t_us, y, color, label=None, lw=1.6, gap_us=5_000_000, zorder=2):
    x = C.dn(t_us); cut = np.where(np.diff(t_us) > gap_us)[0]; first, start = True, 0
    for c in list(cut) + [len(t_us) - 1]:
        sl = slice(start, c + 1)
        if np.isfinite(np.asarray(y[sl], float)).sum() > 1:
            ax.plot(x[sl], y[sl], "-", color=color, lw=lw, zorder=zorder,
                    label=(label if first else None)); first = False
        start = c + 1


def motion_db(dog, win_s=60.0, sub_s=2.0):
    """Rolling-`win_s` RMS (dB) of the MWD triaxial accel AC magnitude — a motion
    indicator. mag=sqrt(sum cᵢ²); slow gravity removed (2 s mean); RMS over win_s;
    20·log10. Returned subsampled to ~sub_s for plotting. (None,None) if no triaxial."""
    g = glob.glob(os.path.join(os.path.dirname(H.HERE), "hive", "username=scg_mwd",
                  f"device={dog}", "stream=45", "date=*", "data_0.parquet"))
    cols = ("c1", "c2", "c3")
    if not g:
        return None, None
    df = pl.read_parquet(g[0])
    if not all(c in df.columns for c in cols):
        return None, None
    ts = df["ts"].to_numpy().astype("int64"); fs = 1e6 / np.median(np.diff(ts))
    m = np.sqrt(sum(df[c].to_numpy().astype(np.float32) ** 2 for c in cols))
    ac = m - uniform_filter1d(m, int(2 * fs))                      # drop slow gravity
    rms = np.sqrt(uniform_filter1d(ac.astype(np.float64) ** 2, int(win_s * fs)))
    db = 20 * np.log10(rms + 1e-9)
    step = max(1, int(sub_s * fs))
    return ts[::step], db[::step]


def rolling_avg(t_us, y, mask=None, win_us=60_000_000, step_us=2_000_000, min_n=3):
    """Time-windowed (60 s) MEAN of y on a uniform query grid; windows with < min_n
    points (e.g. across the BioPac->Polar gap) return NaN so the line breaks."""
    if mask is not None:
        t_us, y = t_us[mask], y[mask]
    if len(t_us) < min_n:
        return np.array([]), np.array([])
    q = np.arange(int(t_us[0]), int(t_us[-1]) + 1, int(step_us))
    lo = np.searchsorted(t_us, q - win_us // 2); hi = np.searchsorted(t_us, q + win_us // 2)
    out = np.full(len(q), np.nan)
    for i in range(len(q)):
        if hi[i] - lo[i] >= min_n:
            out[i] = y[lo[i]:hi[i]].mean()
    return q, out


def bin_agg(t_us, y, t0, fn=np.mean, bin_us=60_000_000):
    """Aggregate y per non-overlapping 60 s bin keyed off a common t0 (so two
    series pair bin-for-bin) — mean for HR, median for SDNN. Finite values only."""
    y = np.asarray(y, float); m = np.isfinite(y)
    t, y = np.asarray(t_us)[m], y[m]
    if len(t) == 0:
        return {}
    b = ((t - t0) // bin_us).astype(int)
    return {int(k): float(fn(y[b == k])) for k in np.unique(b)}


def main():
    dog = sys.argv[1]
    # ---- CORAL HR tracks (full, with validity) ----
    htools = {}
    sp = os.path.join(H.HERE, "coral_scg", dog, "out.csv")
    if os.path.exists(sp):
        htools["SCG (MWD)"] = C.load_coral(sp, os.path.join(H.HERE, "coral_scg", dog, "mask.parquet"))
    for mod, name in [("ecg_biopac", "BioPac CORAL"), ("ecg_polar", "Polar CORAL")]:
        cp = os.path.join(H.HERE, "coral_ecg", f"{dog}_{mod}", "out.csv")
        if os.path.exists(cp):
            htools[name] = C.load_coral(cp, os.path.join(H.HERE, "coral_ecg", f"{dog}_{mod}", "mask.parquet"))
    # ---- beats (RRi) ----
    beats = {}
    for mod, name in [("ecg_biopac", "BioPac"), ("ecg_polar", "Polar")]:
        f = os.path.join(H.HERE, "ecg_hr", f"{dog}_{mod}.parquet")
        if os.path.exists(f):
            e = pl.read_parquet(f); t = e["ts"].to_numpy().astype("int64"); bpm = e["bpm"].to_numpy().astype(float)
            o = np.argsort(t); beats[name] = (t[o], bpm[o])
    bt = np.concatenate([beats[n][0] for n in beats]); bbpm = np.concatenate([beats[n][1] for n in beats])
    o = np.argsort(bt); bt, bbpm = bt[o], bbpm[o]; bn = 60000.0 / bbpm
    WIN_LO, WIN_HI = int(bt.min()) - 60_000_000, int(bt.max()) + 60_000_000
    scg = htools.get("SCG (MWD)")                          # extend to the on-body SCG session
    if scg is not None and scg[3].any():                   # (so the figure shows the whole collar
        sts, sv = scg[0], scg[3]                           #  recording, not just the ECG sub-window)
        WIN_LO = min(WIN_LO, int(sts[sv].min()) - 30_000_000)
        WIN_HI = max(WIN_HI, int(sts[sv].max()) + 30_000_000)
    for n in list(htools):                                 # clip validity to window
        ts, bpm, sqi, v = htools[n]; htools[n] = (ts, bpm, sqi, v & (ts >= WIN_LO) & (ts <= WIN_HI))

    # ---- HRV series ----
    sdnn_rri, rmssd_rri = H.rolling_hrv(bt, bn)
    rri = {"RMSSD": rmssd_rri, "SDNN": sdnn_rri}
    coral_v = {}
    for full, short in HRV_TOOLS:
        path = (os.path.join(H.HERE, "coral_scg", dog, "out.csv") if full == "SCG (MWD)"
                else os.path.join(H.HERE, "coral_ecg", f"{dog}_ecg_{short.lower()}", "out.csv"))
        mpath = path.replace("out.csv", "mask.parquet")
        if os.path.exists(path):
            cts, cbpm = H.load_coral_valid(path, mpath)
            nn = 60000.0 / H.coral_at_beats(bt, cts, cbpm)
            s, r = H.rolling_hrv(bt, nn); coral_v[short] = {"RMSSD": r, "SDNN": s}

    # ============================ figure ============================
    DISP = {"SCG (MWD)": "SCG CORAL", "BioPac CORAL": "BioPac CORAL", "Polar CORAL": "Polar CORAL"}
    REFCOL = {"BioPac": COL["BioPac"], "Polar": COL["Polar"]}
    span = {nm: (int(t.min()), int(t.max())) for nm, (t, _) in beats.items()}   # each ECG's beat coverage
    date_str = datetime.datetime.fromtimestamp(WIN_LO / 1e6, tz=datetime.timezone.utc).astimezone(C.TZ).strftime("%Y-%m-%d")

    fig = plt.figure(figsize=(16, 12), layout="constrained")
    gs = fig.add_gridspec(5, 4, height_ratios=[1.8, 0.78, 0.78, 0.3, 1.4])     # top row tallest (raw data)
    x0, x1 = C.dn(WIN_LO)[0], C.dn(WIN_HI)[0]

    # ---- row 1 (tallest): beat-by-beat ECG HR + CORAL HR tracks (coverage in legend) ----
    axHR = fig.add_subplot(gs[0, :])
    axHR.plot(C.dn(bt), bbpm, ".", ms=2.2, color="0.55", alpha=0.4, zorder=2, label="RRi beats")
    seg(axHR, bt, median_filter(bbpm, 5), "0.3", label="RRi HR", lw=0.9, zorder=3)
    for n in htools:
        ts, bpm, sqi, v = htools[n]
        cov = 100 * v.sum() / max(1, ((ts >= WIN_LO) & (ts <= WIN_HI)).sum())
        xs = C.dn(ts); pts = np.column_stack([xs, bpm]); segs = np.stack([pts[:-1], pts[1:]], 1)
        c = np.array(to_rgba(C.COL[n]))[None, :].repeat(len(segs), 0); c[:, 3] = np.where(v[:-1], 1.0, 0.0)
        lc = LineCollection(segs, colors=c, linewidths=1.8, label=f"{DISP[n]} ({cov:.0f}%)", zorder=6)
        lc.set_path_effects(HALO); axHR.add_collection(lc)
    axHR.set_ylim(40, 210); axHR.set_xlim(x0, x1); axHR.set_ylabel("heart rate (bpm)")
    axHR.grid(True, alpha=0.3); axHR.legend(loc="upper right", ncol=5, framealpha=0.95, fontsize=8)
    axHR.tick_params(labelbottom=False)
    axHR.set_title(f"[{dog}] — heart rate over time", fontsize=12)

    # ---- row 2: heart rate, 60 s rolling mean — CORAL vs RRi ----
    axA = fig.add_subplot(gs[1, :], sharex=axHR)
    qt, qy = rolling_avg(bt, bbpm)
    axA.plot(C.dn(qt), qy, "-", color="0.2", lw=2.2, label="RRi", zorder=4)
    finite = [qy[np.isfinite(qy)]]
    for n in htools:
        ts, bpm, sqi, v = htools[n]
        q2, y2 = rolling_avg(ts, bpm, v)
        if len(q2):
            ln, = axA.plot(C.dn(q2), y2, "-", color=C.COL[n], lw=1.8, label=DISP[n], zorder=6)
            ln.set_path_effects(HALO); finite.append(y2[np.isfinite(y2)])
    af = np.concatenate(finite)
    axA.set_ylim(np.floor(af.min() / 5) * 5 - 2, np.ceil(af.max() / 5) * 5 + 2)
    axA.set_ylabel("HR, 60 s mean\n(bpm)", fontsize=9); axA.grid(True, alpha=0.3); axA.tick_params(labelbottom=False)
    axA.set_title("heart rate — 60 s rolling mean", fontsize=10)
    axA.legend(loc="upper right", ncol=4, framealpha=0.9, fontsize=7)

    # ---- row 3: HRV (SDNN), 60 s rolling — twin axes, full range (no clip) ----
    axS = fig.add_subplot(gs[2, :], sharex=axHR); axS2 = axS.twinx()
    seg(axS, bt, rri["SDNN"], "0.2", lw=1.5)
    for short in coral_v:
        seg(axS2, bt, coral_v[short]["SDNN"], COL[short], lw=1.3)
    axS.set_ylim(0, np.nanmax(rri["SDNN"]) * 1.05)
    cmax = np.nanmax(np.concatenate([coral_v[s]["SDNN"] for s in coral_v]))
    axS2.set_ylim(0, max(1.0, cmax) * 1.1)
    axS.set_ylabel("RRi SDNN\n(ms)", fontsize=9); axS2.set_ylabel("CORAL SDNN\n(ms)", fontsize=9)
    axS.grid(True, alpha=0.3); axS.tick_params(labelbottom=False)
    axS.set_title("HRV — SDNN, 60 s rolling", fontsize=10)
    axS.legend(handles=[Line2D([0], [0], color="0.2", lw=1.5, label="RRi")] +
               [Line2D([0], [0], color=COL[s], lw=1.3, label=f"CORAL {s}") for s in coral_v],
               loc="upper right", ncol=4, framealpha=0.9, fontsize=7)

    # ---- row 4 (shortest): MWD tri-axial accel activity — 60 s RMS (dB) ----
    axM = fig.add_subplot(gs[3, :], sharex=axHR)
    tmm, dbm = motion_db(dog)
    if tmm is not None:
        inwin = (tmm >= WIN_LO) & (tmm <= WIN_HI)
        floor = float(np.nanpercentile(dbm[inwin], 1)) if inwin.any() else float(np.nanmin(dbm))
        top = float(np.nanpercentile(dbm[inwin], 99.7)) + 1 if inwin.any() else float(np.nanmax(dbm))
        axM.fill_between(C.dn(tmm), floor, dbm, color="#b3873a", alpha=0.45, lw=0)
        axM.plot(C.dn(tmm), dbm, color="#6b4f1d", lw=0.7); axM.set_ylim(floor, top)
    axM.set_ylabel("MWD accel\nRMS, 60 s (dB)", fontsize=9); axM.grid(True, alpha=0.3)
    axM.set_xlim(x0, x1); axM.xaxis.set_major_formatter(DateFormatter("%H:%M", tz=C.TZ))
    axM.set_xlabel(f"America/Chicago local time · {date_str}")

    # ---- row 5: SCG cross-validation onto the ECG references (BioPac / Polar) ----
    scg = htools.get("SCG (MWD)")

    def ref_of(k, bin_us=60_000_000):                 # which ECG covers this 60 s bin's centre
        tc = WIN_LO + (k + 0.5) * bin_us
        for nm, (lo, hi) in span.items():
            if lo <= tc <= hi:
                return nm
        return None

    def grouped(scg_bins, ref_bins):
        g = {}
        for k in sorted(set(scg_bins) & set(ref_bins)):
            r = ref_of(k)
            if r:
                g.setdefault(r, []).append((ref_bins[k], scg_bins[k]))   # (reference, SCG)
        return g

    # left: mean HR — SCG-CORAL vs ECG RRi (same scale -> identity, equal aspect)
    axL = fig.add_subplot(gs[4, 0]); allv, yl = [], 0.97
    if scg is not None:
        ts, bpm, sqi, v = scg
        g = grouped(bin_agg(ts[v], bpm[v], WIN_LO, np.mean), bin_agg(bt, bbpm, WIN_LO, np.mean))
        for r in [x for x in ("BioPac", "Polar") if x in g]:
            x = np.array([p[0] for p in g[r]]); y = np.array([p[1] for p in g[r]])
            axL.scatter(x, y, s=20, alpha=0.75, color=REFCOL[r], edgecolor="none", label=f"vs {r}")
            rr = np.corrcoef(x, y)[0, 1] if len(x) > 2 else float("nan")
            axL.text(0.04, yl, f"vs {r}:  r={rr:+.2f}   MAE={np.mean(np.abs(y-x)):.1f}   bias={np.mean(y-x):+.1f} bpm   n={len(x)}",
                     transform=axL.transAxes, va="top", fontsize=8, color=REFCOL[r]); yl -= 0.075
            allv += [x.min(), x.max(), y.min(), y.max()]
    if allv:
        lim = [min(allv) - 4, max(allv) + 4]
        axL.plot(lim, lim, "-", color="0.6", lw=1, alpha=0.7, zorder=0)
        axL.set_xlim(lim); axL.set_ylim(lim)
        axL.legend(loc="lower right", fontsize=8, framealpha=0.9)
    axL.set_box_aspect(1); axL.grid(True, alpha=0.3)
    axL.set_xlabel("ECG HR (RRi), 60 s mean (bpm)"); axL.set_ylabel("SCG-CORAL HR, 60 s mean (bpm)")
    axL.set_title("SCG vs ECG — mean heart rate", fontsize=10)

    # right: HRV — SCG-CORAL SDNN vs ECG RRi SDNN
    axSc = fig.add_subplot(gs[4, 1]); yl = 0.97
    if scg is not None:
        g = grouped(bin_agg(bt, coral_v["SCG"]["SDNN"], WIN_LO, np.median),
                    bin_agg(bt, rri["SDNN"], WIN_LO, np.median))
        for r in [x for x in ("BioPac", "Polar") if x in g]:
            x = np.array([p[0] for p in g[r]]); y = np.array([p[1] for p in g[r]])
            axSc.scatter(x, y, s=20, alpha=0.7, color=REFCOL[r], edgecolor="none", label=f"vs {r}")
            rr = np.corrcoef(x, y)[0, 1] if len(x) > 2 else float("nan")
            axSc.text(0.04, yl, f"vs {r}:  r={rr:+.2f}   bias={np.mean(y-x):+.0f} ms   n={len(x)}",
                      transform=axSc.transAxes, va="top", fontsize=8, color=REFCOL[r]); yl -= 0.075
        axSc.legend(loc="upper right", fontsize=8, framealpha=0.9)
    axSc.set_xlim(left=0); axSc.set_ylim(bottom=0); axSc.set_box_aspect(1); axSc.grid(True, alpha=0.3)
    axSc.set_xlabel("ECG HRV — SDNN (RRi) (ms)"); axSc.set_ylabel("SCG-CORAL SDNN (ms)")
    axSc.set_title("SCG vs ECG — HRV (SDNN)", fontsize=10)

    # right two columns: Poincaré (RR_n vs RR_n+1) per ECG — geometric HRV decomposition.
    # Density-shaded (per-device colormap, log counts) so the dense diagonal core stands out
    # from off-diagonal artifacts. SD1 (⊥ identity) = beat-to-beat HRV (≈RMSSD/√2, what CORAL
    # misses); SD2 (∥) = long-term. One panel per ECG (HRV is within-recording).
    for nm, col, cmap in [("BioPac", 2, "Blues"), ("Polar", 3, "Greens")]:
        ax = fig.add_subplot(gs[4, col])
        if nm not in beats:
            ax.set_visible(False); continue
        t, bpm = beats[nm]; rr = 60000.0 / bpm
        ok = (np.diff(t) / 1e3) < 1500          # true successive beats (drop missed-beat gaps)
        xr, yr = rr[:-1][ok], rr[1:][ok]
        if len(xr) < 5:
            ax.set_visible(False); continue
        lim = [min(xr.min(), yr.min()) - 30, max(xr.max(), yr.max()) + 30]
        ax.hexbin(xr, yr, gridsize=45, cmap=cmap, bins="log", mincnt=1,
                  extent=(lim[0], lim[1], lim[0], lim[1]), zorder=1)
        sd1 = np.std((xr - yr) / np.sqrt(2)); sd2 = np.std((xr + yr) / np.sqrt(2))
        ax.add_patch(Ellipse((xr.mean(), yr.mean()), 2 * sd2, 2 * sd1, angle=45,
                             fill=False, ec=REFCOL[nm], lw=2.2, zorder=8))
        ax.plot(lim, lim, "-", color="0.5", lw=1, alpha=0.7, zorder=2)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_box_aspect(1)
        ax.text(0.04, 0.97, f"SD1 (beat-to-beat) = {sd1:.0f} ms\nSD2 (long-term) = {sd2:.0f} ms",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.9))
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("RR$_n$ (ms)"); ax.set_ylabel("RR$_{n+1}$ (ms)")
        ax.set_title(f"Poincaré — {nm} RRi", fontsize=10)

    out = os.path.join(C.FIGS, f"summary_{dog}.png"); fig.savefig(out, dpi=200); print("->", out)


if __name__ == "__main__":
    main()

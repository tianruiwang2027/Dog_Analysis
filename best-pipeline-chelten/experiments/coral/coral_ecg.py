#!/usr/bin/env python3
"""ECG-configured CORAL: solve canine HR and render the SOLVED CORRELOFORM per dog.

Adapted from the explore-dog-overnight raw-SCG harness, but for single-lead ECG:
  * Conditioning #1 (signal): feed coral a polarity-blind QRS ENERGY ENVELOPE,
    not the bipolar bandpassed ECG. Pan-Tompkins: bandpass 8-40 Hz -> square ->
    moving-window integrate (~150 ms). WHY: the bipolar QRS flips morphology
    beat-to-beat with respiration, so its autocorrelation at one RR (the
    fundamental T) collapses toward 0 while 2T survives -> a sub-octave latch that
    NO alpha can fix (you can't boost a peak that isn't there). The energy
    envelope is sign-independent, so adjacent beats correlate and the fundamental
    dominates. Then sub-harmonic summation (alpha) works as intended and the full
    50-200 search locks the true rate — no need to shrink the search range.
  * Conditioning #2 (exclude bad time): a --mask, never editing the signal. Bad =
    sustained ADC clipping (motion saturation) OR sustained low QRS periodicity.
    coral DP free-jumps masked hops; SQI then gates per hop.
  * Canine search 50-200 bpm. CEBS analysis fs/hop/windows, smooth-track greeks.

Figure = a CLEAN correlogram of the conditioned signal (multi-window autocorr of
the same energy envelope coral solves on, per-column normalized) as the heatmap,
with the solved HR overlaid (continuous line, opacity ∝ SQI, masked regions
blanked), an INDEPENDENT R-peak HR check (ECG lets us see the beats, so this is a
real reference, not circular), the SQI panel, and phase boundaries. We render our
own correlogram rather than coral's --correloform-png because that PNG flickers
hop-to-hop (a render artifact, badly amplified by the fast RSA ridge in Dog1);
the periodicity it represents is identical, just shown cleanly.

Usage: coral_ecg.py <Device>     e.g. coral_ecg.py Chelten
"""
import sys, os, subprocess, shutil
from pathlib import Path
import numpy as np, polars as pl
from scipy import signal as sg
from scipy.ndimage import binary_closing, binary_dilation, median_filter, uniform_filter1d
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import PowerNorm
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from zoneinfo import ZoneInfo
import datetime

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "ecg_cache"
TZ = ZoneInfo("America/Chicago")
CORAL = shutil.which("coral-st") or "coral-st"

# --- ECG CORAL config ---
FC_LO, FC_HI = 8.0, 40.0                  # QRS band (for the energy-envelope conditioning + masking)
MWI_S = float(os.environ.get("MWI_S", 0.15))   # moving-window integration width (s) for the envelope
# Same canine search for EVERY dog — the energy-envelope conditioning makes the
# fundamental dominant, so the sub-octave is rejected on signal merit, not by
# shrinking the search window.
SEARCH_LO = int(os.environ.get("SLO", 50))
SEARCH_HI = int(os.environ.get("SHI", 200))
A_FS, A_HOP, A_WINS = "512", "128", "256,512,1024,1536"   # CEBS temporal structure (s): 0.5/1/2/3, hop 0.25
# alpha MODERATE: the envelope restores the fundamental (does the work); a little
# sub-harmonic summation polishes it, but HIGH alpha (≈1.0) over-boosts octave-UP
# for the slow dogs (their 2× rate is inside the search), so keep it ≈0.5.
ALPHA = os.environ.get("ALPHA", "0.5"); BETA = "30"; GAMMA = "3"
DELTA = "10"; EPS = os.environ.get("EPS", "0.005"); ZETA = os.environ.get("ZETA", "5"); ETA = "4"
MAXD = os.environ.get("MAXD", "20")

# --- mask (conditioning) params ---
WIN_S, HOP_S = 2.0, 0.5
CLIP_FRAC = float(os.environ.get("CLIP_FRAC", 0.02))      # any sustained railing in a window = motion
PER_THR  = float(os.environ.get("PER_THR", 0.30))         # QRS-band autocorr periodicity floor
MIN_BAD_S, PAD_S, CLOSE_S = 4.0, 1.0, 3.0

# phase transitions (local HH:MM:SS) for annotation, from the source files
PHASES = {
    "Dog1": [], "Chuck": [],
    "Dog2": [("pre_intervention", "16:24:09")],
    "Chelten": [("pre_intervention", "12:48:47"), ("intervention", "13:03:50"),
                ("post_intervention", "13:09:00"), ("post_period", "13:24:02")],
}


def dn(us):
    return mdates.date2num([datetime.datetime.fromtimestamp(t / 1e6, tz=datetime.timezone.utc)
                            .astimezone(TZ) for t in np.atleast_1d(us)])


def load(dev):
    p = next(CACHE.glob(f"username=*/device={dev}/stream=*/date=*/data_0.parquet"), None)
    if p is None:
        sys.exit(f"no parquet for {dev}")
    df = pl.read_parquet(p)
    ts = df["ts"].to_numpy().astype("int64")          # i64 µs UTC
    x = df["c1"].to_numpy().astype(float)
    fs = 1e6 / np.median(np.diff(ts))
    return p, ts, x, fs


def build_mask(ts, x, fs, outdir):
    """Sustained clipping OR sustained low QRS periodicity -> masked intervals."""
    b = sg.butter(4, [FC_LO / (fs / 2), FC_HI / (fs / 2)], btype="band")
    xf = sg.filtfilt(*b, x - x.mean())
    rail = np.abs(x).max()
    railed = np.abs(x) >= 0.98 * rail
    w, hop = int(WIN_S * fs), int(HOP_S * fs)
    lo, hi = int(fs * 60 / SEARCH_HI), int(fs * 60 / SEARCH_LO)   # cardiac lag band
    centers = np.arange(w // 2, len(x) - w // 2, hop)
    clipf = np.empty(len(centers)); per = np.empty(len(centers))
    for i, c in enumerate(centers):
        clipf[i] = railed[c - w // 2:c + w // 2].mean()
        seg = xf[c - w // 2:c + w // 2]; seg = seg - seg.mean()
        n = 1 << int(np.ceil(np.log2(2 * len(seg))))
        f = np.fft.rfft(seg, n); ac = np.fft.irfft(f * np.conj(f), n)[:len(seg)]
        per[i] = ac[lo:hi].max() / ac[0] if ac[0] > 0 else 0.0
    bad = (clipf > CLIP_FRAC) | (per < PER_THR)
    # keep only sustained bad (brief motion is DP-recoverable), then pad + close
    min_run = max(1, int(MIN_BAD_S / HOP_S))
    dd = np.diff(np.concatenate([[0], bad.astype(int), [0]]))
    for s, e in zip(np.where(dd == 1)[0], np.where(dd == -1)[0]):
        if e - s < min_run:
            bad[s:e] = False
    bad = binary_dilation(bad, structure=np.ones(2 * int(PAD_S / HOP_S) + 1))
    bad = binary_closing(bad, structure=np.ones(max(3, int(CLOSE_S / HOP_S))))
    d = np.diff(np.concatenate([[0], bad.astype(int), [0]]))
    rows = [(ts[centers[s]] - int(PAD_S * 1e6), ts[centers[min(e - 1, len(centers) - 1)]] + int(PAD_S * 1e6))
            for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0])]
    mp = outdir / "mask.parquet"
    pl.DataFrame({"mask_start": pl.Series([r[0] for r in rows]).cast(pl.Datetime("us")),
                  "mask_end":   pl.Series([r[1] for r in rows]).cast(pl.Datetime("us"))}).write_parquet(mp)
    print(f"  mask: {100*bad.mean():.0f}% of analysis time, {len(rows)} intervals "
          f"(clip>{CLIP_FRAC} or per<{PER_THR})")
    return mp, rows, xf


def rpeak_hr(ts, xf, fs, masked):
    """Independent R-peak HR, polarity-agnostic, in unmasked regions.

    Detect on the analytic envelope of the QRS-band signal against a rolling-
    MEDIAN floor: QRS occupies a small fraction of each beat, so the 3 s median
    tracks the inter-beat noise floor (not the spikes), and QRS rises many× above
    it. A per-sample threshold of K× that floor plus a 0.28 s refractory rejects
    band-limited T-waves and noise without double-counting. This validates the
    CORAL solve (ECG lets us see the beats) — it is NOT used in the solve."""
    env = np.abs(sg.hilbert(xf))
    floor = median_filter(env, size=int(3 * fs) | 1, mode="nearest") + 1e-9
    pk, _ = sg.find_peaks(env, distance=int(fs * 0.28), height=4.0 * floor)   # 0.28 s: > QRS-T gap, < fastest true RR (~0.32 s @185 bpm)
    pk = pk[~masked[pk]]
    if len(pk) < 3:
        return np.array([]), np.array([])
    rr = np.diff(ts[pk]) / 1e6
    hr = 60.0 / rr
    tmid = ts[pk][:-1] + np.diff(ts[pk]) // 2
    ok = (hr >= SEARCH_LO) & (hr <= SEARCH_HI)
    return tmid[ok], hr[ok]


def condition(ts, xf, fs, outdir):
    """Pan-Tompkins QRS energy envelope: square the 8-40 Hz signal, then moving-
    window integrate (~150 ms, < the fastest RR so beats stay separate). Positive,
    sign-independent, one bump per beat -> a clean fundamental in the correlogram.
    Written as the coral input; coral autocorrelates it directly (NO --fc)."""
    env = uniform_filter1d(xf ** 2, max(1, int(MWI_S * fs)))
    cond = outdir / "cond.parquet"
    pl.DataFrame({"ts": ts, "c1": env.astype("float32")}).write_parquet(cond)
    return cond


def run_coral(src, outdir, maskp):
    cmd = [CORAL, "-i", str(src), "-o", str(outdir / "out.csv"),
           "--analysis-fs", A_FS, "--analysis-hop", A_HOP, "--analysis-windows", A_WINS,
           "--alpha", ALPHA, "--beta", BETA, "--gamma", GAMMA, "--delta", DELTA,
           "--epsilon", EPS, "--zeta", ZETA, "--eta", ETA,
           "--search-bpm-low", str(SEARCH_LO), "--search-bpm-high", str(SEARCH_HI),
           "--max-delta-pct-up", MAXD, "--max-delta-pct-down", MAXD,
           "--mask", str(maskp),
           "--correloform-png-path", str(outdir / "cf_coral.png"),
           "--estimates-png-path", str(outdir / "est_coral.png")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:]); raise SystemExit(1)


def correlogram(env, ts, fs, lo, hi, hop_s=2.0, win_s=(8, 11, 14)):
    """Clean STCT for display: autocorrelate the conditioned envelope in several
    windows (averaged), sample onto a bpm grid. The autocorr is already lag-0
    normalized (∈[0,1]) so a true ridge is bright and noise stays dim — we do NOT
    per-column normalize (that amplifies noise/masked columns into hard vertical
    lines). Windows are 8-14 s, LONGER than the canine respiratory cycle (~6 s),
    so each autocorr averages over respiratory sinus arrhythmia and the ridge is a
    smooth band instead of a fast wobble that aliases into vertical stripes at
    full-record scale (the RSA detail survives in the solved HR line)."""
    bpm = np.arange(lo, hi + 0.5, 0.5)
    lags = (60.0 / bpm * fs).astype(int)
    wins = [int(w * fs) for w in win_s]
    W, hop = max(wins), int(hop_s * fs)
    centers = np.arange(W // 2, max(W // 2 + 1, len(env) - W // 2), hop)
    cf = np.zeros((len(bpm), len(centers)))
    for w in wins:
        for j, c in enumerate(centers):
            s = env[c - w // 2:c + w // 2]
            s = s - s.mean()
            n = 1 << int(np.ceil(np.log2(2 * max(2, len(s)))))
            f = np.fft.rfft(s, n)
            ac = np.fft.irfft(f * np.conj(f), n)[:len(s)]
            if ac[0] > 0:
                cf[:, j] += np.clip(ac[lags] / ac[0], 0, None)
    cf /= len(wins)
    return ts[centers], bpm, cf


def figure(dev, outdir, ts, xf, fs, mask_rows):
    k = pl.read_csv(outdir / "out.csv")
    cts = k["ts"].to_numpy().astype("datetime64[us]").astype("int64")
    cb = k["bpm"].to_numpy().astype(float); cq = k["sqi"].to_numpy().astype(float)
    hopdt = max(1e-6, np.median(np.diff(cts)) / 1e6)
    wmed = max(1, int(round(3.0 / hopdt)) | 1)
    cbf = median_filter(cb, size=wmed, mode="nearest")            # 3 s cosmetic smoothing
    x = dn(cts); x0, x1 = x[0], x[-1]

    # masked spans on the coral hop grid
    in_mask = np.zeros(len(cts), bool)
    for s, e in mask_rows:
        in_mask |= (cts >= s) & (cts <= e)
    # per-sample masked flag (for the R-peak check)
    smask = np.zeros(len(ts), bool)
    for s, e in mask_rows:
        smask |= (ts >= s) & (ts <= e)
    rt, rhr = rpeak_hr(ts, xf, fs, smask)

    fig, (ax, axq) = plt.subplots(2, 1, figsize=(22, 10), height_ratios=[3.4, 1.0], sharex=True)

    # row 1: clean correlogram of the conditioned envelope + solved HR (opacity ∝ SQI)
    env = uniform_filter1d(xf ** 2, max(1, int(MWI_S * fs)))
    cgt, cbpm, cg = correlogram(env, ts, fs, SEARCH_LO, SEARCH_HI)
    cmask = np.zeros(len(cgt), bool)                       # blank masked columns (grey axvspan covers them)
    for s, e in mask_rows:
        cmask |= (cgt >= s) & (cgt <= e)
    cg[:, cmask] = np.nan
    vmax = np.nanpercentile(cg, 99.5)
    ax.pcolormesh(dn(cgt), cbpm, cg, cmap="magma", norm=PowerNorm(0.6, vmin=0, vmax=vmax),
                  shading="gouraud", rasterized=True, zorder=0)
    # independent R-peak HR (validation)
    if len(rt):
        ax.plot(dn(rt), rhr, ".", color="white", ms=2.0, alpha=0.55, zorder=4,
                label="R-peak HR (independent)")
    pts = np.column_stack([x, cbf]); segs = np.stack([pts[:-1], pts[1:]], axis=1)
    a = 0.30 + 0.70 * np.clip(cq[:-1] / 0.25, 0.0, 1.0)
    a[in_mask[:-1] | in_mask[1:]] = 0.0
    colors = np.zeros((len(segs), 4)); colors[:, 1] = 0.95; colors[:, 2] = 1.0; colors[:, 3] = a  # cyan
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=2.2, zorder=5,
                                     label="CORAL HR (opacity ∝ SQI)"))
    for s, e in mask_rows:
        ax.axvspan(dn(s)[0], dn(e)[0], color="0.55", alpha=0.5, zorder=2)
        axq.axvspan(dn(s)[0], dn(e)[0], color="0.55", alpha=0.5)
    # phase boundaries
    day = datetime.datetime.fromtimestamp(cts[0] / 1e6, tz=datetime.timezone.utc).astimezone(TZ).date()
    for name, hhmm in PHASES.get(dev, []):
        h, m, s = map(int, hhmm.split(":"))
        pt = mdates.date2num(datetime.datetime(day.year, day.month, day.day, h, m, s, tzinfo=TZ))
        ax.axvline(pt, color="#33ff99", ls="--", lw=1.2, alpha=0.9, zorder=6)
        ax.text(pt, SEARCH_HI - 4, " " + name, color="#33ff99", fontsize=8, va="top", ha="left", zorder=7)

    acc = np.isfinite(cbf) & (cq >= 0.10) & ~in_mask
    med = np.median(cbf[acc]); p10, p90 = np.percentile(cbf[acc], [10, 90])
    ax.set_ylim(SEARCH_LO, SEARCH_HI); ax.set_ylabel("Candidate rate / HR (bpm)")
    ax.legend(loc="upper right", markerscale=4, framealpha=0.85)
    ax.set_title(
        f"[{dev}] Canine ECG heart rate (CORAL) — conditioning = QRS energy envelope "
        f"(bp {FC_LO:.0f}–{FC_HI:.0f} Hz → square → {MWI_S*1000:.0f} ms integrate) + clip/low-periodicity --mask · "
        f"CEBS fs/hop/windows · α={ALPHA} β=30 ε={EPS} ζ={ZETA} η=4 · search {SEARCH_LO}–{SEARCH_HI} bpm\n"
        f"correlogram of the conditioned envelope (2–5 s windows, per-col norm); cyan = CORAL solved HR (opacity ∝ SQI), white = R-peak check · "
        f"HR median {med:.0f} (p10–p90 {p10:.0f}–{p90:.0f}) bpm · {100*acc.mean():.0f}% at SQI≥0.10", fontsize=11)

    # row 2: SQI
    axq.fill_between(x, 0, cq, color="0.7", lw=0, step="mid"); axq.plot(x, cq, color="black", lw=0.3)
    axq.axhline(0.10, color="red", ls="--", lw=1)
    axq.set_ylim(0, max(0.3, np.ceil(np.nanmax(cq) * 10) / 10)); axq.set_ylabel("SQI"); axq.grid(True, alpha=0.3)
    axq.xaxis.set_major_formatter(DateFormatter("%H:%M", tz=TZ)); axq.set_xlabel("America/Chicago time")

    ax.set_xlim(x0, x1)
    fig.tight_layout()
    out = outdir / f"CORRELOFORM_{dev}.png"
    fig.savefig(out, dpi=150); print("  ->", out)
    return med, p10, p90, 100 * acc.mean()


def main():
    dev = sys.argv[1]
    outdir = HERE / "out" / dev; outdir.mkdir(parents=True, exist_ok=True)
    src, ts, x, fs = load(dev)
    print(f"[{dev}] {len(x):,} samples  fs≈{fs:.1f} Hz")
    maskp, rows, xf = build_mask(ts, x, fs, outdir)
    cond = condition(ts, xf, fs, outdir)
    run_coral(cond, outdir, maskp)
    figure(dev, outdir, ts, xf, fs, rows)


if __name__ == "__main__":
    main()

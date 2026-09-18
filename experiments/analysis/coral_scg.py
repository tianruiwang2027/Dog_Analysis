#!/usr/bin/env python3
"""CORAL heart-rate from MWD SCG (3-axis accelerometer, stream 45 @2 kHz).

Adapted from coral_ecg.py for seismocardiography: coral-st locks the canine beat
from the correlogram of the RAW accel axis bandpassed to the SCG vibration band.
Conditioning is HONEST via a --mask (never editing the raw): masked = sustained
ADC saturation (int16 railing = motion) OR sustained low SCG-band periodicity
(no detectable beat). coral DP free-jumps masked hops; SQI gates per hop.

Per dog we feed the accel axis the probe found most cardiac-periodic.

Usage: coral_scg.py <Dog>     e.g. coral_scg.py Chelten
"""
import sys, os, glob, shutil, subprocess
import numpy as np, polars as pl
from scipy import signal as sg
from scipy.ndimage import binary_closing, binary_dilation

HERE = os.path.dirname(os.path.abspath(__file__))
HIVE = os.path.join(os.path.dirname(HERE), "hive")
CORAL = shutil.which("coral-st") or "coral-st"

# axis selected by beat-locked-prominence test (investigate_scg.py); 'mag' = |accel AC|
AXIS = {"Chelten": "mag", "Dasty": "c2", "Chuck": "mag"}
FC_LO, FC_HI = 40.0, 150.0                                  # canine heart-sound (S1/S2) band
SEARCH_LO, SEARCH_HI = 50, 200                              # canine bpm
A_FS, A_HOP, A_WINS = "512", "128", "256,512,1024,1536"     # CEBS temporal structure
ALPHA, BETA, GAMMA, DELTA = "0.0", "30", "3", "10"
EPS, ZETA, ETA, MAXD = "0.005", "5", "4", "20"
SAT = 32767                                                 # int16 rail
WIN_S, HOP_S = 2.0, 0.5
CLIP_FRAC = 0.05                                            # sustained railing in a window = motion
PER_THR = 0.12                                             # SCG-band periodicity floor (lower than ECG)
MIN_BAD_S, PAD_S, CLOSE_S = 4.0, 1.0, 3.0


def ecg_window(dog, margin_s=120):
    """[lo,hi] µs envelope of this dog's ECG (BioPac+Polar) ± margin. This is the
    period the dog was provably instrumented (on-body, validated by concurrent ECG).
    None if no ECG. We scope SCG-CORAL to this — outside it the sensor is unvalidated
    and may be off-body, which we deliberately do not analyze."""
    spans = []
    for mod in ("ecg_biopac", "ecg_polar"):
        for p in glob.glob(f"{HIVE}/username={mod}/device={dog}/stream=*/date=*/data_0.parquet"):
            t = pl.read_parquet(p, columns=["ts"])["ts"].to_numpy().astype("int64")
            if len(t):
                spans.append((int(t.min()), int(t.max())))
    if not spans:
        return None
    return min(s[0] for s in spans) - margin_s * 1_000_000, max(s[1] for s in spans) + margin_s * 1_000_000


def session_window(dog, gap_min=10.0, near_min=5.0):
    """The continuous SCG recording SESSION containing the ECG: split the SCG on
    gaps > gap_min, keep the block(s) overlapping/within near_min of the ECG. This
    captures the whole on-body session (Chuck's 1 h, Dasty's all-day) while EXCLUDING
    lone far blips + the multi-hour gaps before them — interpolating CORAL across
    those fabricates HR (Chuck `full` read 155 bpm). Falls back to the ECG window."""
    ew = ecg_window(dog, margin_s=0)
    p = glob.glob(f"{HIVE}/username=scg_mwd/device={dog}/stream=45/date=*/data_0.parquet")[0]
    ts = pl.read_parquet(p, columns=["ts"])["ts"].to_numpy().astype("int64")
    if ew is None or len(ts) < 2:
        return (int(ts[0]), int(ts[-1])) if len(ts) else None
    d = np.diff(ts); brk = np.where(d > gap_min * 60 * 1_000_000)[0]
    starts = np.concatenate([[0], brk + 1]); ends = np.concatenate([brk, [len(ts) - 1]])
    near = near_min * 60 * 1_000_000; elo, ehi = ew
    sel = [(int(ts[s]), int(ts[e])) for s, e in zip(starts, ends) if ts[e] >= elo - near and ts[s] <= ehi + near]
    if not sel:
        return elo - 120_000_000, ehi + 120_000_000
    return min(a for a, _ in sel) - 30_000_000, max(b for _, b in sel) + 30_000_000


def load(dog, scope=True):
    p = glob.glob(f"{HIVE}/username=scg_mwd/device={dog}/stream=45/date=*/data_0.parquet")[0]
    df = pl.read_parquet(p)
    ts = df["ts"].to_numpy().astype("int64")
    ax = AXIS[dog]
    if ax == "mag":
        x = np.sqrt(sum((df[c].to_numpy().astype(float) - df[c].mean()) ** 2 for c in ("c1", "c2", "c3")))
    else:
        x = df[ax].to_numpy().astype(float)
    d = np.diff(ts); d = d[d > 0]
    fs = float(1e6 / np.median(d))
    w = session_window(dog) if scope else None            # scope to the continuous on-body session
    if w is not None:
        sel = (ts >= w[0]) & (ts <= w[1]); ts, x = ts[sel], x[sel]
    return p, ts, x, fs, ax


def build_mask(ts, x, fs, outdir):
    b = sg.butter(4, [FC_LO / (fs / 2), FC_HI / (fs / 2)], btype="band")
    xf = sg.filtfilt(*b, x - x.mean())
    railed = np.abs(x) >= 0.98 * SAT
    w, hop = int(WIN_S * fs), int(HOP_S * fs)
    lo, hi = int(fs * 60 / SEARCH_HI), int(fs * 60 / SEARCH_LO)
    centers = np.arange(w // 2, len(x) - w // 2, hop)
    clipf = np.empty(len(centers)); per = np.empty(len(centers))
    env_all = np.abs(sg.hilbert(xf))                       # beat energy envelope
    for i, c in enumerate(centers):
        sl = slice(c - w // 2, c + w // 2)
        clipf[i] = railed[sl].mean()
        seg = env_all[sl]; seg = seg - seg.mean()
        n = 1 << int(np.ceil(np.log2(2 * len(seg))))
        f = np.fft.rfft(seg, n); ac = np.fft.irfft(f * np.conj(f), n)[:len(seg)]
        per[i] = ac[lo:hi].max() / ac[0] if ac[0] > 0 else 0.0
    bad = (clipf > CLIP_FRAC) | (per < PER_THR)
    min_run = max(1, int(MIN_BAD_S / HOP_S))
    dd = np.diff(np.concatenate([[0], bad.astype(int), [0]]))
    for s, e in zip(np.where(dd == 1)[0], np.where(dd == -1)[0]):
        if e - s < min_run:
            bad[s:e] = False
    bad = binary_dilation(bad, structure=np.ones(2 * int(PAD_S / HOP_S) + 1))
    bad = binary_closing(bad, structure=np.ones(max(3, int(CLOSE_S / HOP_S))))
    d = np.diff(np.concatenate([[0], bad.astype(int), [0]]))
    rows = [(int(ts[centers[s]] - PAD_S * 1e6),
             int(ts[centers[min(e - 1, len(centers) - 1)]] + PAD_S * 1e6))
            for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0])]
    mp = os.path.join(outdir, "mask.parquet")
    pl.DataFrame({"mask_start": pl.Series([r[0] for r in rows]).cast(pl.Datetime("us")),
                  "mask_end":   pl.Series([r[1] for r in rows]).cast(pl.Datetime("us"))}).write_parquet(mp)
    print(f"  mask: {100*bad.mean():.0f}% of analysis time, {len(rows)} intervals "
          f"(clip>{CLIP_FRAC} or per<{PER_THR})")
    return mp, rows


def main():
    dog = sys.argv[1]
    FULL = len(sys.argv) > 2 and sys.argv[2] == "full"    # 'full' = analyze the whole recording, not just the ECG window
    outdir = os.path.join(HERE, "coral_scg", dog + ("_full" if FULL else "")); os.makedirs(outdir, exist_ok=True)
    src, ts, x, fs, ax = load(dog, scope=not FULL)
    span = (ts[-1] - ts[0]) / 1e6 / 60
    print(f"[{dog}] SCG axis={ax} {len(x):,} samples fs={fs:.1f}Hz span={span:.1f}min")
    # raw chosen accel axis -> coral bandpasses to the 40-150 Hz heart-sound band and
    # locks the beat from the burst-to-burst autocorrelation (channel selection only).
    inp = os.path.join(outdir, "scg_in.parquet")
    pl.DataFrame({"ts": ts, "c1": x}).write_parquet(inp)
    maskp, rows = build_mask(ts, x, fs, outdir)
    cmd = [CORAL, "-i", inp, "-o", os.path.join(outdir, "out.csv"),
           "--fc-low", str(FC_LO), "--fc-high", str(FC_HI),
           "--analysis-fs", A_FS, "--analysis-hop", A_HOP, "--analysis-windows", A_WINS,
           "--alpha", ALPHA, "--beta", BETA, "--gamma", GAMMA, "--delta", DELTA,
           "--epsilon", EPS, "--zeta", ZETA, "--eta", ETA,
           "--search-bpm-low", str(SEARCH_LO), "--search-bpm-high", str(SEARCH_HI),
           "--max-delta-pct-up", MAXD, "--max-delta-pct-down", MAXD, "--mask", maskp,
           "--correloform-png-path", os.path.join(outdir, "cf_coral.png"),
           "--estimates-png-path", os.path.join(outdir, "est_coral.png")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("CORAL FAILED:\n", r.stderr[-2000:]); raise SystemExit(1)
    k = pl.read_csv(os.path.join(outdir, "out.csv"))
    bpm = k["bpm"].to_numpy(); sqi = k["sqi"].to_numpy()
    ok = sqi >= 0.10
    print(f"  -> out.csv  {len(k):,} hops  HR median {np.median(bpm[ok]):.0f}bpm "
          f"(SQI≥0.10 on {100*ok.mean():.0f}% of hops)")


if __name__ == "__main__":
    main()

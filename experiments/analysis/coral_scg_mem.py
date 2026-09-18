#!/usr/bin/env python3
"""Memory-lean re-run of coral_scg.py for large single-axis sessions (e.g. Dasty):
identical algorithm/params, but reads only the needed parquet columns instead of
the full dataframe (the original coral_scg.load() reads all channels even when
axis != 'mag', which OOMs on Dasty's ~5.2h/2kHz/3-axis session)."""
import sys, os, glob, shutil, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, polars as pl
from scipy import signal as sg
from scipy.ndimage import binary_closing, binary_dilation
import coral_scg as C

HERE = C.HERE
HIVE = C.HIVE
CORAL = C.CORAL


def load_lean(dog, scope=True):
    ax = C.AXIS[dog]
    p = glob.glob(f"{HIVE}/username=scg_mwd/device={dog}/stream=45/date=*/data_0.parquet")[0]
    cols = ["ts", "c1", "c2", "c3"] if ax == "mag" else ["ts", ax]
    df = pl.read_parquet(p, columns=cols)
    ts = df["ts"].to_numpy().astype("int64")
    if ax == "mag":
        x = np.sqrt(sum((df[c].to_numpy().astype(np.float32) - df[c].to_numpy().astype(np.float32).mean()) ** 2
                         for c in ("c1", "c2", "c3")))
    else:
        x = df[ax].to_numpy().astype(np.float32)
    del df
    d = np.diff(ts); d = d[d > 0]
    fs = float(1e6 / np.median(d))
    w = C.session_window(dog) if scope else None
    if w is not None:
        sel = (ts >= w[0]) & (ts <= w[1]); ts, x = ts[sel], x[sel]
    return p, ts, x, fs, ax


def build_mask_lean(ts, x, fs, outdir, chunk_s=600.0, overlap_s=2.0):
    """Same thresholds/logic as coral_scg.build_mask, but filtfilt+hilbert run on
    overlapping chunks (float32) to bound peak memory; overlap >> filter/Hilbert
    edge-transient length so results match the whole-array version to within
    floating-point noise away from chunk seams."""
    b = sg.butter(4, [C.FC_LO / (fs / 2), C.FC_HI / (fs / 2)], btype="band")
    n = len(x)
    chunk = int(chunk_s * fs); ov = int(overlap_s * fs)
    xf = np.empty(n, dtype=np.float32)
    env_all = np.empty(n, dtype=np.float32)
    lo_i = 0
    while lo_i < n:
        hi_i = min(n, lo_i + chunk)
        a = max(0, lo_i - ov); bnd = min(n, hi_i + ov)
        seg = x[a:bnd].astype(np.float64)
        seg_f = sg.filtfilt(*b, seg - seg.mean()).astype(np.float32)
        seg_env = np.abs(sg.hilbert(seg_f)).astype(np.float32)
        xf[lo_i:hi_i] = seg_f[lo_i - a:hi_i - a]
        env_all[lo_i:hi_i] = seg_env[lo_i - a:hi_i - a]
        lo_i = hi_i
    railed = np.abs(x) >= 0.98 * C.SAT
    w, hop = int(C.WIN_S * fs), int(C.HOP_S * fs)
    lo, hi = int(fs * 60 / C.SEARCH_HI), int(fs * 60 / C.SEARCH_LO)
    centers = np.arange(w // 2, n - w // 2, hop)
    clipf = np.empty(len(centers)); per = np.empty(len(centers))
    for i, c in enumerate(centers):
        sl = slice(c - w // 2, c + w // 2)
        clipf[i] = railed[sl].mean()
        seg = env_all[sl].astype(np.float64); seg = seg - seg.mean()
        nn = 1 << int(np.ceil(np.log2(2 * len(seg))))
        f = np.fft.rfft(seg, nn); ac = np.fft.irfft(f * np.conj(f), nn)[:len(seg)]
        per[i] = ac[lo:hi].max() / ac[0] if ac[0] > 0 else 0.0
    bad = (clipf > C.CLIP_FRAC) | (per < C.PER_THR)
    min_run = max(1, int(C.MIN_BAD_S / C.HOP_S))
    dd = np.diff(np.concatenate([[0], bad.astype(int), [0]]))
    for s, e in zip(np.where(dd == 1)[0], np.where(dd == -1)[0]):
        if e - s < min_run:
            bad[s:e] = False
    bad = binary_dilation(bad, structure=np.ones(2 * int(C.PAD_S / C.HOP_S) + 1))
    bad = binary_closing(bad, structure=np.ones(max(3, int(C.CLOSE_S / C.HOP_S))))
    d = np.diff(np.concatenate([[0], bad.astype(int), [0]]))
    rows = [(int(ts[centers[s]] - C.PAD_S * 1e6),
             int(ts[centers[min(e - 1, len(centers) - 1)]] + C.PAD_S * 1e6))
            for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0])]
    mp = os.path.join(outdir, "mask.parquet")
    pl.DataFrame({"mask_start": pl.Series([r[0] for r in rows]).cast(pl.Datetime("us")),
                  "mask_end":   pl.Series([r[1] for r in rows]).cast(pl.Datetime("us"))}).write_parquet(mp)
    print(f"  mask: {100*bad.mean():.0f}% of analysis time, {len(rows)} intervals "
          f"(clip>{C.CLIP_FRAC} or per<{C.PER_THR})")
    return mp, rows


def main():
    dog = sys.argv[1]
    outdir = os.path.join(HERE, "coral_scg", dog); os.makedirs(outdir, exist_ok=True)
    _, ts, x, fs, ax = load_lean(dog, scope=True)
    span = (ts[-1] - ts[0]) / 1e6 / 60
    print(f"[{dog}] SCG axis={ax} {len(x):,} samples fs={fs:.1f}Hz span={span:.1f}min (mem-lean)")
    inp = os.path.join(outdir, "scg_in.parquet")
    pl.DataFrame({"ts": ts, "c1": x}).write_parquet(inp)
    maskp, rows = build_mask_lean(ts, x, fs, outdir)
    del x
    cmd = [CORAL, "-i", inp, "-o", os.path.join(outdir, "out.csv"),
           "--fc-low", str(C.FC_LO), "--fc-high", str(C.FC_HI),
           "--analysis-fs", C.A_FS, "--analysis-hop", C.A_HOP, "--analysis-windows", C.A_WINS,
           "--alpha", C.ALPHA, "--beta", C.BETA, "--gamma", C.GAMMA, "--delta", C.DELTA,
           "--epsilon", C.EPS, "--zeta", C.ZETA, "--eta", C.ETA,
           "--search-bpm-low", str(C.SEARCH_LO), "--search-bpm-high", str(C.SEARCH_HI),
           "--max-delta-pct-up", C.MAXD, "--max-delta-pct-down", C.MAXD, "--mask", maskp,
           "--correloform-png-path", os.path.join(outdir, "cf_coral.png"),
           "--estimates-png-path", os.path.join(outdir, "est_coral.png")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("CORAL FAILED:\n", r.stderr[-2000:]); raise SystemExit(1)
    k = pl.read_csv(os.path.join(outdir, "out.csv"))
    bpm = k["bpm"].to_numpy(); sqi = k["sqi"].to_numpy()
    ok = sqi >= 0.10
    print(f"  -> out.csv  {len(k):,} hops  HR median {np.median(bpm[ok]):.0f}bpm "
          f"(SQI>=0.10 on {100*ok.mean():.0f}% of hops)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rebuild the Polar ECG lead-off mask for Chelten with a SYMMETRIC sustained-run
rule: not only must a BAD (low QRS-band periodicity) stretch last >=MIN_BAD_S to
start masking (existing rule), a GOOD stretch must now also last >=MIN_GOOD_S to
END masking. This stops a single noisy periodicity blip from splitting one
continuous recovery-artifact dead-zone into disconnected masked islands with an
unmasked "gap corridor" in between (exactly what produced the 17:13:41 ramp).

Then reruns coral-st on the full Chelten Polar ECG with this improved mask and
writes a new out.csv alongside the original (does not overwrite it).
"""
import os, glob, subprocess, shutil
import numpy as np, polars as pl
from scipy import signal as sg
from scipy.ndimage import binary_dilation, binary_closing

HERE = os.path.dirname(os.path.abspath(__file__))
HIVE = os.path.join(os.path.dirname(HERE), "hive")
CORAL = shutil.which("coral-st") or "coral-st"

FC_LO, FC_HI = 10.0, 30.0          # match ecg_hr.py's contact_mask band
SEARCH_LO, SEARCH_HI = 50, 200
WIN_S, HOP_S = 2.0, 0.5
PER_THR = 0.20
MIN_BAD_S, PAD_S = 3.0, 0.5
CLOSE_GAP_S = 1.0                   # NEW: bridge nearby short bad-runs before length-filtering
                                     # (tuned: 1.0s merges the 17:13:41 dead-zone into one
                                     # continuous mask while only ~doubling global coverage,
                                     # 4.1%->8.0%; >=2.5s causes runaway merging elsewhere, 33-41%)

A_FS, A_HOP, A_WINS = "512", "128", "256,512,1024,1536"


def contact_mask_fixed(ts, xf, fs):
    w, hop = int(WIN_S * fs), int(HOP_S * fs)
    lo, hi = int(fs * 60 / SEARCH_HI), int(fs * 60 / SEARCH_LO)
    env = np.abs(sg.hilbert(xf))
    centers = np.arange(w // 2, len(xf) - w // 2, hop)
    per = np.empty(len(centers))
    for i, c in enumerate(centers):
        seg = env[c - w // 2:c + w // 2]; seg = seg - seg.mean()
        n = 1 << int(np.ceil(np.log2(2 * len(seg))))
        f = np.fft.rfft(seg, n); ac = np.fft.irfft(f * np.conj(f), n)[:len(seg)]
        per[i] = ac[lo:hi].max() / ac[0] if ac[0] > 0 else 0.0
    bad = per < PER_THR

    def drop_short_runs(mask, min_run):
        mask = mask.copy()
        dd = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
        for s, e in zip(np.where(dd == 1)[0], np.where(dd == -1)[0]):
            if e - s < min_run:
                mask[s:e] = False
        return mask

    min_bad_run = max(1, int(MIN_BAD_S / HOP_S))
    close_n = max(1, int(round(CLOSE_GAP_S / HOP_S)))
    # NEW: bridge nearby short bad-runs BEFORE length-filtering, so a cluster of
    # several near-threshold flickers close together (none individually >=3s) is
    # recognized as one continuous unreliable stretch instead of each fragment
    # separately failing the sustained-length test.
    bad = binary_closing(bad, structure=np.ones(close_n))
    bad = drop_short_runs(bad, min_bad_run)          # then drop genuinely isolated brief blips

    bad = binary_dilation(bad, structure=np.ones(2 * int(PAD_S / HOP_S) + 1))
    bad = binary_closing(bad, structure=np.ones(3))
    d = np.diff(np.concatenate([[0], bad.astype(int), [0]]))
    return per, centers, [(int(ts[centers[s]]), int(ts[centers[min(e - 1, len(centers) - 1)]]))
                           for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0])]


def detect_qrs_filt(x, fs):
    b = sg.butter(2, [FC_LO / (fs / 2), FC_HI / (fs / 2)], btype="band")
    return sg.filtfilt(*b, x - np.mean(x))


def main():
    dog, mod = "Chelten", "ecg_polar"
    p = glob.glob(f"{HIVE}/username={mod}/device={dog}/stream=0/date=*/data_0.parquet")[0]
    df = pl.read_parquet(p, columns=["ts", "c1"])
    ts = df["ts"].to_numpy().astype("int64"); x = df["c1"].to_numpy().astype(float)
    fs = float(1e6 / np.median(np.diff(ts)))
    xf = detect_qrs_filt(x, fs)

    per, centers, rows_old_logic = contact_mask_fixed(ts, xf, fs)
    print(f"[fixed mask] {len(rows_old_logic)} intervals, "
          f"{100*sum(e-s for s,e in rows_old_logic)/1e6/ (ts[-1]-ts[0])*1e6:.1f}% of span")

    outdir = os.path.join(HERE, "coral_ecg", f"{dog}_{mod}_maskfix")
    os.makedirs(outdir, exist_ok=True)
    inp = os.path.join(outdir, "ecg_in.parquet")
    pl.DataFrame({"ts": ts, "c1": x}).write_parquet(inp)
    maskp = os.path.join(outdir, "mask.parquet")
    pl.DataFrame({"mask_start": pl.Series([a for a, _ in rows_old_logic]).cast(pl.Datetime("us")),
                  "mask_end": pl.Series([b for _, b in rows_old_logic]).cast(pl.Datetime("us"))}).write_parquet(maskp)

    cmd = [CORAL, "-i", inp, "-o", os.path.join(outdir, "out.csv"),
           "--fc-low", "8.0", "--fc-high", "40.0",
           "--analysis-fs", A_FS, "--analysis-hop", A_HOP, "--analysis-windows", A_WINS,
           "--alpha", "0.0", "--beta", "30", "--gamma", "3", "--delta", "10",
           "--epsilon", "0.005", "--zeta", "5", "--eta", "4",
           "--search-bpm-low", str(SEARCH_LO), "--search-bpm-high", str(SEARCH_HI),
           "--max-delta-pct-up", "20", "--max-delta-pct-down", "20",
           "--mask", maskp]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:]); raise SystemExit(1)
    print("-> ", os.path.join(outdir, "out.csv"))


if __name__ == "__main__":
    main()

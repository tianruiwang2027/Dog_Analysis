#!/usr/bin/env python3
"""Reference heart-rate from ECG — the ground truth the SCG-CORAL HR is compared
against. Pan-Tompkins-style QRS detection (bandpass -> derivative -> square ->
moving-window integration -> adaptive peak picking with a canine refractory),
then RR artifact rejection (Hampel) that KEEPS real respiratory sinus arrhythmia
but removes T-wave double-counts / missed-beat spikes. Lead-off (Polar) and
railed (BioPac) stretches are gated out and saved as excluded intervals.

Outputs ecg_hr/<dog>_<modality>.parquet (ts, bpm = cleaned beat HR) + .json, and
a QC overlay PNG so detection can be eyeballed.

Usage: ecg_hr.py <Dog> <ecg_biopac|ecg_polar>
"""
import sys, os, glob, json
import numpy as np, polars as pl
from scipy import signal as sg
from scipy.ndimage import median_filter, binary_dilation, binary_closing, uniform_filter1d
import datetime
from zoneinfo import ZoneInfo
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
HIVE = os.path.join(os.path.dirname(HERE), "hive")
FIGS = os.path.join(os.path.dirname(HERE), "figs")
TZ = ZoneInfo("America/Chicago")
FC_LO, FC_HI = 10.0, 30.0        # QRS energy band (dodges 60 Hz / EMG / baseline)
SEARCH_LO, SEARCH_HI = 50, 200
REFRACT_S = 0.28                 # canine: caps at ~214 bpm, rejects T-wave re-trigger
LEAD = {"ecg_biopac": "c4", "ecg_polar": "c1"}      # BioPac Lead3 (clean), Polar single lead
RAIL = {"ecg_biopac": 9.999, "ecg_polar": None}
WIN_S, HOP_S = 2.0, 0.5
PER_THR = 0.20
MIN_BAD_S, PAD_S = 3.0, 0.5


def detect_qrs(x, fs):
    """Pan-Tompkins: bandpass, 5-tap derivative, square, MWI, adaptive picking."""
    b = sg.butter(2, [FC_LO / (fs / 2), FC_HI / (fs / 2)], btype="band")
    xf = sg.filtfilt(*b, x - np.mean(x))
    deriv = np.convolve(xf, np.array([1, 2, 0, -2, -1]) * (fs / 8.0), mode="same")
    sqd = deriv ** 2
    mwi = uniform_filter1d(sqd, max(1, int(0.05 * fs)))          # ~50 ms canine QRS
    # adaptive: peak must exceed K x local median of MWI (2 s)
    floor = median_filter(mwi, int(2 * fs) | 1, mode="nearest") + 1e-12
    pk, _ = sg.find_peaks(mwi, distance=int(fs * REFRACT_S), height=None)
    snr = mwi[pk] / floor[pk]
    pk = pk[snr > 4.0]
    # refine each detection to the local |xf| maximum within +-60 ms (true R)
    win = int(0.06 * fs)
    R = []
    for p in pk:
        a, bnd = max(0, p - win), min(len(xf), p + win)
        R.append(a + int(np.argmax(np.abs(xf[a:bnd]))))
    return np.unique(R), xf


def clean_hr(tpk):
    """RR -> HR with Hampel rejection (keeps RSA, drops double/half artifacts)."""
    if len(tpk) < 4:
        return np.array([]), np.array([])
    rr = np.diff(tpk) / 1e6
    hr = 60.0 / rr
    tmid = tpk[:-1] + np.diff(tpk) // 2
    keep = (hr >= SEARCH_LO) & (hr <= SEARCH_HI)
    med = median_filter(hr, 7, mode="nearest")
    mad = median_filter(np.abs(hr - med), 7, mode="nearest") + 1e-6
    keep &= np.abs(hr - med) <= 5.0 * 1.4826 * mad      # generous: RSA swings are real
    return tmid[keep], hr[keep]


def contact_mask(ts, xf, fs):
    """Sustained low QRS-band periodicity -> lead-off / poor-contact intervals."""
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
    min_run = max(1, int(MIN_BAD_S / HOP_S))
    dd = np.diff(np.concatenate([[0], bad.astype(int), [0]]))
    for s, e in zip(np.where(dd == 1)[0], np.where(dd == -1)[0]):
        if e - s < min_run:
            bad[s:e] = False
    bad = binary_dilation(bad, structure=np.ones(2 * int(PAD_S / HOP_S) + 1))
    bad = binary_closing(bad, structure=np.ones(3))
    d = np.diff(np.concatenate([[0], bad.astype(int), [0]]))
    return [(int(ts[centers[s]]), int(ts[centers[min(e - 1, len(centers) - 1)]]))
            for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0])]


def main():
    dog, modality = sys.argv[1], sys.argv[2]
    p = glob.glob(f"{HIVE}/username={modality}/device={dog}/stream=0/date=*/data_0.parquet")[0]
    lead = LEAD[modality]
    df = pl.read_parquet(p, columns=["ts", lead])
    ts = df["ts"].to_numpy().astype("int64")
    x = df[lead].to_numpy().astype(float)
    fs = float(1e6 / np.median(np.diff(ts)))

    R, xf = detect_qrs(x, fs)
    leadoff_rows = contact_mask(ts, xf, fs)
    smask = np.zeros(len(ts), bool)
    for s, e in leadoff_rows:
        smask |= (ts >= s) & (ts <= e)
    if RAIL[modality] is not None:
        rail = binary_dilation(np.abs(x) >= RAIL[modality], structure=np.ones(int(0.1 * fs)))
        smask |= rail
    R = R[~smask[R]]
    tmid, hr = clean_hr(ts[R])

    span = (ts[-1] - ts[0]) / 1e6
    leadoff_s = sum((e - s) for s, e in leadoff_rows) / 1e6
    outdir = os.path.join(HERE, "ecg_hr"); os.makedirs(outdir, exist_ok=True)
    pl.DataFrame({"ts": tmid.astype("int64"), "bpm": hr}).write_parquet(
        os.path.join(outdir, f"{dog}_{modality}.parquet"))
    json.dump({"dog": dog, "modality": modality, "lead": lead, "fs": fs,
               "n_beats": int(len(hr)), "span_s": span,
               "hr_median": float(np.median(hr)) if len(hr) else None,
               "hr_p10_p90": [float(np.percentile(hr, 10)), float(np.percentile(hr, 90))] if len(hr) else None,
               "leadoff_excluded_s": leadoff_s, "leadoff_frac": leadoff_s / span if span else 0,
               "leadoff_intervals_us": leadoff_rows},
              open(os.path.join(outdir, f"{dog}_{modality}.json"), "w"), indent=2)

    # --- QC overlay: 12 s mid-record with detected beats ---
    c0 = len(ts) // 2; w = int(12 * fs)
    sl = slice(max(0, c0 - w // 2), c0 + w // 2)
    tt = (ts[sl] - ts[sl][0]) / 1e6
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(tt, x[sl], lw=0.6, color="0.3")
    Rin = R[(R >= sl.start) & (R < sl.stop)]
    ax.plot((ts[Rin] - ts[sl][0]) / 1e6, x[Rin], "rv", ms=6)
    ax.set_title(f"{dog} {modality} lead={lead} — QC (detected R-peaks) · HR median "
                 f"{np.median(hr):.0f} bpm · n={len(hr):,}")
    ax.set_xlabel("s"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, f"qc_rpeak_{dog}_{modality}.png"), dpi=130)
    print(f"[{dog}/{modality}] lead={lead} fs={fs:.0f}Hz beats={len(hr):,} "
          f"HR median {np.median(hr):.0f} (p10-90 {np.percentile(hr,10):.0f}-{np.percentile(hr,90):.0f}) bpm  "
          f"lead-off {leadoff_s/60:.1f}min ({100*leadoff_s/span:.0f}%)")


if __name__ == "__main__":
    main()

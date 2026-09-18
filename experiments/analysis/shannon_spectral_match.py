#!/usr/bin/env python3
"""Spectral-matching validator: for each single-peak SCG detection, find a
companion burst 140-280ms away in the envelope (as before), then check
whether the two bursts have SIMILAR DOMINANT FREQUENCY in the raw bandpassed
signal (not just similar amplitude/timing). Real S1/S2 should ring at
comparable frequency; a spurious extra detection likely won't."""
import pickle, numpy as np, sqlite3
from scipy import signal as sg

with open("/tmp/shannon_restricted_results_thr0.3.pkl", "rb") as f:
    d = pickle.load(f)
ts_sd, sharp_s, fsd_s = d["ts_sd"], d["sharp_s"], d["fsd_s"]
ts_s, x_s, xf_s, fs_s = d["ts_s"], d["x_s"], d["xf_s"], d["fs_s"]

THR, REFRACT = 0.3, 0.5
cand, _ = sg.find_peaks(sharp_s, height=THR, distance=max(1, int(REFRACT * fsd_s)))
singles = ts_sd[cand]

HALF_WIN_ENV = int(0.32 * fsd_s)     # envelope search window for companion
DIP_FRAC = 0.35
COMP_LO, COMP_HI = 0.14, 0.28        # companion timing window (s)

BURST_HALF_MS = 45.0                 # raw-signal window used to estimate each burst's dominant freq
FREQ_LO, FREQ_HI = 10.0, 100.0       # search band for dominant frequency

def dominant_freq(t_center_us):
    """FFT-based dominant frequency (Hz) of the bandpassed signal in a small
    window around t_center_us, restricted to the 10-100Hz band."""
    lo = np.searchsorted(ts_s, t_center_us - int(BURST_HALF_MS * 1000))
    hi = np.searchsorted(ts_s, t_center_us + int(BURST_HALF_MS * 1000))
    if hi - lo < 8:
        return np.nan
    seg = xf_s[lo:hi] * np.hanning(hi - lo)
    n = 1 << int(np.ceil(np.log2(4 * len(seg))))
    F = np.abs(np.fft.rfft(seg, n))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_s)
    band = (freqs >= FREQ_LO) & (freqs <= FREQ_HI)
    if not band.any() or F[band].max() <= 0:
        return np.nan
    return freqs[band][np.argmax(F[band])]

results = []  # (candidate_time, has_companion, companion_time, f_main, f_comp, freq_ratio, validated)
for c in cand:
    tc = ts_sd[c]
    lo, hi = max(0, c - HALF_WIN_ENV), min(len(sharp_s), c + HALF_WIN_ENV + 1)
    seg = sharp_s[lo:hi]; seg_t = ts_sd[lo:hi]
    c_local = c - lo
    pk, _ = sg.find_peaks(seg, prominence=0.05 * max(seg.max(), 1e-6))
    dt = (seg_t[pk].astype("int64") - tc) / 1e6
    comp_idx = pk[(np.abs(dt) >= COMP_LO) & (np.abs(dt) <= COMP_HI)]

    best = None
    for ci in comp_idx:
        a, b = sorted([c_local, ci])
        if b > a:
            dip = seg[a:b + 1].min(); smaller = min(seg[a], seg[b])
            if smaller > 0 and dip <= DIP_FRAC * smaller:
                best = ci; break
    if best is None:
        results.append((tc, False, None, np.nan, np.nan, np.nan, None))
        continue
    tcomp = seg_t[best]
    f_main = dominant_freq(tc)
    f_comp = dominant_freq(tcomp)
    ratio = np.nan
    if np.isfinite(f_main) and np.isfinite(f_comp) and f_main > 0 and f_comp > 0:
        ratio = max(f_main, f_comp) / min(f_main, f_comp)
    results.append((tc, True, tcomp, f_main, f_comp, ratio, None))

with open("/tmp/spectral_match_results.pkl", "wb") as f:
    pickle.dump(dict(results=results, singles=singles), f)
print(f"processed {len(results)} candidates")
n_with_comp = sum(1 for r in results if r[1])
print(f"  {n_with_comp} had a structurally-valid companion (amplitude dip test)")
ratios = np.array([r[5] for r in results if r[1] and np.isfinite(r[5])])
print(f"  {len(ratios)} of those had valid frequency estimates for both bursts")
print(f"  freq ratio (larger/smaller) stats: median={np.median(ratios):.2f} p25={np.percentile(ratios,25):.2f} p75={np.percentile(ratios,75):.2f} p90={np.percentile(ratios,90):.2f}")

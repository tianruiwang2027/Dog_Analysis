"""
Signal-processing utilities reused from PIPELINE_SUMMARY.md (Sections 0-3), unchanged.

Kept as a separate module so the rest of the teacher pipeline stays consistent with the
already-validated SCG/ECG pipeline instead of re-deriving equivalent-but-different code.
Only additions here are thin wrappers needed to expose intermediate arrays (env_sd,
ts_sd, primary candidate indices, R-peak timestamps) that the original scripts computed
inline but the teacher pipeline needs to consume directly.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sg
from scipy.ndimage import gaussian_filter1d, uniform_filter1d, percentile_filter, median_filter

FSD = 200.0  # decimated envelope sample rate (Section 1)

# ---------------------------------------------------------------------------
# Section 1, Stage 1 -- Shannon-energy envelope + local adaptive sharpening
# ---------------------------------------------------------------------------


def bandpass(x, fs, lo, hi):
    hi = min(hi, 0.45 * fs)
    b = sg.butter(2, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return sg.filtfilt(*b, x - np.mean(x))


def shannon_envelope_decimated(xf, ts, fs, avg_win_s=0.02, gauss_sigma_s=0.01, ref_max=None):
    """ref_max=None self-normalizes on this window's own max|xf| (device-scale
    invariance for a new, independent window); pass a fixed ref_max only to stay
    consistent with a previously-calibrated window of the SAME recording."""
    denom = ref_max if ref_max is not None else np.max(np.abs(xf))
    xn = xf / (denom + 1e-12)
    se = -(xn**2) * np.log(xn**2 + 1e-9)
    se_avg = uniform_filter1d(se, max(1, int(avg_win_s * fs)))
    step = max(1, int(round(fs / FSD)))
    se_d = se_avg[::step]
    ts_d = ts[::step]
    fsd = fs / step
    se_smooth = gaussian_filter1d(se_d, sigma=max(1, gauss_sigma_s * fsd))
    return ts_d, se_smooth, fsd


def sharpen_local(env, fsd, q995_win_s=8.0, q=99.5):
    win = max(3, int(q995_win_s * fsd)) | 1
    local_q995 = percentile_filter(env, q, size=win, mode="nearest")
    return np.exp(env / np.maximum(local_q995, 1e-9)) - 1.0, local_q995


PRIMARY_THR = 0.3
PRIMARY_REFRACT_S = 0.50


def primary_candidates(sharp, ts_sd, fsd, thr=PRIMARY_THR, refract_s=PRIMARY_REFRACT_S):
    """Returns (candidate indices into sharp/ts_sd, candidate timestamps)."""
    pk_idx, _ = sg.find_peaks(sharp, height=thr, distance=max(1, int(refract_s * fsd)))
    return pk_idx, ts_sd[pk_idx]


def scg_pipeline_stage1(x_raw, ts_raw, fs, ref_max=None, refract_s=PRIMARY_REFRACT_S,
                         bp_lo=10.0, bp_hi=100.0):
    """Convenience wrapper: raw SCG channel -> (ts_sd, env_sd, sharp, cand_idx, cand_t).

    refract_s defaults to the dog-tuned 0.5s (>=120bpm ceiling), which is too coarse for
    domains with meaningfully different heart rates (e.g. a resting human at ~60bpm, or
    an epicardial dog under a fast intervention) -- callers pooling multiple domains
    (teacher/multi_domain_data.py) should pass a value scaled to that recording's own
    ECG-derived median RR instead of relying on this default."""
    xf = bandpass(x_raw, fs, bp_lo, bp_hi)
    ts_sd, env_sd, fsd = shannon_envelope_decimated(xf, ts_raw, fs, ref_max=ref_max)
    sharp, _ = sharpen_local(env_sd, fsd)
    cand_idx, cand_t = primary_candidates(sharp, ts_sd, fsd, refract_s=refract_s)
    return ts_sd, env_sd, sharp, cand_idx, cand_t, fsd


def snippet_at(env_sd, idx, half_n=80):
    """Extract the HALF_N=80 window SCGNet/S2Net/TeacherMatcher all consume, zero-padded
    at recording edges."""
    lo, hi = idx - half_n, idx + half_n + 1
    if lo < 0 or hi > len(env_sd):
        pad_lo, pad_hi = max(0, -lo), max(0, hi - len(env_sd))
        seg = np.pad(env_sd, (pad_lo, pad_hi))
        lo, hi = lo + pad_lo, hi + pad_lo
        return seg[lo:hi]
    return env_sd[lo:hi]


# ---------------------------------------------------------------------------
# Section 2 -- fully-tuned Pan-Tompkins ECG detector
# ---------------------------------------------------------------------------

FC_LO, FC_HI = 8.0, 45.0
REFRACT_S = 0.40
MWI_S = 0.010
FLOOR_S = 2.0
SNR_THR = 100.0


def detect_qrs(x, fs):
    b = sg.butter(2, [FC_LO / (fs / 2), FC_HI / (fs / 2)], btype="band")
    xf = sg.filtfilt(*b, x - np.mean(x))
    deriv = np.convolve(xf, np.array([1, 2, 0, -2, -1]) * (fs / 8.0), mode="same")
    sqd = deriv**2
    mwi = uniform_filter1d(sqd, max(1, int(MWI_S * fs)))
    floor = median_filter(mwi, max(3, int(FLOOR_S * fs)) | 1, mode="nearest") + 1e-12
    pk, _ = sg.find_peaks(mwi, distance=int(fs * REFRACT_S), height=None)
    snr = mwi[pk] / floor[pk]
    pk = pk[snr > SNR_THR]
    win = int(0.06 * fs)
    R = []
    for p_ in pk:
        a, bnd = max(0, p_ - win), min(len(xf), p_ + win)
        R.append(a + int(np.argmax(np.abs(xf[a:bnd]))))
    # .astype(int64) matters even (especially) when R is empty: np.unique([]) on an
    # empty python list defaults to float64, and indexing a timestamp array with a
    # float array raises rather than just returning zero beats -- a real case here
    # (e.g. the epicardial set's LBBB-intervention recordings, whose distorted QRS
    # morphology this SNR-thresholded detector doesn't find matches for at all).
    return np.unique(R).astype(np.int64)


# ---------------------------------------------------------------------------
# Section 3 -- shared comparison utilities
# ---------------------------------------------------------------------------


def beat_hr(peaks):
    rr = np.diff(peaks) / 1e6
    hr = 60.0 / rr
    tmid = peaks[:-1] + np.diff(peaks) // 2
    return tmid, hr, rr


def gap_aware_hr(merged_beats, max_rr_s=1.5, gap_thresh_s=1.5, win_s=3.0):
    tmid, hr_raw, rr = beat_hr(merged_beats)
    ok = rr <= max_rr_s
    tmid_ok, hr_ok = tmid[ok], hr_raw[ok]

    win_us, gap_us = int(win_s * 1e6), int(gap_thresh_s * 1e6)
    gap_len = np.diff(merged_beats)
    has_gap = gap_len >= gap_us
    gap_windows = list(zip(merged_beats[:-1][has_gap], merged_beats[1:][has_gap]))

    def touches_gap(t):
        lo, hi = t - win_us, t + win_us
        return any(ge >= lo and gs <= hi for gs, ge in gap_windows)

    hr_sm = np.full(len(tmid_ok), np.nan)
    for i, t in enumerate(tmid_ok):
        if touches_gap(t):
            continue
        sel = (tmid_ok >= t - win_us) & (tmid_ok <= t + win_us)
        hr_sm[i] = hr_ok[sel].mean()
    keep = ~np.isnan(hr_sm)
    return tmid_ok[keep], hr_sm[keep]


def align(tsA, hrA, vA, tsB, hrB, vB, win_us=2_000_000, lag_us=0, restrict=None):
    """For each valid sample of series A, average nearby (+-win_us) valid samples of B."""
    bts, bhr = tsB[vB], hrB[vB]
    xa, yb, ta = [], [], []
    for i in np.where(vA)[0]:
        traw = tsA[i]
        if restrict is not None and not (restrict[0] <= traw <= restrict[1]):
            continue
        t = traw + lag_us
        sel = (bts >= t - win_us) & (bts <= t + win_us)
        if sel.sum() >= 1:
            xa.append(hrA[i])
            yb.append(bhr[sel].mean())
            ta.append(traw)
    return np.array(xa), np.array(yb), np.array(ta, dtype="int64")


def stats(a, b):
    d = a - b
    return dict(
        n=int(len(a)),
        bias=float(d.mean()) if len(a) else float("nan"),
        mae=float(np.abs(d).mean()) if len(a) else float("nan"),
        r=float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan"),
    )

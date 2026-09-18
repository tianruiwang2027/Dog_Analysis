"""Fully-tuned Pan-Tompkins QRS/R-peak detector.

Validated on Chelten (hand-click ground truth, ecg_polar "c1"): r=0.9820,
n=2495, MAE=1.82bpm. Hyperparameters were found via a staged grid search
(~8000 combos across 3 refinement passes) and picked from the middle of a
broad r~0.98 plateau rather than the single top grid cell -- with only one
hand-click set to both tune and evaluate against, chasing the literal best
cell risks fitting noise rather than finding a real optimum. See
docs/PIPELINE_SUMMARY.md Section 2 for the full tuning history.

Generalized well cross-dog to Dasty without retuning (median HR ~123bpm,
matching the pre-existing CORAL-ECG benchmark's 125.9bpm almost exactly) --
see docs/PIPELINE_SUMMARY.md Section 6.
"""
import numpy as np
from scipy import signal as sg
from scipy.ndimage import median_filter, uniform_filter1d

from . import constants as C


def detect_qrs(x, fs,
                fc_lo=C.PT_FC_LO, fc_hi=C.PT_FC_HI, refract_s=C.PT_REFRACT_S,
                mwi_s=C.PT_MWI_S, floor_s=C.PT_FLOOR_S, snr_thr=C.PT_SNR_THR):
    """x: raw ECG samples (1D array). fs: sample rate (Hz).
    Returns sorted, unique sample indices of detected R-peaks."""
    b = sg.butter(2, [fc_lo / (fs / 2), fc_hi / (fs / 2)], btype="band")
    xf = sg.filtfilt(*b, x - np.mean(x))
    deriv = np.convolve(xf, np.array([1, 2, 0, -2, -1]) * (fs / 8.0), mode="same")
    sqd = deriv ** 2
    mwi = uniform_filter1d(sqd, max(1, int(mwi_s * fs)))
    floor = median_filter(mwi, max(3, int(floor_s * fs)) | 1, mode="nearest") + 1e-12
    pk, _ = sg.find_peaks(mwi, distance=int(fs * refract_s), height=None)
    snr = mwi[pk] / floor[pk]
    pk = pk[snr > snr_thr]
    win = int(0.06 * fs)
    R = []
    for p_ in pk:
        a, bnd = max(0, p_ - win), min(len(xf), p_ + win)
        R.append(a + int(np.argmax(np.abs(xf[a:bnd]))))
    return np.unique(R)

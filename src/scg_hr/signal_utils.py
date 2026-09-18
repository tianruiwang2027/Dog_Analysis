"""Stage-1 signal processing: bandpass -> Shannon-energy envelope -> local sharpening.

No learned parameters live here; everything in this module is deterministic DSP.
"""
import numpy as np
from scipy import signal as sg
from scipy.ndimage import gaussian_filter1d, uniform_filter1d, percentile_filter

from .constants import FSD


def bandpass(x, fs, lo, hi):
    """2nd-order Butterworth bandpass, zero-phase (filtfilt), mean-removed input."""
    hi = min(hi, 0.45 * fs)
    b = sg.butter(2, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return sg.filtfilt(*b, x - np.mean(x))


def shannon_envelope_decimated(xf, ts, fs, avg_win_s=0.02, gauss_sigma_s=0.01, ref_max=None):
    """Shannon-energy envelope of a bandpassed SCG signal, decimated to FSD Hz.

    ref_max=None -> self-normalize on this window's own max|xf| (the pipeline's
    designed-in device-scale invariance). Use this for any NEW, independent
    recording/window (e.g. a different dog, or the first time you process a
    given span).

    ref_max=<value> -> use a FIXED external scale instead of self-normalizing.
    Only needed to keep amplitude-scale consistency with an already-CNN-
    calibrated window when processing an ADJACENT segment of the *same*
    recording (see docs/PIPELINE_SUMMARY.md Section 5, "global-normalization
    contamination" bug) -- otherwise a big motion/handling spike elsewhere in a
    long recording can silently rescale the envelope the CNN sees.
    """
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
    """Ratio of env to its own local high-percentile background.

    This is what makes candidate DETECTION (as opposed to CNN amplitude
    scoring) robust to a uniform global rescale of the envelope.
    """
    win = max(3, int(q995_win_s * fsd)) | 1
    local_q995 = percentile_filter(env, q, size=win, mode="nearest")
    return np.exp(env / np.maximum(local_q995, 1e-9)) - 1.0, local_q995

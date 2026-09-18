"""
Automatic per-dog ECG<->SCG clock-offset estimation (DESIGN_PROPOSAL.md section 4,
step 3). PIPELINE_SUMMARY.md's `LAG_US=7.0s` for Chelten was measured once via hand-click
cross-correlation and explicitly does not transfer to other dogs/rigs. For a new,
unlabeled dog there are no hand clicks to cross-correlate, so this estimates the offset
from something that needs no labels at all: both the ECG-derived instantaneous-HR curve
and the SCG primary-candidate density curve should show the same physiological
HR-modulation pattern over time, even before any per-beat labels exist.

Convention (matches teacher/synth_data.py's generator): scg_time ~= ecg_time + lag_us,
i.e. estimate the lag_us that best aligns a coarse "activity curve" from the SCG
candidates (shifted by -lag_us) with the same curve from the ECG beats.
"""
from __future__ import annotations

import numpy as np


def _binned_rate(event_times_us: np.ndarray, t0_us: float, t1_us: float, bin_s: float = 2.0) -> np.ndarray:
    bin_us = bin_s * 1e6
    n_bins = max(1, int((t1_us - t0_us) / bin_us))
    edges = t0_us + np.arange(n_bins + 1) * bin_us
    counts, _ = np.histogram(event_times_us, bins=edges)
    return counts.astype(np.float64)


def estimate_lag_us(
    ecg_beat_times_us: np.ndarray,
    scg_candidate_times_us: np.ndarray,
    search_lo_us: float = -15_000_000.0,
    search_hi_us: float = 15_000_000.0,
    step_us: float = 50_000.0,
    bin_s: float = 2.0,
) -> tuple[float, float]:
    """Returns (best_lag_us, best_correlation). Coarse grid search -- fast enough for a
    one-time offline per-recording estimate, and robust to the exact search resolution
    mattering less than getting into the right multi-second ballpark (auto-labeling
    itself then only needs beat-level, not perfect, alignment -- see DESIGN_PROPOSAL.md
    section 4 step 4's physiologically-plausible search window)."""
    t0 = min(ecg_beat_times_us.min(), scg_candidate_times_us.min() - search_hi_us)
    t1 = max(ecg_beat_times_us.max(), scg_candidate_times_us.max() - search_lo_us)

    ecg_rate = _binned_rate(ecg_beat_times_us, t0, t1, bin_s)
    ecg_rate = ecg_rate - ecg_rate.mean()

    best_lag, best_corr = 0.0, -np.inf
    for lag in np.arange(search_lo_us, search_hi_us + step_us, step_us):
        shifted = scg_candidate_times_us - lag  # bring SCG candidates into ECG clock
        scg_rate = _binned_rate(shifted, t0, t1, bin_s)
        scg_rate = scg_rate - scg_rate.mean()
        denom = np.linalg.norm(ecg_rate) * np.linalg.norm(scg_rate)
        corr = float(ecg_rate @ scg_rate / denom) if denom > 1e-9 else -np.inf
        if corr > best_corr:
            best_lag, best_corr = float(lag), corr
    return best_lag, best_corr

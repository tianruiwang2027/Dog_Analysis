"""Shared utilities for comparing two HR time series (e.g. SCG pipeline vs
Pan-Tompkins ECG, or either vs CORAL's own output)."""
import numpy as np

from . import constants as C


def smooth_plain(tmid, hr, win_s=3.0):
    win_us = int(win_s * 1e6)
    out = np.empty(len(tmid))
    for i, t in enumerate(tmid):
        sel = (tmid >= t - win_us) & (tmid <= t + win_us)
        out[i] = hr[sel].mean()
    return out


def stats(a, b):
    d = a - b
    return dict(n=int(len(a)), bias=float(d.mean()), mae=float(np.abs(d).mean()),
                r=float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan"))


def align(tsA, hrA, vA, tsB, hrB, vB, win_us=2_000_000, lag_us=0, restrict=None, t_ref=None):
    """For each valid sample of series A, average nearby (+/- win_us) valid
    samples of series B. lag_us shifts A's timestamps into B's clock frame
    before matching -- pass either a fixed number (constants.LAG_US) or a
    callable lag_us(t_elapsed_s) -> lag_us for a time-varying correction (see
    lag_at() below / constants.LAG_US_INTERCEPT+LAG_DRIFT_US_PER_S for
    Chelten's measured drift). t_ref is the epoch-us origin for t_elapsed_s
    when lag_us is callable; defaults to tsA's own first (valid) sample."""
    bts, bhr = tsB[vB], hrB[vB]
    if callable(lag_us) and t_ref is None:
        t_ref = int(tsA[vA][0]) if vA.any() else int(tsA[0])
    xa, yb, ta = [], [], []
    for i in np.where(vA)[0]:
        traw = tsA[i]
        if restrict is not None and not (restrict[0] <= traw <= restrict[1]):
            continue
        this_lag = lag_us((traw - t_ref) / 1e6) if callable(lag_us) else lag_us
        t = traw + this_lag
        sel = (bts >= t - win_us) & (bts <= t + win_us)
        if sel.sum() >= 1:
            xa.append(hrA[i])
            yb.append(bhr[sel].mean())
            ta.append(traw)
    return np.array(xa), np.array(yb), np.array(ta, dtype="int64")


def lag_at(t_elapsed_s, intercept_us=C.LAG_US_INTERCEPT, drift_us_per_s=C.LAG_DRIFT_US_PER_S):
    """Chelten-specific time-varying lag correction (see docs/PIPELINE_SUMMARY.md
    Section 10). Use instead of the fixed LAG_US for sessions spanning more than
    ~30 minutes; within the original hand-click window the fixed constant is fine."""
    return intercept_us + drift_us_per_s * t_elapsed_s


# ---------------------------------------------------------------------------
# CORAL quality-gating (reused as-is when comparing against a CORAL .csv output)
# valid = sqi >= CORAL_SQI_THR, minus (in-band 95-125bpm AND sqi < CORAL_BAND_THR),
# minus frozen-bpm runs (>=8 identical consecutive hops), minus any leadoff-mask
# interval from mask.parquet. Implement per your own CORAL output schema; the
# thresholds are the validated part, reproduced here for reference:
CORAL_SQI_THR = C.CORAL_SQI_THR
CORAL_ATTRACTOR_BAND = C.CORAL_ATTRACTOR_BAND
CORAL_BAND_THR = C.CORAL_BAND_THR

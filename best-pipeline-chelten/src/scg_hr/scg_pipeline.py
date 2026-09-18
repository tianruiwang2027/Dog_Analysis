"""High-level SCG heartbeat-detection pipeline: Stage 1 (candidates, via
signal_utils) -> Stage 2 (main CNN confirm) -> Stage 3 (S2-recovery chain) ->
Stage 4 (gap-aware HR reporting).

Validated result on Chelten (hand-click ground truth, 17:26:30-18:00:09 UTC):
r=0.9095, n=1285, MAE=2.20bpm. See docs/PIPELINE_SUMMARY.md Section 1 for the
full narrative and Section 6 for the (currently poor) cross-dog generalization
result on Dasty -- this pipeline's CNN/S2Net were trained on Chelten only and do
NOT currently transfer well to other dogs.
"""
from pathlib import Path

import numpy as np
import torch
from scipy import signal as sg

from . import constants as C
from .models import SCGNet, S2Net


# ---------------------------------------------------------------------------
# Stage 1 (candidate generation) lives in signal_utils.py (bandpass,
# shannon_envelope_decimated, sharpen_local). This module picks up from the
# resulting `sharp_s` array.

def primary_candidates(sharp_s, ts_sd, fsd_s,
                        thr=C.PRIMARY_THR, refract_s=C.PRIMARY_REFRACT_S):
    """Peak-pick the sharpened envelope. refract_s caps detectable HR at
    60/refract_s bpm -- loosen for dogs with faster resting HR than ~120bpm
    (see docs/PIPELINE_SUMMARY.md Section 6, Dasty refractory-0.3s retest)."""
    pk_idx, _ = sg.find_peaks(sharp_s, height=thr, distance=max(1, int(refract_s * fsd_s)))
    return ts_sd[pk_idx]


# ---------------------------------------------------------------------------
# Stages 2/3: main CNN confirm + S2-recovery chain

def load_scg_models(weights_dir):
    """Load the frozen SCGNet + S2Net weights and their training-time scale
    constants. weights_dir should contain cnn_model_relabel.pt,
    cnn_train_relabel_result.pkl, s2_model.pt, s2_train_result.pkl."""
    import pickle
    weights_dir = Path(weights_dir)

    with open(weights_dir / "cnn_train_relabel_result.pkl", "rb") as f:
        scale_main = pickle.load(f)["scale"]
    net_main = SCGNet()
    net_main.load_state_dict(torch.load(weights_dir / "cnn_model_relabel.pt", map_location="cpu"))
    net_main.eval()

    with open(weights_dir / "s2_train_result.pkl", "rb") as f:
        scale_s2 = pickle.load(f)["scale"]
    net_s2 = S2Net()
    net_s2.load_state_dict(torch.load(weights_dir / "s2_model.pt", map_location="cpu"))
    net_s2.eval()

    return dict(net_main=net_main, scale_main=scale_main, net_s2=net_s2, scale_s2=scale_s2)


def _snippet(t, ts_sd, env_sd, half_n=C.HALF_N):
    i = np.searchsorted(ts_sd, t)
    if i - half_n < 0 or i + half_n + 1 > len(env_sd):
        return None
    return env_sd[i - half_n:i + half_n + 1]


def score_candidates(net, scale, cand_t, ts_sd, env_sd, batch_size=512):
    """Score every candidate time with a trained net; returns an array the same
    length as cand_t (NaN where the snippet fell outside the data range)."""
    scores = np.full(len(cand_t), np.nan)
    batch, idxs = [], []

    def flush():
        if not batch:
            return
        with torch.no_grad():
            x = torch.tensor(np.array(batch), dtype=torch.float32).unsqueeze(1)
            p = torch.sigmoid(net(x)).numpy().ravel()
        scores[idxs] = p
        batch.clear()
        idxs.clear()

    for i, tc in enumerate(cand_t):
        s = _snippet(tc, ts_sd, env_sd)
        if s is None:
            continue
        batch.append(s / scale)
        idxs.append(i)
        if len(batch) >= batch_size:
            flush()
    flush()
    return scores


def _backward_local_max(tc, ts_sd, env_sd,
                         search_lo_us=C.SEARCH_LO_US, search_hi_us=C.SEARCH_HI_US):
    lo, hi = tc - search_hi_us, tc - search_lo_us
    i_lo, i_hi = np.searchsorted(ts_sd, lo), np.searchsorted(ts_sd, hi)
    if i_hi <= i_lo:
        return None
    seg = env_sd[i_lo:i_hi]
    return ts_sd[i_lo + np.argmax(seg)]


def detect_beats(ts_sd, env_sd, sharp_s, fsd_s, models,
                  primary_thr=C.PRIMARY_THR, primary_refract_s=C.PRIMARY_REFRACT_S,
                  cutoff_main=C.CUTOFF_MAIN, s2_cutoff=C.S2_CUTOFF):
    """Run the full Stage 1->2->3 chain. Returns (merged_beats, info_dict).

    merged_beats is a sorted int64 array of confirmed beat timestamps (pre-
    gap-aware -- pass through gap_aware_hr() for Stage 4 honest reporting).
    """
    cand_t = primary_candidates(sharp_s, ts_sd, fsd_s, thr=primary_thr, refract_s=primary_refract_s)

    scores_main = score_candidates(models["net_main"], models["scale_main"], cand_t, ts_sd, env_sd)
    valid = ~np.isnan(scores_main)
    confirmed0 = set(cand_t[valid][scores_main[valid] >= cutoff_main].tolist())
    rejected_t = cand_t[valid][scores_main[valid] < cutoff_main]

    n_flagged, n_accepted, recovered = 0, 0, []
    if len(rejected_t):
        s2_scores = score_candidates(models["net_s2"], models["scale_s2"], rejected_t, ts_sd, env_sd)
        flagged_t = rejected_t[(~np.isnan(s2_scores)) & (s2_scores >= s2_cutoff)]
        n_flagged = len(flagged_t)

        backward_ts = np.array([_backward_local_max(tc, ts_sd, env_sd) for tc in flagged_t])
        has_back = np.array([b is not None for b in backward_ts])
        backward_ts_valid = backward_ts[has_back].astype("int64") if has_back.any() else np.array([], dtype="int64")

        if len(backward_ts_valid):
            p_main_arr = score_candidates(models["net_main"], models["scale_main"], backward_ts_valid, ts_sd, env_sd)
            accept_mask = (~np.isnan(p_main_arr)) & (p_main_arr >= cutoff_main)
            recovered = backward_ts_valid[accept_mask].tolist()
            n_accepted = len(recovered)

    merged = set(confirmed0)
    for rt in recovered:
        nearby = [c for c in merged if abs(c - rt) < C.MERGE_TOL_US]
        if not nearby:
            merged.add(int(rt))
    merged_arr = np.array(sorted(merged), dtype="int64")

    info = dict(n_candidates=len(cand_t), n_confirmed_main=len(confirmed0),
                n_rejected=len(rejected_t), n_s2_flagged=n_flagged, n_s2_accepted=n_accepted)
    return merged_arr, info


# ---------------------------------------------------------------------------
# Stage 4: gap-aware honest HR reporting (shared with the ECG side; also
# re-exported from compare_utils for convenience)

def beat_hr(peaks):
    rr = np.diff(peaks) / 1e6
    hr = 60.0 / rr
    tmid = peaks[:-1] + np.diff(peaks) // 2
    return tmid, hr, rr


def gap_aware_hr(merged_beats, max_rr_s=1.5, gap_thresh_s=C.GAP_THRESH_S, win_s=C.WIN_S):
    """3s-smoothed HR, with any window that touches a >=gap_thresh_s detection
    gap suppressed (NaN'd) rather than silently smoothed across it."""
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

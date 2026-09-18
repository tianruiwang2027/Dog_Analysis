"""
Weakly-supervised training of TeacherMatcher across a pool of dogs (Option B,
DESIGN_PROPOSAL.md section 3). No hand clicks are used anywhere in this file -- only
the existing Stage-1 SCG candidate generator, the existing Pan-Tompkins ECG detector,
and the automatic clock-offset estimator in teacher/clock_align.py. That is the whole
point: this is what lets a new, never-hand-clicked dog contribute training signal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from . import clock_align
from common import signal_utils
from .models import TeacherMatcher
from .synth_data import SynthRecording

S1_SEARCH_LO_US = -50_000.0   # a genuine S1 should land close to the ECG beat itself
S1_SEARCH_HI_US = 150_000.0   # allow for electromechanical delay


@dataclass
class RecordingFeatures:
    dog: str
    snippets: np.ndarray          # (n_cand, 161) raw env_sd windows
    cand_t_ecg_clock: np.ndarray  # (n_cand,) candidate timestamps, ECG clock, us
    cand_t_native: np.ndarray     # (n_cand,) candidate timestamps, SCG clock, us (for downstream HR calc)
    beat_times: np.ndarray        # (n_beats,) ECG R-peak timestamps, us
    rr_context: np.ndarray        # (n_beats, 4)
    estimated_lag_us: float
    lag_corr: float
    # Full decimated envelope (native SCG clock) -- not used by the original
    # TeacherMatcher/MIL-query pipelines (they only ever consume the pre-cut per-
    # candidate snippets), but teacher/shape_mil.py needs it to mine hard-negative
    # snippets from time points that were never Stage-1 candidates at all.
    ts_sd: np.ndarray = None
    env_sd: np.ndarray = None
    fsd: float = None


def _rr_context(beat_times_us: np.ndarray) -> np.ndarray:
    n = len(beat_times_us)
    rr_prev = np.full(n, np.nan)
    rr_next = np.full(n, np.nan)
    rr_prev[1:] = np.diff(beat_times_us) / 1e6
    rr_next[:-1] = np.diff(beat_times_us) / 1e6
    rr_prev = np.where(np.isnan(rr_prev), rr_next, rr_prev)
    rr_next = np.where(np.isnan(rr_next), rr_prev, rr_next)
    rr_prev = np.nan_to_num(rr_prev, nan=0.7)
    rr_next = np.nan_to_num(rr_next, nan=0.7)
    mean_rr = np.convolve(rr_prev, np.ones(5) / 5, mode="same")
    mean_rr = np.clip(mean_rr, 1e-3, None)
    hr_bpm = 60.0 / mean_rr
    return np.stack([rr_prev, rr_next, mean_rr, hr_bpm / 150.0], axis=1).astype(np.float32)


def extract_features(
    rec: SynthRecording,
    lag_us_override: float | None = None,
    beat_times_override: np.ndarray | None = None,
    adaptive_refract: bool = False,
) -> RecordingFeatures:
    """beat_times_override lets a caller hand in ECG-side beat times it already trusts
    (e.g. a human dataset's own hand-verified R-peak annotations) instead of running
    detect_qrs -- same role as Pan-Tompkins output, just already given to us. Kept
    False by default so existing Chelten/Chuck/Dasty runs stay bit-for-bit comparable
    to previously-reported numbers; multi_domain_data.py pretraining explicitly opts
    into adaptive_refract=True since pooling domains with very different heart rates
    (a resting human at ~60bpm vs a dog under a fast intervention) makes the single
    dog-tuned 0.5s refractory period too coarse for some of them (see
    signal_utils.scg_pipeline_stage1's docstring)."""
    hint = beat_times_override if beat_times_override is not None else getattr(rec, "ecg_beats_hint", None)
    if hint is not None:
        beat_times = hint
    else:
        R_idx = signal_utils.detect_qrs(rec.x_ecg, rec.fs_ecg)
        beat_times = rec.ts_ecg[R_idx]

    refract_s = signal_utils.PRIMARY_REFRACT_S
    if adaptive_refract and len(beat_times) > 3:
        med_rr_s = float(np.median(np.diff(np.sort(beat_times))) / 1e6)
        refract_s = float(np.clip(0.6 * med_rr_s, 0.2, 0.6))

    ts_sd, env_sd, sharp, cand_idx, cand_t, fsd = signal_utils.scg_pipeline_stage1(
        rec.x_scg, rec.ts_scg, rec.fs_scg, refract_s=refract_s
    )

    if lag_us_override is None:
        lag_us, corr = clock_align.estimate_lag_us(beat_times, cand_t)
    else:
        lag_us, corr = lag_us_override, float("nan")

    cand_t_ecg_clock = cand_t - lag_us
    snippets = np.stack([signal_utils.snippet_at(env_sd, i) for i in cand_idx]).astype(np.float32)

    return RecordingFeatures(
        dog=rec.dog,
        snippets=snippets,
        cand_t_ecg_clock=cand_t_ecg_clock,
        cand_t_native=cand_t,
        beat_times=beat_times,
        rr_context=_rr_context(beat_times),
        estimated_lag_us=lag_us,
        lag_corr=corr,
        ts_sd=ts_sd,
        env_sd=env_sd,
        fsd=fsd,
    )


def matching_loss(model: TeacherMatcher, feats: RecordingFeatures, device: str = "cpu") -> torch.Tensor | None:
    if len(feats.snippets) == 0 or len(feats.beat_times) == 0:
        return None
    snippets_t = torch.tensor(feats.snippets, device=device).unsqueeze(1)  # (n_cand,1,161)
    rr_t = torch.tensor(feats.rr_context, device=device)  # (n_beats,4)

    cand_emb = model.embed_candidates(snippets_t)  # (n_cand, d)
    beat_emb = model.embed_beats(rr_t)             # (n_beats, d)
    sims = model.logit_scale * (beat_emb @ cand_emb.t())  # (n_beats, n_cand)

    # Vectorized rather than a per-beat Python loop -- real recordings can have
    # thousands of beats/candidates, where per-beat torch calls dominate runtime.
    delta = feats.cand_t_ecg_clock[None, :] - feats.beat_times[:, None]  # (n_beats, n_cand)
    in_window = (delta >= S1_SEARCH_LO_US) & (delta <= S1_SEARCH_HI_US)
    valid_beats = in_window.any(axis=1)
    if not valid_beats.any():
        return None

    delta_abs_masked = np.where(in_window, np.abs(delta), np.inf)
    target_idx = np.argmin(delta_abs_masked, axis=1)  # (n_beats,) -- nearest-in-window candidate

    mask_t = torch.tensor(in_window, device=device)
    sims_masked = sims.masked_fill(~mask_t, float("-inf"))

    valid_idx = np.where(valid_beats)[0]
    rows = sims_masked[valid_idx]
    target = torch.tensor(target_idx[valid_idx], device=device, dtype=torch.long)
    return F.cross_entropy(rows, target)


def train_teacher_matcher(
    recordings: list[SynthRecording],
    n_epochs: int = 15,
    lr: float = 3e-4,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[TeacherMatcher, list[RecordingFeatures]]:
    model = TeacherMatcher().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    feats_list = [extract_features(r) for r in recordings]
    if verbose:
        for f, r in zip(feats_list, recordings):
            print(f"  [{f.dog}] estimated lag={f.estimated_lag_us/1e6:+.2f}s "
                  f"(true={r.true_lag_us/1e6:+.2f}s, corr={f.lag_corr:.3f}), "
                  f"{len(f.snippets)} candidates, {len(f.beat_times)} ECG beats")

    for epoch in range(n_epochs):
        epoch_losses = []
        for feats in feats_list:
            loss = matching_loss(model, feats, device)
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        if verbose and (epoch == 0 or (epoch + 1) % 5 == 0):
            print(f"  epoch {epoch+1}/{n_epochs}  mean matching loss={np.mean(epoch_losses):.4f}")

    return model, feats_list

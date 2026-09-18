"""
Wires up Option A (`teacher/models.py:JointMaskedAutoencoder`) for real: builds
co-registered SCG+ECG windows across the unlabeled multi-domain pool, pretrains the
masked autoencoder on them (no labels, no hand clicks -- exactly like the npj
Cardiovascular Health paper's approach, applied here to ECG+SCG instead of ECG+PCG),
then exposes the pretrained encoder as a drop-in trunk for teacher/shape_mil.py's
ShapeMILModel, so the exact same weakly-supervised MIL training/grading code can be
reused unmodified with a foundation-model-pretrained trunk instead of a from-scratch one.

Design choices (see docstrings below for the "why"):
  - Window size matches the existing 161-sample candidate-snippet convention exactly
    (HALF_N=80 in teacher/signal_utils.py), via patch_len=23, n_patches_per_channel=7
    (23*7=161) -- NOT the DESIGN_PROPOSAL default (patch_len=40, n_patches=20, a 4s
    window). This keeps pretraining and downstream use on identical window geometry
    (no resizing/pooling mismatch when the pretrained encoder is later fed real
    candidate snippets), at the cost of a shorter context window (~0.8s instead of 4s)
    -- still enough to span a full S1/S2 cardiac cycle, which is the structure that
    actually matters for the downstream shape classifier.
  - Windows are sliding windows across the WHOLE recording (not anchored on Stage-1
    candidates), so pretraining sees every phase of the cardiac cycle, not just where
    the existing peak-picker happened to fire -- true self-supervised use of all the
    unlabeled signal, independent of Stage-1's own recall ceiling.
  - Each window is self-normalized (teacher.models.self_normalize) per channel before
    being fed to the model, consistent with how CandidateEncoder/CandidateTokenEncoder
    already normalize internally elsewhere in this codebase -- JointMaskedAutoencoder's
    forward() does NOT do this itself, so callers must.
  - At downstream (shape-only) use, ECG tokens are replaced entirely by the model's own
    learned mask token (not real ECG data) -- this guarantees no timing/ECG information
    leaks into the shape classifier, preserving shape_mil.py's whole design point, while
    still benefiting from an encoder that was pretrained to relate SCG shape to ECG
    timing/morphology.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from common import signal_utils
from .models import JointMaskedAutoencoder, self_normalize
from common.real_data import RealRecording
from .shape_mil import InstanceScorer, mil_shape_loss, mine_hard_negatives
from .mil_matcher import DOMAINS, DOMAIN_TO_ID

PATCH_LEN = 23
N_PATCHES_PER_CHANNEL = 7
WINDOW_LEN = PATCH_LEN * N_PATCHES_PER_CHANNEL  # 161, matches HALF_N=80 snippet convention

_ZERO_LAG_DOMAINS = {"dog_epicardial", "human"}


def co_register(rec: RealRecording, lag_us_override: float | None = None):
    """Returns (ts_sd, env_sd, ecg_seg, fsd): SCG's own decimated envelope/time-grid
    (from signal_utils.scg_pipeline_stage1, same as the rest of the pipeline) plus a
    bandpassed ECG segment resampled onto that SAME time grid (in ECG clock), so
    env_sd[i] and ecg_seg[i] are the co-registered SCG/ECG values at the same instant.
    """
    from . import clock_align

    ts_sd, env_sd, sharp, cand_idx, cand_t, fsd = signal_utils.scg_pipeline_stage1(
        rec.x_scg, rec.ts_scg, rec.fs_scg
    )

    if lag_us_override is not None:
        lag_us = lag_us_override
    else:
        R_idx = signal_utils.detect_qrs(rec.x_ecg, rec.fs_ecg)
        beat_times = rec.ts_ecg[R_idx]
        if len(beat_times) < 3 or len(cand_t) < 3:
            lag_us = 0.0
        else:
            lag_us, _ = clock_align.estimate_lag_us(beat_times, cand_t)

    # bandpass ECG in the QRS-relevant band (same band as detect_qrs), then resample
    # onto ts_sd's grid, brought into ECG's own clock (ts_sd - lag_us), via linear
    # interpolation against the ECG signal's own native timestamps.
    ecg_f = signal_utils.bandpass(rec.x_ecg, rec.fs_ecg, signal_utils.FC_LO, signal_utils.FC_HI)
    ts_sd_ecg_clock = ts_sd - lag_us
    ecg_seg = np.interp(ts_sd_ecg_clock, rec.ts_ecg, ecg_f, left=0.0, right=0.0).astype(np.float32)

    return ts_sd, env_sd.astype(np.float32), ecg_seg, fsd, lag_us


def sliding_windows(env_sd: np.ndarray, ecg_seg: np.ndarray, win_len: int = WINDOW_LEN, stride: int = 40):
    """Returns (scg_windows, ecg_windows), each (N, win_len) float32 arrays."""
    n = len(env_sd)
    starts = np.arange(0, max(0, n - win_len + 1), stride)
    if len(starts) == 0:
        return np.zeros((0, win_len), np.float32), np.zeros((0, win_len), np.float32)
    scg_w = np.stack([env_sd[s:s + win_len] for s in starts])
    ecg_w = np.stack([ecg_seg[s:s + win_len] for s in starts])
    return scg_w.astype(np.float32), ecg_w.astype(np.float32)


def build_pretrain_windows(recordings: list[RealRecording], stride: int = 40):
    """Pools sliding SCG/ECG windows across every recording (each domain's own zero-lag
    convention applied, same as teacher/pretrain_mil.py's `_extract`)."""
    all_scg, all_ecg = [], []
    for rec in recordings:
        lag_override = 0.0 if rec.domain in _ZERO_LAG_DOMAINS else None
        try:
            ts_sd, env_sd, ecg_seg, fsd, lag_us = co_register(rec, lag_us_override=lag_override)
        except Exception as e:  # pragma: no cover -- real-data edge cases
            print(f"  [build_pretrain_windows] skipping {rec.dog}: {e}")
            continue
        scg_w, ecg_w = sliding_windows(env_sd, ecg_seg, stride=stride)
        if len(scg_w) == 0:
            continue
        all_scg.append(scg_w)
        all_ecg.append(ecg_w)
        print(f"  [{rec.domain:16s}] {rec.dog:20s} lag={lag_us/1e6:+.2f}s  {len(scg_w)} windows")
    if not all_scg:
        return np.zeros((0, WINDOW_LEN), np.float32), np.zeros((0, WINDOW_LEN), np.float32)
    return np.concatenate(all_scg), np.concatenate(all_ecg)


def train_joint_mae(
    scg_windows: np.ndarray,
    ecg_windows: np.ndarray,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 4,
    mask_ratio: float = 0.5,
    n_epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: str = "cpu",
    verbose: bool = True,
) -> JointMaskedAutoencoder:
    model = JointMaskedAutoencoder(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        patch_len=PATCH_LEN, n_patches_per_channel=N_PATCHES_PER_CHANNEL, mask_ratio=mask_ratio,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    scg_t = torch.tensor(scg_windows, device=device)
    ecg_t = torch.tensor(ecg_windows, device=device)
    # per-window, per-channel self-normalization (see module docstring)
    scg_t = self_normalize(scg_t)
    ecg_t = self_normalize(ecg_t)

    n = len(scg_t)
    if verbose:
        print(f"  pretraining JointMaskedAutoencoder on {n} windows "
              f"(d_model={d_model}, layers={n_layers}, patch_len={PATCH_LEN}, "
              f"n_patches/channel={N_PATCHES_PER_CHANNEL}, mask_ratio={mask_ratio})")

    for epoch in range(n_epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            loss = model.reconstruction_loss(scg_t[idx], ecg_t[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        if verbose and (epoch == 0 or (epoch + 1) % 5 == 0):
            print(f"    epoch {epoch+1}/{n_epochs}  mean recon MSE={epoch_loss/max(1,n_batches):.4f}")

    return model


class MAEShapeEncoder(nn.Module):
    """Drop-in replacement for CandidateTokenEncoder, backed by a pretrained
    JointMaskedAutoencoder's encoder. ECG token positions are filled ENTIRELY with the
    model's own learned mask token (no real ECG data ever reaches this path) -- shape
    classification stays strictly shape-only, matching teacher/shape_mil.py's design,
    while still transferring what the encoder learned about SCG structure from joint
    pretraining."""

    def __init__(self, mae: JointMaskedAutoencoder, freeze: bool = False):
        super().__init__()
        self.mae = mae
        self.d_model = mae.patch_embed.out_features
        self.n_patches = mae.n_patches_per_channel
        if freeze:
            for p in self.mae.parameters():
                p.requires_grad_(False)

    def forward(self, snippet: torch.Tensor) -> torch.Tensor:
        """snippet: (B, 1, WINDOW_LEN) raw candidate snippet, NOT pre-normalized --
        normalized internally, same calling convention as CandidateTokenEncoder."""
        x = self_normalize(snippet).squeeze(1)  # (B, WINDOW_LEN)
        B = x.shape[0]
        scg_p = x.view(B, self.n_patches, self.mae.patch_len)
        scg_tokens = self.mae.patch_embed(scg_p)  # (B, n_patches, d_model)

        mod0 = self.mae.modality_embed(torch.zeros(self.n_patches, dtype=torch.long, device=x.device))
        scg_tokens = scg_tokens + mod0[None] + self.mae.pos[:, : self.n_patches]

        mod1 = self.mae.modality_embed(torch.ones(self.n_patches, dtype=torch.long, device=x.device))
        ecg_tokens = self.mae.mask_token.expand(B, self.n_patches, -1) + mod1[None] + self.mae.pos[:, self.n_patches:]

        tokens = torch.cat([scg_tokens, ecg_tokens], dim=1)
        encoded = self.mae.encoder(tokens)
        return encoded[:, : self.n_patches].mean(dim=1)  # pool over SCG positions only


class ShapeMILModelMAE(nn.Module):
    """Same interface as teacher.shape_mil.ShapeMILModel (.embed/.score), so
    mil_shape_loss/mine_hard_negatives/score_recording/labels_at_threshold all work
    unmodified -- only cand_enc's trunk differs."""

    def __init__(self, mae: JointMaskedAutoencoder, freeze_encoder: bool = False):
        super().__init__()
        self.cand_enc = MAEShapeEncoder(mae, freeze=freeze_encoder)
        self.d_model = self.cand_enc.d_model
        self.domain_embed = nn.Embedding(len(DOMAINS), self.d_model)
        self.scorer = InstanceScorer(self.d_model)

    def embed(self, snippets_t: torch.Tensor, domain_id: int, chunk_size: int = 512) -> torch.Tensor:
        n = snippets_t.shape[0]
        if n == 0:
            return torch.zeros(0, self.d_model, device=snippets_t.device)
        if n <= chunk_size:
            e = self.cand_enc(snippets_t)
        else:
            e = torch.cat([self.cand_enc(snippets_t[i:i + chunk_size]) for i in range(0, n, chunk_size)], dim=0)
        return e + self.domain_embed(torch.tensor(domain_id, device=e.device))

    def score(self, e: torch.Tensor) -> torch.Tensor:
        return self.scorer(e)


def train_shape_mil_given_model(
    model: nn.Module, recordings, extract_fn, n_epochs: int = 20, lr: float = 3e-4,
    device: str = "cpu", verbose: bool = True,
):
    """Identical training loop to teacher.shape_mil.train_shape_mil, except it takes an
    already-constructed model instead of always building a fresh ShapeMILModel --
    needed so we can plug in ShapeMILModelMAE instead."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    feats_list = [extract_fn(r) for r in recordings]
    neg_list = [mine_hard_negatives(f) for f in feats_list]
    domain_ids = [DOMAIN_TO_ID[r.domain] for r in recordings]

    for epoch in range(n_epochs):
        epoch_losses = []
        for feats, negs, did in zip(feats_list, neg_list, domain_ids):
            loss = mil_shape_loss(model, feats, negs, did, device=device)
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        if verbose and (epoch == 0 or (epoch + 1) % 5 == 0):
            print(f"    epoch {epoch+1}/{n_epochs}  mean loss={np.mean(epoch_losses):.4f}")

    return model, feats_list

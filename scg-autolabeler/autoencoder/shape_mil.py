"""
A genuinely different model from teacher/mil_matcher.py, built in response to a direct
critique of that one: mil_matcher's "MIL" target was defined as "always the nearest
candidate in the ECG-anchored window," so the model never had any training signal that
would reward it for looking at candidate SHAPE rather than just imitating that timing
rule -- which is exactly what it did (its numbers were statistically identical to the
much simpler original TeacherMatcher heuristic).

This module fixes that by removing timing entirely from what the model is allowed to
look at. The classifier below (ShapeMILModel) sees ONLY the raw candidate snippet
waveform (plus which domain it came from) -- never its time offset from any ECG beat --
so it cannot shortcut to "pick whichever one is closest in time." It has to learn what
a real S1/S2 shape actually looks like.

The weak-supervision design, following the user's framing directly:

  - ECG tells us WHEN a heartbeat happened. That's a real, trustworthy fact.
  - It does NOT tell us the SCG channel was clean enough, at that moment, to show a
    recognizable heartbeat shape -- motion artifact, poor skin contact, or a genuinely
    quiet cardiac cycle can all mean no candidate in the window looks like a real S1,
    even though a heartbeat definitely happened. That is what makes the ECG anchor a
    WEAK label rather than a strong one: it tells us a positive probably exists
    somewhere nearby, not which candidate (if any) it is.

  - This is textbook Multiple Instance Learning, not the MIL-flavored approximation in
    mil_matcher.py: each ECG beat defines a BAG (the Stage-1 candidates in its
    physiological window), the bag is weakly labeled positive (a heartbeat happened
    here), and instances within the bag are individually unlabeled -- exactly the
    "which instance" ambiguity MIL formalizes, pooled with a smooth-max (logsumexp),
    the standard differentiable relaxation of the noisy-OR/max MIL assumption
    ("a bag is positive if at least one instance is").

  - The missing ingredient mil_matcher.py never had: real, individually-confident
    NEGATIVE instances. A candidate (or any other local bump in the envelope) that
    sits nowhere near any ECG beat has no physiological reason to be a real S1/S2 --
    it's noise, motion, or an artifact the Stage-1 peak-picker fired on. Training the
    same shape classifier against these (label 0) is what actually forces it to learn
    a decision boundary in SHAPE space instead of collapsing onto a timing shortcut --
    there IS no timing information available to it at all, so shape is the only thing
    left for the loss to push on.

All three domains (dog_noninvasive, dog_epicardial, human) are pooled through the same
domain-conditioned encoder, exactly as in mil_matcher.py. Chelten is never in the
training pool -- see run_shape_mil.py.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal as sg

from common import signal_utils
from .mil_matcher import DOMAIN_TO_ID, DOMAINS, CandidateTokenEncoder
from .pretrain_matcher import RecordingFeatures, S1_SEARCH_HI_US, S1_SEARCH_LO_US

# How far beyond the physiological eligibility window (S1_SEARCH_LO/HI_US) a point in
# time must sit before we trust it as a genuine negative -- a safety margin so we never
# accidentally train against a real S1/S2 that just happened to fall near a window edge.
NEG_GUARD_US = 100_000.0

# A much lower bar than signal_utils.PRIMARY_THR (0.3) used to mine SUPPLEMENTAL "noise
# bump" negatives from quiet spans, for domains/recordings where Stage-1's own official
# candidates rarely fire away from real events (so there isn't much natural negative
# material to learn from otherwise). These are deliberately still peak-like (not flat
# baseline) -- the useful negative is "a bump that isn't a heartbeat", not silence.
NEG_PEAK_THR = 0.03
NEG_PEAK_REFRACT_S = 0.05
MAX_NEG_PER_RECORDING = 200


class InstanceScorer(nn.Module):
    """One scalar logit per candidate: "does this snippet look like a real S1/S2
    shape", independent of any timing information."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.mlp(e).squeeze(-1)


class ShapeMILModel(nn.Module):
    """Deliberately simpler than MILQueryMatcher: no cross-attention query, no
    candidate-set self-attention -- both of those let timing/neighbor information leak
    in. This model only ever sees one candidate snippet at a time (plus which domain it
    came from), by design, so "shape" is the only thing it CAN learn from."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.d_model = d_model
        self.cand_enc = CandidateTokenEncoder(d_model)
        self.domain_embed = nn.Embedding(len(DOMAINS), d_model)
        self.scorer = InstanceScorer(d_model)

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


def mine_hard_negatives(feats: RecordingFeatures, max_neg: int = MAX_NEG_PER_RECORDING) -> np.ndarray:
    """Snippets with no physiological reason to be a real S1/S2: (1) Stage-1's own
    candidates that sit far (with NEG_GUARD_US margin) from every ECG beat, plus
    (2) lower-threshold "noise bump" peaks mined from those same safe/quiet spans, so
    domains where Stage-1 rarely fires away from real events still contribute negative
    material. Recordings with too few/fast beats to have any safe span return zero
    negatives -- fine, they just contribute nothing to the negative side of the loss
    that epoch."""
    beat_times = feats.beat_times
    if len(beat_times) < 2 or feats.env_sd is None:
        return np.zeros((0, 161), dtype=np.float32)

    # (1) existing candidates far from every beat
    delta = feats.cand_t_ecg_clock[None, :] - beat_times[:, None]
    min_abs = np.min(np.abs(delta), axis=0) if delta.size else np.array([])
    safe_cand_mask = min_abs > (S1_SEARCH_HI_US + NEG_GUARD_US)
    neg_from_cand = feats.snippets[safe_cand_mask] if len(feats.snippets) else np.zeros((0, 161), np.float32)

    # (2) supplemental low-threshold peaks from the same safe spans
    bt_sorted = np.sort(beat_times)
    ts_sd_ecg = feats.ts_sd - feats.estimated_lag_us
    pos = np.searchsorted(bt_sorted, ts_sd_ecg)
    pos_c = np.clip(pos, 1, len(bt_sorted) - 1)
    d_right = np.abs(ts_sd_ecg - bt_sorted[pos_c])
    d_left = np.abs(ts_sd_ecg - bt_sorted[pos_c - 1])
    min_d = np.minimum(d_left, d_right)
    safe_zone = min_d > (S1_SEARCH_HI_US + NEG_GUARD_US)

    sharp, _ = signal_utils.sharpen_local(feats.env_sd, feats.fsd)
    sharp_masked = np.where(safe_zone, sharp, -np.inf)
    pk_idx, _ = sg.find_peaks(
        sharp_masked, height=NEG_PEAK_THR, distance=max(1, int(NEG_PEAK_REFRACT_S * feats.fsd))
    )
    extra_neg = (
        np.stack([signal_utils.snippet_at(feats.env_sd, i) for i in pk_idx]).astype(np.float32)
        if len(pk_idx) else np.zeros((0, 161), np.float32)
    )

    negs = np.concatenate([neg_from_cand, extra_neg], axis=0).astype(np.float32)
    if len(negs) > max_neg:
        idx = np.random.default_rng(0).choice(len(negs), size=max_neg, replace=False)
        negs = negs[idx]
    return negs


def mil_shape_loss(
    model: ShapeMILModel, feats: RecordingFeatures, neg_snippets: np.ndarray, domain_id: int,
    beta: float = 6.0, device: str = "cpu",
):
    """Positive side: for every ECG beat with >=1 eligible candidate, at least one of
    them should look real -- smooth-max (logsumexp/beta) pooled bag loss, the
    differentiable relaxation of the MIL noisy-OR assumption. Negative side: mined
    hard negatives (mine_hard_negatives) should score as "not real" individually.
    Averaged with equal weight regardless of how many instances feed each side."""
    if len(feats.snippets) == 0 or len(feats.beat_times) == 0:
        pos_loss = None
    else:
        snippets_t = torch.tensor(feats.snippets, device=device).unsqueeze(1)
        cand_emb = model.embed(snippets_t, domain_id)
        scores = model.score(cand_emb)

        delta = feats.cand_t_ecg_clock[None, :] - feats.beat_times[:, None]
        in_window = (delta >= S1_SEARCH_LO_US) & (delta <= S1_SEARCH_HI_US)
        valid = in_window.any(axis=1)

        pos_loss = None
        if valid.any():
            losses = []
            for i in np.where(valid)[0]:
                elig = np.where(in_window[i])[0]
                bag_scores = scores[elig]
                bag_logit = torch.logsumexp(beta * bag_scores, dim=0) / beta
                losses.append(F.binary_cross_entropy_with_logits(bag_logit, torch.ones((), device=device)))
            pos_loss = torch.stack(losses).mean()

    neg_loss = None
    if neg_snippets is not None and len(neg_snippets) > 0:
        neg_t = torch.tensor(neg_snippets, device=device).unsqueeze(1)
        neg_emb = model.embed(neg_t, domain_id)
        neg_scores = model.score(neg_emb)
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))

    if pos_loss is None and neg_loss is None:
        return None
    if pos_loss is None:
        return neg_loss
    if neg_loss is None:
        return pos_loss
    return 0.5 * pos_loss + 0.5 * neg_loss


def train_shape_mil(
    recordings, extract_fn, n_epochs: int = 20, lr: float = 3e-4, device: str = "cpu", verbose: bool = True,
):
    """extract_fn: caller-supplied RealRecording -> RecordingFeatures (e.g.
    pretrain_mil._extract, which handles zero-lag domains + adaptive refractory)."""
    model = ShapeMILModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    feats_list = [extract_fn(r) for r in recordings]
    neg_list = [mine_hard_negatives(f) for f in feats_list]
    domain_ids = [DOMAIN_TO_ID[r.domain] for r in recordings]

    if verbose:
        for f, negs, r in zip(feats_list, neg_list, recordings):
            n_beats_with_cand = int(
                ((f.cand_t_ecg_clock[None, :] - f.beat_times[:, None] >= S1_SEARCH_LO_US)
                 & (f.cand_t_ecg_clock[None, :] - f.beat_times[:, None] <= S1_SEARCH_HI_US)).any(axis=1).sum()
            ) if len(f.snippets) and len(f.beat_times) else 0
            print(f"  [{r.domain:16s}] {r.dog:20s} {len(f.snippets):4d} candidates, "
                  f"{n_beats_with_cand:4d}/{len(f.beat_times):4d} beats w/ >=1 candidate (pos bags), "
                  f"{len(negs):3d} mined negatives")

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
            print(f"  epoch {epoch+1}/{n_epochs}  mean loss={np.mean(epoch_losses):.4f}")

    return model, feats_list


@torch.no_grad()
def score_recording(model: ShapeMILModel, feats: RecordingFeatures, domain_id: int, device: str = "cpu") -> np.ndarray:
    """Per-candidate P(this looks like a real S1/S2 shape), from shape alone."""
    n_cand = len(feats.snippets)
    if n_cand == 0:
        return np.zeros(0, dtype=np.float32)
    snippets_t = torch.tensor(feats.snippets, device=device).unsqueeze(1)
    emb = model.embed(snippets_t, domain_id)
    return torch.sigmoid(model.score(emb)).cpu().numpy()


def labels_at_threshold(feats: RecordingFeatures, probs: np.ndarray, threshold: float = 0.5):
    """For each ECG beat, among its eligible candidates, pick the highest-scoring one;
    accept it as S1 only if its shape-score clears `threshold`, else REJECT (this is
    the meaningful reject the earlier mil_matcher.py never actually exercised -- here
    it fires whenever even the best candidate in the window doesn't look real)."""
    n_cand = len(feats.snippets)
    n_beats = len(feats.beat_times)
    s1_labels = np.zeros(n_cand, dtype=np.float32)
    n_matched = 0
    n_rejected = 0
    if n_cand and n_beats:
        delta = feats.cand_t_ecg_clock[None, :] - feats.beat_times[:, None]
        in_window = (delta >= S1_SEARCH_LO_US) & (delta <= S1_SEARCH_HI_US)
        for i in range(n_beats):
            elig = np.where(in_window[i])[0]
            if len(elig) == 0:
                continue
            best_local = int(np.argmax(probs[elig]))
            best_idx = elig[best_local]
            if probs[best_idx] >= threshold:
                s1_labels[best_idx] = 1.0
                n_matched += 1
            else:
                n_rejected += 1
    return s1_labels, n_matched, n_rejected

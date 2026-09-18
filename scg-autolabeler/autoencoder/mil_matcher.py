"""
Query-based, reject-aware auto-labeler -- the non-autoencoder alternative discussed
alongside DESIGN_PROPOSAL.md's Option A/B. Three concrete upgrades over
teacher/models.py's TeacherMatcher (Option B), all task-aligned rather than
reconstruction-based, and all trainable on CPU at this project's data scale:

1. Candidates are no longer embedded in isolation: after each SCG candidate snippet is
   encoded independently (CandidateTokenEncoder, architecturally identical to
   CandidateEncoder), a second self-attention pass (CandidateSetContext) lets nearby
   candidates within a recording see each other -- "is this one of three competing
   peaks, or the only one nearby" is now representable, which a purely independent
   embedding could never capture. Chunked (not full-recording) attention, since
   candidate count scales with recording length and full O(n^2) attention over a whole
   85-minute recording (thousands of candidates) is not worth the memory on CPU for
   what would mostly be very-long-range, low-value context anyway.

2. A per-beat query (built from RR context, same inputs as BeatEncoder) cross-attends
   over the candidates in its own physiological search window PLUS a learned "null"
   token, instead of a plain cosine similarity into an independently-pooled embedding.
   This generalizes TeacherMatcher's single dot product into genuine learned
   cross-attention with richer per-candidate context.

3. The null token gives the model an explicit way to abstain -- "nothing in this
   window looks like a real S1" -- rather than being forced to always name some
   candidate. The weak-supervision target is still ultimately anchored on ECG-beat
   timing (never a hand click), but a TIGHT timing tolerance decides whether the
   nearest-in-time candidate is trustworthy enough to be the target, or whether null
   is the safer target -- directly targeting the false-positive/precision problem the
   Chelten grading surfaced (77.8%), which plain nearest-in-time pseudo-labeling has no
   mechanism to address.

A domain embedding (dog_noninvasive / dog_epicardial / human) is added to both sides
of the cross-attention so one shared model can be conditioned on which "shape" of SCG
signal a window came from, letting Chuck/Dasty, the epicardial dogs, and the human CEBS
recordings all contribute to one pretraining pool despite being three genuinely
different signal domains.

Deliberately NOT built as a full textbook Multiple-Instance-Learning pooling formula
(e.g. noisy-OR over an unordered bag) -- the actual ambiguity here is "which instance",
not "does a positive exist in the bag" (that part is already known from the ECG side),
so a query-based selector with an explicit reject class is a more direct fit, described
honestly as an MIL-flavored approximation rather than textbook MIL.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import SimpleTransformerEncoder, self_normalize

DOMAINS = ["dog_noninvasive", "dog_epicardial", "human"]
DOMAIN_TO_ID = {d: i for i, d in enumerate(DOMAINS)}

# First version: the "reject" target is used only for beats with literally zero
# eligible candidates (which never reach assign_targets at all -- see mil_loss's
# `valid` mask), not via a second, tighter timing threshold. An earlier version of
# this file used a much tighter window here (-20ms/+100ms vs the wider S1_SEARCH
# eligibility window of -50ms/+150ms) hoping to teach the model when to abstain even
# among eligible candidates -- in practice, real electromechanical delay on Chelten
# routinely lands past 100ms, so that tighter window assigned "null" as the training
# target for the MAJORITY of real beats, and the model dutifully learned to always
# reject (0 matched / 1677 rejected out of 5117 beats on the real Chelten run).
# Matching the eligibility window exactly reproduces the original TeacherMatcher's
# safe "always pick the nearest eligible candidate" behavior; the null token and
# cross-attention machinery stay in place for a better-calibrated reject rule later
# (e.g. one learned from validation data rather than hand-picked), rather than ripped
# out, since the architecture upgrade (candidates attending to each other, a real
# cross-attention query, domain conditioning) is the part validated as working here.
from .pretrain_matcher import S1_SEARCH_HI_US, S1_SEARCH_LO_US  # noqa: E402

TARGET_TOL_LO_US = S1_SEARCH_LO_US
TARGET_TOL_HI_US = S1_SEARCH_HI_US


class CandidateTokenEncoder(nn.Module):
    """Architecturally identical to models.CandidateEncoder -- conv stem + a couple of
    transformer layers, mean-pooled to one embedding per candidate snippet. Kept as a
    separate class (rather than importing CandidateEncoder directly) so this module
    can evolve independently, but intentionally reuses the same building blocks that
    are already known not to trip this environment's PyTorch CPU attention bug."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, snippet_len: int = 161):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, padding=7),
            nn.GELU(),
            nn.Conv1d(16, d_model, kernel_size=9, padding=4),
            nn.GELU(),
        )
        self.transformer = SimpleTransformerEncoder(d_model, n_heads, n_layers, 4 * d_model, dropout=0.1)
        self.pos = nn.Parameter(torch.randn(1, snippet_len, d_model) * 0.02)

    def forward(self, snippet: torch.Tensor) -> torch.Tensor:
        x = self_normalize(snippet)
        x = self.stem(x).transpose(1, 2)
        x = self.transformer(x + self.pos[:, : x.shape[1]])
        return x.mean(dim=1)


class CandidateSetContext(nn.Module):
    """Lets nearby candidates see each other. Operates on chunks (not the whole
    recording) to bound attention memory -- see module docstring point 1."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, chunk_size: int = 256):
        super().__init__()
        self.chunk_size = chunk_size
        self.time_mlp = nn.Sequential(nn.Linear(1, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, d_model))
        self.transformer = SimpleTransformerEncoder(d_model, n_heads, n_layers, 4 * d_model, dropout=0.1)

    def forward(self, cand_emb: torch.Tensor, cand_t_us: np.ndarray) -> torch.Tensor:
        """cand_emb: (n_cand, d_model). cand_t_us: (n_cand,) candidate timestamps (for
        a lightweight relative-time signal within each chunk). Returns contextualized
        embeddings, same shape as input."""
        n = cand_emb.shape[0]
        if n == 0:
            return cand_emb
        out = torch.empty_like(cand_emb)
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            chunk = cand_emb[start:end].unsqueeze(0)  # (1, L, d_model)
            t_rel = (cand_t_us[start:end] - cand_t_us[start]) / 1e6  # seconds, chunk-local
            t_rel_t = torch.tensor(t_rel, dtype=chunk.dtype, device=chunk.device).unsqueeze(0).unsqueeze(-1)
            chunk = chunk + self.time_mlp(t_rel_t)
            out[start:end] = self.transformer(chunk).squeeze(0)
        return out


class BeatEncoder(nn.Module):
    """Same inputs as models.BeatEncoder (RR context), used here to build the
    per-beat cross-attention query rather than an embedding to be cosine-compared."""

    def __init__(self, d_model: int = 64, n_features: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_features, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, rr_context: torch.Tensor) -> torch.Tensor:
        return self.mlp(rr_context)


class MILQueryMatcher(nn.Module):
    """Ties the pieces together. embed_candidates / contextualize are separated so
    callers can cache the (expensive) per-candidate encoding and re-run only the
    cheap per-beat cross-attention when experimenting."""

    def __init__(self, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.cand_enc = CandidateTokenEncoder(d_model)
        self.set_context = CandidateSetContext(d_model)
        self.beat_enc = BeatEncoder(d_model)
        self.domain_embed = nn.Embedding(len(DOMAINS), d_model)
        self.null_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.logit_scale = nn.Parameter(torch.tensor(4.0))

    def embed_candidates(self, snippets: torch.Tensor, cand_t_us: np.ndarray, domain_id: int,
                          chunk_size: int = 512) -> torch.Tensor:
        if snippets.shape[0] <= chunk_size:
            e = self.cand_enc(snippets)
        else:
            e = torch.cat([self.cand_enc(snippets[i:i + chunk_size]) for i in range(0, snippets.shape[0], chunk_size)], dim=0)
        e = e + self.domain_embed(torch.tensor(domain_id, device=e.device))
        return self.set_context(e, cand_t_us)

    def beat_query(self, rr_context: torch.Tensor, domain_id: int) -> torch.Tensor:
        q = self.beat_enc(rr_context)
        return q + self.domain_embed(torch.tensor(domain_id, device=q.device))

    def beat_logits(self, query: torch.Tensor, cand_emb_window: torch.Tensor) -> torch.Tensor:
        """query: (d_model,). cand_emb_window: (m, d_model) -- the candidates eligible
        for this one beat (already sliced by the caller's window mask). Returns (m+1,)
        logits over [candidates..., null]."""
        keys = torch.cat([cand_emb_window, self.null_token.unsqueeze(0)], dim=0)
        q = self.q_proj(query)
        k = self.k_proj(keys)
        return self.logit_scale * (k @ q) / (self.d_model ** 0.5)


def assign_targets(delta_us: np.ndarray, in_window: np.ndarray) -> np.ndarray:
    """delta_us: (n_beats, n_cand) candidate time minus beat time. in_window: same
    shape, the (wider) eligibility mask used elsewhere (S1_SEARCH_LO/HI_US). Returns,
    per beat with >=1 eligible candidate, the target index into
    [eligible-candidates-in-order, null] -- the last index (== number of eligible
    candidates for that beat) means "target is null" because the nearest-in-time
    candidate fell outside the tighter TARGET_TOL window even though it was inside the
    wider eligibility one."""
    n_beats = delta_us.shape[0]
    targets = []
    valid = []
    for i in range(n_beats):
        elig = np.where(in_window[i])[0]
        if len(elig) == 0:
            valid.append(False)
            targets.append(-1)
            continue
        valid.append(True)
        d = delta_us[i, elig]
        nearest_local = int(np.argmin(np.abs(d)))
        if TARGET_TOL_LO_US <= d[nearest_local] <= TARGET_TOL_HI_US:
            targets.append(nearest_local)  # index within elig
        else:
            targets.append(len(elig))  # null
    return np.array(valid), np.array(targets, dtype=np.int64)

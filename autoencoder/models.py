"""
Teacher model architectures. See DESIGN_PROPOSAL.md sections 3-4 for the reasoning.

Option B (`TeacherMatcher`, functional, used by the demo pipeline): a small transformer
candidate encoder + ECG-beat encoder trained to match true (SCG candidate, ECG beat)
pairs via cosine similarity in a shared embedding space -- a learned, multi-dog
generalization of the fixed-template NCC idea in PIPELINE_SUMMARY.md section 9.4/9.6.

Option A (`JointMaskedAutoencoder`, architecture stub): a joint SCG+ECG masked
autoencoder in the style of the npj Cardiovascular Health PCG/ECG foundation model
(https://www.nature.com/articles/s44325-024-00027-5). Forward pass is functional and
shape-checked, but it is not wired into the demo training loop -- see DESIGN_PROPOSAL.md
section 3 for why Option B is validated first. `CandidateEncoder` is written so its
`d_model`-dim pooled output could later be replaced by this encoder's patch embeddings
with no other code changes.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _SelfAttention(nn.Module):
    """Plain, unfused multi-head self-attention (matmul + softmax), deliberately NOT
    using nn.MultiheadAttention/nn.TransformerEncoderLayer. This environment's CPU
    build of PyTorch (2.14.0) has a reproducible bug where TWO CHAINED
    TransformerEncoderLayers, under grad tracking, blow up to multi-GB memory and get
    OOM-killed on some real (structured, non-random) inputs -- confirmed by isolating
    layer-by-layer with and without torch.no_grad(). A hand-rolled attention block
    avoids whatever fused/fastpath kernel that bug lives in."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, L, hd)
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out(out)


class _TransformerBlock(nn.Module):
    """Pre-norm transformer block: self-attention + feedforward, both residual."""

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _SelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Linear(dim_feedforward, d_model)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class SimpleTransformerEncoder(nn.Module):
    """Stack of `_TransformerBlock`s -- a drop-in, bug-avoiding replacement for
    nn.TransformerEncoder for this project (see `_SelfAttention` docstring)."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [_TransformerBlock(d_model, n_heads, dim_feedforward, dropout) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def self_normalize(snippet, eps=1e-6):
    """Per-snippet scale normalization -- the fix PIPELINE_SUMMARY.md section 9.5
    identifies as needed (SCGNet's fixed `scale_main` is ~32x off for Dasty). Dividing
    by each snippet's own robust scale instead makes the encoder invariant to
    per-dog/per-sensor gain differences by construction, not by luck."""
    scale = snippet.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    return snippet / scale


class CandidateEncoder(nn.Module):
    """Encodes one SCG candidate snippet (env_sd window, HALF_N=80 -> length 161) into a
    d_model embedding. Conv stem for local shape features, then a couple of transformer
    layers so the encoder can relate different parts of the S1/S2 waveform to each
    other -- more representational capacity than SCGNet's flatten+MLP head, which is the
    point: this network is trained across many dogs, not one."""

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
        """snippet: (B, 1, L) raw env_sd window, NOT pre-normalized -- normalization
        happens inside so callers never have to remember the fixed-scale pitfall."""
        x = self_normalize(snippet)
        x = self.stem(x).transpose(1, 2)  # (B, L, d_model)
        x = self.transformer(x + self.pos[:, : x.shape[1]])
        return x.mean(dim=1)  # (B, d_model)


class BeatEncoder(nn.Module):
    """Encodes local ECG-beat context (already dog-general via Pan-Tompkins, section 6)
    into the same embedding space as CandidateEncoder. rr_context is cheap and already
    available from the existing detect_qrs/beat_hr code: no new signal processing."""

    def __init__(self, d_model: int = 64, n_features: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_features, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, rr_context: torch.Tensor) -> torch.Tensor:
        return self.mlp(rr_context)


class TeacherMatcher(nn.Module):
    """Scores an (SCG candidate, ECG beat) pair by cosine similarity of their
    embeddings. Trained with a weakly-supervised contrastive objective (section 3/4):
    within each ECG beat's plausible search window, the candidate closest in time to
    the beat is treated as the positive, the rest as in-batch negatives -- no hand
    clicks required, only the existing ECG detector and candidate generator."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.cand_enc = CandidateEncoder(d_model)
        self.beat_enc = BeatEncoder(d_model)
        self.logit_scale = nn.Parameter(torch.tensor(4.0))

    def embed_candidates(self, snippets: torch.Tensor, chunk_size: int = 512) -> torch.Tensor:
        """Chunked rather than one giant forward pass -- self-attention memory scales
        with batch size here (each candidate attends only within its own 161-sample
        snippet, but all candidates go through the encoder as one batch dimension), and
        a real recording can have several thousand candidates where a synthetic demo
        has a few hundred. Chunking bounds peak memory regardless of recording length."""
        if snippets.shape[0] <= chunk_size:
            e = self.cand_enc(snippets)
        else:
            e = torch.cat([self.cand_enc(snippets[i : i + chunk_size]) for i in range(0, snippets.shape[0], chunk_size)], dim=0)
        return e / e.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def embed_beats(self, rr_context: torch.Tensor) -> torch.Tensor:
        e = self.beat_enc(rr_context)
        return e / e.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def forward(self, snippets: torch.Tensor, rr_context: torch.Tensor) -> torch.Tensor:
        """Returns pairwise similarity matrix (n_beats, n_candidates) scaled by a
        learned temperature, ready for softmax/cross-entropy matching."""
        c = self.embed_candidates(snippets)
        b = self.embed_beats(rr_context)
        return self.logit_scale * (b @ c.t())


# ---------------------------------------------------------------------------
# Option A -- architecture stub, not used by the demo pipeline (see module docstring)
# ---------------------------------------------------------------------------


class JointMaskedAutoencoder(nn.Module):
    """Joint SCG+ECG masked-autoencoder pretraining, in the style of the npj
    Cardiovascular Health PCG/ECG foundation model. Both channels are cut into patches
    over a shared multi-second window, concatenated into one token sequence tagged by
    modality, a fraction of tokens are masked, and a lightweight decoder reconstructs
    them from the visible tokens of BOTH channels -- forcing the encoder to learn the
    SCG<->ECG timing/morphology relationship without any labels at all. Intended as a
    drop-in replacement for CandidateEncoder's trunk once Option B's simpler matcher has
    been validated (DESIGN_PROPOSAL.md section 3); not exercised by demo_end_to_end.py.
    """

    def __init__(self, d_model: int = 96, n_heads: int = 4, n_layers: int = 4,
                 patch_len: int = 40, n_patches_per_channel: int = 20, mask_ratio: float = 0.5):
        super().__init__()
        self.patch_len = patch_len
        self.n_patches_per_channel = n_patches_per_channel
        self.mask_ratio = mask_ratio

        self.patch_embed = nn.Linear(patch_len, d_model)
        self.modality_embed = nn.Embedding(2, d_model)  # 0=SCG, 1=ECG
        n_tokens = 2 * n_patches_per_channel
        self.pos = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.encoder = SimpleTransformerEncoder(d_model, n_heads, n_layers, 4 * d_model, dropout=0.1)
        self.decoder = SimpleTransformerEncoder(d_model, n_heads, 2, 2 * d_model, dropout=0.1)
        self.decoder_pred = nn.Linear(d_model, patch_len)

    def patchify(self, scg_window: torch.Tensor, ecg_window: torch.Tensor) -> torch.Tensor:
        """scg_window/ecg_window: (B, n_patches_per_channel * patch_len) each,
        already-aligned per-channel envelope/segment windows. Returns (B, n_tokens, P)."""
        B = scg_window.shape[0]
        scg_p = scg_window.view(B, self.n_patches_per_channel, self.patch_len)
        ecg_p = ecg_window.view(B, self.n_patches_per_channel, self.patch_len)
        return torch.cat([scg_p, ecg_p], dim=1)

    def forward(self, scg_window: torch.Tensor, ecg_window: torch.Tensor):
        B = scg_window.shape[0]
        patches = self.patchify(scg_window, ecg_window)  # (B, n_tokens, P)
        n_tokens = patches.shape[1]
        tokens = self.patch_embed(patches)

        mod_ids = torch.cat([
            torch.zeros(self.n_patches_per_channel, dtype=torch.long),
            torch.ones(self.n_patches_per_channel, dtype=torch.long),
        ]).to(patches.device)
        tokens = tokens + self.modality_embed(mod_ids)[None] + self.pos

        n_mask = int(n_tokens * self.mask_ratio)
        noise = torch.rand(B, n_tokens, device=patches.device)
        mask = torch.zeros(B, n_tokens, dtype=torch.bool, device=patches.device)
        mask.scatter_(1, noise.argsort(dim=1)[:, :n_mask], True)

        visible = tokens.clone()
        visible[mask] = self.mask_token.expand(B, n_tokens, -1)[mask]

        encoded = self.encoder(visible)
        decoded = self.decoder(encoded)
        recon = self.decoder_pred(decoded)  # (B, n_tokens, patch_len)
        return recon, patches, mask

    def reconstruction_loss(self, scg_window: torch.Tensor, ecg_window: torch.Tensor) -> torch.Tensor:
        recon, patches, mask = self.forward(scg_window, ecg_window)
        err = (recon - patches).pow(2).mean(dim=-1)  # (B, n_tokens)
        return (err * mask).sum() / mask.sum().clamp_min(1)

"""
Retrains SCGNet (or S2Net) on whatever labels it's given -- hand-click-derived, as
today, or teacher-auto-generated, as this proposal adds. The architecture and training
recipe are otherwise unchanged from PIPELINE_SUMMARY.md; the only addition is
self-normalizing each snippet before feeding the net (section 9.5's proposed fix),
which matters more here than it did for Chelten-only training since auto-labels may be
pooled across dogs with very different sensor gains.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .small_nets import SCGNet, count_params


def _self_normalize(snippets: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    scale = np.abs(snippets).max(axis=1, keepdims=True)
    scale = np.maximum(scale, eps)
    return snippets / scale


def train_small_net(
    snippets: np.ndarray,
    labels: np.ndarray,
    val_snippets: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    epochs: int = 40,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[SCGNet, dict]:
    net = SCGNet().to(device)
    if verbose:
        print(f"  SCGNet params: {count_params(net)} (unchanged architecture)")

    x = torch.tensor(_self_normalize(snippets), device=device).unsqueeze(1)
    y = torch.tensor(labels, device=device).float()

    # Auto-labeled (and real hand-click-derived) beat/non-beat data is usually
    # imbalanced -- real beats vastly outnumber surviving Stage-1 negatives, similar to
    # why S2Net's own training set needed a stratified split (PIPELINE_SUMMARY.md
    # section 1 Stage 3). Balanced per-sample weighting keeps the tiny net from just
    # learning to predict the majority class.
    n_pos = max(float((y == 1).sum().item()), 1.0)
    n_neg = max(float((y == 0).sum().item()), 1.0)
    sample_weight = torch.where(y == 1, y.new_full((), 0.5 / n_pos), y.new_full((), 0.5 / n_neg))
    sample_weight = sample_weight * len(y)  # keep the loss on a familiar scale

    opt = torch.optim.Adam(net.parameters(), lr=lr)

    n = len(x)
    history = {"train_loss": []}
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            out = net(x[idx]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(out, y[idx], weight=sample_weight[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        history["train_loss"].append(epoch_loss / n)
        if verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
            print(f"    epoch {epoch+1}/{epochs}  train BCE={history['train_loss'][-1]:.4f}")

    if val_snippets is not None and len(val_snippets):
        net.eval()
        with torch.no_grad():
            xv = torch.tensor(_self_normalize(val_snippets), device=device).unsqueeze(1)
            preds = torch.sigmoid(net(xv).squeeze(-1)).cpu().numpy()
        history["val_preds"] = preds
        history["val_labels"] = val_labels
    return net, history


@torch.no_grad()
def confirm_beats(net: SCGNet, snippets: np.ndarray, cutoff: float = 0.5, device: str = "cpu") -> np.ndarray:
    """Returns a boolean mask of which candidates the trained net confirms."""
    net.eval()
    x = torch.tensor(_self_normalize(snippets), device=device).unsqueeze(1)
    probs = torch.sigmoid(net(x).squeeze(-1)).cpu().numpy()
    return probs >= cutoff

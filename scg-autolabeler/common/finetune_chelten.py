"""
Genuine supervised fine-tuning on real Chelten hand labels -- the actual thing the user
asked for: "teach it the function of auto-labeling the SCG, with 30 minutes worth of
[real, hand-clicked] SCG labeling data."

Unlike everywhere else in this codebase (which deliberately never uses hand clicks for
training, only for final grading), this module DOES train on them -- on purpose, as a
second, complementary protocol to weak supervision: a true few-label fine-tune, the
same recipe the npj Cardiovascular Health paper uses downstream of its foundation-model
pretraining. Chelten's ~37.6-minute hand-annotated window is split in time: the first
~30 minutes gives real positive/negative per-candidate labels to fine-tune on, the
remaining ~7.6 minutes is held out and graded with the exact same metric used
elsewhere (grade_against_clicks-style precision/recall @ 60ms), so this is directly
comparable to the fully-blind weakly-supervised numbers.

Both annotation SQLite files' timestamps are each in their OWN recording's native
clock (SCG clicks vs the MWD_Chelten SCG waveform, ECG clicks vs the ecg_polar_chelten
ECG waveform) -- confirmed by matching real_data.load_chelten_scg_clicks' existing use
of cand_t_native (no lag correction) for grading. So no clock alignment is needed at
all for this module: SCG candidates are matched directly against SCG clicks, and the
ECG hand clicks are used directly as trusted beat times, both in their own native
clocks.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import signal_utils
from .real_data import RealRecording

TOL_US = 60_000.0


def load_click_events(sqlite_path: str) -> np.ndarray:
    """Hand-clicked timestamps (empty-label rows only -- see teacher/real_data.py's
    load_chelten_scg_clicks, which this generalizes to also read the ECG annotation
    file using the identical schema)."""
    con = sqlite3.connect(sqlite_path)
    try:
        cur = con.cursor()
        cur.execute("select timestamp from events where label='' order by timestamp")
        return np.array([r[0] for r in cur.fetchall()], dtype=np.int64)
    finally:
        con.close()


@dataclass
class ChelChentenSplit:
    train_lo: int
    train_hi: int
    test_lo: int
    test_hi: int
    train_scg_clicks: np.ndarray
    test_scg_clicks: np.ndarray
    train_ecg_clicks: np.ndarray
    test_ecg_clicks: np.ndarray


def make_split(scg_clicks: np.ndarray, ecg_clicks: np.ndarray, train_minutes: float = 30.0) -> ChelChentenSplit:
    lo = max(scg_clicks.min(), ecg_clicks.min())
    hi = min(scg_clicks.max(), ecg_clicks.max())
    split = lo + int(train_minutes * 60 * 1e6)
    return ChelChentenSplit(
        train_lo=lo, train_hi=split, test_lo=split, test_hi=hi,
        train_scg_clicks=scg_clicks[(scg_clicks >= lo) & (scg_clicks < split)],
        test_scg_clicks=scg_clicks[(scg_clicks >= split) & (scg_clicks <= hi)],
        train_ecg_clicks=ecg_clicks[(ecg_clicks >= lo) & (ecg_clicks < split)],
        test_ecg_clicks=ecg_clicks[(ecg_clicks >= split) & (ecg_clicks <= hi)],
    )


@dataclass
class LabeledCandidates:
    snippets: np.ndarray      # (n, 161)
    cand_t: np.ndarray        # (n,) SCG-native clock, us
    labels: np.ndarray        # (n,) 1.0 = matched a real hand click within TOL_US


def stage1_candidates_in_window(chelten_rec: RealRecording, t_lo: float, t_hi: float):
    """Runs the unchanged Stage-1 pipeline over the recording, then restricts to
    candidates whose timestamp falls in [t_lo, t_hi)."""
    ts_sd, env_sd, sharp, cand_idx, cand_t, fsd = signal_utils.scg_pipeline_stage1(
        chelten_rec.x_scg, chelten_rec.ts_scg, chelten_rec.fs_scg
    )
    keep = (cand_t >= t_lo) & (cand_t < t_hi)
    cand_idx, cand_t = cand_idx[keep], cand_t[keep]
    snippets = np.stack([signal_utils.snippet_at(env_sd, i) for i in cand_idx]).astype(np.float32) if len(cand_idx) else np.zeros((0, 161), np.float32)
    return snippets, cand_t


def label_candidates(snippets: np.ndarray, cand_t: np.ndarray, clicks: np.ndarray, tol_us: float = TOL_US) -> LabeledCandidates:
    labels = np.zeros(len(cand_t), dtype=np.float32)
    if len(cand_t) and len(clicks):
        d = np.abs(cand_t[:, None] - clicks[None, :])
        nearest = d.min(axis=1)
        labels = (nearest <= tol_us).astype(np.float32)
    return LabeledCandidates(snippets=snippets, cand_t=cand_t, labels=labels)


def supervised_finetune(
    model: nn.Module,  # any model exposing .embed(snippets_t, domain_id) / .score(e)
    train_data: LabeledCandidates,
    domain_id: int,
    n_epochs: int = 60,
    lr: float = 5e-4,
    batch_size: int = 64,
    device: str = "cpu",
    verbose: bool = True,
):
    """Plain balanced-BCE fine-tune on real per-candidate labels -- no MIL/bag ambiguity
    needed here, since these labels are ground truth, not ECG-timing pseudo-labels."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.tensor(train_data.snippets, device=device).unsqueeze(1)
    y = torch.tensor(train_data.labels, device=device)

    n_pos = max(float((y == 1).sum().item()), 1.0)
    n_neg = max(float((y == 0).sum().item()), 1.0)
    weight_per_sample = torch.where(y == 1, y.new_full((), 0.5 / n_pos), y.new_full((), 0.5 / n_neg)) * len(y)

    n = len(x)
    if verbose:
        print(f"  fine-tuning on {n} real-labeled candidates ({int(y.sum())} positive, {int((1-y).sum())} negative)")

    for epoch in range(n_epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            e = model.embed(x[idx], domain_id)
            logits = model.score(e)
            loss = F.binary_cross_entropy_with_logits(logits, y[idx], weight=weight_per_sample[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        if verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
            print(f"    epoch {epoch+1}/{n_epochs}  train BCE={epoch_loss/n:.4f}")

    return model


@torch.no_grad()
def evaluate_on_held_out(
    model: nn.Module, test_data: LabeledCandidates, domain_id: int, threshold: float = 0.5, device: str = "cpu",
) -> dict:
    """Precision/recall at a fixed score threshold on the held-out real labels --
    a genuinely supervised metric (no Stage-1-ceiling framing needed here, since we
    already restricted to Stage-1 candidates on both sides consistently)."""
    if len(test_data.snippets) == 0:
        return dict(precision=float("nan"), recall=float("nan"), f1=float("nan"), n=0)
    x = torch.tensor(test_data.snippets, device=device).unsqueeze(1)
    e = model.embed(x, domain_id)
    probs = torch.sigmoid(model.score(e)).cpu().numpy()
    pred = probs >= threshold
    y = test_data.labels.astype(bool)

    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision and recall) else float("nan")
    return dict(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn, n=len(y), n_pos=int(y.sum()))

"""
Shared U-Net training/evaluation logic, factored out of run_unet.py so the same
leakage-free fine-tune+evaluate procedure can be reused for the from-scratch baseline
AND for pretrained-then-fine-tuned variants (run_unet_pretrain.py) without duplicating
(and risking re-diverging) the same code three times.
"""
from __future__ import annotations

import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import real_data, signal_utils
from common.finetune_chelten import load_click_events, make_split, TOL_US
from .unet1d import UNet1D
from common.config import DATA_DIR, ANNOTATION_SCG_PATH, ANNOTATION_ECG_PATH

WINDOW_SAMPLES = 800     # 4s at FSD=200Hz
STRIDE_SAMPLES = 400     # 2s -- 50% overlap
MIN_PEAK_DISTANCE_S = 0.50   # same PRIMARY_REFRACT_S convention used everywhere else
VAL_MINUTES = 5.0
THRESHOLD_GRID = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
BATCH_SIZE = 32
LR = 1e-3


def build_windows(env: np.ndarray, mask: np.ndarray, window: int, stride: int):
    """Chops a continuous (env, mask) pair into overlapping fixed-length windows.
    Both arrays must be the same length and already restricted to one contiguous,
    fully-labeled time span (e.g. just the training portion)."""
    n = len(env)
    xs, ys = [], []
    for start in range(0, max(1, n - window + 1), stride):
        xs.append(env[start:start + window])
        ys.append(mask[start:start + window])
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def masks_to_peaks(probs: np.ndarray, env: np.ndarray, ts: np.ndarray, fsd: float,
                    threshold: float, min_distance_s: float = MIN_PEAK_DISTANCE_S):
    """Post-processing: threshold the predicted per-sample mask, find each contiguous
    positive run, take the envelope's local max within that run as the exact peak
    sample, then greedily drop any candidate closer than min_distance_s to an
    already-accepted earlier one -- the same physiological-refractory-period cleanup
    `signal_utils.primary_candidates` already applies elsewhere in this project."""
    pred = probs >= threshold
    peak_idx = []
    n = len(pred)
    i = 0
    while i < n:
        if pred[i]:
            j = i
            while j < n and pred[j]:
                j += 1
            run_peak = i + int(np.argmax(env[i:j]))
            peak_idx.append(run_peak)
            i = j
        else:
            i += 1

    min_dist_samples = int(min_distance_s * fsd)
    accepted = []
    for idx in peak_idx:
        if not accepted or idx - accepted[-1] >= min_dist_samples:
            accepted.append(idx)
    return ts[accepted]


def grade(pred_t: np.ndarray, true_clicks: np.ndarray, tol_us: float = TOL_US) -> dict:
    """Identical precision/recall/F1 @ tolerance metric used throughout this project."""
    if len(pred_t) == 0 or len(true_clicks) == 0:
        return dict(precision=float("nan"), recall=float("nan"), f1=float("nan"),
                    tp=0, fp=len(pred_t), fn=len(true_clicks),
                    n_pred=int(len(pred_t)), n_true=int(len(true_clicks)))
    d_pred_to_true = np.abs(pred_t[:, None] - true_clicks[None, :])
    matched_true = set()
    tp = 0
    for row in d_pred_to_true:
        j = int(np.argmin(row))
        if row[j] <= tol_us and j not in matched_true:
            matched_true.add(j)
            tp += 1
    fp = len(pred_t) - tp
    fn = len(true_clicks) - len(matched_true)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision and recall) else float("nan")
    return dict(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn,
                n_pred=int(len(pred_t)), n_true=int(len(true_clicks)))


def best_threshold(probs: np.ndarray, env: np.ndarray, ts: np.ndarray, fsd: float,
                    true_clicks: np.ndarray):
    """Sweeps THRESHOLD_GRID on a (validation) span and returns (best_thr, best_result)."""
    best_thr, best_r = None, None
    for thr in THRESHOLD_GRID:
        pred_t = masks_to_peaks(probs, env, ts, fsd, threshold=thr)
        r = grade(pred_t, true_clicks)
        f1 = r["f1"] if r["f1"] == r["f1"] else -1.0  # nan-safe
        if best_r is None or f1 > (best_r["f1"] if best_r["f1"] == best_r["f1"] else -1.0):
            best_thr, best_r = thr, r
    return best_thr, best_r


def load_chelten_split():
    """Everything needed for the fit/val/test split, envelope-only (no pseudo-labels)."""
    ecg_x, ecg_ts, ecg_fs = real_data.load_ecg(f"{DATA_DIR}/ecg_polar_chelten.parquet", "c1")
    scg_x, scg_ts, scg_fs = real_data.load_scg_mag(f"{DATA_DIR}/MWD_Chelten.parquet", ecg_ts[0], ecg_ts[-1])
    ts_sd, env_sd, sharp, cand_idx, cand_t, fsd = signal_utils.scg_pipeline_stage1(scg_x, scg_ts, scg_fs)

    scg_clicks = load_click_events(ANNOTATION_SCG_PATH)
    ecg_clicks = load_click_events(ANNOTATION_ECG_PATH)
    split = make_split(scg_clicks, ecg_clicks, train_minutes=30.0)

    val_split_t = split.train_hi - int(VAL_MINUTES * 60 * 1e6)
    fit_lo, fit_hi = split.train_lo, val_split_t
    val_lo, val_hi = val_split_t, split.train_hi
    fit_clicks = split.train_scg_clicks[split.train_scg_clicks < val_split_t]
    val_clicks = split.train_scg_clicks[split.train_scg_clicks >= val_split_t]

    fit_i0, fit_i1 = np.searchsorted(ts_sd, [fit_lo, fit_hi])
    val_i0, val_i1 = np.searchsorted(ts_sd, [val_lo, val_hi])
    test_i0, test_i1 = np.searchsorted(ts_sd, [split.test_lo, split.test_hi])

    return dict(
        fsd=fsd,
        env_fit=env_sd[fit_i0:fit_i1], ts_fit=ts_sd[fit_i0:fit_i1], fit_clicks=fit_clicks,
        env_val=env_sd[val_i0:val_i1], ts_val=ts_sd[val_i0:val_i1], val_clicks=val_clicks,
        env_test=env_sd[test_i0:test_i1], ts_test=ts_sd[test_i0:test_i1], test_clicks=split.test_scg_clicks,
        fit_minutes=(fit_hi - fit_lo) / 1e6 / 60, val_minutes=(val_hi - val_lo) / 1e6 / 60,
        test_minutes=(split.test_hi - split.test_lo) / 1e6 / 60,
    )


def build_click_mask(ts_slice: np.ndarray, clicks: np.ndarray, fsd: float, tol_us: float = TOL_US) -> np.ndarray:
    tol_samples = max(1, int(round((tol_us / 1e6) * fsd)))
    mask = np.zeros(len(ts_slice), dtype=np.float32)
    if len(clicks) == 0:
        return mask
    click_idx = np.searchsorted(ts_slice, clicks)
    for idx in click_idx:
        lo, hi = max(0, idx - tol_samples), min(len(mask), idx + tol_samples + 1)
        mask[lo:hi] = 1.0
    return mask


def finetune_and_evaluate(
    run_name: str,
    checkpoint_path: str,
    best_path: str,
    results_path: str,
    n_epochs: int,
    pretrained_state_dict: dict | None = None,
    lr: float = LR,
    verbose: bool = True,
) -> dict:
    """The full leakage-free procedure: fine-tune on Chelten's fit portion (optionally
    starting from a pretrained state dict), select (epoch, threshold) on the internal
    validation slice every epoch, then evaluate exactly once on the true held-out test
    span using the best validation checkpoint. Checkpointed every epoch so it survives
    a sandbox restart."""
    d = load_chelten_split()
    fsd = d["fsd"]
    if verbose:
        print(f"[{run_name}] fit={d['fit_minutes']:.2f}min ({len(d['fit_clicks'])} clicks)  "
              f"val={d['val_minutes']:.2f}min ({len(d['val_clicks'])} clicks)  "
              f"test={d['test_minutes']:.2f}min ({len(d['test_clicks'])} clicks)")

    mask_fit = build_click_mask(d["ts_fit"], d["fit_clicks"], fsd)
    env_mean, env_std = d["env_fit"].mean(), d["env_fit"].std()
    env_fit_n = (d["env_fit"] - env_mean) / (env_std + 1e-12)
    env_val_n = (d["env_val"] - env_mean) / (env_std + 1e-12)
    env_test_n = (d["env_test"] - env_mean) / (env_std + 1e-12)

    X_fit, Y_fit = build_windows(env_fit_n, mask_fit, WINDOW_SAMPLES, STRIDE_SAMPLES)
    frac_pos = Y_fit.mean()
    pos_weight = torch.tensor((1 - frac_pos) / max(frac_pos, 1e-6))
    if verbose:
        print(f"[{run_name}] {len(X_fit)} fit windows, {frac_pos*100:.2f}% positive, "
              f"pos_weight={pos_weight.item():.2f}")

    model = UNet1D(in_channels=1, n_classes=1, base_channels=16, depth=4)
    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)
        if verbose:
            print(f"[{run_name}] initialized from pretrained weights")
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 0
    best_val_f1, best_val_thr, best_val_epoch = -1.0, 0.5, -1
    history = []
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1, best_val_thr, best_val_epoch = ckpt["best_val_f1"], ckpt["best_val_thr"], ckpt["best_val_epoch"]
        history = ckpt["history"]
        if verbose:
            print(f"[{run_name}] resuming from checkpoint: epoch {start_epoch}/{n_epochs} done "
                  f"(best val F1={best_val_f1:.3f} @ epoch {best_val_epoch+1})")

    x = torch.tensor(X_fit).unsqueeze(1)
    y = torch.tensor(Y_fit)
    n = len(x)

    t0 = time.time()
    for epoch in range(start_epoch, n_epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            opt.zero_grad()
            logits = model(x[idx]).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, y[idx], pos_weight=pos_weight)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            x_val = torch.tensor(env_val_n, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            probs_val = torch.sigmoid(model(x_val).squeeze()).numpy()
        thr, r = best_threshold(probs_val, d["env_val"], d["ts_val"], fsd, d["val_clicks"])
        val_f1 = r["f1"] if r["f1"] == r["f1"] else 0.0
        history.append(dict(epoch=epoch, train_bce=epoch_loss / n, val_thr=thr, val_f1=val_f1))

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1, best_val_thr, best_val_epoch = val_f1, thr, epoch
            torch.save(model.state_dict(), best_path)

        if verbose:
            elapsed = time.time() - t0
            flag = "  <-- best" if is_best else ""
            print(f"[{run_name}] epoch {epoch+1}/{n_epochs}  BCE={epoch_loss/n:.4f}  "
                  f"val F1={val_f1:.3f}@{thr}  ({elapsed:.0f}s){flag}", flush=True)

        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=epoch,
                         best_val_f1=best_val_f1, best_val_thr=best_val_thr,
                         best_val_epoch=best_val_epoch, history=history), checkpoint_path)

    model.load_state_dict(torch.load(best_path, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        x_test = torch.tensor(env_test_n, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        probs_test = torch.sigmoid(model(x_test).squeeze()).numpy()
    pred_peak_t = masks_to_peaks(probs_test, d["env_test"], d["ts_test"], fsd, threshold=best_val_thr)
    result = grade(pred_peak_t, d["test_clicks"])
    if verbose:
        print(f"[{run_name}] HELD-OUT (best epoch {best_val_epoch+1}, thr={best_val_thr}): "
              f"precision={result['precision']:.3f} recall={result['recall']:.3f} f1={result['f1']:.3f}")

    out = dict(run_name=run_name, n_params=n_params, n_fit_windows=len(X_fit),
               frac_pos_fit=float(frac_pos), n_epochs=n_epochs,
               best_val_epoch=best_val_epoch, best_val_thr=best_val_thr, best_val_f1=best_val_f1,
               history=history, held_out_result=result, pretrained=pretrained_state_dict is not None)
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    return out

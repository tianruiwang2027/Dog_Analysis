"""
Repeated-trials (Monte Carlo cross-validation) evaluation on Chelten's real hand-clicked
SCG peaks -- the only recording in this dataset with actual hand-clicked SCG ground
truth.

Instead of one arbitrary 30-min-train / 7.6-min-test split, this draws MANY different
random contiguous held-out test windows from within Chelten's ~37.6-minute annotated
span, and for each one fine-tunes a FRESH from-scratch model and a FRESH MAE-pretrained
model (same saved pretrained encoder every trial, fresh downstream head each time) on
whatever remains, then grades both on that trial's held-out window. Same test window
used for both models within a trial, so the per-trial (MAE - scratch) difference is a
paired comparison, controlling for that segment's particular difficulty.

Stage-1 candidates + their ground-truth labels are computed ONCE across the whole
annotated span (the label -- does this candidate match a real click -- never changes),
so each trial only redoes the cheap part (train/test split + fine-tune + grade), not
the expensive signal-processing part.

Run after run_option_a.py has produced mae_pretrained.pt in this same folder. Supports
MAX_NEW_TRIALS=<n> to do a bounded number of new trials per invocation and resume from
a checkpoint (useful if a run gets interrupted).
"""
from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import real_data, signal_utils
from common.config import DATA_DIR, ANNOTATION_SCG_PATH, ANNOTATION_ECG_PATH
from autoencoder.mil_matcher import DOMAIN_TO_ID
from autoencoder.models import JointMaskedAutoencoder
from autoencoder.shape_mil import ShapeMILModel
from autoencoder.pretrain_mae import ShapeMILModelMAE, PATCH_LEN, N_PATCHES_PER_CHANNEL
from common.finetune_chelten import load_click_events, supervised_finetune, evaluate_on_held_out, LabeledCandidates

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "repeated_trials_checkpoint.json")

N_TRIALS = 15
TEST_WINDOW_MIN = 7.5
MIN_TEST_POS = 15
N_EPOCHS = 40


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    domain_id = DOMAIN_TO_ID["dog_noninvasive"]

    print("Loading Chelten + computing Stage-1 candidates/labels ONCE across the full")
    print("annotated span (the label itself never changes across trials)")
    ecg_x, ecg_ts, ecg_fs = real_data.load_ecg(f"{DATA_DIR}/ecg_polar_chelten.parquet", "c1")
    scg_x, scg_ts, scg_fs = real_data.load_scg_mag(f"{DATA_DIR}/MWD_Chelten.parquet", ecg_ts[0], ecg_ts[-1])
    chelten_rec = real_data.RealRecording("Chelten", ecg_x, ecg_ts, ecg_fs, scg_x, scg_ts, scg_fs, domain="dog_noninvasive")

    scg_clicks = load_click_events(ANNOTATION_SCG_PATH)
    ecg_clicks = load_click_events(ANNOTATION_ECG_PATH)
    lo = max(scg_clicks.min(), ecg_clicks.min())
    hi = min(scg_clicks.max(), ecg_clicks.max())
    print(f"  annotated span: {(hi-lo)/1e6/60:.2f} min, {len(scg_clicks)} real SCG clicks")

    ts_sd, env_sd, sharp, cand_idx, cand_t, fsd = signal_utils.scg_pipeline_stage1(
        chelten_rec.x_scg, chelten_rec.ts_scg, chelten_rec.fs_scg
    )
    keep = (cand_t >= lo) & (cand_t < hi)
    cand_idx, cand_t = cand_idx[keep], cand_t[keep]
    snippets = np.stack([signal_utils.snippet_at(env_sd, i) for i in cand_idx]).astype(np.float32)
    d = np.abs(cand_t[:, None] - scg_clicks[None, :])
    labels_all = (d.min(axis=1) <= 60_000.0).astype(np.float32)
    print(f"  {len(cand_t)} Stage-1 candidates in the annotated span, {int(labels_all.sum())} match a real click")

    mae_base = JointMaskedAutoencoder(d_model=64, n_heads=4, n_layers=4, patch_len=PATCH_LEN,
                                       n_patches_per_channel=N_PATCHES_PER_CHANNEL, mask_ratio=0.5)
    mae_base.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "mae_pretrained.pt"), map_location="cpu"))

    test_len_us = TEST_WINDOW_MIN * 60 * 1e6
    rng = np.random.default_rng(42)

    # --- resume support: checkpointed to disk after EVERY trial, and on startup we
    # replay the same deterministic rng draws (cheap) to get back to the same point in
    # the sequence rather than losing everything already computed. ---
    trials = []
    attempts = 0
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            ckpt = json.load(f)
        trials = ckpt["trials"]
        attempts = ckpt["attempts"]
        print(f"resuming from checkpoint: {len(trials)} trials already done, {attempts} attempts so far")
        for _ in range(attempts):
            rng.uniform(0, (hi - lo) - test_len_us)  # replay to reach the same rng state

    max_new = int(os.environ.get("MAX_NEW_TRIALS", "999"))
    n_new_this_run = 0
    trial_i = len(trials)
    while trial_i < N_TRIALS and attempts < N_TRIALS * 8 and n_new_this_run < max_new:
        attempts += 1
        test_start = lo + rng.uniform(0, (hi - lo) - test_len_us)
        test_end = test_start + test_len_us
        test_mask = (cand_t >= test_start) & (cand_t < test_end)
        train_mask = ~test_mask
        n_pos_test = int(labels_all[test_mask].sum())
        if n_pos_test < MIN_TEST_POS:
            continue

        train_data = LabeledCandidates(snippets[train_mask], cand_t[train_mask], labels_all[train_mask])
        test_data = LabeledCandidates(snippets[test_mask], cand_t[test_mask], labels_all[test_mask])

        print(f"\n--- trial {trial_i+1}/{N_TRIALS}  test window [{test_start:.0f},{test_end:.0f}] "
              f"({int(test_mask.sum())} cand, {n_pos_test} pos) | train {int(train_mask.sum())} cand ---", flush=True)

        torch.manual_seed(1000 + trial_i)
        model_scratch = ShapeMILModel()
        model_scratch = supervised_finetune(model_scratch, train_data, domain_id, n_epochs=N_EPOCHS, verbose=False)
        grade_scratch = evaluate_on_held_out(model_scratch, test_data, domain_id)

        torch.manual_seed(1000 + trial_i)
        model_mae = ShapeMILModelMAE(copy.deepcopy(mae_base))
        model_mae = supervised_finetune(model_mae, train_data, domain_id, n_epochs=N_EPOCHS, verbose=False)
        grade_mae = evaluate_on_held_out(model_mae, test_data, domain_id)

        print(f"  scratch: P={grade_scratch['precision']:.3f} R={grade_scratch['recall']:.3f} F1={grade_scratch['f1']:.3f}", flush=True)
        print(f"  mae:     P={grade_mae['precision']:.3f} R={grade_mae['recall']:.3f} F1={grade_mae['f1']:.3f}", flush=True)

        trials.append(dict(
            trial=trial_i, test_start=float(test_start), test_end=float(test_end),
            n_test=int(test_mask.sum()), n_test_pos=n_pos_test, n_train=int(train_mask.sum()),
            scratch=grade_scratch, mae=grade_mae,
        ))
        trial_i += 1
        n_new_this_run += 1

        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(dict(trials=trials, attempts=attempts), f, default=float)
        print(f"  [checkpoint saved: {len(trials)}/{N_TRIALS} trials done]", flush=True)

    print(f"\ncompleted {len(trials)} trials so far ({attempts} attempts)")

    if len(trials) < N_TRIALS:
        print(f"stopping this invocation after {n_new_this_run} new trial(s) this run "
              f"(MAX_NEW_TRIALS={max_new}); re-run the script to continue from checkpoint.")
        return

    def arr(key, sub):
        return np.array([t[sub][key] for t in trials], dtype=float)

    summary = {}
    for sub in ("scratch", "mae"):
        summary[sub] = {}
        for key in ("precision", "recall", "f1"):
            a = arr(key, sub)
            summary[sub][key] = dict(mean=float(a.mean()), std=float(a.std()), min=float(a.min()), max=float(a.max()))

    f1_scratch = arr("f1", "scratch")
    f1_mae = arr("f1", "mae")
    diff = f1_mae - f1_scratch
    summary["paired_f1_diff"] = dict(
        mean=float(diff.mean()), std=float(diff.std()),
        n_mae_wins=int((diff > 0).sum()), n_scratch_wins=int((diff < 0).sum()), n_ties=int((diff == 0).sum()),
        n_trials=len(trials),
    )

    print(json.dumps(summary, indent=2))

    with open(os.path.join(RESULTS_DIR, "repeated_trials_results.json"), "w") as f:
        json.dump(dict(trials=trials, summary=summary), f, indent=2, default=float)
    print("saved repeated_trials_results.json")


if __name__ == "__main__":
    main()

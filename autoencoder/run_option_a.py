"""
Implements Option A end-to-end (masked-autoencoder foundation-model approach) and runs
the four-way comparison this exercise is about:

  (1) ShapeMIL, from-scratch trunk, weak MIL supervision on the pool only -- the
      existing baseline (numbers pasted in by hand, from a separate run).
  (2) ShapeMIL, MAE-PRETRAINED trunk, weak MIL supervision on the pool only -- isolates
      whether foundation-model pretraining helps under the SAME weak-supervision recipe.
  (3) From-scratch trunk, SUPERVISED fine-tune on 30 real min of Chelten hand labels,
      graded on the held-out ~7.6 min -- no pretraining at all, just real few-shot labels.
  (4) MAE-PRETRAINED trunk, SUPERVISED fine-tune on the SAME 30 real min -- this is the
      actual thing asked for: self-supervised pretrain on unlabeled data, fine-tune a
      small head on real few-shot labels, applied to ECG+SCG.

Chelten is excluded from Option A's own pretraining pool (same rule as everywhere else
in this project) -- its labels are only ever used in step (3)/(4)'s fine-tune/grade,
never in pretraining.

Run from anywhere: `python autoencoder/run_option_a.py` (or `python run_option_a.py`
from within this folder) -- the sys.path bootstrap below makes the sibling
`common`/`autoencoder` packages importable either way. Set SCG_DATA_ROOT if your real
data doesn't live in the repo's `data/` folder (see common/config.py).
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import real_data, signal_utils
from common.config import DATA_DIR, ANNOTATION_SCG_PATH, ANNOTATION_ECG_PATH, EPI_BASE, HUMAN_BASE
from autoencoder.mil_matcher import DOMAIN_TO_ID
from autoencoder.multi_domain_data import load_epicardial_dogs, load_human_cebs
from autoencoder.pretrain_mil import _extract
from autoencoder.shape_mil import ShapeMILModel, labels_at_threshold, score_recording
from autoencoder.pretrain_mae import (
    build_pretrain_windows, train_joint_mae, ShapeMILModelMAE, train_shape_mil_given_model,
)
from common.finetune_chelten import (
    load_click_events, make_split, stage1_candidates_in_window, label_candidates,
    supervised_finetune, evaluate_on_held_out,
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

EPI_FILES = [
    f"{EPI_BASE}/MK/MK21/crt/MK2116_crt_b_000.json",
    f"{EPI_BASE}/MK/MK22/ischemia/MK2211_ischemia_b_000.json",
    f"{EPI_BASE}/MK/MK23/lbbbisc/MK2313_lbbbisc_b_000.json",
    f"{EPI_BASE}/MV/MV06/lbbb/MV0629_lbbb_b_000.json",
    f"{EPI_BASE}/MV/MV07/baseline/MV0703_baseline_b_000.json",
    f"{EPI_BASE}/MV/MV08/baseline/MV0802_baseline_b_000.json",
    f"{EPI_BASE}/MV/MV09/baseline/MV0903_baseline_b_000.json",
    f"{EPI_BASE}/MV/MV10/lbbbdob/MV1068_lbbbdob_b_000.json",
    f"{EPI_BASE}/MV/MV13/MV1361_lbbb_b_000.json",
    f"{EPI_BASE}/MV/MV14/baseline/MV1414_baseline_b_000.json",
    f"{EPI_BASE}/MV/MV15/lbbbdob/MV15101_lbbbdob_b_000.json",
    f"{EPI_BASE}/MV/MV17/baseline/MV1702_baseline_b_000.json",
    f"{EPI_BASE}/MV/MV18/baseline/MV1801_baseline_b_000.json",
    f"{EPI_BASE}/MV/MV19/lbbb/MV1980_lbbb_b_000.json",
    f"{EPI_BASE}/MV/MV20/baseline/MV2001_baseline_b_000.json",
    f"{EPI_BASE}/MV/MV21/lbbb/MV2171_lbbb_b_000.json",
    f"{EPI_BASE}/MV/MV22/baseline/MV2201_baseline_b_000.json",
    f"{EPI_BASE}/MV/MV23/crt/MV2352_crt_b_000.json",
]
HUMAN_RECORDS = ["b001", "b002", "b003", "p001"]


def grade_against_clicks(cand_t_native, s1_labels, clicks, tol_us=60_000.0):
    """Identical grader used elsewhere in this project, for direct comparability."""
    lo, hi = clicks.min() - 2_000_000, clicks.max() + 2_000_000
    in_range = (cand_t_native >= lo) & (cand_t_native <= hi)
    cand_in = cand_t_native[in_range]
    pred_in = s1_labels[in_range].astype(bool)
    pred_idx = np.where(pred_in)[0]
    if len(pred_idx) and len(clicks):
        d_pred = np.abs(cand_in[pred_idx][:, None] - clicks[None, :])
        precision = float((d_pred.min(axis=1) <= tol_us).mean())
    else:
        precision = float("nan")
    if len(cand_in):
        d_click = np.abs(clicks[:, None] - cand_in[None, :])
        nearest_idx = np.argmin(d_click, axis=1)
        nearest_dist = d_click[np.arange(len(clicks)), nearest_idx]
        has_candidate = nearest_dist <= tol_us
        matched_positive = has_candidate & pred_in[nearest_idx]
    else:
        has_candidate = np.zeros(len(clicks), dtype=bool)
        matched_positive = np.zeros(len(clicks), dtype=bool)
    return dict(
        precision=precision,
        recall_overall=float(matched_positive.mean()) if len(clicks) else float("nan"),
        recall_conditional=float(matched_positive.sum() / has_candidate.sum()) if has_candidate.sum() else float("nan"),
        stage1_ceiling=float(has_candidate.mean()) if len(clicks) else float("nan"),
        n_clicks=len(clicks),
    )


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t_start = time.time()
    results = {}

    print("=" * 78)
    print("1) Building the multi-domain pretraining pool (Chelten stays blind)")
    print("=" * 78)
    chuck_ecg, chuck_ts_ecg, chuck_fs_ecg = real_data.load_ecg(f"{DATA_DIR}/ecg_biopac_Chuck.parquet", "c4")
    chuck_scg, chuck_ts_scg, chuck_fs_scg = real_data.load_scg_mag(f"{DATA_DIR}/MWD_Chuck.parquet", chuck_ts_ecg[0], chuck_ts_ecg[-1])
    chuck_rec = real_data.RealRecording("Chuck", chuck_ecg, chuck_ts_ecg, chuck_fs_ecg, chuck_scg, chuck_ts_scg, chuck_fs_scg, domain="dog_noninvasive")

    dasty_ecg, dasty_ts_ecg, dasty_fs_ecg = real_data.load_ecg(f"{DATA_DIR}/ecg_Dasty_Biopac.parquet", "c4")
    dasty_scg, dasty_ts_scg, dasty_fs_scg = real_data.load_scg_axis(f"{DATA_DIR}/MWD_Dasty.parquet", "c2", dasty_ts_ecg[0], dasty_ts_ecg[-1])
    dasty_rec = real_data.RealRecording("Dasty", dasty_ecg, dasty_ts_ecg, dasty_fs_ecg, dasty_scg, dasty_ts_scg, dasty_fs_scg, domain="dog_noninvasive")

    epi_recs = load_epicardial_dogs(EPI_FILES)
    human_recs = load_human_cebs(HUMAN_RECORDS, HUMAN_BASE)
    pool = [chuck_rec, dasty_rec] + epi_recs + human_recs
    for dom in set(r.domain for r in pool):
        recs = [r for r in pool if r.domain == dom]
        total_min = sum((r.ts_ecg[-1] - r.ts_ecg[0]) / 1e6 for r in recs) / 60
        print(f"  {dom:16s}: {len(recs):2d} recordings, {total_min:.1f} min total")

    print()
    print("=" * 78)
    print("2) OPTION A: pretraining JointMaskedAutoencoder (masked SCG+ECG reconstruction,")
    print("   NO labels, NO hand clicks)")
    print("=" * 78)
    scg_w, ecg_w = build_pretrain_windows(pool, stride=60)
    print(f"  total pretraining windows: {len(scg_w)}")
    mae = train_joint_mae(scg_w, ecg_w, n_epochs=12, batch_size=512, verbose=True)
    torch.save(mae.state_dict(), os.path.join(OUTPUT_DIR, "mae_pretrained.pt"))
    print(f"  [{time.time()-t_start:.0f}s elapsed]")

    print()
    print("=" * 78)
    print("3) Weak-supervision comparison: MAE-pretrained trunk + SAME MIL objective as")
    print("   the existing ShapeMIL baseline, graded BLIND on Chelten's full window")
    print("=" * 78)
    model_mae_weak = ShapeMILModelMAE(copy.deepcopy(mae))
    model_mae_weak, _ = train_shape_mil_given_model(model_mae_weak, pool, extract_fn=_extract, n_epochs=20, verbose=True)
    print(f"  [{time.time()-t_start:.0f}s elapsed]")

    ecg_x, ecg_ts, ecg_fs = real_data.load_ecg(f"{DATA_DIR}/ecg_polar_chelten.parquet", "c1")
    scg_x, scg_ts, scg_fs = real_data.load_scg_mag(f"{DATA_DIR}/MWD_Chelten.parquet", ecg_ts[0], ecg_ts[-1])
    chelten_rec = real_data.RealRecording("Chelten", ecg_x, ecg_ts, ecg_fs, scg_x, scg_ts, scg_fs, domain="dog_noninvasive")

    domain_id = DOMAIN_TO_ID["dog_noninvasive"]
    feats = _extract(chelten_rec)
    probs_mae_weak = score_recording(model_mae_weak, feats, domain_id)
    s1_mae_weak, n_matched, n_rejected = labels_at_threshold(feats, probs_mae_weak, threshold=0.5)
    scg_clicks_full = real_data.load_chelten_scg_clicks(ANNOTATION_SCG_PATH)
    grade_mae_weak = grade_against_clicks(feats.cand_t_native, s1_mae_weak, scg_clicks_full)
    print(f"  MAE-pretrained + weak MIL supervision (blind, full window): {grade_mae_weak}")
    results["weak_mae"] = grade_mae_weak

    print()
    print("=" * 78)
    print("4) SUPERVISED fine-tune experiments: real Chelten hand labels")
    print("   (first ~30 real minutes to fine-tune, held-out ~7.6 min to grade)")
    print("=" * 78)
    scg_clicks = load_click_events(ANNOTATION_SCG_PATH)
    ecg_clicks = load_click_events(ANNOTATION_ECG_PATH)
    split = make_split(scg_clicks, ecg_clicks, train_minutes=30.0)
    print(f"  train window: {split.train_lo} - {split.train_hi} ({(split.train_hi-split.train_lo)/1e6/60:.1f} min), "
          f"{len(split.train_scg_clicks)} SCG clicks")
    print(f"  held-out window: {split.test_lo} - {split.test_hi} ({(split.test_hi-split.test_lo)/1e6/60:.1f} min), "
          f"{len(split.test_scg_clicks)} SCG clicks")

    train_snip, train_t = stage1_candidates_in_window(chelten_rec, split.train_lo, split.train_hi)
    test_snip, test_t = stage1_candidates_in_window(chelten_rec, split.test_lo, split.test_hi)
    train_data = label_candidates(train_snip, train_t, split.train_scg_clicks)
    test_data = label_candidates(test_snip, test_t, split.test_scg_clicks)
    print(f"  train candidates: {len(train_data.snippets)} ({int(train_data.labels.sum())} positive)")
    print(f"  test candidates:  {len(test_data.snippets)} ({int(test_data.labels.sum())} positive)")

    print()
    print("  --- (3) FROM-SCRATCH trunk, supervised fine-tune on 30 real min ---")
    model_scratch_sup = ShapeMILModel()
    model_scratch_sup = supervised_finetune(model_scratch_sup, train_data, domain_id, n_epochs=60, verbose=True)
    grade_scratch_sup = evaluate_on_held_out(model_scratch_sup, test_data, domain_id)
    print(f"  held-out grading: {grade_scratch_sup}")
    results["scratch_supervised"] = grade_scratch_sup

    print()
    print("  --- (4) MAE-PRETRAINED trunk, supervised fine-tune on 30 real min ---")
    print("      (THE ACTUAL ASK: foundation-model pretrain -> few-label fine-tune)")
    model_mae_sup = ShapeMILModelMAE(copy.deepcopy(mae))
    model_mae_sup = supervised_finetune(model_mae_sup, train_data, domain_id, n_epochs=60, verbose=True)
    grade_mae_sup = evaluate_on_held_out(model_mae_sup, test_data, domain_id)
    print(f"  held-out grading: {grade_mae_sup}")
    results["mae_supervised"] = grade_mae_sup

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(json.dumps(results, indent=2, default=float))
    with open(os.path.join(RESULTS_DIR, "option_a_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[{time.time()-t_start:.0f}s total elapsed]")


if __name__ == "__main__":
    main()

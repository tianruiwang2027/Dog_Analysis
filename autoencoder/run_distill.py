"""
Closes the loop: MAE-pretrained + 30-min-supervised-fine-tuned "teacher" auto-labels
Chelten's FULL ECG<->SCG window, then a tiny SCGNet (the same ~6.5k-parameter
architecture already deployed on the MCU, unchanged) is distilled from those
auto-labels. Evaluated ONLY on the held-out ~7.6-minute span the teacher was never
fine-tuned on, so there's no label leakage into the number that matters: can a
per-dog, MCU-sized model trained this way actually track real HR.

Run after run_option_a.py has produced mae_pretrained.pt in this same folder.
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
from autoencoder import distill
from autoencoder.mil_matcher import DOMAIN_TO_ID
from autoencoder.models import JointMaskedAutoencoder
from autoencoder.pretrain_mae import ShapeMILModelMAE, PATCH_LEN, N_PATCHES_PER_CHANNEL
from common.finetune_chelten import (
    load_click_events, make_split, stage1_candidates_in_window, label_candidates,
    supervised_finetune,
)
from autoencoder.small_nets import count_params

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    domain_id = DOMAIN_TO_ID["dog_noninvasive"]

    print("1) Reloading pretrained MAE + re-running the 30-min supervised fine-tune")
    mae = JointMaskedAutoencoder(d_model=64, n_heads=4, n_layers=4, patch_len=PATCH_LEN,
                                  n_patches_per_channel=N_PATCHES_PER_CHANNEL, mask_ratio=0.5)
    mae.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "mae_pretrained.pt"), map_location="cpu"))

    ecg_x, ecg_ts, ecg_fs = real_data.load_ecg(f"{DATA_DIR}/ecg_polar_chelten.parquet", "c1")
    scg_x, scg_ts, scg_fs = real_data.load_scg_mag(f"{DATA_DIR}/MWD_Chelten.parquet", ecg_ts[0], ecg_ts[-1])
    chelten_rec = real_data.RealRecording("Chelten", ecg_x, ecg_ts, ecg_fs, scg_x, scg_ts, scg_fs, domain="dog_noninvasive")

    scg_clicks = load_click_events(ANNOTATION_SCG_PATH)
    ecg_clicks = load_click_events(ANNOTATION_ECG_PATH)
    split = make_split(scg_clicks, ecg_clicks, train_minutes=30.0)
    train_snip, train_t = stage1_candidates_in_window(chelten_rec, split.train_lo, split.train_hi)
    test_snip, test_t = stage1_candidates_in_window(chelten_rec, split.test_lo, split.test_hi)
    train_data = label_candidates(train_snip, train_t, split.train_scg_clicks)
    test_data = label_candidates(test_snip, test_t, split.test_scg_clicks)

    teacher_model = ShapeMILModelMAE(copy.deepcopy(mae))
    teacher_model = supervised_finetune(teacher_model, train_data, domain_id, n_epochs=60, verbose=False)
    print("  teacher fine-tuned.")

    print()
    print("2) Auto-labeling Chelten's FULL ECG<->SCG window with the fine-tuned teacher")
    ts_sd, env_sd, sharp, cand_idx, cand_t, fsd = signal_utils.scg_pipeline_stage1(
        chelten_rec.x_scg, chelten_rec.ts_scg, chelten_rec.fs_scg
    )
    full_snippets = np.stack([signal_utils.snippet_at(env_sd, i) for i in cand_idx]).astype(np.float32)
    print(f"  {len(full_snippets)} Stage-1 candidates across the full {(chelten_rec.ts_ecg[-1]-chelten_rec.ts_ecg[0])/1e6/60:.1f}-min window")

    with torch.no_grad():
        x = torch.tensor(full_snippets).unsqueeze(1)
        e = teacher_model.embed(x, domain_id)
        probs = torch.sigmoid(teacher_model.score(e)).numpy()
    auto_labels = (probs >= 0.5).astype(np.float32)
    print(f"  auto-labeled {int(auto_labels.sum())} positive / {len(auto_labels)} total")

    print()
    print("3) Distilling a tiny SCGNet (unchanged MCU architecture) on the FULL-window auto-labels")
    net, hist = distill.train_small_net(full_snippets, auto_labels, epochs=40, verbose=True)
    n_params = count_params(net)
    print(f"  SCGNet: {n_params} params, {n_params*4} bytes fp32, {n_params} bytes int8-quantized")

    print()
    print("4) Evaluating the distilled SCGNet on the HELD-OUT ~7.6 min (never used for")
    print("   teacher fine-tuning OR distillation labels)")
    confirmed = distill.confirm_beats(net, test_data.snippets)
    y = test_data.labels.astype(bool)
    tp = int((confirmed & y).sum())
    fp = int((confirmed & ~y).sum())
    fn = int((~confirmed & y).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision and recall) else float("nan")
    print(f"  distilled SCGNet on held-out: precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} "
          f"(tp={tp} fp={fp} fn={fn} n_pos={int(y.sum())})")

    confirmed_t_test = test_data.cand_t[confirmed]
    t_scg, hr_scg = signal_utils.gap_aware_hr(np.sort(confirmed_t_test).astype(np.int64))
    t_click, hr_click, rr_click = signal_utils.beat_hr(np.sort(split.test_scg_clicks).astype(np.int64))
    v_click = rr_click <= 1.5
    v_scg = np.ones(len(t_scg), dtype=bool)
    a, b, _ = signal_utils.align(t_scg, hr_scg, v_scg, t_click, hr_click, v_click, win_us=2_000_000)
    hr_stats = signal_utils.stats(a, b)
    print(f"  HR agreement on held-out only: r={hr_stats['r']:.4f} mae={hr_stats['mae']:.2f}bpm n={hr_stats['n']}")

    torch.save(net.state_dict(), os.path.join(OUTPUT_DIR, "scgnet_distilled_chelten.pt"))

    results = dict(
        n_stage1_candidates_full_window=len(full_snippets),
        n_auto_labeled_positive=int(auto_labels.sum()),
        scgnet_params=n_params,
        scgnet_bytes_fp32=n_params * 4,
        held_out_precision=precision, held_out_recall=recall, held_out_f1=f1,
        held_out_hr_r=hr_stats["r"], held_out_hr_mae=hr_stats["mae"], held_out_hr_n=hr_stats["n"],
    )
    with open(os.path.join(RESULTS_DIR, "distill_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print()
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()

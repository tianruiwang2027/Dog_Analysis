#!/usr/bin/env python3
"""Evaluate the SCG pipeline against Chelten's hand-click ground truth, over
the original validated window.

Uses ground_truth/singles_labels.pkl (hand-annotated beat times, key
"hand_good") and ground_truth/valid_regions.pkl (a small set of time
intervals where the ground truth itself is trustworthy -- excludes stretches
around 24 known click-count mismatches and hand-labeled bad-SCG-quality
edges, each with a 3.0s radius matching the pipeline's own 3.0s smoothing
window). Both are small hand-annotation files, not raw sensor data.

IMPORTANT (see docs/PIPELINE_SUMMARY.md Section 11.1): without the
valid-region mask, this script reproduces r~0.85 instead of the documented
r=0.9095 -- the difference is NOT a bug in the detection pipeline (verified
bit-exact/self-consistent at every stage) but ~64 confident CNN false
positives concentrated in exactly the low-SCG-quality stretches the original
hand-labeling flagged and excluded from evaluation. WITH the mask this
script gets to r~0.88 -- closer, but still short of 0.9095, meaning the
valid-region mask closes most but not all of the gap; some smaller
evaluation-scoping detail from the original session was not fully
reconstructed. Treat r=0.9095 as the documented historical result and
whatever this script reports as its best current reproduction, not as
identical.

Example:
    python scripts/evaluate_vs_hand_clicks.py --hive-dir ./hive --weights-dir ./weights
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scg_hr import constants as C
from scg_hr import io_hive, signal_utils, scg_pipeline, compare_utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hive-dir", default=io_hive.DEFAULT_HIVE_DIR)
    ap.add_argument("--weights-dir", default="./weights")
    ap.add_argument("--ground-truth-dir", default="./ground_truth")
    ap.add_argument("--no-mask", action="store_true", help="skip the valid-region mask (shows the ~0.85 unmasked number)")
    args = ap.parse_args()

    t0, t1 = C.RESTRICT_CHELTEN
    ts_s, x_s, axis = io_hive.load_scg_window("Chelten", t0, t1, pad_s=C.PAD_S, hive_dir=args.hive_dir)
    fs_s = io_hive.fs_estimate(ts_s)
    xf_s = signal_utils.bandpass(x_s, fs_s, 10.0, 100.0)
    ts_sd, env_sd, fsd_s = signal_utils.shannon_envelope_decimated(xf_s, ts_s, fs_s, ref_max=C.CHELTEN_SCG_REF_MAX)
    sharp_s, _ = signal_utils.sharpen_local(env_sd, fsd_s, q995_win_s=8.0)

    models = scg_pipeline.load_scg_models(args.weights_dir)
    merged, info = scg_pipeline.detect_beats(ts_sd, env_sd, sharp_s, fsd_s, models)
    print(f"confirmed: {len(merged)}  ({info})")
    scg_tmid, scg_hr = scg_pipeline.gap_aware_hr(merged)

    gt_dir = Path(args.ground_truth_dir)
    with open(gt_dir / "singles_labels.pkl", "rb") as f:
        hand = np.sort(pickle.load(f)["hand_good"])
    h_tmid, h_hr, h_rr = scg_pipeline.beat_hr(hand)
    ok = h_rr <= 1.5
    h_tmid_ok, h_hr_ok = h_tmid[ok], h_hr[ok]
    h_hr_sm = compare_utils.smooth_plain(h_tmid_ok, h_hr_ok)

    vA = np.ones(len(scg_tmid), dtype=bool)
    vB = np.ones(len(h_tmid_ok), dtype=bool)
    xa, yb, ta = compare_utils.align(scg_tmid, scg_hr, vA, h_tmid_ok, h_hr_sm, vB, lag_us=0)

    if not args.no_mask:
        with open(gt_dir / "valid_regions.pkl", "rb") as f:
            valid_ivs = pickle.load(f)["valid_ivs"]
        mask = np.array([any(a <= t <= b for a, b in valid_ivs) for t in ta])
        xa, yb = xa[mask], yb[mask]
        print(f"valid-region mask kept {mask.sum()}/{len(mask)} points")

    s = compare_utils.stats(xa, yb)
    print(f"\n=== SCG pipeline vs hand clicks ({'no mask' if args.no_mask else 'valid-region masked'}) ===")
    print(f"r={s['r']:.4f}  n={s['n']}  MAE={s['mae']:.2f}bpm  bias={s['bias']:.2f}bpm")
    print("(documented headline: r=0.9095, n=1285, MAE=2.20bpm, bias=-0.25bpm -- see docs/PIPELINE_SUMMARY.md Section 11.1)")


if __name__ == "__main__":
    main()

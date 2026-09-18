#!/usr/bin/env python3
"""End-to-end SCG-vs-ECG comparison for one dog: runs the SCG pipeline (Stages
1-4) and the tuned Pan-Tompkins ECG detector over the same time span, aligns
them, and reports r/MAE/bias. This is an ALGORITHM-VS-ALGORITHM comparison --
it does not require a hand-click ground-truth file (which is not included in
this repo; see docs/PIPELINE_SUMMARY.md for the headline hand-click-validated
numbers on Chelten, r=0.9095 SCG / r=0.9820 ECG, which this script cannot
reproduce on its own since it has no hand clicks to compare against).

Example (Chelten, using the measured clock-drift correction):
    python scripts/compare_scg_vs_ecg.py --dog Chelten --ecg-stream ecg_polar \\
        --hive-dir ./hive --weights-dir ./weights --use-chelten-drift-correction

Example (Dasty, cross-dog generalization check -- SCG side is NOT expected to
work well; see docs/PIPELINE_SUMMARY.md Section 6):
    python scripts/compare_scg_vs_ecg.py --dog Dasty --ecg-stream ecg_biopac \\
        --hive-dir ./hive --weights-dir ./weights --lag-us 0 --refract-s 0.3
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scg_hr import constants as C
from scg_hr import io_hive, signal_utils, scg_pipeline, ecg_pantompkins, compare_utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dog", required=True)
    ap.add_argument("--ecg-stream", default="ecg_biopac", choices=["ecg_biopac", "ecg_polar"])
    ap.add_argument("--hive-dir", default=io_hive.DEFAULT_HIVE_DIR)
    ap.add_argument("--weights-dir", default="./weights")
    ap.add_argument("--axis", default=None)
    ap.add_argument("--lead", default=None)
    ap.add_argument("--refract-s", type=float, default=C.PRIMARY_REFRACT_S)
    ap.add_argument("--lag-us", type=float, default=None,
                     help=f"fixed ECG->SCG-clock lag in microseconds; default = constants.LAG_US "
                          f"({C.LAG_US}), meaningful for Chelten only -- pass 0 for a same-clock "
                          f"dog like Dasty")
    ap.add_argument("--use-chelten-drift-correction", action="store_true",
                     help="use the measured time-varying lag (constants.LAG_US_INTERCEPT + "
                          "LAG_DRIFT_US_PER_S*t) instead of a fixed --lag-us; Chelten only, "
                          "see docs/PIPELINE_SUMMARY.md Section 10")
    ap.add_argument("--ref-max", default="auto",
                     help="envelope normalization reference (see constants.CHELTEN_SCG_REF_MAX). "
                          "'auto' (default) = CHELTEN_SCG_REF_MAX for Chelten, self-normalize "
                          "otherwise -- required to avoid a known normalization-contamination bug "
                          "over long/full-session spans; 'self' forces self-normalization.")
    args = ap.parse_args()

    if args.ref_max == "auto":
        ref_max = C.CHELTEN_SCG_REF_MAX if args.dog == "Chelten" else None
    elif args.ref_max == "self":
        ref_max = None
    else:
        ref_max = float(args.ref_max)

    axis = args.axis or C.SCG_CHANNEL.get(args.dog) or C.AXIS.get(args.dog)
    lead = args.lead or C.LEAD.get(args.ecg_stream)
    if axis is None or lead is None:
        raise SystemExit("unknown axis/lead for this dog/stream; pass --axis/--lead explicitly")

    # ---- SCG side ----
    e_lo, e_hi = io_hive.full_span(args.ecg_stream, args.dog, 0, hive_dir=args.hive_dir)
    s_lo, s_hi = io_hive.full_span("scg_mwd", args.dog, 45, hive_dir=args.hive_dir)
    t0, t1 = max(e_lo, s_lo), min(e_hi, s_hi)
    print(f"overlap span: {(t1-t0)/1e6/60:.2f} min")

    ts_s, x_s, axis = io_hive.load_scg_window(args.dog, t0, t1, axis,
                                               pad_s=C.PAD_S, hive_dir=args.hive_dir)
    fs_s = io_hive.fs_estimate(ts_s)
    xf_s = signal_utils.bandpass(x_s, fs_s, 10.0, 100.0)
    print(f"envelope ref_max: {'self-normalizing' if ref_max is None else ref_max}")
    ts_sd, env_sd, fsd_s = signal_utils.shannon_envelope_decimated(xf_s, ts_s, fs_s, ref_max=ref_max)
    sharp_s, _ = signal_utils.sharpen_local(env_sd, fsd_s, q995_win_s=8.0)

    models = scg_pipeline.load_scg_models(args.weights_dir)
    merged, info = scg_pipeline.detect_beats(ts_sd, env_sd, sharp_s, fsd_s, models,
                                              primary_refract_s=args.refract_s)
    print(f"SCG: candidates={info['n_candidates']}  confirmed={len(merged)} "
          f"({len(merged)/((t1-t0)/1e6/60):.1f}/min)")
    scg_tmid, scg_hr = scg_pipeline.gap_aware_hr(merged)
    print(f"SCG: {len(scg_tmid)} gap-aware HR points")

    # ---- ECG side ----
    ts_e, x_e = io_hive.load_window(args.ecg_stream, args.dog, 0, lead, t0, t1,
                                     hive_dir=args.hive_dir)
    fs_e = io_hive.fs_estimate(ts_e)
    R_idx = ecg_pantompkins.detect_qrs(x_e, fs_e)
    pt_pk = ts_e[R_idx]
    print(f"ECG: {len(pt_pk)} R-peaks")
    pt_tmid, pt_hr_raw, pt_rr = scg_pipeline.beat_hr(pt_pk)
    rr_ok = (pt_rr >= 0.3) & (pt_rr <= 1.5)
    pt_tmid_ok, pt_hr_ok = pt_tmid[rr_ok], pt_hr_raw[rr_ok]
    pt_hr_sm = compare_utils.smooth_plain(pt_tmid_ok, pt_hr_ok)

    # ---- align + compare ----
    if args.use_chelten_drift_correction:
        lag = compare_utils.lag_at
        print(f"using time-varying lag: {C.LAG_US_INTERCEPT/1e6:.4f}s + {C.LAG_DRIFT_US_PER_S:.2f}us/s * t_elapsed")
    else:
        lag = args.lag_us if args.lag_us is not None else float(C.LAG_US)
        print(f"using fixed lag: {lag/1e6:.4f}s")

    vA = np.ones(len(scg_tmid), dtype=bool)
    vB = np.ones(len(pt_tmid_ok), dtype=bool)
    xa, yb, ta = compare_utils.align(scg_tmid, scg_hr, vA, pt_tmid_ok, pt_hr_sm, vB, lag_us=lag)

    s = compare_utils.stats(xa, yb)
    print(f"\n=== SCG pipeline vs Pan-Tompkins ECG ({args.dog}) ===")
    print(f"r={s['r']:.4f}  n={s['n']}  MAE={s['mae']:.2f}bpm  bias={s['bias']:.2f}bpm")


if __name__ == "__main__":
    main()

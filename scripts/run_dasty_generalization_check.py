#!/usr/bin/env python3
"""Reproduce docs/PIPELINE_SUMMARY.md Section 6: run the Chelten-trained SCG
pipeline unchanged on Dasty, at both the original 0.5s primary refractory
period and the loosened 0.3s one, and print the confirm-rate comparison table.

This is a NEGATIVE result by design (SCGNet/S2Net were trained on Chelten
only and do not currently generalize to Dasty's SCG morphology/amplitude
scale) -- see docs/PIPELINE_SUMMARY.md Section 9 for the planned fix
(NCC feature fusion + self-normalization) that is NOT yet implemented.

Example:
    python scripts/run_dasty_generalization_check.py --hive-dir ./hive --weights-dir ./weights
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scg_hr import constants as C
from scg_hr import io_hive, signal_utils, scg_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hive-dir", default=io_hive.DEFAULT_HIVE_DIR)
    ap.add_argument("--weights-dir", default="./weights")
    ap.add_argument("--dog", default="Dasty")
    args = ap.parse_args()

    axis = C.SCG_CHANNEL.get(args.dog) or C.AXIS[args.dog]
    e_lo, e_hi = io_hive.full_span("ecg_biopac", args.dog, 0, hive_dir=args.hive_dir)
    s_lo, s_hi = io_hive.full_span("scg_mwd", args.dog, 45, hive_dir=args.hive_dir)
    t0, t1 = max(e_lo, s_lo), min(e_hi, s_hi)
    span_min = (t1 - t0) / 1e6 / 60
    print(f"{args.dog}: ecg_biopac/scg_mwd overlap = {span_min:.2f} min")

    ts_s, x_s, axis = io_hive.load_scg_window(args.dog, t0, t1, axis,
                                               pad_s=C.PAD_S, hive_dir=args.hive_dir)
    fs_s = io_hive.fs_estimate(ts_s)
    xf_s = signal_utils.bandpass(x_s, fs_s, 10.0, 100.0)
    # self-normalize (no calibrated reference exists for any dog but Chelten)
    ref_max = C.CHELTEN_SCG_REF_MAX if args.dog == "Chelten" else None
    ts_sd, env_sd, fsd_s = signal_utils.shannon_envelope_decimated(xf_s, ts_s, fs_s, ref_max=ref_max)
    sharp_s, _ = signal_utils.sharpen_local(env_sd, fsd_s, q995_win_s=8.0)

    models = scg_pipeline.load_scg_models(args.weights_dir)

    print(f"\n{'refractory':>10} {'candidates':>11} {'confirmed':>10} {'confirm%':>9} {'beats/min':>10}")
    for refract_s in (0.50, 0.30):
        merged, info = scg_pipeline.detect_beats(ts_sd, env_sd, sharp_s, fsd_s, models,
                                                  primary_refract_s=refract_s)
        confirm_pct = 100 * info["n_confirmed_main"] / max(1, info["n_candidates"])
        print(f"{refract_s:10.2f} {info['n_candidates']:11d} {len(merged):10d} "
              f"{confirm_pct:8.1f}% {len(merged)/span_min:10.1f}")

    print(f"\nExternal reference (pre-existing CORAL run, algorithm vs algorithm on Dasty):")
    print(f"  r=0.7170, n=3547, mae=4.94bpm, bias=-2.00bpm "
          f"(see analysis/coral_ecg/{args.dog}_ecg_biopac/agree.json if present)")


if __name__ == "__main__":
    main()

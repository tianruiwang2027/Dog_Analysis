#!/usr/bin/env python3
"""Run the full SCG heartbeat-detection pipeline (Stages 1-4) for one dog over
a given time window and print/save the resulting gap-aware HR trace.

Example:
    python scripts/run_scg_pipeline.py --dog Chelten \\
        --t0 1782494790000000 --t1 1782496808699088 \\
        --hive-dir ./hive --weights-dir ./weights

For a dog other than Chelten, the frozen CNN/S2Net were NOT trained on that
dog's data and are known to generalize poorly (see docs/PIPELINE_SUMMARY.md
Section 6) -- this script will still run, but treat its output as an
unvalidated cross-dog generalization test, not a trustworthy HR trace.
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scg_hr import constants as C
from scg_hr import io_hive, signal_utils, scg_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dog", required=True)
    ap.add_argument("--t0", type=int, default=None, help="epoch microseconds; default = whole recording")
    ap.add_argument("--t1", type=int, default=None)
    ap.add_argument("--hive-dir", default=io_hive.DEFAULT_HIVE_DIR)
    ap.add_argument("--weights-dir", default="./weights")
    ap.add_argument("--axis", default=None, help="override constants.AXIS[dog]")
    ap.add_argument("--refract-s", type=float, default=C.PRIMARY_REFRACT_S,
                     help="primary-candidate refractory period; loosen (e.g. 0.3) for fast-HR dogs")
    ap.add_argument("--out", default=None, help="save result dict to this pickle path")
    ap.add_argument("--ref-max", default="auto",
                     help="envelope normalization reference (see constants.CHELTEN_SCG_REF_MAX). "
                          "'auto' (default) = CHELTEN_SCG_REF_MAX for Chelten, self-normalize "
                          "otherwise; 'self' forces self-normalization; or pass a float.")
    args = ap.parse_args()

    if args.ref_max == "auto":
        ref_max = C.CHELTEN_SCG_REF_MAX if args.dog == "Chelten" else None
    elif args.ref_max == "self":
        ref_max = None
    else:
        ref_max = float(args.ref_max)

    if args.axis is None and args.dog not in C.SCG_CHANNEL:
        raise SystemExit(f"no known scg_mwd axis for dog={args.dog!r}; pass --axis explicitly")

    ts_s, x_s, axis = io_hive.load_scg_window(args.dog, args.t0, args.t1, args.axis,
                                               pad_s=C.PAD_S, hive_dir=args.hive_dir)
    fs_s = io_hive.fs_estimate(ts_s)
    print(f"loaded {len(ts_s)} scg samples @ {fs_s:.2f}Hz (axis={axis})")

    xf_s = signal_utils.bandpass(x_s, fs_s, 10.0, 100.0)
    print(f"envelope ref_max: {'self-normalizing' if ref_max is None else ref_max}")
    ts_sd, env_sd, fsd_s = signal_utils.shannon_envelope_decimated(xf_s, ts_s, fs_s, ref_max=ref_max)
    sharp_s, _ = signal_utils.sharpen_local(env_sd, fsd_s, q995_win_s=8.0)

    models = scg_pipeline.load_scg_models(args.weights_dir)
    merged, info = scg_pipeline.detect_beats(ts_sd, env_sd, sharp_s, fsd_s, models,
                                              primary_refract_s=args.refract_s)
    print(f"candidates={info['n_candidates']}  confirmed(main CNN)={info['n_confirmed_main']}  "
          f"S2-flagged={info['n_s2_flagged']}  S2-recovered={info['n_s2_accepted']}  "
          f"final confirmed={len(merged)}")

    tmid, hr = scg_pipeline.gap_aware_hr(merged)
    span_min = (ts_s[-1] - ts_s[0]) / 1e6 / 60
    print(f"confirmed beats/min = {len(merged)/span_min:.1f}   "
          f"gap-aware HR points = {len(tmid)}/{max(1,len(merged)-1)} raw RR intervals")
    if len(hr):
        print(f"HR range: {hr.min():.0f}-{hr.max():.0f} bpm, median {sorted(hr)[len(hr)//2]:.0f} bpm")

    if args.out:
        with open(args.out, "wb") as f:
            pickle.dump(dict(merged=merged, tmid=tmid, hr=hr, info=info, axis=axis,
                              t0=int(ts_s[0]), t1=int(ts_s[-1])), f)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()

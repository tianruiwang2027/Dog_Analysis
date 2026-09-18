#!/usr/bin/env python3
"""Run the fully-tuned Pan-Tompkins R-peak detector for one dog/stream over a
given time window.

Example:
    python scripts/run_ecg_pantompkins.py --dog Chelten --stream ecg_polar \\
        --hive-dir ./hive
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scg_hr import constants as C
from scg_hr import io_hive, ecg_pantompkins, scg_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dog", required=True)
    ap.add_argument("--stream", default="ecg_biopac", choices=["ecg_biopac", "ecg_polar"])
    ap.add_argument("--t0", type=int, default=None)
    ap.add_argument("--t1", type=int, default=None)
    ap.add_argument("--hive-dir", default=io_hive.DEFAULT_HIVE_DIR)
    ap.add_argument("--lead", default=None, help="override constants.LEAD[stream]")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lead = args.lead or C.LEAD.get(args.stream)
    ts, x = io_hive.load_window(args.stream, args.dog, 0, lead, args.t0, args.t1,
                                 hive_dir=args.hive_dir)
    fs = io_hive.fs_estimate(ts)
    print(f"loaded {len(ts)} ecg samples @ {fs:.2f}Hz (stream={args.stream}, lead={lead})")

    R_idx = ecg_pantompkins.detect_qrs(x, fs)
    pk = ts[R_idx]
    print(f"detected {len(pk)} R-peaks")

    tmid, hr_raw, rr = scg_pipeline.beat_hr(pk)
    rr_ok = (rr >= 0.3) & (rr <= 1.5)
    if rr_ok.sum():
        print(f"median HR (RR in [0.3,1.5]s): {sorted(hr_raw[rr_ok])[rr_ok.sum()//2]:.0f} bpm")

    if args.out:
        with open(args.out, "wb") as f:
            pickle.dump(dict(pk=pk, tmid=tmid, hr_raw=hr_raw, rr=rr,
                              t0=int(ts[0]), t1=int(ts[-1])), f)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()

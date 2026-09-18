#!/usr/bin/env python3
"""Reproduce docs/PIPELINE_SUMMARY.md Section 10: estimate the ECG-SCG lag
independently in windows spanning the full session, via a cross-correlogram
(histogram-of-time-differences mode) method on actual detected beats -- and
fit a line to check for clock drift beyond the fixed constants.LAG_US.

Chelten result (already measured, baked into constants.py as
LAG_US_INTERCEPT/LAG_DRIFT_US_PER_S): ~25.1 ppm drift rate, t-stat=14.7,
r=0.977 across 12 high-confidence windows -- a real, physically-expected
clock-skew effect between the two independently-clocked recording rigs, NOT
"aliasing" in the classical signal-processing sense. See docs/PIPELINE_SUMMARY.md
Section 10.5 for the terminology note.

Example:
    python scripts/run_lag_drift_check.py --dog Chelten --ecg-stream ecg_polar \\
        --hive-dir ./hive --weights-dir ./weights --n-windows 18 --fig-out lag_drift.png
"""
import argparse
import datetime
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scg_hr import constants as C
from scg_hr import io_hive, signal_utils, scg_pipeline, ecg_pantompkins

UTC = datetime.timezone.utc


def mode_lag(ecg_win, scg_all, search_lo_us, search_hi_us, bin_us=1000):
    """Histogram-of-time-differences method: for every ECG peak in this
    window, look at every SCG beat within [t_e - search_hi, t_e - search_lo]
    and record diff = t_e - t_s. True matching pairs cluster tightly around
    the real lag; false pairs scatter roughly uniformly. The histogram mode
    (parabolically refined) is the lag estimate."""
    diffs = []
    for t_e in ecg_win:
        lo_t, hi_t = t_e - search_hi_us, t_e - search_lo_us
        i0, i1 = np.searchsorted(scg_all, lo_t), np.searchsorted(scg_all, hi_t)
        if i1 > i0:
            diffs.append(t_e - scg_all[i0:i1])
    if not diffs:
        return None
    diffs = np.concatenate(diffs)
    if len(diffs) < 20:
        return None
    edges = np.arange(search_lo_us, search_hi_us + bin_us, bin_us)
    hist, _ = np.histogram(diffs, bins=edges)
    i_peak = np.argmax(hist)
    if 0 < i_peak < len(hist) - 1:
        y0, y1, y2 = hist[i_peak - 1], hist[i_peak], hist[i_peak + 1]
        denom = y0 - 2 * y1 + y2
        delta = np.clip(0.5 * (y0 - y2) / denom, -1, 1) if denom != 0 else 0.0
    else:
        delta = 0.0
    lag_center = edges[i_peak] + bin_us * (0.5 + delta)
    peak_h = hist[i_peak]
    bg = np.median(hist[hist > 0]) if (hist > 0).any() else 0
    return dict(lag_us=lag_center, n_ecg=len(ecg_win), n_diffs=len(diffs),
                snr=peak_h / (bg + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dog", default="Chelten")
    ap.add_argument("--ecg-stream", default="ecg_polar")
    ap.add_argument("--hive-dir", default=io_hive.DEFAULT_HIVE_DIR)
    ap.add_argument("--weights-dir", default="./weights")
    ap.add_argument("--prior-lag-us", type=float, default=float(C.LAG_US))
    ap.add_argument("--n-windows", type=int, default=18)
    ap.add_argument("--min-snr", type=float, default=10.0)
    ap.add_argument("--min-ecg", type=int, default=50)
    ap.add_argument("--fig-out", default=None)
    args = ap.parse_args()

    axis = C.SCG_CHANNEL.get(args.dog) or C.AXIS[args.dog]
    lead = C.LEAD[args.ecg_stream]

    ts_e, x_e = io_hive.load_window(args.ecg_stream, args.dog, 0, lead, hive_dir=args.hive_dir)
    fs_e = io_hive.fs_estimate(ts_e)
    pt_pk = ts_e[ecg_pantompkins.detect_qrs(x_e, fs_e)]
    print(f"Pan-Tompkins: {len(pt_pk)} R-peaks over full session")

    ts_s, x_s, axis = io_hive.load_scg_window(args.dog, axis=axis, hive_dir=args.hive_dir)
    fs_s = io_hive.fs_estimate(ts_s)
    xf_s = signal_utils.bandpass(x_s, fs_s, 10.0, 100.0)
    # full-session load -> MUST use a fixed ref_max, not self-normalization, or a stray
    # motion/handling spike elsewhere in the recording silently rescales the envelope
    # the CNN sees (see constants.CHELTEN_SCG_REF_MAX docstring / PIPELINE_SUMMARY.md Section 5)
    ref_max = C.CHELTEN_SCG_REF_MAX if args.dog == "Chelten" else None
    ts_sd, env_sd, fsd_s = signal_utils.shannon_envelope_decimated(xf_s, ts_s, fs_s, ref_max=ref_max)
    sharp_s, _ = signal_utils.sharpen_local(env_sd, fsd_s, q995_win_s=8.0)
    models = scg_pipeline.load_scg_models(args.weights_dir)
    merged, _ = scg_pipeline.detect_beats(ts_sd, env_sd, sharp_s, fsd_s, models)
    print(f"SCG pipeline: {len(merged)} confirmed beats over full session")

    search_lo_us = args.prior_lag_us - 3_000_000
    search_hi_us = args.prior_lag_us + 3_000_000
    t0_all, t1_all = pt_pk.min(), pt_pk.max()
    edges_t = np.linspace(t0_all, t1_all, args.n_windows + 1).astype("int64")

    rows = []
    print(f"\n{'win':>3} {'time':>9} {'n_ecg':>6} {'lag(s)':>9} {'snr':>7}  keep?")
    for i in range(args.n_windows):
        w0, w1 = edges_t[i], edges_t[i + 1]
        ecg_win = pt_pk[(pt_pk >= w0) & (pt_pk < w1)]
        res = mode_lag(ecg_win, merged, search_lo_us, search_hi_us)
        tmid = (w0 + w1) / 2
        tstr = datetime.datetime.fromtimestamp(tmid / 1e6, tz=UTC).strftime("%H:%M:%S")
        if res is None or res["snr"] < args.min_snr or res["n_ecg"] < args.min_ecg:
            print(f"{i:3d} {tstr:>9} {len(ecg_win):6d}      --      --  DROP")
            continue
        rows.append((tmid, res["lag_us"], res["n_ecg"]))
        print(f"{i:3d} {tstr:>9} {res['n_ecg']:6d} {res['lag_us']/1e6:9.4f} {res['snr']:7.1f}  keep")

    if len(rows) < 3:
        print("\ntoo few high-confidence windows to fit a trend")
        return

    rows = np.array(rows)
    tmid_arr, lag_arr, n_arr = rows[:, 0], rows[:, 1], rows[:, 2]
    t_elapsed_s = (tmid_arr - t0_all) / 1e6
    A = np.vstack([t_elapsed_s, np.ones_like(t_elapsed_s)]).T
    w = np.sqrt(n_arr)
    coef, *_ = np.linalg.lstsq(A * w[:, None], lag_arr * w, rcond=None)
    slope_us_per_s, intercept_us = coef
    yhat = A @ coef
    dof = len(rows) - 2
    sigma2 = (((lag_arr - yhat) * w) ** 2).sum() / dof
    cov = sigma2 * np.linalg.inv((A * w[:, None]).T @ (A * w[:, None]))
    slope_se = np.sqrt(cov[0, 0])
    r_lag_time = np.corrcoef(t_elapsed_s, lag_arr)[0, 1]

    print(f"\nlag(t) = {intercept_us/1e6:.4f}s + ({slope_us_per_s:.2f} +/- {slope_se:.2f} us/s) * t_elapsed")
    print(f"drift rate: {slope_us_per_s:.2f} ppm  (t-stat={slope_us_per_s/slope_se:.1f}, r={r_lag_time:.3f}, "
          f"n={len(rows)} windows kept)")

    if args.fig_out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.errorbar(t_elapsed_s / 60, lag_arr / 1000, yerr=1000 / np.sqrt(n_arr), fmt="o",
                     color="tab:purple", capsize=3, label="windowed event-matched lag")
        tt = np.linspace(t_elapsed_s.min(), t_elapsed_s.max(), 50)
        ax.plot(tt / 60, (intercept_us + slope_us_per_s * tt) / 1000, color="black", lw=1.5, ls="--",
                label=f"fit: {slope_us_per_s:.2f} us/s")
        ax.axhline(args.prior_lag_us / 1000, color="tab:gray", lw=1, ls=":", label="prior fixed lag")
        ax.set_xlabel("time (min, elapsed)")
        ax.set_ylabel("estimated ECG-SCG lag (ms)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(args.fig_out, dpi=140)
        print(f"saved {args.fig_out}")


if __name__ == "__main__":
    main()

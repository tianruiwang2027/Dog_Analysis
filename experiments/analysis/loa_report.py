#!/usr/bin/env python3
"""Bland-Altman limits of agreement: MWD (SCG autocorrelation HR, scg_hr.py) vs
each ECG reference (Pan-Tompkins beat HR, ecg_hr.py), for dog Chelten.

Reuses the exact alignment convention of compare.py: ECG beat-HR linearly
interpolated onto each valid SCG hop time, NaN'd out if the local beat gap
exceeds 2s (lead-off). Reports in BOTH directions so it's unambiguous.
"""
import os, numpy as np, polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
DOG = "Chelten"
SQI_THR = 0.30  # scg_hr.py's own gate, already applied in its 'valid' column


def ecg_at(t, tmid, bpm, max_gap_us=2_000_000):
    out = np.full(len(t), np.nan)
    if len(tmid) < 2:
        return out
    idx = np.searchsorted(tmid, t)
    v = (idx > 0) & (idx < len(tmid))
    i = idx[v]; lo = tmid[i - 1]; hi = tmid[i]; gap = hi - lo
    frac = (t[v] - lo) / np.maximum(gap, 1)
    val = bpm[i - 1] * (1 - frac) + bpm[i] * frac
    val[gap > max_gap_us] = np.nan
    out[v] = val
    return out


def stats(x, y, label):
    """x = MWD (SCG) HR, y = reference ECG HR. Returns dict with both directions."""
    d_mwd_minus_ref = x - y
    d_ref_minus_mwd = y - x
    n = len(x)
    bias_r = float(d_ref_minus_mwd.mean()); sd_r = float(d_ref_minus_mwd.std(ddof=1))
    bias_m = float(d_mwd_minus_ref.mean()); sd_m = float(d_mwd_minus_ref.std(ddof=1))
    r = float(np.corrcoef(x, y)[0, 1]) if n > 2 else float("nan")
    return dict(
        label=label, n=n, pearson_r=r,
        mae_bpm=float(np.abs(d_mwd_minus_ref).mean()),
        rmse_bpm=float(np.sqrt((d_mwd_minus_ref ** 2).mean())),
        within5=float(np.mean(np.abs(d_mwd_minus_ref) <= 5)),
        within10=float(np.mean(np.abs(d_mwd_minus_ref) <= 10)),
        bias_ref_minus_mwd=bias_r, sd_diff=sd_r,
        loa_ref_minus_mwd=[bias_r - 1.96 * sd_r, bias_r + 1.96 * sd_r],
        bias_mwd_minus_ref=bias_m,
        loa_mwd_minus_ref=[bias_m - 1.96 * sd_m, bias_m + 1.96 * sd_m],
    )


def main():
    s = pl.read_parquet(os.path.join(HERE, "scg_hr", f"{DOG}.parquet"))
    sts = s["ts"].to_numpy().astype("int64")
    sbpm = s["bpm"].to_numpy().astype(float)
    svalid = s["valid"].to_numpy().astype(bool)
    print(f"MWD (SCG autocorrelation HR) hops: {len(sts):,}  valid (SQI>={SQI_THR}): {svalid.sum():,} "
          f"({100*svalid.mean():.0f}%)")

    results = {}
    for mod, label in [("ecg_biopac", "BioPac"), ("ecg_polar", "Polar")]:
        e = pl.read_parquet(os.path.join(HERE, "ecg_hr", f"{DOG}_{mod}.parquet"))
        rts = e["ts"].to_numpy().astype("int64"); rbpm = e["bpm"].to_numpy().astype(float)
        w0, w1 = max(sts[0], rts[0]), min(sts[-1], rts[-1])
        sel = svalid & (sts >= w0) & (sts <= w1)
        x = sbpm[sel]
        y = ecg_at(sts[sel], rts, rbpm)
        ok = np.isfinite(y)
        x, y = x[ok], y[ok]
        print(f"\n[{label}] overlap window with MWD: n valid-and-overlapping = {len(x)}")
        if len(x) < 10:
            print("  insufficient overlap, skipping")
            continue
        R = stats(x, y, label)
        results[label] = R
        print(f"  n={R['n']}  r={R['pearson_r']:.2f}  MAE={R['mae_bpm']:.1f} bpm  RMSE={R['rmse_bpm']:.1f} bpm  "
              f"within5={100*R['within5']:.0f}%  within10={100*R['within10']:.0f}%")
        print(f"  bias ({label} - MWD) = {R['bias_ref_minus_mwd']:+.2f} bpm,  SD of differences = {R['sd_diff']:.2f} bpm")
        print(f"  95% Limits of Agreement ({label} - MWD) = "
              f"[{R['loa_ref_minus_mwd'][0]:+.2f}, {R['loa_ref_minus_mwd'][1]:+.2f}] bpm")

    import json
    json.dump(results, open(os.path.join(HERE, f"loa_{DOG}.json"), "w"), indent=2)
    print(f"\n-> {os.path.join(HERE, f'loa_{DOG}.json')}")


if __name__ == "__main__":
    main()

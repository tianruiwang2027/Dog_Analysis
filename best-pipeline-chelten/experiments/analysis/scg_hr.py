#!/usr/bin/env python3
"""SCG heart rate by INTENTIONAL conditioning + sliding-window autocorrelation,
fully visually verifiable:

  raw accel -> bandpass 40-150 Hz (S1/S2 heart sounds) -> energy envelope
  -> sliding 3 s autocorrelation -> fundamental peak in the 50-200 bpm lag band.

Why autocorrelation, not peak-picking: the envelope is a clean but doubled
(S1+S2) pulse train; per-beat peak detection trips on S2 and motion. The
autocorrelation of a 3 s window locks the BEAT period robustly, and because the
S1->S2 gap (~170 ms) is shorter than the 0.3 s (=200 bpm) minimum lag, the
doublet cannot cause the octave error that broke CORAL here. SQI = autocorr peak
height; low-SQI and ADC-saturated stretches are excluded.

Outputs scg_hr/<dog>.parquet (ts,bpm,sqi) + scg_hr/<dog>_mask.json and, if an ECG
reference exists, a verification overlay PNG.

Usage: scg_hr.py <Dog> [ecg_modality_for_QC]
"""
import sys, os, glob, json
import numpy as np, polars as pl
from scipy import signal as sg
from scipy.ndimage import binary_dilation, median_filter
import datetime
from zoneinfo import ZoneInfo
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
HIVE = os.path.join(os.path.dirname(HERE), "hive")
FIGS = os.path.join(os.path.dirname(HERE), "figs")
TZ = ZoneInfo("America/Chicago"); UTC = datetime.timezone.utc
AXIS = {"Dasty": "c2", "Chelten": "c1", "Chuck": "c1"}
FC_LO, FC_HI = 40.0, 150.0
ENV_LP = 20.0
SEARCH_LO, SEARCH_HI = 50, 200
FSD = 200                       # envelope analysis rate (Hz)
WIN_S, HOP_S = 3.0, 0.25
SQI_THR = 0.30
SAT = 32767


def dn(us):
    return mdates.date2num([datetime.datetime.fromtimestamp(t/1e6, tz=UTC).astimezone(TZ)
                            for t in np.atleast_1d(np.asarray(us, dtype="int64"))])


def main():
    dog = sys.argv[1]
    qc_mod = sys.argv[2] if len(sys.argv) > 2 else None
    ax = AXIS.get(dog, "c2")
    p = glob.glob(f"{HIVE}/username=scg_mwd/device={dog}/stream=45/date=*/data_0.parquet")[0]
    df = pl.read_parquet(p)
    ts = df["ts"].to_numpy().astype("int64")
    fs = float(1e6 / np.median(np.diff(ts)))
    x = df[ax].to_numpy().astype(float)
    print(f"[{dog}] SCG axis={ax} n={len(x):,} fs={fs:.0f}Hz span={(ts[-1]-ts[0])/1e6/60:.0f}min")

    # --- condition: bandpass 40-150 -> rectified energy envelope -> low-pass ---
    b = sg.butter(4, [FC_LO/(fs/2), FC_HI/(fs/2)], btype="band")
    xf = sg.filtfilt(*b, x - x.mean())
    bl = sg.butter(4, ENV_LP/(fs/2), btype="low")
    env = sg.filtfilt(*bl, np.abs(xf))
    railed = np.abs(x) >= 0.98 * SAT

    # --- decimate envelope to FSD for fast autocorrelation ---
    step = max(1, int(round(fs / FSD)))
    env_d = env[::step]; ts_d = ts[::step]; rail_d = railed[::step]; fsd = fs / step

    win = int(WIN_S * fsd); hop = int(HOP_S * fsd)
    lagmin = int(fsd * 60 / SEARCH_HI); lagmax = int(fsd * 60 / SEARCH_LO)
    centers = np.arange(win // 2, len(env_d) - win // 2, hop)
    bpm = np.full(len(centers), np.nan); sqi = np.zeros(len(centers)); satf = np.zeros(len(centers))
    cts = np.empty(len(centers), dtype="int64")
    for i, c in enumerate(centers):
        sl = slice(c - win // 2, c + win // 2)
        seg = env_d[sl].astype(float); seg = seg - seg.mean()
        cts[i] = ts_d[c]; satf[i] = rail_d[sl].mean()
        n = 1 << int(np.ceil(np.log2(2 * len(seg))))
        f = np.fft.rfft(seg, n); ac = np.fft.irfft(f * np.conj(f), n)[:len(seg)]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        pk = lagmin + int(np.argmax(ac[lagmin:lagmax]))
        # octave-DOWN correction: a pulse-train autocorr peaks at T,2T,3T; if a
        # near-as-strong peak sits at ~half the chosen lag, that shorter lag is the
        # true fundamental (we'd locked the 2nd harmonic -> HR too low).
        for _ in range(2):
            hl = pk // 2
            if hl >= lagmin:
                w2 = max(1, int(0.04 * fsd))
                lo2, hi2 = max(lagmin, hl - w2), min(lagmax, hl + w2 + 1)
                if hi2 > lo2:
                    hpk = lo2 + int(np.argmax(ac[lo2:hi2]))
                    if ac[hpk] >= 0.80 * ac[pk]:
                        pk = hpk; continue
            break
        if 1 <= pk < len(ac) - 1:
            y0, y1, y2 = ac[pk-1], ac[pk], ac[pk+1]
            denom = (y0 - 2*y1 + y2)
            off = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        else:
            off = 0.0
        lag = (pk + off) / fsd
        bpm[i] = 60.0 / lag; sqi[i] = ac[pk]

    valid = (sqi >= SQI_THR) & (satf < 0.05) & np.isfinite(bpm)
    # continuity: reject hops whose HR jumps >20% from the local median (octave
    # slips / lone glitches), then lightly median-smooth what remains.
    bpm_s = bpm.copy()
    if valid.sum() > 9:
        idx = np.where(valid)[0]
        med = median_filter(bpm[idx], 9, mode="nearest")
        keep = np.abs(bpm[idx] - med) / np.maximum(med, 1) <= 0.20
        valid[idx[~keep]] = False
        if valid.sum() > 5:
            bpm_s[valid] = median_filter(bpm[valid], 5, mode="nearest")

    outdir = os.path.join(HERE, "scg_hr"); os.makedirs(outdir, exist_ok=True)
    pl.DataFrame({"ts": cts, "bpm": bpm_s, "sqi": sqi, "valid": valid}).write_parquet(
        os.path.join(outdir, f"{dog}.parquet"))
    span = (ts[-1]-ts[0])/1e6
    print(f"  hops={len(centers):,}  valid={100*valid.mean():.0f}%  "
          f"HR median {np.nanmedian(bpm_s[valid]):.0f}bpm (valid)  axis={ax}")

    # ---- QC overlay vs ECG ----
    if qc_mod:
        ef = os.path.join(HERE, "ecg_hr", f"{dog}_{qc_mod}.parquet")
        if os.path.exists(ef):
            e = pl.read_parquet(ef); ets = e["ts"].to_numpy().astype("int64"); ehr = e["bpm"].to_numpy()
            emid = ets; ebh = ehr            # stored Pan-Tompkins beat HR (already cleaned)
            w0, w1 = max(cts[0], ets[0]), min(cts[-1], ets[-1])
            fig, axs = plt.subplots(2, 1, figsize=(16, 9), height_ratios=[1, 1.2])
            # envelope window (30 s near overlap mid)
            mid = (w0 + w1)//2; ws = (ts_d >= mid) & (ts_d < mid + 20_000_000)
            axs[0].plot((ts_d[ws]-ts_d[ws][0])/1e6, env_d[ws], color="darkorange", lw=0.8)
            rs = ets[(ets>=mid)&(ets<mid+20_000_000)]
            for r in rs: axs[0].axvline((r-ts_d[ws][0])/1e6, color="r", lw=0.6, alpha=0.5)
            axs[0].set_title(f"{dog} conditioned SCG envelope (20 s) + ECG R-peaks (red)"); axs[0].set_xlabel("s")
            # HR overlay on overlap
            sel = valid & (cts>=w0) & (cts<=w1)
            axs[1].plot(dn(cts[sel]), bpm_s[sel], ".", ms=3, color="#d62728", label="SCG-CORR HR (autocorr)")
            esel = (emid>=w0)&(emid<=w1)
            axs[1].plot(dn(emid[esel]), ebh[esel], ".", ms=2, color="#1f77b4", alpha=0.5, label=f"ECG ({qc_mod}) beat HR")
            axs[1].set_ylim(40,210); axs[1].set_ylabel("HR (bpm)"); axs[1].legend(markerscale=3)
            axs[1].xaxis.set_major_formatter(DateFormatter("%H:%M", tz=TZ))
            axs[1].set_title("SCG-derived HR vs ECG over the overlap window")
            # quick error on windowed ECG
            yy=np.full(sel.sum(),np.nan); tt=cts[sel]
            for i,t in enumerate(tt):
                m=(emid>=t-2_000_000)&(emid<=t+2_000_000)
                if m.sum()>=2: yy[i]=ebh[m].mean()
            xx=bpm_s[sel]; v=np.isfinite(yy)
            if v.sum()>5:
                d=xx[v]-yy[v]
                axs[1].text(0.02,0.04,f"n={v.sum()} MAE {np.abs(d).mean():.1f} bias {d.mean():+.1f} "
                            f"<=10bpm {100*np.mean(np.abs(d)<=10):.0f}% r={np.corrcoef(xx[v],yy[v])[0,1]:.2f}",
                            transform=axs[1].transAxes, bbox=dict(boxstyle="round",fc="white",alpha=0.8))
                print(f"  QC vs {qc_mod}: n={v.sum()} MAE {np.abs(d).mean():.1f} bias {d.mean():+.1f} "
                      f"<=10bpm {100*np.mean(np.abs(d)<=10):.0f}% r={np.corrcoef(xx[v],yy[v])[0,1]:.2f}")
            fig.tight_layout(); fig.savefig(os.path.join(FIGS,f"scghr_qc_{dog}.png"),dpi=140)
            print("  ->", os.path.join(FIGS,f"scghr_qc_{dog}.png"))


if __name__ == "__main__":
    main()

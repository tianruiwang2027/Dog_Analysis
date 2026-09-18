import pickle
import numpy as np

with open("/tmp/combined_detector.pkl","rb") as f:
    c = pickle.load(f)
with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)

LAG_US = int(7.0e6)  # SCG clock reads ~7s behind ECG/Polar clock; shift SCG timestamps forward to align

def beat_hr(peaks):
    rr = np.diff(peaks)/1e6
    hr = 60.0/rr
    tmid = peaks[:-1] + np.diff(peaks)//2
    return tmid, hr

def smooth(tmid, hr, win_s=3.0):
    win_us = int(win_s*1e6)
    out = np.empty(len(tmid))
    for i,t in enumerate(tmid):
        sel = (tmid>=t-win_us)&(tmid<=t+win_us)
        out[i] = hr[sel].mean()
    return out

def stats(a,b):
    dd=a-b
    return dict(n=int(len(a)), bias=float(dd.mean()), mae=float(np.abs(dd).mean()),
                r=float(np.corrcoef(a,b)[0,1]) if len(a)>2 else float("nan"))

def align_lagcorrected(tsA, hrA, tsB, hrB, lag_us=LAG_US, win_us=2_000_000):
    """tsA/hrA = SCG-derived (shifted +lag_us to match ECG clock). tsB/hrB = ECG (hand or none)."""
    xa,yb=[],[]
    for i in range(len(tsA)):
        t = tsA[i] + lag_us
        sel=(tsB>=t-win_us)&(tsB<=t+win_us)
        if sel.sum()>=1:
            xa.append(hrA[i]); yb.append(hrB[sel].mean())
    return np.array(xa), np.array(yb)

ecg_good = c["ecg_good"]; scg_hand_good = c["scg_hand_good"]
ecg_tmid, ecg_hr = beat_hr(ecg_good); ecg_hr_sm = smooth(ecg_tmid, ecg_hr)
scg_tmid, scg_hr = beat_hr(scg_hand_good); scg_hr_sm = smooth(scg_tmid, scg_hr)

def run_vs_ecg(name, peaks):
    tmid, hr_raw = beat_hr(peaks)
    hr_sm = smooth(tmid, hr_raw)
    xa,yb = align_lagcorrected(tmid, hr_sm, ecg_tmid, ecg_hr_sm)
    s = stats(xa,yb)
    print(f"{name:34s} vs hand ECG (lag+7.0s corrected)  n_peaks={len(peaks):5d} n_pairs={s['n']:5d}  r={s['r']:.3f}  MAE={s['mae']:.2f}  bias={s['bias']:+.2f}")

print("=== LAG-CORRECTED (+7.0s, SCG clock -> ECG clock) vs hand ECG ===")
run_vs_ecg("hand-clicked SCG (gold std)", scg_hand_good)
run_vs_ecg("baseline singles (thr0.3,rfr0.5)", lab["singles_good"])
run_vs_ecg("+ wide-template NCC>=0.5", c["primary"])
run_vs_ecg("+ NCC filter + search-back", c["combined"])

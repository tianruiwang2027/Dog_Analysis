import pickle
import numpy as np

with open("/tmp/combined_detector.pkl","rb") as f:
    c = pickle.load(f)
with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)

SMOOTH_S = 3.0
def beat_hr(peaks):
    rr = np.diff(peaks)/1e6
    hr = 60.0/rr
    tmid = peaks[:-1] + np.diff(peaks)//2
    return tmid, hr

def smooth(tmid, hr, win_s=SMOOTH_S):
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

def align(tsA,hrA,tsB,hrB,win_us=2_000_000):
    xa,yb=[],[]
    for i in range(len(tsA)):
        t=tsA[i]; sel=(tsB>=t-win_us)&(tsB<=t+win_us)
        if sel.sum()>=1:
            xa.append(hrA[i]); yb.append(hrB[sel].mean())
    return np.array(xa), np.array(yb)

ecg_good = c["ecg_good"]; scg_hand_good = c["scg_hand_good"]
ecg_tmid, ecg_hr = beat_hr(ecg_good); ecg_hr_sm = smooth(ecg_tmid, ecg_hr)
scg_tmid, scg_hr = beat_hr(scg_hand_good); scg_hr_sm = smooth(scg_tmid, scg_hr)

def run_vs(name, peaks, ref_tmid, ref_hr_sm, ref_name):
    tmid, hr_raw = beat_hr(peaks)
    hr_sm = smooth(tmid, hr_raw)
    xa,yb = align(tmid, hr_sm, ref_tmid, ref_hr_sm)
    s = stats(xa,yb)
    print(f"{name:34s} vs {ref_name:10s} n_peaks={len(peaks):5d} n_pairs={s['n']:5d}  r={s['r']:.3f}  MAE={s['mae']:.2f}  bias={s['bias']:+.2f}")

print("=== vs hand ECG (the real cross-modality gold-standard target) ===")
run_vs("hand-clicked SCG (gold std)", scg_hand_good, ecg_tmid, ecg_hr_sm, "hand ECG")
run_vs("baseline singles (thr0.3,rfr0.5)", lab["singles_good"], ecg_tmid, ecg_hr_sm, "hand ECG")
run_vs("+ wide-template NCC>=0.5", c["primary"], ecg_tmid, ecg_hr_sm, "hand ECG")
run_vs("+ NCC filter + search-back", c["combined"], ecg_tmid, ecg_hr_sm, "hand ECG")

print("\n=== vs hand-clicked SCG (does the automated pipeline reproduce the human SCG annotator?) ===")
run_vs("baseline singles (thr0.3,rfr0.5)", lab["singles_good"], scg_tmid, scg_hr_sm, "hand SCG")
run_vs("+ wide-template NCC>=0.5", c["primary"], scg_tmid, scg_hr_sm, "hand SCG")
run_vs("+ NCC filter + search-back", c["combined"], scg_tmid, scg_hr_sm, "hand SCG")

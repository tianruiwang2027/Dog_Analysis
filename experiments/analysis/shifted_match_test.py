import pickle
import numpy as np

with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)
with open("/tmp/shifted_detector.pkl","rb") as f:
    sd = pickle.load(f)

hand_good = np.sort(lab["hand_good"])
TOL_US = 100_000

def match(det):
    det = np.sort(det)
    used = np.zeros(len(hand_good), bool)
    tp=0; fp=0
    for s in det:
        idx = np.searchsorted(hand_good, s)
        best=None; bestd=TOL_US+1
        for ci in (idx-1, idx):
            if 0<=ci<len(hand_good) and not used[ci]:
                dd = abs(hand_good[ci]-s)
                if dd<bestd: bestd=dd; best=ci
        if best is not None and bestd<=TOL_US:
            used[best]=True; tp+=1
        else:
            fp+=1
    missed = (~used).sum()
    return tp, fp, missed

tp,fp,missed = match(sd["shifted"])
print(f"AFTER earlier-candidate shift correction (tight +-100ms matching only):")
print(f"  TP={tp}  FP={fp}  missed={missed}  (of {len(hand_good)} hand clicks)")
print(f"  match rate={tp/len(hand_good)*100:.1f}%   FP rate={fp/len(sd['shifted'])*100:.1f}%")

tp0,fp0,missed0 = match(lab["singles_good"])
print(f"\nBEFORE (original singles_good, same tight matching):")
print(f"  TP={tp0}  FP={fp0}  missed={missed0}")

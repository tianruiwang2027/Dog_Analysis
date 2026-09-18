import pickle
import numpy as np

with open("/tmp/combined_detector.pkl","rb") as f:
    c = pickle.load(f)

scg_hand_good = np.sort(c["scg_hand_good"])
TOL_US = 100_000

def match(det):
    det = np.sort(det)
    used = np.zeros(len(scg_hand_good), bool)
    tp = 0; fp = 0
    for s in det:
        idx = np.searchsorted(scg_hand_good, s)
        best=None; bestd=TOL_US+1
        for ci in (idx-1, idx):
            if 0<=ci<len(scg_hand_good) and not used[ci]:
                dd=abs(scg_hand_good[ci]-s)
                if dd<bestd: bestd=dd; best=ci
        if best is not None and bestd<=TOL_US:
            used[best]=True; tp+=1
        else:
            fp+=1
    missed = (~used).sum()
    return tp, fp, missed

for name, det in [("primary (NCC>=0.5)", c["primary"]), ("combined (+searchback)", c["combined"]), ("extra beats only", c["extra"])]:
    tp, fp, missed = match(det)
    print(f"{name:26s} n={len(det):5d}  TP={tp:5d}  FP={fp:4d}  missed(of {len(scg_hand_good)})={missed:4d}  FP_rate={fp/len(det)*100:.1f}%")

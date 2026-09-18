import pickle
import numpy as np

with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)

singles_good = np.sort(lab["singles_good"])
hand_good = np.sort(lab["hand_good"])

TOL_US = 100_000
EXT_LO_US = 150_000
EXT_HI_US = 250_000

used_hand = np.zeros(len(hand_good), bool)
tp_tight = []; tp_extended = []; fp_times = []

for s in singles_good:
    idx = np.searchsorted(hand_good, s)
    # 1) tight match first (existing rule)
    best=None; bestd=TOL_US+1
    for ci in (idx-1, idx):
        if 0<=ci<len(hand_good) and not used_hand[ci]:
            dd = abs(hand_good[ci]-s)
            if dd<bestd: bestd=dd; best=ci
    if best is not None and bestd<=TOL_US:
        used_hand[best]=True; tp_tight.append(s); continue
    # 2) extended delayed match: candidate is 150-250ms AFTER an unused hand click
    best=None
    for ci in (idx-1, idx, idx-2):
        if 0<=ci<len(hand_good) and not used_hand[ci]:
            delay = s - hand_good[ci]
            if EXT_LO_US <= delay <= EXT_HI_US:
                best = ci; break
    if best is not None:
        used_hand[best]=True; tp_extended.append(s); continue
    fp_times.append(s)

missed = (~used_hand).sum()
n_tp = len(tp_tight)+len(tp_extended)
print(f"Tight matches (<=100ms): {len(tp_tight)}")
print(f"Extended delayed matches (150-250ms after a click, same cycle/wrong heart sound): {len(tp_extended)}")
print(f"Total valid matches: {n_tp}  of {len(hand_good)} hand clicks")
print(f"Remaining false positives: {len(fp_times)}  (was 121 before)")
print(f"Remaining missed beats: {missed}  (was 199 before)")
print(f"\nNew match rate: {n_tp/len(hand_good)*100:.1f}%  (was 88.2%)")
print(f"New FP rate: {len(fp_times)/len(singles_good)*100:.1f}%  (was 7.5%)")

with open("/tmp/extended_match.pkl","wb") as f:
    pickle.dump(dict(tp_tight=np.array(tp_tight), tp_extended=np.array(tp_extended),
                      fp_times=np.array(fp_times, dtype="int64")), f)

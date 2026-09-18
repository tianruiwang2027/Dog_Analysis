import pickle
import numpy as np

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)

cand_scg = np.sort(d["cand_scg"])  # fine-grained (80ms refractory) raw candidates, thr=0.3
t0 = int(lab["t0"]); t1 = int(lab["t1"])

def apply_shift(detections, cand_pool, lo_us=150_000, hi_us=250_000, min_gap_us=300_000):
    detections = np.sort(detections)
    out = []
    prev_final = -10**18
    n_shifted = 0
    for t in detections:
        # look for a fine candidate strictly between (t-hi_us, t-lo_us)
        a = np.searchsorted(cand_pool, t - hi_us)
        b = np.searchsorted(cand_pool, t - lo_us + 1)
        cands = cand_pool[a:b]
        new_t = t
        if len(cands) > 0:
            # take the earliest such candidate (closest to being "the" S1)
            candidate = cands[0]
            if candidate - prev_final >= min_gap_us:
                new_t = candidate
                n_shifted += 1
        out.append(new_t)
        prev_final = new_t
    return np.array(sorted(out), dtype="int64"), n_shifted

singles_good = np.sort(lab["singles_good"])
shifted, n_shifted = apply_shift(singles_good, cand_scg)
print(f"n singles: {len(singles_good)}   n shifted to an earlier candidate: {n_shifted}")

with open("/tmp/shifted_detector.pkl","wb") as f:
    pickle.dump(dict(shifted=shifted, n_shifted=n_shifted, t0=t0, t1=t1), f)

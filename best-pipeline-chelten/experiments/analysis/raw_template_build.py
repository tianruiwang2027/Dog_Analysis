import pickle
import numpy as np

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)

ts_s = d["ts_s"]; xf_s = d["xf_s"]; fs_s = d["fs_s"]
tp_times = np.sort(lab["tp_times"])

HALF_MS = 300  # +-300ms raw window, native 2000Hz
HALF_N = int(HALF_MS/1000*fs_s)
LAG_MAX_MS = 40
LAG_MAX_N = int(LAG_MAX_MS/1000*fs_s)

def snippet(t, half_n=HALF_N):
    i = np.searchsorted(ts_s, t)
    if i-half_n-LAG_MAX_N < 0 or i+half_n+LAG_MAX_N+1 > len(xf_s):
        return None
    # extract a slightly wider window so we can shift within +-LAG_MAX
    return xf_s[i-half_n-LAG_MAX_N : i+half_n+LAG_MAX_N+1].copy()

# training set: first half of TPs (time-ordered), held out for evaluation later
n_train = len(tp_times)//2
train_times = tp_times[:n_train]

wide = [snippet(t) for t in train_times]
wide = [w for w in wide if w is not None]
wide = np.array(wide)
print("n training exemplars:", len(wide))
L = 2*HALF_N+1

def extract_center(w, shift):
    c = len(w)//2
    return w[c-HALF_N+shift : c+HALF_N+1+shift]

# init reference: center-aligned (no shift) mean
ref = np.mean([extract_center(w,0) for w in wide], axis=0)
ref = ref - ref.mean()

N_ITERS = 4
for it in range(N_ITERS):
    aligned = []
    for w in wide:
        best_lag = 0; best_score = -1e18
        for lag in range(-LAG_MAX_N, LAG_MAX_N+1):
            seg = extract_center(w, lag)
            seg0 = seg - seg.mean()
            denom = np.linalg.norm(seg0)*np.linalg.norm(ref)
            score = (seg0*ref).sum()/denom if denom>1e-9 else -1e18
            if score > best_score:
                best_score = score; best_lag = lag
        aligned.append(extract_center(w, best_lag))
    aligned = np.array(aligned)
    new_ref = aligned.mean(axis=0)
    new_ref = new_ref - new_ref.mean()
    shift_amt = np.linalg.norm(new_ref-ref)
    ref = new_ref
    print(f"iter {it}: template updated, ||delta||={shift_amt:.3f}, template peak-to-peak={ref.max()-ref.min():.3f}")

with open("/tmp/raw_template.pkl","wb") as f:
    pickle.dump(dict(template=ref, HALF_N=HALF_N, fs_s=fs_s, LAG_MAX_N=LAG_MAX_N), f)
print("saved raw template, length", len(ref))

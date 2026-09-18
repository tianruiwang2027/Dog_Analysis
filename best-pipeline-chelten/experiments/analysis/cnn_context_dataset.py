import pickle
import numpy as np

# ---- add one context feature per candidate: the background RMS of the envelope in the
# region SURROUNDING the candidate but EXCLUDING the beat shape itself (the same ±400ms
# core the shape-snippet already sees). This directly tests: "a real beat sits in an
# otherwise flat/quiet neighborhood; a motion-driven false positive sits in a neighborhood
# that's noisy even away from the peak itself." ----

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; fsd_s = d["fsd_s"]

with open("/tmp/cnn_dataset.pkl","rb") as f:
    D = pickle.load(f)
t = D["t"]; label = D["label"]; snips = D["snips"]; HALF_N = D["HALF_N"]

CORE_S = 0.40      # matches the existing snippet half-window -- excluded from the context calc
FLANK_S = 2.0       # look out to +-2.0s for background level

def bg_rms(tc):
    lo1, hi1 = tc-FLANK_S*1e6, tc-CORE_S*1e6           # left flank
    lo2, hi2 = tc+CORE_S*1e6,  tc+FLANK_S*1e6           # right flank
    i1a, i1b = np.searchsorted(ts_sd, lo1), np.searchsorted(ts_sd, hi1)
    i2a, i2b = np.searchsorted(ts_sd, lo2), np.searchsorted(ts_sd, hi2)
    seg = np.concatenate([env_sd[i1a:i1b], env_sd[i2a:i2b]])
    if len(seg) < 20:
        return np.nan
    return float(np.sqrt(np.mean(seg.astype(np.float64)**2)))

print(f"computing background RMS for {len(t)} candidates (flanks +-{FLANK_S}s, excluding core +-{CORE_S}s)...")
bg = np.array([bg_rms(tc) for tc in t], dtype=np.float64)
n_nan = np.isnan(bg).sum()
print(f"  {n_nan} candidates too close to session edges (dropped)")

keep = ~np.isnan(bg)
t2, label2, snips2, bg2 = t[keep], label[keep], snips[keep], bg[keep]

print(f"final n={len(t2)}  positives={int(label2.sum())}  negatives={int((1-label2).sum())}")
print(f"background RMS: positives mean={bg2[label2==1].mean():.3f}  negatives mean={bg2[label2==0].mean():.3f}")

with open("/tmp/cnn_context_dataset.pkl","wb") as f:
    pickle.dump(dict(t=t2, label=label2, snips=snips2, bg=bg2, HALF_N=HALF_N, fsd_s=fsd_s,
                      CORE_S=CORE_S, FLANK_S=FLANK_S), f)
print("saved /tmp/cnn_context_dataset.pkl")

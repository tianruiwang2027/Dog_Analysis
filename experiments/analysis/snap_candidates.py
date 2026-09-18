import pickle, sqlite3
import numpy as np
from scipy.signal import find_peaks

# ---------- Snap step: for every primary-detector candidate, check whether there's a
# comparably-large (or larger) envelope peak 100-300ms BEFORE it. If so, that earlier
# peak is almost certainly the true dominant S1 and the original candidate was really
# sitting on S2 -- re-anchor the candidate there before doing anything else (matching,
# labeling, snippet extraction). ----------

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; sharp_s = d["sharp_s"]; fsd_s = d["fsd_s"]
t0 = int(d["t0"]); t1 = int(d["t1"])

PRIMARY_THR = 0.3; PRIMARY_REFRACT_S = 0.50
pk_idx, _ = find_peaks(sharp_s, height=PRIMARY_THR, distance=max(1,int(PRIMARY_REFRACT_S*fsd_s)))
cand_t = ts_sd[pk_idx]
cand_t = cand_t[(cand_t>=t0)&(cand_t<=t1)]
print(f"n candidates (primary detector, whole session): {len(cand_t)}")

SEARCH_LO_US, SEARCH_HI_US = 100_000, 300_000
RATIO_THRESH = 0.9   # backward peak must be >= 90% of the candidate's own envelope value to trigger a snap

def snap_one(tc):
    i_c = np.searchsorted(ts_sd, tc)
    if i_c >= len(env_sd): return tc, False
    cand_val = env_sd[i_c]
    lo, hi = tc-SEARCH_HI_US, tc-SEARCH_LO_US
    i_lo, i_hi = np.searchsorted(ts_sd, lo), np.searchsorted(ts_sd, hi)
    if i_hi <= i_lo: return tc, False
    seg = env_sd[i_lo:i_hi]
    j = np.argmax(seg)
    back_val = seg[j]
    if back_val >= cand_val * RATIO_THRESH:
        return ts_sd[i_lo+j], True
    return tc, False

snapped_t = np.empty_like(cand_t)
was_snapped = np.zeros(len(cand_t), dtype=bool)
for i,tc in enumerate(cand_t):
    st, sn = snap_one(tc)
    snapped_t[i] = st
    was_snapped[i] = sn

# after snapping, re-sort and de-duplicate (two originally-separate candidates could snap
# to nearly the same true S1, e.g. one lands ON S1 already [no snap needed] and the very
# next candidate, which is that same beat's S2, snaps backward onto the same S1)
order = np.argsort(snapped_t)
snapped_t_sorted = snapped_t[order]
was_snapped_sorted = was_snapped[order]
keep = np.ones(len(snapped_t_sorted), bool)
MERGE_TOL_US = 50_000
last = -1e18
for i,t in enumerate(snapped_t_sorted):
    if t - last < MERGE_TOL_US:
        keep[i] = False   # duplicate of previous, drop
    else:
        last = t
final_t = snapped_t_sorted[keep]
final_was_snapped = was_snapped_sorted[keep]

print(f"n candidates snapped: {was_snapped.sum()} / {len(cand_t)}  ({was_snapped.mean()*100:.1f}%)")
print(f"n duplicate merges after snapping: {(~keep).sum()}")
print(f"final candidate count after snap+merge: {len(final_t)}")

with open("/tmp/snapped_candidates.pkl","wb") as f:
    pickle.dump(dict(cand_t_orig=cand_t, final_t=final_t, final_was_snapped=final_was_snapped,
                      SEARCH_LO_US=SEARCH_LO_US, SEARCH_HI_US=SEARCH_HI_US, RATIO_THRESH=RATIO_THRESH), f)
print("saved /tmp/snapped_candidates.pkl")

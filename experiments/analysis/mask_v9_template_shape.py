import pickle, sqlite3, datetime
import numpy as np
from scipy.signal import find_peaks

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; fsd_s = d["fsd_s"]
with open("/tmp/env_template_wide.pkl","rb") as f:
    tpl = pickle.load(f)
template = tpl["template"]; HALF_N = tpl["HALF_N"]
template_energy = np.sqrt((template**2).sum())

def ncc_score(t):
    i = np.searchsorted(ts_sd, t)
    if i-HALF_N < 0 or i+HALF_N+1 > len(env_sd):
        return np.nan
    s = env_sd[i-HALF_N:i+HALF_N+1]
    s = s - s.mean()
    denom = np.sqrt((s**2).sum()) * template_energy
    if denom < 1e-12: return np.nan
    return float((s*template).sum() / denom)

# KEY DESIGN CHOICE: candidate generation is amplitude-PERMISSIVE (low bar), because the point of
# this feature is to be amplitude-independent -- a weak-but-real beat should still be found as a
# candidate, then NCC (normalized -> insensitive to absolute amplitude) decides if its SHAPE matches.
RELAX_REFRACT_S = 0.30
# no height/prominence gate at all -- pure local-maxima + refractory spacing, so a weak-but-real
# beat is just as much a "candidate" as a loud one. NCC (shape, normalized) does all the filtering.
pk_idx, _ = find_peaks(env_sd, distance=max(1,int(RELAX_REFRACT_S*fsd_s)))
cand_t = ts_sd[pk_idx]
print(f"n candidates (relaxed, amplitude-agnostic pool): {len(cand_t)}")

cand_ncc = np.array([ncc_score(t) for t in cand_t])
valid = ~np.isnan(cand_ncc)
cand_t = cand_t[valid]; cand_ncc = cand_ncc[valid]
print(f"n candidates with valid NCC: {len(cand_t)}")

with open("/tmp/mask_v9_candidates.pkl","wb") as f:
    pickle.dump(dict(cand_t=cand_t, cand_ncc=cand_ncc), f)
print("saved /tmp/mask_v9_candidates.pkl")

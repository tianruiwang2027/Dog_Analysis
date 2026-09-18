import pickle, datetime
import numpy as np

with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)
with open("/tmp/combined_detector.pkl","rb") as f:
    c = pickle.load(f)
with open("/tmp/illustrate_15missed.pkl","rb") as f:
    I = pickle.load(f)
with open("/tmp/env_template_wide.pkl","rb") as f:
    tpl = pickle.load(f)
with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)

ts_sd = d["ts_sd"]; env_sd = d["env_sd"]
template = tpl["template"]; HALF_N = tpl["HALF_N"]
template_energy = np.sqrt((template**2).sum())
def ncc_score(t):
    i = np.searchsorted(ts_sd, t)
    if i-HALF_N < 0 or i+HALF_N+1 > len(env_sd): return np.nan
    s = env_sd[i-HALF_N:i+HALF_N+1]; s = s - s.mean()
    denom = np.sqrt((s**2).sum()) * template_energy
    if denom < 1e-12: return np.nan
    return float((s*template).sum() / denom)

singles_good = np.sort(lab["singles_good"])
det_primary = set(np.sort(c["primary"]).tolist())

UTC = datetime.timezone.utc
def fmt(us):
    return datetime.datetime.fromtimestamp(us/1e6, tz=UTC).strftime("%H:%M:%S.%f")[:-3]

ta = I["ta"]; bad_idx = I["bad_idx"]
pts = np.sort(ta[bad_idx])
WIN_US = int(2.5e6)

for t_center in pts:
    lo, hi = t_center-WIN_US, t_center+WIN_US
    cands = singles_good[(singles_good>=lo)&(singles_good<=hi)]
    removed = [t for t in cands if t not in det_primary]
    print(f"=== {fmt(t_center)} ===  n_pre-filter-candidates={len(cands)}  n_removed_by_template={len(removed)}")
    for t in cands:
        score = ncc_score(t)
        status = "KEPT" if t in det_primary else "REMOVED(NCC<0.5)"
        print(f"    {fmt(t)}  ncc={score:.3f}  {status}")
    print()

import pickle, sqlite3
import numpy as np
from scipy.signal import find_peaks

# ---------- 3-class dataset: S1 (the real click point), S2 (the second heart sound --
# a real cardiac event the primary detector sometimes locks onto INSTEAD of S1), and
# noise (neither). Built from the same candidate pool as before, no new hand labeling. ----------

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; sharp_s = d["sharp_s"]; fsd_s = d["fsd_s"]
t0 = int(d["t0"]); t1 = int(d["t1"])

PRIMARY_THR = 0.3; PRIMARY_REFRACT_S = 0.50
pk_idx, _ = find_peaks(sharp_s, height=PRIMARY_THR, distance=max(1,int(PRIMARY_REFRACT_S*fsd_s)))
cand_t = ts_sd[pk_idx]
cand_t = cand_t[(cand_t>=t0)&(cand_t<=t1)]

def load_peaks(path):
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = ''")
    return np.array([r[0] for r in cur.fetchall()], dtype="int64")
scg_pk = load_peaks("/tmp/annotation_chelten_scg2.db")

TOL_US = 150_000
S2_LO_US, S2_HI_US = 140_000, 280_000

order = np.argsort(cand_t)
cand_t_sorted = cand_t[order]
used_hand = np.zeros(len(scg_pk), bool)
cls = np.full(len(cand_t_sorted), -1, dtype=int)   # -1=unset, 0=S1, 1=S2, 2=noise

# pass 1: match S1 (same greedy 1:1 as before)
for i, s in enumerate(cand_t_sorted):
    idx = np.searchsorted(scg_pk, s)
    best = None; bestd = TOL_US+1
    for cidx in (idx-1, idx):
        if 0 <= cidx < len(scg_pk) and not used_hand[cidx]:
            dd = abs(scg_pk[cidx]-s)
            if dd < bestd:
                bestd = dd; best = cidx
    if best is not None and bestd <= TOL_US:
        used_hand[best] = True
        cls[i] = 0

# pass 2: among the rest, label S2 if it sits 140-280ms AFTER the nearest hand click
for i, s in enumerate(cand_t_sorted):
    if cls[i] == 0: continue
    idx = np.searchsorted(scg_pk, s)
    is_s2 = False
    if idx > 0:
        off = s - scg_pk[idx-1]
        if S2_LO_US <= off <= S2_HI_US:
            is_s2 = True
    cls[i] = 1 if is_s2 else 2

n_s1 = (cls==0).sum(); n_s2 = (cls==1).sum(); n_noise = (cls==2).sum()
print(f"S1 (real beat): {n_s1}   S2 (second sound): {n_s2}   noise: {n_noise}")

HALF_S = 0.40
HALF_N = int(HALF_S*fsd_s)
def snippet(t):
    i = np.searchsorted(ts_sd, t)
    if i-HALF_N < 0 or i+HALF_N+1 > len(env_sd): return None
    return env_sd[i-HALF_N:i+HALF_N+1].copy()

snips = []; keep_t = []; keep_cls = []
for t, c in zip(cand_t_sorted, cls):
    s = snippet(t)
    if s is None: continue
    snips.append(s); keep_t.append(t); keep_cls.append(c)

snips = np.array(snips, dtype=np.float32)
keep_t = np.array(keep_t, dtype="int64")
keep_cls = np.array(keep_cls, dtype=np.int64)
print(f"final: {len(keep_t)} candidates  S1={np.sum(keep_cls==0)} S2={np.sum(keep_cls==1)} noise={np.sum(keep_cls==2)}")

with open("/tmp/cnn3_dataset.pkl","wb") as f:
    pickle.dump(dict(t=keep_t, cls=keep_cls, snips=snips, HALF_N=HALF_N, fsd_s=fsd_s, t0=t0, t1=t1), f)
print("saved /tmp/cnn3_dataset.pkl")

import pickle, sqlite3
import numpy as np

# ---------- Build a training set for a small S2-vs-noise classifier, using candidates
# REJECTED by the main (already-validated) CNN. The label comes for free from data we
# already have: a rejected candidate sitting 140-280ms AFTER an SCG hand click (which
# marks S1) is almost certainly the real S2 sound of that same beat, not noise -- this
# is exactly the mechanism that the earlier envelope diagnostic (chelten_low_score_envelope_view.png)
# uncovered by hand. Rejected candidates far (>400ms) from any preceding click are clean
# noise negatives. Everything in between is ambiguous and dropped. ----------

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; fsd_s = d["fsd_s"]

with open("/tmp/full_session_scores_all_models.pkl","rb") as f:
    S = pickle.load(f)
cand_t = S["cand_t"]; scores_main = S["scores_relabel"]

with open("/tmp/three_model_r_compare.pkl","rb") as f:
    prev = pickle.load(f)
CUTOFF_MAIN = prev["CNN-SCG+ECG (envelope, relabeled)"]["cutoff"]
print(f"main CNN cutoff (best operating point): {CUTOFF_MAIN:.2f}")

rejected_t = np.sort(cand_t[scores_main < CUTOFF_MAIN])
print(f"n rejected by main CNN: {len(rejected_t)} / {len(cand_t)}")

def load_peaks(path):
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = ''")
    return np.array([r[0] for r in cur.fetchall()], dtype="int64")
scg_pk = np.sort(load_peaks("/tmp/annotation_chelten_scg2.db"))

S2_LO_US, S2_HI_US = 140_000, 280_000
NOISE_MIN_US = 400_000

label = np.full(len(rejected_t), -1, dtype=int)  # -1 = ambiguous, drop
for i, t in enumerate(rejected_t):
    idx = np.searchsorted(scg_pk, t)
    prev_idx = idx - 1
    if prev_idx >= 0:
        delta = t - scg_pk[prev_idx]
    else:
        delta = np.inf
    if S2_LO_US <= delta <= S2_HI_US:
        label[i] = 1
    elif delta > NOISE_MIN_US:
        label[i] = 0
    # else: ambiguous, leave as -1

n_s2 = (label == 1).sum(); n_noise = (label == 0).sum(); n_drop = (label == -1).sum()
print(f"S2-labeled (positive): {n_s2}")
print(f"noise-labeled (negative): {n_noise}")
print(f"dropped as ambiguous: {n_drop}")

keep = label >= 0
t2 = rejected_t[keep]; label2 = label[keep]

HALF_S = 0.40
HALF_N = int(HALF_S*fsd_s)
def snippet(t):
    i = np.searchsorted(ts_sd, t)
    if i-HALF_N < 0 or i+HALF_N+1 > len(env_sd):
        return None
    return env_sd[i-HALF_N:i+HALF_N+1].copy()

snips = []; keep_t = []; keep_label = []
for t, lab in zip(t2, label2):
    s = snippet(t)
    if s is None: continue
    snips.append(s); keep_t.append(t); keep_label.append(lab)

snips = np.array(snips, dtype=np.float32)
keep_t = np.array(keep_t, dtype="int64")
keep_label = np.array(keep_label, dtype=np.float32)
print(f"final S2-classifier dataset: n={len(keep_t)}  S2={int(keep_label.sum())}  noise={int((1-keep_label).sum())}")

with open("/tmp/s2_dataset.pkl","wb") as f:
    pickle.dump(dict(t=keep_t, label=keep_label, snips=snips, HALF_N=HALF_N, fsd_s=fsd_s,
                      CUTOFF_MAIN=CUTOFF_MAIN, S2_LO_US=S2_LO_US, S2_HI_US=S2_HI_US, NOISE_MIN_US=NOISE_MIN_US), f)
print("saved /tmp/s2_dataset.pkl")

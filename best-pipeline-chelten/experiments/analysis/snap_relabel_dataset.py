import pickle, sqlite3
import numpy as np

# ---------- Build the ECG-corroborated relabeled dataset using the SNAPPED candidate
# positions (final_t from snap_candidates.py) instead of the original primary-detector
# positions. Same matching/relabeling logic as cnn_relabel_ecg.py, just applied to the
# snapped times, and snippets are freshly extracted centered on the SNAPPED position. ----------

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; fsd_s = d["fsd_s"]
t0 = int(d["t0"]); t1 = int(d["t1"])

with open("/tmp/snapped_candidates.pkl","rb") as f:
    SN = pickle.load(f)
cand_t = SN["final_t"]
was_snapped = SN["final_was_snapped"]
print(f"n snapped candidates: {len(cand_t)}  (snapped={was_snapped.sum()})")

def load_peaks(path):
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = ''")
    return np.array([r[0] for r in cur.fetchall()], dtype="int64")

scg_pk = load_peaks("/tmp/annotation_chelten_scg2.db")
ecg_pk = load_peaks("/tmp/annotation_chelten_ecg2.db")
LAG_US = 7_000_000
ecg_shift = ecg_pk - LAG_US   # CORRECTED sign

# step 1: SCG match (same TOL_US=150ms, greedy 1:1) on the snapped positions
TOL_US = 150_000
order = np.argsort(cand_t)
cand_t_sorted = cand_t[order]
was_snapped_sorted = was_snapped[order]
used_hand = np.zeros(len(scg_pk), bool)
label_old = np.zeros(len(cand_t_sorted), dtype=int)
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
        label_old[i] = 1

n_pos_scg = label_old.sum()
print(f"SCG-matched (positive): {n_pos_scg}   unmatched: {len(label_old)-n_pos_scg}")

# step 2: ECG corroboration rule
RESCUE_US = 50_000
CLEAN_NEG_US = 250_000
d_ecg = np.array([np.min(np.abs(ecg_shift-tc)) for tc in cand_t_sorted])

label_new = np.full(len(cand_t_sorted), -1, dtype=int)
n_rescued = 0; n_clean_neg = 0; n_kept_pos = 0; n_dropped = 0
for i in range(len(cand_t_sorted)):
    if label_old[i] == 1:
        label_new[i] = 1; n_kept_pos += 1
    elif d_ecg[i] <= RESCUE_US:
        label_new[i] = 1; n_rescued += 1
    elif d_ecg[i] > CLEAN_NEG_US:
        label_new[i] = 0; n_clean_neg += 1
    else:
        n_dropped += 1

keep = label_new >= 0
t2 = cand_t_sorted[keep]; label2 = label_new[keep]; snap2 = was_snapped_sorted[keep]
print(f"kept as positive (SCG-matched): {n_kept_pos}")
print(f"ECG-rescued (was negative, now positive): {n_rescued}")
print(f"clean negative (unmatched, ECG also >250ms away): {n_clean_neg}")
print(f"dropped as ambiguous: {n_dropped}")
print(f"final relabeled+snapped dataset: n={len(t2)}  positive={int(label2.sum())}  negative={int((1-label2).sum())}")

# step 3: extract +-400ms envelope snippets centered on the SNAPPED positions
HALF_S = 0.40
HALF_N = int(HALF_S*fsd_s)

def snippet(t):
    i = np.searchsorted(ts_sd, t)
    if i-HALF_N < 0 or i+HALF_N+1 > len(env_sd):
        return None
    return env_sd[i-HALF_N:i+HALF_N+1].copy()

snips = []; keep_t = []; keep_label = []; keep_snap = []
for t, lab, sn in zip(t2, label2, snap2):
    s = snippet(t)
    if s is None: continue
    snips.append(s); keep_t.append(t); keep_label.append(lab); keep_snap.append(sn)

snips = np.array(snips, dtype=np.float32)
keep_t = np.array(keep_t, dtype="int64")
keep_label = np.array(keep_label, dtype=np.float32)
keep_snap = np.array(keep_snap, dtype=bool)
print(f"final dataset after snippet extraction: n={len(keep_t)}  "
      f"positive={int(keep_label.sum())}  negative={int((1-keep_label).sum())}  "
      f"(of which {keep_snap.sum()} were snap-recovered candidates)")

with open("/tmp/cnn_dataset_snap_relabel.pkl","wb") as f:
    pickle.dump(dict(t=keep_t, label=keep_label, snips=snips, HALF_N=HALF_N, fsd_s=fsd_s,
                      was_snapped=keep_snap, n_rescued=n_rescued, n_clean_neg=n_clean_neg, n_dropped=n_dropped), f)
print("saved /tmp/cnn_dataset_snap_relabel.pkl")

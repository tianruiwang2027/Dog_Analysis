import pickle, datetime
import numpy as np

with open("/tmp/reconcile_check_cache.pkl","rb") as f:
    C = pickle.load(f)
diffs, ta_scg, err = C["diffs"], C["ta_scg"], C["err"]

with open("/tmp/jitter_gold_pairs.pkl","rb") as f:
    J = pickle.load(f)
t0, t1, LAG_US = J["t0"], J["t1"], J["LAG_US"]
pairs = J["pairs"]
matched_scg_times = set(int(a) for a in pairs[:,0])

def load_peaks(path):
    import sqlite3
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = '' ORDER BY timestamp")
    return np.array([r[0] for r in cur.fetchall()], dtype="int64")

ecg_pk = load_peaks("/tmp/annotation_chelten_ecg2.db")
scg_pk = load_peaks("/tmp/annotation_chelten_scg2.db")
ecg_on_scg = ecg_pk - LAG_US

unmatched_scg = np.array(sorted(t for t in scg_pk if int(t) not in matched_scg_times))
matched_ecg_times = set(int(b) for b in pairs[:,1])
unmatched_ecg = np.array(sorted(t for t in ecg_on_scg if int(t) not in matched_ecg_times))
anomaly_locs = np.sort(np.concatenate([unmatched_scg, unmatched_ecg]))

nonzero_idx = np.where(diffs!=0)[0]
WIN = int(2.5e6)

near_true_anomaly = 0
boundary_only = 0
for i in nonzero_idx:
    t = ta_scg[i]
    if np.any(np.abs(anomaly_locs - t) <= WIN):
        near_true_anomaly += 1
    else:
        boundary_only += 1

print(f"Of the 202 'window-count mismatch' points:")
print(f"  within 2.5s of a TRUE individual click anomaly:  {near_true_anomaly}")
print(f"  NOT near any true anomaly (pure window-edge/boundary artifact): {boundary_only}")

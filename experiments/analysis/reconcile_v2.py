import pickle, sqlite3, datetime
import numpy as np

with open("/tmp/jitter_gold_pairs.pkl","rb") as f:
    J = pickle.load(f)
t0, t1, LAG_US = J["t0"], J["t1"], J["LAG_US"]

with open("/tmp/final_4way_compare.pkl","rb") as f:
    D = pickle.load(f)
xa_scg, yb_scg, ta_scg = D["xa_scg"], D["yb_scg"], D["ta_scg"]
err = xa_scg - yb_scg

def load_peaks(path):
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = '' ORDER BY timestamp")
    return np.array([r[0] for r in cur.fetchall()], dtype="int64")

ecg_pk = load_peaks("/tmp/annotation_chelten_ecg2.db")
scg_pk = load_peaks("/tmp/annotation_chelten_scg2.db")
ecg_on_scg = ecg_pk - LAG_US

# EXACT original methodology from last time: raw full click lists (not good-restricted), +-2.5s window
WIN = int(2.5e6)
diffs = np.empty(len(ta_scg))
for i,t in enumerate(ta_scg):
    n_scg = ((scg_pk>=t-WIN)&(scg_pk<=t+WIN)).sum()
    n_ecg = ((ecg_on_scg>=t-WIN)&(ecg_on_scg<=t+WIN)).sum()
    diffs[i] = n_scg - n_ecg

nonzero = diffs != 0
print(f"REPRODUCED original check: {nonzero.sum()}/{len(diffs)} points with nonzero window-count diff ({100*nonzero.mean():.1f}%)")

# now let's look at a few of these "diff!=0 but not near a true anomaly" points directly
with open("/tmp/reconcile_check_cache.pkl","wb") as f:
    pickle.dump(dict(diffs=diffs, ta_scg=ta_scg, err=err), f)

# distribution of |diff|
for dv in sorted(set(diffs.astype(int).tolist())):
    if abs(dv)>4: continue
    m = diffs.astype(int)==dv
    print(f"  diff={dv:+d}: n={m.sum()}")

import pickle, datetime, sqlite3
import numpy as np

with open("/tmp/final_4way_compare.pkl","rb") as f:
    D = pickle.load(f)

LAG_US = D["LAG_US"]
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

WIN = int(2.5e6)
diffs = np.empty(len(ta_scg))
for i,t in enumerate(ta_scg):
    n_scg = ((scg_pk>=t-WIN)&(scg_pk<=t+WIN)).sum()
    n_ecg = ((ecg_on_scg>=t-WIN)&(ecg_on_scg<=t+WIN)).sum()
    diffs[i] = n_scg - n_ecg

r_clickdiff_err = np.corrcoef(diffs, err)[0,1]
print("correlation between (n_SCG_clicks - n_ECG_clicks) in a local window and err:", r_clickdiff_err)
print("R^2 (variance explained):", r_clickdiff_err**2)

# how many points have zero click-count diff, and what's r/MAE restricted to those?
zero_diff = diffs==0
print(f"\npoints with matched click counts (diff=0): {zero_diff.sum()} / {len(diffs)} ({100*zero_diff.mean():.1f}%)")
print("  r restricted to those:", np.corrcoef(xa_scg[zero_diff], yb_scg[zero_diff])[0,1])
print("  MAE restricted to those:", np.abs(err[zero_diff]).mean())
print("\nfull dataset r:", np.corrcoef(xa_scg,yb_scg)[0,1], " MAE:", np.abs(err).mean())

print("\nerr stats by diff bucket:")
for dv in sorted(set(diffs.astype(int).tolist())):
    if abs(dv)>4: continue
    m = diffs.astype(int)==dv
    if m.sum()<3: continue
    print(f"  diff={dv:+d}  n={m.sum():4d}  mean_err={err[m].mean():+6.2f}  std={err[m].std():5.2f}")

# overall click totals
print("\ntotal ECG hand clicks:", len(ecg_pk), " total SCG hand clicks:", len(scg_pk))

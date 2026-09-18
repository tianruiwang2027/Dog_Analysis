import pickle, datetime, sqlite3
import numpy as np

with open("/tmp/reconcile_check_cache.pkl","rb") as f:
    C = pickle.load(f)
diffs, ta_scg, err = C["diffs"], C["ta_scg"], C["err"]

with open("/tmp/jitter_gold_pairs.pkl","rb") as f:
    J = pickle.load(f)
t0, t1, LAG_US = J["t0"], J["t1"], J["LAG_US"]

def load_peaks(path):
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = '' ORDER BY timestamp")
    return np.array([r[0] for r in cur.fetchall()], dtype="int64")

ecg_pk = load_peaks("/tmp/annotation_chelten_ecg2.db")
scg_pk = load_peaks("/tmp/annotation_chelten_scg2.db")
ecg_on_scg = ecg_pk - LAG_US

# the 24 true anomaly locations from before
anomaly_locs = np.array([
1782495997308,1782496017062,1782496040992,1782496078257,1782496471679,1782496541176,
1782496798294,1782497046167,1782497096647,1782497096985,1782497971814,1782497996389,
1782497996995,1782497997582,1782497998306,1782497999853,1782498000878,1782498001583,
1782498002783,1782498003856,1782498004493,1782498005866,1782498006985,1782498007696], dtype="float64")
# NOTE: need actual int64 not this pasted approx -- instead recompute directly below

UTC = datetime.timezone.utc
def fmt(us): return datetime.datetime.fromtimestamp(us/1e6, tz=UTC).strftime("%H:%M:%S.%f")[:-3]

nonzero_idx = np.where(diffs!=0)[0]
WIN = int(2.5e6)

print("Checking a sample of 'diff != 0' points to see WHY the count differs (boundary effect vs real gap):\n")
for i in nonzero_idx[:8]:
    t = ta_scg[i]
    scg_near = scg_pk[(scg_pk>=t-WIN-500000)&(scg_pk<=t+WIN+500000)]
    ecg_near = ecg_on_scg[(ecg_on_scg>=t-WIN-500000)&(ecg_on_scg<=t+WIN+500000)]
    print(f"=== point t={fmt(t)}  diff={diffs[i]:+.0f}  err={err[i]:+.1f} ===")
    print(f"   window = [{fmt(t-WIN)}, {fmt(t+WIN)}]")
    # show clicks right AT the boundary (within 150ms of either edge)
    for edge_name, edge in [("LEFT edge", t-WIN), ("RIGHT edge", t+WIN)]:
        s_near_edge = scg_near[np.abs(scg_near-edge)<150000]
        e_near_edge = ecg_near[np.abs(ecg_near-edge)<150000]
        if len(s_near_edge) or len(e_near_edge):
            print(f"   near {edge_name} ({fmt(edge)}): SCG clicks={[fmt(x) for x in s_near_edge]}  ECG clicks={[fmt(x) for x in e_near_edge]}")
    print()

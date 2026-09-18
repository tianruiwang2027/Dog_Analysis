import pickle
import numpy as np

with open("/tmp/final_4way_compare.pkl","rb") as f:
    D = pickle.load(f)
with open("/tmp/valid_regions.pkl","rb") as f:
    V = pickle.load(f)
valid_ivs = V["valid_ivs"]

def valid_mask(ts):
    ts = np.asarray(ts)
    g = np.zeros(len(ts), bool)
    for a,b in valid_ivs: g |= (ts>=a)&(ts<b)
    return g

def stats(a,b):
    dd=a-b
    return dict(n=int(len(a)), bias=float(dd.mean()), mae=float(np.abs(dd).mean()),
                r=float(np.corrcoef(a,b)[0,1]) if len(a)>2 else float("nan"))

series = [
    ("HAND SCG (gold standard)", D["xa_scg"], D["yb_scg"], D["ta_scg"], D["s_scg"]),
    ("BEST PIPELINE",            D["xa_det"], D["yb_det"], D["ta_det"], D["s_det"]),
    ("CORAL",                    D["xa_cor"], D["yb_cor"], D["ta_cor"], D["s_cor"]),
]

print(f"{'series':28s} {'OLD (hand-good-both)':>28s}   ->   {'NEW (valid regions)':>26s}")
results = {}
for name, xa, yb, ta, s_old in series:
    m = valid_mask(ta)
    s_new = stats(xa[m], yb[m])
    print(f"{name:28s} n={s_old['n']:5d} r={s_old['r']:.4f}          ->   n={s_new['n']:5d} r={s_new['r']:.4f}  MAE={s_new['mae']:.2f}  bias={s_new['bias']:+.2f}")
    results[name] = dict(old=s_old, new=s_new)

with open("/tmp/recompute_with_valid_regions.pkl","wb") as f:
    pickle.dump(results, f)
print("\nsaved /tmp/recompute_with_valid_regions.pkl")

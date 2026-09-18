import pickle, datetime
import numpy as np

with open("/tmp/final_compare_strict_mask.pkl","rb") as f:
    M = pickle.load(f)
with open("/tmp/final_full_comparison_data.pkl","rb") as f:
    D = pickle.load(f)

xa, yb, ta = M["xa_new"], M["yb_new"], M["ta_new"]
err = xa - yb
bad_idx = np.where(err < -18)[0]
print("n points total (strict mask):", len(xa))
print("n flagged missed-beat points:", len(bad_idx))

def stats(a,b):
    dd=a-b
    return dict(n=int(len(a)), bias=float(dd.mean()), mae=float(np.abs(dd).mean()),
                r=float(np.corrcoef(a,b)[0,1]) if len(a)>2 else float("nan"))

s_all = stats(xa, yb)
keep = np.ones(len(xa), bool); keep[bad_idx] = False
s_removed = stats(xa[keep], yb[keep])
print("\nWITH the 15 points:   ", s_all)
print("WITHOUT the 15 points:", s_removed)

UTC = datetime.timezone.utc
def fmt(us):
    return datetime.datetime.fromtimestamp(us/1e6, tz=UTC).strftime("%H:%M:%S.%f")[:-3]
print("\nthe 15 points:")
for i in bad_idx:
    print(f"  t={fmt(ta[i])}  det_hr={xa[i]:.1f}  ecg_hr={yb[i]:.1f}  err={err[i]:.1f}")

with open("/tmp/illustrate_15missed.pkl","wb") as f:
    pickle.dump(dict(xa=xa, yb=yb, ta=ta, bad_idx=bad_idx, keep=keep, s_all=s_all, s_removed=s_removed), f)

import pickle, datetime
import numpy as np

with open("/tmp/final_4way_compare.pkl","rb") as f:
    D = pickle.load(f)

xa_scg, yb_scg, ta_scg = D["xa_scg"], D["yb_scg"], D["ta_scg"]
err = xa_scg - yb_scg
print("n:", len(xa_scg), " r:", np.corrcoef(xa_scg,yb_scg)[0,1], " MAE:", np.abs(err).mean(), " bias:", err.mean())
print("err std:", err.std())
print("percentiles of |err|:", np.percentile(np.abs(err), [50,75,90,95,99]))

UTC = datetime.timezone.utc
def fmt(us):
    return datetime.datetime.fromtimestamp(us/1e6, tz=UTC).strftime("%H:%M:%S.%f")[:-3]

idx = np.argsort(-np.abs(err))[:25]
print("\ntop 25 largest |error| points:")
for i in idx:
    print(f"  t={fmt(ta_scg[i])}  scg_hr={xa_scg[i]:.1f}  ecg_hr={yb_scg[i]:.1f}  err={err[i]:+.1f}")

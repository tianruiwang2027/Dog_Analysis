import pickle, datetime
import numpy as np

with open("/tmp/final_full_comparison_data.pkl","rb") as f:
    D = pickle.load(f)

UTC = datetime.timezone.utc
def fmt(us):
    return datetime.datetime.fromtimestamp(us/1e6, tz=UTC).strftime("%H:%M:%S.%f")[:-3]

ta_det = D["ta_det"]; xa_det = D["xa_det"]; yb_det = D["yb_det"]
ta_scg = D["ta_scg"]; xa_scg = D["xa_scg"]; yb_scg = D["yb_scg"]

err_det = xa_det - yb_det
err_scg = xa_scg - yb_scg

print("=== DET series large errors (err_det < -18) ===")
idx = np.where(err_det < -18)[0]
for i in idx:
    print(f"  t={fmt(ta_det[i])}  det_hr={xa_det[i]:.1f}  ecg_hr={yb_det[i]:.1f}  err={err_det[i]:.1f}")

print(f"\n=== SCG(hand) series large errors (err_scg < -18) ===")
idx2 = np.where(err_scg < -18)[0]
for i in idx2:
    print(f"  t={fmt(ta_scg[i])}  scg_hr={xa_scg[i]:.1f}  ecg_hr={yb_scg[i]:.1f}  err={err_scg[i]:.1f}")

print(f"\ncounts: det<-18: {len(idx)}   scg<-18: {len(idx2)}")

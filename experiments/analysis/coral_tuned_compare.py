import pickle, datetime
import numpy as np
import polars as pl

with open("/tmp/final_4way_compare.pkl","rb") as f:
    D = pickle.load(f)

t0, t1, LAG_US = D["t0"], D["t1"], D["LAG_US"]
scg_ivs = sorted(D["scg_ivs"])
ecg_tmid, ecg_hr_sm, ecg_v = D["ecg_tmid"], D["ecg_hr_sm"], D["ecg_v"]
UTC = datetime.timezone.utc

def stats(a,b):
    dd=a-b
    return dict(n=int(len(a)), bias=float(dd.mean()), mae=float(np.abs(dd).mean()),
                r=float(np.corrcoef(a,b)[0,1]) if len(a)>2 else float("nan"))

def align(tsA, hrA, vA, tsB, hrB, vB, lag_us=0, win_us=2_000_000, restrict=(t0,t1)):
    bts, bhr = tsB[vB], hrB[vB]
    A = np.where(vA)[0]
    xa, yb, ta = [], [], []
    for i in A:
        traw = tsA[i]
        if restrict is not None and not (restrict[0] <= traw <= restrict[1]): continue
        t = traw + lag_us
        sel = (bts>=t-win_us)&(bts<=t+win_us)
        if sel.sum()>=1:
            xa.append(hrA[i]); yb.append(bhr[sel].mean()); ta.append(traw)
    return np.array(xa), np.array(yb), np.array(ta, dtype="int64")

def scg_good_point(ts):
    g = np.zeros(len(ts), bool)
    for a,b in scg_ivs: g |= (ts>=a)&(ts<b)
    return g

def load_coral(path):
    cdf = pl.read_csv(path)
    c_ts = np.array([datetime.datetime.fromisoformat(s).replace(tzinfo=UTC).timestamp()*1e6 for s in cdf["ts"].to_list()], dtype="int64")
    c_bpm = cdf["bpm"].to_numpy(); c_sqi = cdf["sqi"].to_numpy()
    return c_ts, c_bpm, c_sqi

def eval_coral(path, sqi_min=0.10, label=""):
    c_ts, c_bpm, c_sqi = load_coral(path)
    in_win = (c_ts>=t0)&(c_ts<=t1)
    coverage = float((c_sqi[in_win]>=sqi_min).mean())*100
    c_v = in_win & (c_sqi>=sqi_min) & scg_good_point(c_ts)
    xa, yb, ta = align(c_ts, c_bpm, c_v, ecg_tmid, ecg_hr_sm, ecg_v, lag_us=LAG_US)
    s = stats(xa, yb)
    medHR = float(np.median(c_bpm[in_win & (c_sqi>=sqi_min)]))
    print(f"{label:22s} coverage(SQI>=.10)={coverage:5.1f}%  medHR={medHR:5.1f}  n_pairs={s['n']:5d}  r={s['r']:.3f}  MAE={s['mae']:5.2f}  bias={s['bias']:+.2f}")
    return dict(c_ts=c_ts, c_bpm=c_bpm, c_sqi=c_sqi, c_v=c_v, xa=xa, yb=yb, ta=ta, s=s, coverage=coverage, medHR=medHR)

print("Comparing original vs mentor-tuned CORAL parameters (Chelten, same ECG reference/masking as before):\n")
R_orig  = eval_coral("coral_scg/Chelten/out.csv",       label="ORIGINAL (a=0,b=30,g=3,d=10,e=.005,z=5)")
R_tuned = eval_coral("coral_scg/Chelten_tuned/out.csv", label="TUNED (a=0,b=60,g=3,d=100,e=.001,z=.01)")

with open("/tmp/coral_tuned_compare.pkl","wb") as f:
    pickle.dump(dict(orig=R_orig, tuned=R_tuned, t0=t0, t1=t1, LAG_US=LAG_US, scg_ivs=D["scg_ivs"],
                      ecg_tmid=ecg_tmid, ecg_hr_sm=ecg_hr_sm, ecg_v=ecg_v), f)
print("\nsaved")

print("\n--- single-knob ablation (each vs the ORIGINAL baseline) ---")
R_delta   = eval_coral("coral_scg/Chelten_ab_delta/out.csv",   label="delta only: 10->100")
R_epszeta = eval_coral("coral_scg/Chelten_ab_epszeta/out.csv", label="epsilon/zeta only: loosened")
R_windows = eval_coral("coral_scg/Chelten_ab_windows/out.csv", label="windows only: drop 0.5s")
R_beta    = eval_coral("coral_scg/Chelten_ab_beta/out.csv",    label="beta only: 30->60")

with open("/tmp/coral_tuned_compare.pkl","wb") as f:
    pickle.dump(dict(orig=R_orig, tuned=R_tuned, delta=R_delta, epszeta=R_epszeta, windows=R_windows, beta=R_beta,
                      t0=t0, t1=t1, LAG_US=LAG_US, scg_ivs=D["scg_ivs"],
                      ecg_tmid=ecg_tmid, ecg_hr_sm=ecg_hr_sm, ecg_v=ecg_v), f)
print("\nsaved (with ablation)")

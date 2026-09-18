import pickle, sqlite3, datetime
import numpy as np
from scipy.signal import find_peaks

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_s = d["ts_s"]; xf_s = d["xf_s"]; fs_s = d["fs_s"]
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; fsd_s = d["fsd_s"]
# xf_s = bandpassed 10-100Hz -> respiration already removed.

def load_peaks_and_good(path):
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = '' ORDER BY timestamp")
    pk = np.array([r[0] for r in cur.fetchall()], dtype="int64")
    cur.execute("SELECT DISTINCT timestamp, label FROM events WHERE label != '' ORDER BY timestamp")
    rows = cur.fetchall()
    ev_ts = np.array([r[0] for r in rows], dtype="int64")
    ev_lab = [r[1].lower() for r in rows]
    return pk, ev_ts, ev_lab

def good_intervals(ev_ts, ev_lab, span_end):
    ivs=[]; state="bad"; cur_start=None
    for t,l in zip(ev_ts, ev_lab):
        if "good" in l and "start" in l:
            if state!="good": cur_start=t; state="good"
        elif "bad" in l and "start" in l:
            if state=="good": ivs.append((cur_start,t)); state="bad"
        elif "bad" in l and "end" in l:
            if state!="good": cur_start=t; state="good"
        elif "good" in l and "end" in l:
            if state=="good": ivs.append((cur_start,t)); state="bad"
        elif "stop" in l:
            if state=="good": ivs.append((cur_start,t)); state="bad"
    if state=="good": ivs.append((cur_start, span_end))
    return ivs

scg_pk, scg_ev_ts, scg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_scg2.db")
span_end = max(scg_pk.max(), ts_s.max())
scg_ivs = good_intervals(scg_ev_ts, scg_ev_lab, span_end)
def label_for(tc):
    return "good" if any(a<=tc<b for a,b in scg_ivs) else "bad"

WIN_S = 3.0
STEP_S = 1.0
win_us = int(WIN_S*1e6); step_us = int(STEP_S*1e6)
t0_all, t1_all = ts_s[0], ts_s[-1]
centers = np.arange(t0_all+win_us//2, t1_all-win_us//2, step_us)

ROLL_MS = 200.0          # rolling-average window for the "flat trend" feature
roll_n = max(1, int(ROLL_MS/1000*fs_s))
N_SUB = 6                # sub-segments per 3s window for the ratio-constancy feature
sub_us = win_us // N_SUB

def window_features(tc):
    lo, hi = tc-win_us//2, tc+win_us//2

    # ---- Feature A: rolling-average trend flatness ----
    i0 = np.searchsorted(ts_s, lo); i1 = np.searchsorted(ts_s, hi)
    if i1-i0 < roll_n*2: return None
    xf_w = xf_s[i0:i1]
    kernel = np.ones(roll_n)/roll_n
    roll = np.convolve(xf_w, kernel, mode="valid")
    trend_range = float(roll.max() - roll.min())

    # ---- Feature B: cardiac/noise proportion constancy across sub-segments ----
    j0 = np.searchsorted(ts_sd, lo); j1 = np.searchsorted(ts_sd, hi)
    if j1-j0 < 20: return None
    env_w = env_sd[j0:j1]; ts_w = ts_sd[j0:j1]
    med = np.median(env_w)
    mad = np.median(np.abs(env_w-med)) + 1e-15
    prom_thresh = max(3.0*mad, 0.15*(env_w.max()-med))
    pk_idx, _ = find_peaks(env_w, distance=max(1,int(0.08*fsd_s)), prominence=prom_thresh)

    ratios = []
    for k in range(N_SUB):
        a, b = lo+k*sub_us, lo+(k+1)*sub_us
        m = (ts_w>=a)&(ts_w<b)
        if m.sum() < 5: continue
        seg = env_w[m]
        seg_pk_h = env_w[pk_idx][ (ts_w[pk_idx]>=a)&(ts_w[pk_idx]<b) ]
        if len(seg_pk_h)==0: continue
        card = np.percentile(seg_pk_h, 75)
        noisef = np.percentile(seg, 20) + 1e-15
        ratios.append(card/noisef)
    ratios = np.array(ratios)
    if len(ratios) >= 3:
        ratio_cv = float(ratios.std()/ (ratios.mean()+1e-12))
        ratio_mean = float(ratios.mean())
        n_valid_sub = len(ratios)
    else:
        ratio_cv = np.nan
        ratio_mean = np.nan
        n_valid_sub = len(ratios)

    return dict(tc=tc, trend_range=trend_range, ratio_cv=ratio_cv, ratio_mean=ratio_mean, n_valid_sub=n_valid_sub)

feats = []
for tc in centers:
    f = window_features(tc)
    if f is None: continue
    f["label"] = label_for(tc)
    feats.append(f)

print(f"n windows: {len(feats)}")
good = [f for f in feats if f["label"]=="good"]
bad  = [f for f in feats if f["label"]=="bad"]
print(f"n good={len(good)}  n bad={len(bad)}")

g_tr = np.array([f["trend_range"] for f in good]); b_tr = np.array([f["trend_range"] for f in bad])
g_cv = np.array([f["ratio_cv"] for f in good]); b_cv = np.array([f["ratio_cv"] for f in bad])
print()
print("trend_range (rolling-avg peak-to-peak, raw units):")
print(" GOOD pct[5,25,50,75,90,95]:", np.percentile(g_tr,[5,25,50,75,90,95]))
print(" BAD  pct[5,25,50,75,90,95]:", np.percentile(b_tr,[5,25,50,75,90,95]))
print()
print("ratio_cv (coefficient of variation of cardiac/noise proportion across sub-segments):")
print(f" GOOD: n_valid={np.sum(~np.isnan(g_cv))}/{len(g_cv)}  median={np.nanmedian(g_cv):.3f}  pct[25,75]=[{np.nanpercentile(g_cv,25):.3f},{np.nanpercentile(g_cv,75):.3f}]")
print(f" BAD : n_valid={np.sum(~np.isnan(b_cv))}/{len(b_cv)}  median={np.nanmedian(b_cv):.3f}  pct[25,75]=[{np.nanpercentile(b_cv,25):.3f},{np.nanpercentile(b_cv,75):.3f}]")

with open("/tmp/mask_v6_trend_ratio.pkl","wb") as f:
    pickle.dump(feats, f)
print("saved /tmp/mask_v6_trend_ratio.pkl")

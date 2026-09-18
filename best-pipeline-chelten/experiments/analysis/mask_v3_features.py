import pickle, sqlite3
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_s = d["ts_s"]; x_s = d["x_s"]; xf_s = d["xf_s"]; fs_s = d["fs_s"]
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; fsd_s = d["fsd_s"]

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

b_lp, a_lp = butter(2, 1.0/(fs_s/2), btype="low")
x_lp = filtfilt(b_lp, a_lp, x_s)

WIN_S = 3.0
STEP_S = 1.0
t0_all, t1_all = ts_s[0], ts_s[-1]
win_us = int(WIN_S*1e6); step_us = int(STEP_S*1e6)
centers = np.arange(t0_all+win_us//2, t1_all-win_us//2, step_us)

def window_features(tc):
    lo, hi = tc-win_us//2, tc+win_us//2
    i0 = np.searchsorted(ts_s, lo); i1 = np.searchsorted(ts_s, hi)
    if i1-i0 < 100: return None
    xf_w = xf_s[i0:i1]
    xlp_w = x_lp[i0:i1]

    j0 = np.searchsorted(ts_sd, lo); j1 = np.searchsorted(ts_sd, hi)
    if j1-j0 < 20: return None
    env_w = env_sd[j0:j1]
    ts_w = ts_sd[j0:j1]

    med = np.median(env_w)
    mad = np.median(np.abs(env_w-med)) + 1e-15
    prom_thresh = max(3.0*mad, 0.15*(env_w.max()-med))
    pk_idx, props = find_peaks(env_w, distance=max(1,int(0.08*fsd_s)), prominence=prom_thresh)

    if len(pk_idx) == 0:
        cardiac_amp = 0.0
        noise_floor = float(np.percentile(env_w, 20)) + 1e-15
        n_pk = 0
        regularity = 0.0
        s1s2_std_ms = np.nan
    else:
        pk_h = env_w[pk_idx]
        cardiac_amp = float(np.percentile(pk_h, 75))
        pk_t = ts_w[pk_idx]
        mask_near_peak = np.zeros(len(ts_w), bool)
        for pt in pk_t:
            mask_near_peak |= np.abs(ts_w-pt) < int(0.05e6)
        far = env_w[~mask_near_peak]
        noise_floor = float(np.percentile(far, 50)) + 1e-15 if len(far)>5 else float(np.percentile(env_w,20))+1e-15
        n_pk = len(pk_idx)

        # timing REGULARITY: fraction of the window's peak-to-peak gaps that fit a clean
        # alternating S1-S2 (100-320ms) / diastole (320-1200ms) pattern, by total duration covered
        pk_t_sorted = np.sort(pk_t)
        gaps = np.diff(pk_t_sorted)/1e3  # ms
        is_s1s2 = (gaps>=100)&(gaps<=320)
        is_dia  = (gaps>320)&(gaps<=1200)
        explained = is_s1s2 | is_dia
        total_span_ms = (pk_t_sorted[-1]-pk_t_sorted[0])/1e3 if len(pk_t_sorted)>1 else 0
        explained_ms = gaps[explained].sum() if len(gaps) else 0
        regularity = explained_ms/total_span_ms if total_span_ms>0 else 0.0
        s1s2_vals = gaps[is_s1s2]
        s1s2_std_ms = float(s1s2_vals.std()) if len(s1s2_vals)>=2 else np.nan

    snr = cardiac_amp / noise_floor if cardiac_amp>0 else 0.0
    drift_amp = float(xlp_w.max()-xlp_w.min())
    cardiac_bp_amp = float(np.percentile(xf_w,99) - np.percentile(xf_w,1)) + 1e-12
    drift_ratio = drift_amp / cardiac_bp_amp

    return dict(tc=tc, cardiac_amp=cardiac_amp, noise_floor=noise_floor, snr=snr,
                drift_amp=drift_amp, cardiac_bp_amp=cardiac_bp_amp, drift_ratio=drift_ratio,
                n_peaks=n_pk, regularity=regularity, s1s2_std_ms=s1s2_std_ms)

def label_for(tc):
    good = any(a<=tc<b for a,b in scg_ivs)
    return "good" if good else "bad"

print(f"computing features for {len(centers)} windows...")
feats = []
for i,tc in enumerate(centers):
    f = window_features(tc)
    if f is None: continue
    f["label"] = label_for(tc)
    feats.append(f)

print(f"done: {len(feats)} windows")
with open("/tmp/mask_v3_all_features.pkl","wb") as f:
    pickle.dump(feats, f)

good = [f for f in feats if f["label"]=="good"]
bad = [f for f in feats if f["label"]=="bad"]
print(f"n good={len(good)}  n bad={len(bad)}")
for key in ["snr","drift_ratio","n_peaks","regularity"]:
    gv = np.array([f[key] for f in good]); bv = np.array([f[key] for f in bad])
    print(f"{key}: GOOD median={np.median(gv):.3f} (p10={np.percentile(gv,10):.3f},p90={np.percentile(gv,90):.3f})   BAD median={np.median(bv):.3f} (p10={np.percentile(bv,10):.3f},p90={np.percentile(bv,90):.3f})")

gs = np.array([f["s1s2_std_ms"] for f in good]); bs = np.array([f["s1s2_std_ms"] for f in bad])
print(f"s1s2_std_ms: GOOD median={np.nanmedian(gs):.1f}  BAD median={np.nanmedian(bs):.1f}  (nan fraction good={np.isnan(gs).mean():.2f} bad={np.isnan(bs).mean():.2f})")

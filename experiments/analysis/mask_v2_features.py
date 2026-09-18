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

WIN_S = 2.5
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

    # ROBUST prominence-based peak detection: only count peaks that stick up meaningfully
    # above the window's own noise floor (MAD-based), so we don't count every tiny wiggle.
    med = np.median(env_w)
    mad = np.median(np.abs(env_w-med)) + 1e-15
    prom_thresh = max(3.0*mad, 0.15*(env_w.max()-med))
    pk_idx, props = find_peaks(env_w, distance=max(1,int(0.08*fsd_s)), prominence=prom_thresh)

    if len(pk_idx) == 0:
        cardiac_amp = 0.0
        noise_floor = float(np.percentile(env_w, 20)) + 1e-15
    else:
        pk_h = env_w[pk_idx]
        cardiac_amp = float(np.percentile(pk_h, 75))
        pk_t = ts_w[pk_idx]
        mask_near_peak = np.zeros(len(ts_w), bool)
        for pt in pk_t:
            mask_near_peak |= np.abs(ts_w-pt) < int(0.05e6)
        far = env_w[~mask_near_peak]
        noise_floor = float(np.percentile(far, 50)) + 1e-15 if len(far)>5 else float(np.percentile(env_w,20))+1e-15

    snr = cardiac_amp / noise_floor if cardiac_amp>0 else 0.0

    drift_amp = float(xlp_w.max()-xlp_w.min())
    cardiac_bp_amp = float(np.percentile(xf_w,99) - np.percentile(xf_w,1)) + 1e-12
    drift_ratio = drift_amp / cardiac_bp_amp

    timing_ok = False
    n_pk = len(pk_idx)
    if n_pk >= 2:
        pk_t_sorted = np.sort(ts_w[pk_idx])
        ipi_ms = np.diff(pk_t_sorted)/1e3
        short = (ipi_ms>=100)&(ipi_ms<=320)
        long_ = (ipi_ms>320)&(ipi_ms<=1200)
        timing_ok = bool((short.sum()>=1) and (long_.sum()>=1))

    return dict(tc=tc, cardiac_amp=cardiac_amp, noise_floor=noise_floor, snr=snr,
                drift_amp=drift_amp, cardiac_bp_amp=cardiac_bp_amp, drift_ratio=drift_ratio,
                n_peaks=n_pk, timing_ok=timing_ok)

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
with open("/tmp/mask_v2_all_features.pkl","wb") as f:
    pickle.dump(feats, f)

good = [f for f in feats if f["label"]=="good"]
bad = [f for f in feats if f["label"]=="bad"]
print(f"n good={len(good)}  n bad={len(bad)}")
for key in ["snr","drift_ratio","n_peaks"]:
    gv = np.array([f[key] for f in good]); bv = np.array([f[key] for f in bad])
    print(f"{key}: GOOD median={np.median(gv):.3f} (p10={np.percentile(gv,10):.3f},p90={np.percentile(gv,90):.3f})   BAD median={np.median(bv):.3f} (p10={np.percentile(bv,10):.3f},p90={np.percentile(bv,90):.3f})")
gt = np.array([f["timing_ok"] for f in good]); bt = np.array([f["timing_ok"] for f in bad])
print(f"timing_ok: GOOD={100*gt.mean():.1f}%  BAD={100*bt.mean():.1f}%")

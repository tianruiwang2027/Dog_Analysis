import pickle, sqlite3, datetime
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

# precompute a low-pass (respiration-band) version of the RAW signal, native fs
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

    # candidate cardiac bursts: peaks in the envelope
    pk_idx, _ = find_peaks(env_w, distance=max(1,int(0.08*fsd_s)))
    if len(pk_idx) == 0:
        cardiac_amp = 0.0
        noise_floor = float(np.median(env_w)) + 1e-12
    else:
        pk_h = env_w[pk_idx]
        cardiac_amp = float(np.percentile(pk_h, 90))
        # noise floor: envelope value away from any peak (>60ms from nearest peak)
        pk_t = ts_w[pk_idx]
        mask_near_peak = np.zeros(len(ts_w), bool)
        for pt in pk_t:
            mask_near_peak |= np.abs(ts_w-pt) < int(0.06e6)
        far = env_w[~mask_near_peak]
        noise_floor = float(np.median(far)) + 1e-12 if len(far)>3 else float(np.percentile(env_w,20))+1e-12

    snr = cardiac_amp / noise_floor

    # baseline drift: peak-to-peak of the lowpass raw trend within window
    drift_amp = float(xlp_w.max()-xlp_w.min())
    cardiac_bp_amp = float(np.percentile(xf_w,99) - np.percentile(xf_w,1)) + 1e-12
    drift_ratio = drift_amp / cardiac_bp_amp

    # timing score: inter-peak intervals -- do they show an alternating short/long pattern?
    timing_ok = False
    n_pk = len(pk_idx)
    if n_pk >= 3:
        pk_t_sorted = np.sort(ts_w[pk_idx])
        ipi_ms = np.diff(pk_t_sorted)/1e3
        short = (ipi_ms>=100)&(ipi_ms<=320)   # S1-to-S2 range
        long_ = (ipi_ms>320)&(ipi_ms<=1200)   # S2-to-next-S1 (diastole) range
        # look for at least one short followed eventually by a long (alternation)
        timing_ok = bool((short.sum()>=1) and (long_.sum()>=1))

    return dict(tc=tc, cardiac_amp=cardiac_amp, noise_floor=noise_floor, snr=snr,
                drift_amp=drift_amp, cardiac_bp_amp=cardiac_bp_amp, drift_ratio=drift_ratio,
                n_peaks=n_pk, timing_ok=timing_ok)

def label_for(tc):
    good = any(a<=tc<b for a,b in scg_ivs)
    return "good" if good else "bad"

print(f"computing features for {len(centers)} windows (step={STEP_S}s, win={WIN_S}s)...")
feats = []
for i,tc in enumerate(centers):
    f = window_features(tc)
    if f is None: continue
    f["label"] = label_for(tc)
    feats.append(f)
    if i % 500 == 0: print(f"  {i}/{len(centers)}")

print(f"done: {len(feats)} windows")
with open("/tmp/mask_v1_all_features.pkl","wb") as f:
    pickle.dump(feats, f)

import pickle, sqlite3
import numpy as np

with open("/tmp/combined_detector.pkl","rb") as f:
    c = pickle.load(f)
with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)
with open("/tmp/shifted_detector.pkl","rb") as f:
    sd = pickle.load(f)
with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
with open("/tmp/env_template_wide.pkl","rb") as f:
    tpl = pickle.load(f)

t0 = int(lab["t0"]); t1 = int(lab["t1"])

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

def good_fn(ev_ts, ev_lab, span_end):
    ivs = good_intervals(ev_ts, ev_lab, span_end)
    def good(ts):
        g = np.zeros(len(ts), bool)
        for a,b in ivs: g |= (ts>=a)&(ts<b)
        return g
    return good

ecg_pk_full, ecg_ev_ts, ecg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_ecg2.db")
scg_pk_full, scg_ev_ts, scg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_scg2.db")
span_end = max(ecg_pk_full.max(), scg_pk_full.max(), t1)
ecg_good = good_fn(ecg_ev_ts, ecg_ev_lab, span_end)
scg_good = good_fn(scg_ev_ts, scg_ev_lab, span_end)

def beat_hr(peaks):
    rr = np.diff(peaks)/1e6; hr = 60.0/rr
    tmid = peaks[:-1] + np.diff(peaks)//2
    return tmid, hr

def smooth(tmid, hr, win_s=3.0):
    win_us = int(win_s*1e6)
    out = np.empty(len(tmid))
    for i,t in enumerate(tmid):
        sel = (tmid>=t-win_us)&(tmid<=t+win_us)
        out[i] = hr[sel].mean()
    return out

def stats(a,b):
    dd=a-b
    return dict(n=int(len(a)), bias=float(dd.mean()), mae=float(np.abs(dd).mean()),
                r=float(np.corrcoef(a,b)[0,1]) if len(a)>2 else float("nan"))

def align(tsA, hrA, vA, tsB, hrB, vB, lag_us=0, win_us=2_000_000, restrict=(t0,t1)):
    bts, bhr = tsB[vB], hrB[vB]
    A = np.where(vA)[0]
    xa, yb = [], []
    for i in A:
        traw = tsA[i]
        if restrict is not None and not (restrict[0] <= traw <= restrict[1]):
            continue
        t = traw + lag_us
        sel = (bts>=t-win_us)&(bts<=t+win_us)
        if sel.sum()>=1:
            xa.append(hrA[i]); yb.append(bhr[sel].mean())
    return np.array(xa), np.array(yb)

LAG_US = int(7.0e6)
ecg_tmid, ecg_hr_raw = beat_hr(ecg_pk_full)
ecg_hr_sm = smooth(ecg_tmid, ecg_hr_raw)
ecg_v = ecg_good(ecg_tmid)

def compare_vs_ecg(name, det_peaks):
    tmid, hr_raw = beat_hr(det_peaks)
    hr_sm = smooth(tmid, hr_raw)
    v = scg_good(tmid)
    xa, yb = align(tmid, hr_sm, v, ecg_tmid, ecg_hr_sm, ecg_v, lag_us=LAG_US)
    s = stats(xa, yb)
    print(f"{name:38s} n_pairs={s['n']:5d}  r={s['r']:.3f}  MAE={s['mae']:.2f}  bias={s['bias']:+.2f}")

print("=== lag-corrected (+7.0s) vs hand ECG ===")
compare_vs_ecg("hand-clicked SCG (gold std)", scg_pk_full)
compare_vs_ecg("baseline singles", np.sort(lab["singles_good"]))
compare_vs_ecg("+ wide-template NCC>=0.5", c["primary"])
compare_vs_ecg("+ earlier-candidate shift correction", sd["shifted"])

# now combine: shift correction AND template filter
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]
template = tpl["template"]; HALF_N = tpl["HALF_N"]
template_energy = np.sqrt((template**2).sum())
def ncc_score(t):
    i = np.searchsorted(ts_sd, t)
    if i-HALF_N < 0 or i+HALF_N+1 > len(env_sd): return np.nan
    s = env_sd[i-HALF_N:i+HALF_N+1]; s=s-s.mean()
    denom = np.sqrt((s**2).sum())*template_energy
    return float((s*template).sum()/denom) if denom>1e-12 else np.nan

shifted_scores = np.array([ncc_score(t) for t in sd["shifted"]])
shifted_and_filtered = sd["shifted"][shifted_scores>=0.5]
compare_vs_ecg("+ shift correction + template filter", shifted_and_filtered)

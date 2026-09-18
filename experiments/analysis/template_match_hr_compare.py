import pickle, sqlite3
import numpy as np

with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)
with open("/tmp/env_template.pkl","rb") as f:
    tpl = pickle.load(f)
with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)

ts_sd = d["ts_sd"]; env_sd = d["env_sd"]
template = tpl["template"]; HALF_N = tpl["HALF_N"]
template_energy = np.sqrt((template**2).sum())

def ncc_score(t):
    i = np.searchsorted(ts_sd, t)
    if i-HALF_N < 0 or i+HALF_N+1 > len(env_sd):
        return np.nan
    s = env_sd[i-HALF_N:i+HALF_N+1]
    s = s - s.mean()
    denom = np.sqrt((s**2).sum()) * template_energy
    if denom < 1e-12: return np.nan
    return float((s*template).sum() / denom)

singles_good_all = np.sort(lab["singles_good"])

SMOOTH_S = 3.0
def beat_hr(peaks):
    rr = np.diff(peaks)/1e6
    hr = 60.0/rr
    tmid = peaks[:-1] + np.diff(peaks)//2
    return tmid, hr

def smooth(tmid, hr, win_s=SMOOTH_S):
    win_us = int(win_s*1e6)
    out = np.empty(len(tmid))
    for i,t in enumerate(tmid):
        sel = (tmid>=t-win_us)&(tmid<=t+win_us)
        out[i] = hr[sel].mean()
    return out

def stats(a,b):
    dd = a-b
    return dict(n=int(len(a)), bias=float(dd.mean()), mae=float(np.abs(dd).mean()),
                r=float(np.corrcoef(a,b)[0,1]) if len(a)>2 else float("nan"))

def align(tsA,hrA,tsB,hrB,win_us=2_000_000):
    xa,yb=[],[]
    for i in range(len(tsA)):
        t=tsA[i]; sel=(tsB>=t-win_us)&(tsB<=t+win_us)
        if sel.sum()>=1:
            xa.append(hrA[i]); yb.append(hrB[sel].mean())
    return np.array(xa), np.array(yb)

# ECG reference: TRUE ECG hand clicks (not SCG hand clicks), restricted to same t0/t1 window and ECG's own good-data mask
con = sqlite3.connect("/tmp/annotation_chelten_ecg2.db")
cur = con.cursor()
cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = '' ORDER BY timestamp")
ecg_pk = np.array([r[0] for r in cur.fetchall()], dtype="int64")
cur.execute("SELECT DISTINCT timestamp, label FROM events WHERE label != '' ORDER BY timestamp")
rows = cur.fetchall()
ev_ts = np.array([r[0] for r in rows], dtype="int64")
ev_lab = [r[1].lower() for r in rows]

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

t0 = int(lab["t0"]); t1 = int(lab["t1"])
ivs = good_intervals(ev_ts, ev_lab, max(ecg_pk.max(), t1))
def good_mask(ts):
    g = np.zeros(len(ts), bool)
    for a,b in ivs: g |= (ts>=a)&(ts<b)
    return g

# SCG's own good/bad intervals (independent of ECG's)
con2 = sqlite3.connect("/tmp/annotation_chelten_scg2.db")
cur2 = con2.cursor()
cur2.execute("SELECT DISTINCT timestamp, label FROM events WHERE label != '' ORDER BY timestamp")
rows2 = cur2.fetchall()
scg_ev_ts = np.array([r[0] for r in rows2], dtype="int64")
scg_ev_lab = [r[1].lower() for r in rows2]
scg_ivs = good_intervals(scg_ev_ts, scg_ev_lab, max(ecg_pk.max(), t1))
def scg_good_mask(ts):
    g = np.zeros(len(ts), bool)
    for a,b in scg_ivs: g |= (ts>=a)&(ts<b)
    return g

def both_good(ts):
    return good_mask(ts) & scg_good_mask(ts)

ecg_pk_w = ecg_pk[(ecg_pk>=t0)&(ecg_pk<=t1)]
ecg_good = ecg_pk_w[both_good(ecg_pk_w)]
print("n ECG hand clicks in-window & good on BOTH masks:", len(ecg_good))
ecg_tmid, ecg_hr_raw = beat_hr(ecg_good)
ecg_hr_sm = smooth(ecg_tmid, ecg_hr_raw)

# require BOTH channels' good-data mask (intersection), matching the stricter
# methodology used earlier in this investigation
singles_good = singles_good_all[both_good(singles_good_all)]
print("n singles in-window & good on BOTH scg+ecg masks:", len(singles_good), "(was", len(singles_good_all), "with scg mask only)")
scores = np.array([ncc_score(t) for t in singles_good])

def run(name, mask):
    peaks = singles_good[mask]
    tmid, hr_raw = beat_hr(peaks)
    hr_sm = smooth(tmid, hr_raw)
    xa,yb = align(tmid, hr_sm, ecg_tmid, ecg_hr_sm)
    s = stats(xa,yb)
    print(f"{name:28s} n_peaks={len(peaks):5d}  n_pairs={s['n']:5d}  r={s['r']:.3f}  MAE={s['mae']:.2f}  bias={s['bias']:+.2f}")

run("baseline (no NCC filter)", np.ones(len(singles_good), bool))
for cutoff in [0.5,0.6,0.65,0.7,0.75,0.8,0.85]:
    run(f"NCC cutoff={cutoff}", scores>=cutoff)

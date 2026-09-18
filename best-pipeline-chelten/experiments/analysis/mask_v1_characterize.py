import pickle, sqlite3, datetime
import numpy as np

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_s = d["ts_s"]; x_s = d["x_s"]; xf_s = d["xf_s"]; fs_s = d["fs_s"]
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; sharp_s = d["sharp_s"]; fsd_s = d["fsd_s"]

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

def in_ivs(t, ivs):
    return any(a<=t<b for a,b in ivs)

good_clicks = np.array([t for t in scg_pk if in_ivs(t, scg_ivs)])
print(f"n hand-clicked SCG beats in GOOD-labeled spans: {len(good_clicks)}")

# for each good click, extract a window and find fine peaks in the RAW BANDPASSED signal
# to check the "2 peaks for S1, gap ~170ms, 1 peak for S2, gap 200-300ms" structure
from scipy.signal import find_peaks

WIN_MS = 500
results = []
rng = np.random.default_rng(0)
sample = rng.choice(good_clicks, size=min(400,len(good_clicks)), replace=False)
for t in sample:
    lo = np.searchsorted(ts_s, t - int(WIN_MS*1e3))
    hi = np.searchsorted(ts_s, t + int(WIN_MS*1e3))
    if hi-lo < 10: continue
    seg = xf_s[lo:hi]
    t_ms = (ts_s[lo:hi]-t)/1e3
    # find peaks in the ENVELOPE within this window to locate S1/S2 bursts robustly
    lo2 = np.searchsorted(ts_sd, t - int(WIN_MS*1e3))
    hi2 = np.searchsorted(ts_sd, t + int(WIN_MS*1e3))
    envseg = env_sd[lo2:hi2]
    t_ms2 = (ts_sd[lo2:hi2]-t)/1e3
    pk_idx, props = find_peaks(envseg, height=envseg.max()*0.25, distance=max(1,int(0.03*fsd_s)))
    pk_t = t_ms2[pk_idx]; pk_h = envseg[pk_idx]
    results.append(dict(t=t, seg=seg, t_ms=t_ms, pk_t=pk_t, pk_h=pk_h, env=envseg, t_ms2=t_ms2))

print(f"processed {len(results)} good beats")
with open("/tmp/mask_v1_good_samples.pkl","wb") as f:
    pickle.dump(dict(results=results, scg_ivs=scg_ivs, good_clicks=good_clicks), f)

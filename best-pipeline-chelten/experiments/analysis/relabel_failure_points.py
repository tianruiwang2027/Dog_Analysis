import pickle, sqlite3, datetime
import numpy as np

with open("/tmp/full_session_scores_all_models.pkl","rb") as f:
    S = pickle.load(f)
cand_t = S["cand_t"]; scores = S["scores_relabel"]
with open("/tmp/cnn_mask_apply.pkl","rb") as f:
    CA = pickle.load(f)
LAG_US = CA["LAG_US"]; RESTRICT = CA["RESTRICT"]

def load_peaks_and_good(path):
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = ''")
    pk = np.array([r[0] for r in cur.fetchall()], dtype="int64")
    cur.execute("SELECT DISTINCT timestamp, label FROM events WHERE label != ''")
    rows = cur.fetchall()
    ev_ts = np.array([r[0] for r in rows], dtype="int64"); ev_lab=[r[1].lower() for r in rows]
    return pk, ev_ts, ev_lab
def good_intervals(ev_ts, ev_lab, span_end):
    ivs=[]; state="bad"; cur_start=None
    for tt,l in zip(ev_ts, ev_lab):
        if "good" in l and "start" in l:
            if state!="good": cur_start=tt; state="good"
        elif "bad" in l and "start" in l:
            if state=="good": ivs.append((cur_start,tt)); state="bad"
        elif "bad" in l and "end" in l:
            if state!="good": cur_start=tt; state="good"
        elif "good" in l and "end" in l:
            if state=="good": ivs.append((cur_start,tt)); state="bad"
        elif "stop" in l:
            if state=="good": ivs.append((cur_start,tt)); state="bad"
    if state=="good": ivs.append((cur_start, span_end))
    return ivs
def good_mask(ts, ivs):
    g = np.zeros(len(ts), bool)
    for a,b in ivs: g |= (ts>=a)&(ts<b)
    return g
def beat_hr(peaks):
    rr = np.diff(peaks)/1e6; hr = 60.0/rr
    tmid = peaks[:-1] + np.diff(peaks)//2
    return tmid, hr, rr
def smooth(tmid, hr, win_s=3.0):
    win_us2 = int(win_s*1e6); out = np.empty(len(tmid))
    for i,t in enumerate(tmid):
        sel = (tmid>=t-win_us2)&(tmid<=t+win_us2); out[i]=hr[sel].mean()
    return out
def align(tsA, hrA, vA, tsB, hrB, vB, lag_us=0, win_us=2_000_000, restrict=None):
    bts, bhr = tsB[vB], hrB[vB]
    A_ = np.where(vA)[0]
    xa, yb, ta = [], [], []
    for i in A_:
        traw = tsA[i]
        if restrict is not None and not (restrict[0] <= traw <= restrict[1]): continue
        t = traw + lag_us
        sel = (bts>=t-win_us)&(bts<=t+win_us)
        if sel.sum()>=1:
            xa.append(hrA[i]); yb.append(bhr[sel].mean()); ta.append(traw)
    return np.array(xa), np.array(yb), np.array(ta, dtype="int64")

ecg_pk, ecg_ev_ts, ecg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_ecg2.db")
span_end = max(ecg_pk.max(), cand_t.max())
ecg_ivs = good_intervals(ecg_ev_ts, ecg_ev_lab, span_end)
ecg_tmid, ecg_hr, _ = beat_hr(ecg_pk)
ecg_hr_sm = smooth(ecg_tmid, ecg_hr)
ecg_v_full = good_mask(ecg_tmid, ecg_ivs)

CUTOFF = 0.25; MAX_RR_S = 1.5
valid = ~np.isnan(scores)
confirmed = np.sort(cand_t[valid][scores[valid] >= CUTOFF])
tmid, hr_raw, rr = beat_hr(confirmed)
ok = rr <= MAX_RR_S
tmid_ok, hr_ok = tmid[ok], hr_raw[ok]
hr_sm = smooth(tmid_ok, hr_ok, win_s=3.0)
ecg_v_at = good_mask(tmid_ok, ecg_ivs)
xa, yb, ta = align(tmid_ok, hr_sm, ecg_v_at, ecg_tmid, ecg_hr_sm, ecg_v_full, lag_us=LAG_US, restrict=RESTRICT)
err = xa - yb
r = np.corrcoef(xa,yb)[0,1]
print(f"n={len(xa)}  r={r:.4f}  MAE={np.abs(err).mean():.2f}  bias={err.mean():+.2f}")

# find the worst-error clusters: sort by |err| and cluster nearby times
order = np.argsort(-np.abs(err))
top = order[:60]
top_t = np.sort(ta[top])
print("\ntop 60 worst-error points, session-relative time (s) and error (bpm):")
t0_session = 1782494791107337
for i in order[:20]:
    tstr = datetime.datetime.utcfromtimestamp(ta[i]/1e6).strftime("%H:%M:%S")
    print(f"  {tstr}  algo={xa[i]:.1f} ecg={yb[i]:.1f} err={err[i]:+.1f}")

# cluster the worst points in time (gap > 20s = new cluster)
gaps = np.diff(top_t)
breaks = np.where(gaps>20_000_000)[0]
clusters = np.split(top_t, breaks+1)
print(f"\n{len(clusters)} distinct worst-error time clusters:")
for c in clusters:
    tstr0 = datetime.datetime.utcfromtimestamp(c.min()/1e6).strftime("%H:%M:%S")
    tstr1 = datetime.datetime.utcfromtimestamp(c.max()/1e6).strftime("%H:%M:%S")
    print(f"  {tstr0} - {tstr1}  (n={len(c)})")

with open("/tmp/relabel_failure_points.pkl","wb") as f:
    pickle.dump(dict(xa=xa, yb=yb, ta=ta, err=err, r=r, confirmed=confirmed), f)
print("saved")

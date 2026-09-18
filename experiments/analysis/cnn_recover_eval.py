import pickle, sqlite3
import numpy as np

with open("/tmp/cnn_mask_apply.pkl","rb") as f:
    CA = pickle.load(f)
cand_t = CA["cand_t"]; cand_cnn = CA["cand_cnn"]
LAG_US = CA["LAG_US"]; RESTRICT = CA["RESTRICT"]

with open("/tmp/cnn_recover_result.pkl","rb") as f:
    REC = pickle.load(f)
recovered = REC["recovered"]   # list of (t_rec, score)
rec_t = np.array([r[0] for r in recovered], dtype="int64")

CUTOFF = 0.40
confirmed_old = cand_t[cand_cnn >= CUTOFF]
confirmed_new = np.sort(np.concatenate([confirmed_old, rec_t]))
print(f"confirmed beats: old (no recovery)={len(confirmed_old)}  new (with recovery)={len(confirmed_new)}  "
      f"(+{len(confirmed_new)-len(confirmed_old)})")

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_s = d["ts_s"]

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
def stats(a,b):
    dd=a-b
    return dict(n=int(len(a)), bias=float(dd.mean()), mae=float(np.abs(dd).mean()),
                r=float(np.corrcoef(a,b)[0,1]) if len(a)>2 else float("nan"))
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
span_end = max(ecg_pk.max(), ts_s.max())
ecg_ivs = good_intervals(ecg_ev_ts, ecg_ev_lab, span_end)
ecg_tmid, ecg_hr, _ = beat_hr(ecg_pk)
ecg_hr_sm = smooth(ecg_tmid, ecg_hr)
ecg_v_full = good_mask(ecg_tmid, ecg_ivs)

MAX_RR_S = 1.5
def run(confirmed, label):
    tmid, hr_raw, rr = beat_hr(confirmed)
    ok = rr <= MAX_RR_S
    tmid_ok, hr_ok = tmid[ok], hr_raw[ok]
    hr_sm = smooth(tmid_ok, hr_ok, win_s=3.0)
    ecg_v_at = good_mask(tmid_ok, ecg_ivs)
    xa, yb, ta = align(tmid_ok, hr_sm, ecg_v_at, ecg_tmid, ecg_hr_sm, ecg_v_full, lag_us=LAG_US, restrict=RESTRICT)
    s = stats(xa, yb)
    print(f"{label:34s} n_beats={len(confirmed):5d}  pairs_kept={ok.sum():5d}/{len(rr):5d}  "
          f"r={s['r']:.4f}  MAE={s['mae']:.2f}  bias={s['bias']:+.2f}  n={s['n']}")
    return dict(s=s, xa=xa, yb=yb, ta=ta, tmid_ok=tmid_ok, hr_ok=hr_ok)

res_old = run(confirmed_old, "old (no recovery)")
res_new = run(confirmed_new, "new (with S2 recovery)")

with open("/tmp/cnn_recover_eval.pkl","wb") as f:
    pickle.dump(dict(res_old=res_old, res_new=res_new, confirmed_old=confirmed_old,
                      confirmed_new=confirmed_new, rec_t=rec_t, LAG_US=LAG_US, RESTRICT=RESTRICT), f)
print("saved /tmp/cnn_recover_eval.pkl")

# ---- specific check: the big end cluster (17:53:11 - 17:55:41) ----
import datetime
t0w = int(datetime.datetime(2026,6,26,17,53,11, tzinfo=datetime.timezone.utc).timestamp()*1e6)
t1w = int(datetime.datetime(2026,6,26,17,55,41, tzinfo=datetime.timezone.utc).timestamp()*1e6)
for label, res in [("old", res_old), ("new", res_new)]:
    sel = (res["ta"]>=t0w)&(res["ta"]<t1w)
    if sel.sum()==0: continue
    err = res["xa"][sel]-res["yb"][sel]
    print(f"end-cluster window ({label}): n={sel.sum()}  mean SCG-ECG error={err.mean():+.2f} bpm  MAE={np.abs(err).mean():.2f}")

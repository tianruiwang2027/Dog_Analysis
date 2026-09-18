import pickle, sqlite3
import numpy as np

# ---------- NEW architecture: detector proposes a peak -> CNN directly confirms/rejects
# THAT peak (pruned straight out of the beat sequence used to compute RR intervals),
# instead of the old approach (keep every detected peak, and separately decide whether
# a 3s WINDOW around it is "good enough" via a gap/min_n rule on the side). ----------

with open("/tmp/cnn_mask_apply.pkl","rb") as f:
    CA = pickle.load(f)
cand_t = CA["cand_t"]; cand_cnn = CA["cand_cnn"]     # already time-sorted, whole session
LAG_US = CA["LAG_US"]; RESTRICT = CA["RESTRICT"]

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

def beat_hr(peaks):
    rr = np.diff(peaks)/1e6; hr = 60.0/rr
    tmid = peaks[:-1] + np.diff(peaks)//2
    return tmid, hr, rr

ecg_tmid, ecg_hr, _ = beat_hr(ecg_pk)
def smooth(tmid, hr, win_s=3.0):
    win_us2 = int(win_s*1e6); out = np.empty(len(tmid))
    for i,t in enumerate(tmid):
        sel = (tmid>=t-win_us2)&(tmid<=t+win_us2); out[i]=hr[sel].mean()
    return out
ecg_hr_sm = smooth(ecg_tmid, ecg_hr)
ecg_v_full = good_mask(ecg_tmid, ecg_ivs)

def confirm_and_run(cutoff, max_rr_s):
    """Prune candidates by CNN score BEFORE computing RR/HR (direct per-beat confirmation),
    then additionally drop any RR interval that's too long to be a real single beat-to-beat
    gap (i.e. a stretch where a real beat was almost certainly rejected or missed) so those
    don't inject a spuriously low instantaneous HR."""
    confirmed = np.sort(cand_t[cand_cnn >= cutoff])
    tmid, hr_raw, rr = beat_hr(confirmed)
    ok = rr <= max_rr_s
    tmid_ok, hr_ok = tmid[ok], hr_raw[ok]
    if len(tmid_ok) < 10:
        return None
    hr_sm = smooth(tmid_ok, hr_ok, win_s=3.0)
    ecg_v_at = good_mask(tmid_ok, ecg_ivs)
    xa, yb, ta = align(tmid_ok, hr_sm, ecg_v_at, ecg_tmid, ecg_hr_sm, ecg_v_full, lag_us=LAG_US, restrict=RESTRICT)
    s = stats(xa, yb)
    return dict(cutoff=cutoff, max_rr_s=max_rr_s, n_confirmed=len(confirmed), n_pairs_kept=int(ok.sum()),
                n_pairs_total=len(rr), s=s, xa=xa, yb=yb, ta=ta)

print(f"{'cutoff':>7} {'max_rr':>7} {'n_conf':>7} {'pairs_kept':>10} {'r':>7} {'MAE':>6} {'bias':>7} {'n':>6}")
results = {}
for cutoff in [0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80]:
    for max_rr_s in [1.5, 2.0]:
        res = confirm_and_run(cutoff, max_rr_s)
        if res is None: continue
        s = res["s"]
        print(f"{cutoff:7.2f} {max_rr_s:7.1f} {res['n_confirmed']:7d} {res['n_pairs_kept']:10d} "
              f"{s['r']:7.3f} {s['mae']:6.2f} {s['bias']:+7.2f} {s['n']:6d}")
        results[(cutoff, max_rr_s)] = res

best_key = max(results, key=lambda k: results[k]["s"]["r"])
print(f"\nBEST: cutoff={best_key[0]} max_rr={best_key[1]}  ->  {results[best_key]['s']}")

with open("/tmp/cnn_confirm_pipeline.pkl","wb") as f:
    pickle.dump(dict(results=results, best_key=best_key), f)
print("saved /tmp/cnn_confirm_pipeline.pkl")

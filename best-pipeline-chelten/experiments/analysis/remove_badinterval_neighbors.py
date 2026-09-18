import pickle, sqlite3, datetime
import numpy as np

with open("/tmp/unmatched_clicks.pkl","rb") as f:
    U = pickle.load(f)
mismatch_times = np.concatenate([U["unmatched_scg"], U["unmatched_ecg"]])

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

def good_mask(ts, ivs):
    g = np.zeros(len(ts), bool)
    for a,b in ivs: g |= (ts>=a)&(ts<b)
    return g

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

def align(tsA, hrA, vA, tsB, hrB, vB, lag_us=0, win_us=2_000_000, restrict=None):
    bts, bhr = tsB[vB], hrB[vB]
    A = np.where(vA)[0]
    xa, yb, ta = [], [], []
    for i in A:
        traw = tsA[i]
        if restrict is not None and not (restrict[0] <= traw <= restrict[1]): continue
        t = traw + lag_us
        sel = (bts>=t-win_us)&(bts<=t+win_us)
        if sel.sum()>=1:
            xa.append(hrA[i]); yb.append(bhr[sel].mean()); ta.append(traw)
    return np.array(xa), np.array(yb), np.array(ta, dtype="int64")

ecg_pk, ecg_ev_ts, ecg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_ecg2.db")
scg_pk, scg_ev_ts, scg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_scg2.db")

with open("/tmp/final_4way_compare.pkl","rb") as f:
    D = pickle.load(f)
LAG_US = D["LAG_US"]; RESTRICT = (D["t0"], D["t1"])

span_end = max(ecg_pk.max(), scg_pk.max(), D["t1"])
ecg_ivs = good_intervals(ecg_ev_ts, ecg_ev_lab, span_end)
scg_ivs_hand = good_intervals(scg_ev_ts, scg_ev_lab, span_end)

ecg_tmid, ecg_hr = beat_hr(ecg_pk); ecg_hr_sm = smooth(ecg_tmid, ecg_hr)
scg_tmid, scg_hr = beat_hr(scg_pk); scg_hr_sm = smooth(scg_tmid, scg_hr)
ecg_v_full = good_mask(ecg_tmid, ecg_ivs)
ecg_v_at_scg = good_mask(scg_tmid, ecg_ivs)
scg_v_hand = good_mask(scg_tmid, scg_ivs_hand) & ecg_v_at_scg

xa_h, yb_h, ta_h = align(scg_tmid, scg_hr_sm, scg_v_hand, ecg_tmid, ecg_hr_sm, ecg_v_full, lag_us=LAG_US, restrict=RESTRICT)
s_h = stats(xa_h, yb_h)
print(f"BASELINE (hand labels, official):        n={s_h['n']:5d}  r={s_h['r']:.4f}  MAE={s_h['mae']:.2f}")

RADIUS_US = int(3.0e6)
def near_any(ts, times, radius_us=RADIUS_US):
    bad = np.zeros(len(ts), bool)
    for t in times:
        bad |= np.abs(ts - t) <= radius_us
    return bad

# step 1: mismatch-cleaned only (as before)
clean1 = scg_v_hand & ~near_any(scg_tmid, mismatch_times)
xa_1, yb_1, ta_1 = align(scg_tmid, scg_hr_sm, clean1, ecg_tmid, ecg_hr_sm, ecg_v_full, lag_us=LAG_US, restrict=RESTRICT)
s_1 = stats(xa_1, yb_1)
print(f"+ 24 mismatch pts removed (+-3s):        n={s_1['n']:5d}  r={s_1['r']:.4f}  MAE={s_1['mae']:.2f}")

# step 2: ALSO remove +-3s around every SCG bad-interval EDGE (start and end of every bad stretch)
bad_edges = []
scg_ivs_sorted = sorted(scg_ivs_hand)
# the "bad" stretches are the GAPS between consecutive good intervals
for i in range(len(scg_ivs_sorted)-1):
    gap_start = scg_ivs_sorted[i][1]   # end of a good interval = start of a bad stretch
    gap_end = scg_ivs_sorted[i+1][0]   # start of next good interval = end of that bad stretch
    if gap_end > gap_start:
        bad_edges.append(gap_start); bad_edges.append(gap_end)
bad_edges = np.array(bad_edges, dtype="int64")
print(f"\nn SCG bad-interval edges: {len(bad_edges)} ({len(bad_edges)//2} bad stretches)")

clean2 = scg_v_hand & ~near_any(scg_tmid, mismatch_times) & ~near_any(scg_tmid, bad_edges)
xa_2, yb_2, ta_2 = align(scg_tmid, scg_hr_sm, clean2, ecg_tmid, ecg_hr_sm, ecg_v_full, lag_us=LAG_US, restrict=RESTRICT)
s_2 = stats(xa_2, yb_2)
print(f"+ ALSO +-3s around every bad-interval edge: n={s_2['n']:5d}  r={s_2['r']:.4f}  MAE={s_2['mae']:.2f}  bias={s_2['bias']:+.2f}")
print(f"  -> total removed vs baseline: {s_h['n']-s_2['n']} points ({(s_h['n']-s_2['n'])/s_h['n']*100:.1f}%)")

with open("/tmp/remove_badinterval_result.pkl","wb") as f:
    pickle.dump(dict(s_h=s_h, s_1=s_1, s_2=s_2, xa_h=xa_h, yb_h=yb_h, ta_h=ta_h,
                      xa_2=xa_2, yb_2=yb_2, ta_2=ta_2, bad_edges=bad_edges, mismatch_times=mismatch_times,
                      RESTRICT=RESTRICT, LAG_US=LAG_US), f)
print("\nsaved /tmp/remove_badinterval_result.pkl")

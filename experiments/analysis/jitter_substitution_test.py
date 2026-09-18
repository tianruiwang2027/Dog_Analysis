import pickle, sqlite3
import numpy as np

with open("/tmp/combined_detector.pkl","rb") as f:
    c = pickle.load(f)
with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)

t0 = int(lab["t0"]); t1 = int(lab["t1"])
LAG_US = int(7.0e6)

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

ecg_tmid, ecg_hr_raw = beat_hr(ecg_pk_full)
ecg_hr_sm = smooth(ecg_tmid, ecg_hr_raw)
ecg_v = ecg_good(ecg_tmid)

det = np.sort(c["primary"])
hand_full = scg_pk_full
TOL=100_000
used = np.zeros(len(hand_full), bool)
matched_hand_time = np.full(len(det), -1, dtype="int64")
for i,s in enumerate(det):
    idx = np.searchsorted(hand_full, s)
    best=None; bestd=TOL+1
    for ci in (idx-1, idx):
        if 0<=ci<len(hand_full) and not used[ci]:
            dd=abs(hand_full[ci]-s)
            if dd<bestd: bestd=dd; best=ci
    if best is not None and bestd<=TOL:
        used[best]=True
        matched_hand_time[i] = hand_full[best]

is_tp = matched_hand_time >= 0
clean_mask = is_tp[:-1] & is_tp[1:]
print(f"n clean consecutive pairs: {clean_mask.sum()}")

# jitter stats on clean matched pairs
jitter_ms = (det[is_tp] - matched_hand_time[is_tp]) / 1000.0
print(f"detector-vs-hand-click timing offset on matches: mean={jitter_ms.mean():+.1f}ms  std={jitter_ms.std():.1f}ms")

# --- Test A: detector's OWN times, clean-only (already have: r=0.831) ---
tmid_det, hr_det = beat_hr(det)
clean_tmid_det = tmid_det[clean_mask]; clean_hr_det = hr_det[clean_mask]
hr_det_sm = smooth(clean_tmid_det, clean_hr_det)
v = scg_good(clean_tmid_det)
xa,yb = align(clean_tmid_det, hr_det_sm, v, ecg_tmid, ecg_hr_sm, ecg_v, lag_us=LAG_US)
print(f"\nA) detector's OWN timestamps, clean pairs only: r={stats(xa,yb)['r']:.3f}  MAE={stats(xa,yb)['mae']:.2f}")

# --- Test B: SUBSTITUTE the matched hand-click times for the exact same clean cycles ---
hand_substituted = matched_hand_time.copy()
tmid_hs, hr_hs = beat_hr(hand_substituted[is_tp])  # need contiguous array; but is_tp already selects only matched ones
# careful: beat_hr needs strictly the sequence of TIMES; using only is_tp positions changes adjacency vs det's own indexing
# so instead directly recompute RR from hand_substituted at the SAME clean_mask positions using det's own indexing
# use full-length array with -1 placeholders replaced only where matched, then compute RR only for clean consecutive TP pairs directly:
rr_hand = (matched_hand_time[1:] - matched_hand_time[:-1]) / 1e6
hr_hand_at_detpos = 60.0/rr_hand
tmid_hand_at_detpos = (matched_hand_time[:-1] + matched_hand_time[1:])//2
clean_tmid_hs = tmid_hand_at_detpos[clean_mask]
clean_hr_hs = hr_hand_at_detpos[clean_mask]
hr_hs_sm = smooth(clean_tmid_hs, clean_hr_hs)
v2 = scg_good(clean_tmid_hs)
xa2,yb2 = align(clean_tmid_hs, hr_hs_sm, v2, ecg_tmid, ecg_hr_sm, ecg_v, lag_us=LAG_US)
print(f"B) SAME cycles, but using the matched HAND-CLICK timestamps instead: r={stats(xa2,yb2)['r']:.3f}  MAE={stats(xa2,yb2)['mae']:.2f}")

print(f"\nGold standard (hand SCG vs hand ECG, all clicks): r=0.897")

# --- range-restriction check: is the "clean" subset just covering a narrower HR range? ---
print(f"\n--- range-restriction check ---")
print(f"ECG-HR range in the CLEAN-only comparison: std={yb.std():.2f}  min={yb.min():.1f}  max={yb.max():.1f}  n={len(yb)}")

# full gold-standard comparison, for reference
scg_tmid_full, scg_hr_full = beat_hr(scg_pk_full)
scg_hr_full_sm = smooth(scg_tmid_full, scg_hr_full)
scg_v_full = scg_good(scg_tmid_full)
xa_gold, yb_gold = align(scg_tmid_full, scg_hr_full_sm, scg_v_full, ecg_tmid, ecg_hr_sm, ecg_v, lag_us=LAG_US)
print(f"ECG-HR range in the GOLD STANDARD comparison:  std={yb_gold.std():.2f}  min={yb_gold.min():.1f}  max={yb_gold.max():.1f}  n={len(yb_gold)}")

# also: what if we restrict the GOLD STANDARD to the SAME time windows the clean-only set covers?
clean_times_set = set(clean_tmid_det.tolist())
# find gold standard points whose time is close to a clean-only point (within 3s, i.e. same local stretch)
clean_times_arr = np.sort(clean_tmid_det)
mask_gold_in_clean_regions = np.zeros(len(xa_gold), bool)
scg_v_idx = np.where(scg_v_full)[0]
matched_gold_times = scg_tmid_full[scg_v_idx]
# recompute alignment restricting source points to those within +-2s of ANY clean-only point
for i, t in enumerate(matched_gold_times):
    if not (t0<=t<=t1): continue
    j = np.searchsorted(clean_times_arr, t)
    near = False
    if j>0 and t - clean_times_arr[j-1] <= 2_000_000: near=True
    if j<len(clean_times_arr) and clean_times_arr[j]-t <= 2_000_000: near=True
    if near: mask_gold_in_clean_regions[i]=True

xa_gold2, yb_gold2 = align(matched_gold_times[mask_gold_in_clean_regions], scg_hr_full_sm[scg_v_idx][mask_gold_in_clean_regions], np.ones(mask_gold_in_clean_regions.sum(),bool), ecg_tmid, ecg_hr_sm, ecg_v, lag_us=LAG_US, restrict=None)
s3 = stats(xa_gold2, yb_gold2)
print(f"\nGOLD STANDARD restricted to the SAME time regions as clean-only: n={s3['n']} r={s3['r']:.3f}  ECG-HR std={yb_gold2.std():.2f}")

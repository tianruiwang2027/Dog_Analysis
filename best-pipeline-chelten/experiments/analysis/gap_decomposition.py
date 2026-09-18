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

# --- template-filtered detector (our best) ---
det = np.sort(c["primary"])

# tight match each detection against hand-clicked SCG (same modality, precise)
hand_good_full = scg_pk_full  # full click list already
used = np.zeros(len(hand_good_full), bool)
is_tp = np.zeros(len(det), bool)
TOL=100_000
for i,s in enumerate(det):
    idx = np.searchsorted(hand_good_full, s)
    best=None; bestd=TOL+1
    for ci in (idx-1, idx):
        if 0<=ci<len(hand_good_full) and not used[ci]:
            dd=abs(hand_good_full[ci]-s)
            if dd<bestd: bestd=dd; best=ci
    if best is not None and bestd<=TOL:
        used[best]=True; is_tp[i]=True

print(f"detector n={len(det)}  TP={is_tp.sum()}  FP={(~is_tp).sum()}")

# clean-only: consecutive pairs of detections that are BOTH validated TPs
clean_mask = is_tp[:-1] & is_tp[1:]
tmid_all, hr_all = beat_hr(det)
clean_tmid = tmid_all[clean_mask]
clean_hr = hr_all[clean_mask]
print(f"n clean consecutive RR intervals: {clean_mask.sum()} of {len(clean_mask)} total intervals")

clean_hr_sm = smooth(clean_tmid, clean_hr)
v_clean = scg_good(clean_tmid)
xa, yb = align(clean_tmid, clean_hr_sm, v_clean, ecg_tmid, ecg_hr_sm, ecg_v, lag_us=LAG_US)
s_clean = stats(xa, yb)
print(f"\nCLEAN-ONLY (detector, error-free RR intervals only) vs hand ECG:")
print(f"  n_pairs={s_clean['n']}  r={s_clean['r']:.3f}  MAE={s_clean['mae']:.2f}  bias={s_clean['bias']:+.2f}")

# full (unfiltered-for-cleanliness) template-filtered detector, for reference
tmid_full, hr_full = beat_hr(det)
hr_full_sm = smooth(tmid_full, hr_full)
v_full = scg_good(tmid_full)
xa2, yb2 = align(tmid_full, hr_full_sm, v_full, ecg_tmid, ecg_hr_sm, ecg_v, lag_us=LAG_US)
s_full = stats(xa2, yb2)
print(f"\nFULL template-filtered detector vs hand ECG (for reference): r={s_full['r']:.3f} MAE={s_full['mae']:.2f}")

print(f"\nGold standard (hand SCG vs hand ECG): r=0.897 (established)")

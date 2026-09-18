import pickle, datetime, sqlite3
import numpy as np

with open("/tmp/mask_v5_flatness.pkl","rb") as f:
    M = pickle.load(f)
tc = M["tc"]; rms = M["rms"]; step_us = M["step_us"]

THR = 20.0  # flag BAD if rms >= THR  (86.9% acc, 98.1% specificity, 48.8% sensitivity)
is_good = rms < THR
order = np.argsort(tc)
tc_s = tc[order]; good_s = is_good[order]

auto_ivs = []
cur_start = None
for i in range(len(tc_s)):
    if good_s[i]:
        if cur_start is None: cur_start = tc_s[i]-step_us//2
    else:
        if cur_start is not None:
            auto_ivs.append((cur_start, tc_s[i]-step_us//2)); cur_start=None
if cur_start is not None:
    auto_ivs.append((cur_start, tc_s[-1]+step_us//2))
print(f"auto-mask (flatness, thr={THR}): {len(auto_ivs)} good intervals, "
      f"{sum(b-a for a,b in auto_ivs)/1e6:.0f}s of {(tc_s[-1]-tc_s[0])/1e6:.0f}s span")

def auto_good_mask(ts):
    g = np.zeros(len(ts), bool)
    for a,b in auto_ivs: g |= (ts>=a)&(ts<b)
    return g

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
LAG_US = D["LAG_US"]

span_end = max(ecg_pk.max(), scg_pk.max(), D["t1"])
ecg_ivs = good_intervals(ecg_ev_ts, ecg_ev_lab, span_end)
scg_ivs_hand = good_intervals(scg_ev_ts, scg_ev_lab, span_end)

ecg_tmid, ecg_hr = beat_hr(ecg_pk); ecg_hr_sm = smooth(ecg_tmid, ecg_hr)
scg_tmid, scg_hr = beat_hr(scg_pk); scg_hr_sm = smooth(scg_tmid, scg_hr)

ecg_v_at_scg = good_mask(scg_tmid, ecg_ivs)
scg_v_hand = good_mask(scg_tmid, scg_ivs_hand) & ecg_v_at_scg
scg_v_auto = auto_good_mask(scg_tmid) & ecg_v_at_scg

RESTRICT = (D["t0"], D["t1"])
print()
print("=== hand SCG vs ECG: HAND-labeled mask vs simple FLATNESS mask ===")
ecg_v_full = good_mask(ecg_tmid, ecg_ivs)
xa_h, yb_h, ta_h = align(scg_tmid, scg_hr_sm, scg_v_hand, ecg_tmid, ecg_hr_sm, ecg_v_full, lag_us=LAG_US, restrict=RESTRICT)
s_h = stats(xa_h, yb_h)
print(f"HAND      mask:  n={s_h['n']:5d}  r={s_h['r']:.4f}  MAE={s_h['mae']:.2f}  bias={s_h['bias']:+.2f}")

xa_a, yb_a, ta_a = align(scg_tmid, scg_hr_sm, scg_v_auto, ecg_tmid, ecg_hr_sm, ecg_v_full, lag_us=LAG_US, restrict=RESTRICT)
s_a = stats(xa_a, yb_a)
print(f"FLATNESS  mask:  n={s_a['n']:5d}  r={s_a['r']:.4f}  MAE={s_a['mae']:.2f}  bias={s_a['bias']:+.2f}")

in_win = (scg_tmid>=RESTRICT[0])&(scg_tmid<=RESTRICT[1])
hv = scg_v_hand[in_win]; av = scg_v_auto[in_win]
agree = (hv==av).mean()
both_good=(hv&av).sum(); hand_only=(hv&~av).sum(); auto_only=(~hv&av).sum(); both_bad=(~hv&~av).sum()
print()
print(f"beat-level agreement: {agree*100:.1f}%   both good={both_good}  hand-only={hand_only}  auto-only={auto_only}  both bad={both_bad}")

with open("/tmp/mask_v5_apply.pkl","wb") as f:
    pickle.dump(dict(auto_ivs=auto_ivs, scg_ivs_hand=scg_ivs_hand, THR=THR,
                      xa_h=xa_h, yb_h=yb_h, ta_h=ta_h, s_h=s_h, xa_a=xa_a, yb_a=yb_a, ta_a=ta_a, s_a=s_a,
                      scg_tmid=scg_tmid, scg_hr_sm=scg_hr_sm, scg_v_hand=scg_v_hand, scg_v_auto=scg_v_auto,
                      LAG_US=LAG_US, RESTRICT=RESTRICT), f)
print("saved /tmp/mask_v5_apply.pkl")

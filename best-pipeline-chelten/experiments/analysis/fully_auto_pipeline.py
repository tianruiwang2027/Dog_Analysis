import pickle, sqlite3
import numpy as np

# ---------- fully automatic chain: primary detector (whole session, NO hand-label
# restriction at all) -> mask decides which stretches to trust -> compare to ECG hand-picked ----------

with open("/tmp/cnn_mask_apply.pkl","rb") as f:
    CA = pickle.load(f)
cand_t = CA["cand_t"]; cand_cnn = CA["cand_cnn"]          # primary candidates, WHOLE session, unrestricted
LAG_US = CA["LAG_US"]; RESTRICT = CA["RESTRICT"]

with open("/tmp/mask_v9_rows.pkl","rb") as f:
    NCC0 = pickle.load(f)
ncc_cand_t = NCC0["cand_t"]; ncc_cand_ncc = NCC0["cand_ncc"]   # NCC-scored candidates, ALSO whole session

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
    return tmid, hr

def smooth(tmid, hr, win_s=3.0):
    win_us2 = int(win_s*1e6)
    out = np.empty(len(tmid))
    for i,t in enumerate(tmid):
        sel = (tmid>=t-win_us2)&(tmid<=t+win_us2)
        out[i] = hr[sel].mean()
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

# ECG side: only restriction we keep is ECG's OWN hand-good annotation (defines when
# the reference itself is trustworthy) -- this is not an SCG-quality restriction.
ecg_pk, ecg_ev_ts, ecg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_ecg2.db")
span_end = max(ecg_pk.max(), ts_s.max())
ecg_ivs = good_intervals(ecg_ev_ts, ecg_ev_lab, span_end)
ecg_tmid, ecg_hr = beat_hr(ecg_pk)
ecg_hr_sm = smooth(ecg_tmid, ecg_hr)
ecg_v_full = good_mask(ecg_tmid, ecg_ivs)

# ---------- fully automatic PIPELINE beat stream: ALL primary candidates, whole session,
# NO SCG hand-label filtering of any kind ----------
det_tmid_auto, det_hr_auto = beat_hr(np.sort(cand_t))
det_hr_sm_auto = smooth(det_tmid_auto, det_hr_auto)
ecg_v_at_det = good_mask(det_tmid_auto, ecg_ivs)
print(f"fully-automatic primary-detector beat stream: {len(cand_t)} peaks -> {len(det_tmid_auto)} HR points (whole session)")

WIN_S = 3.0; STEP_S = 1.0
win_us = int(WIN_S*1e6); step_us = int(STEP_S*1e6)
t0_all, t1_all = ts_s[0], ts_s[-1]
centers = np.arange(t0_all+win_us//2, t1_all-win_us//2, step_us)
pad = int(1.0e6)

def make_auto_ivs(cand_t_, cscore, cutoff, gap_max_us, min_n):
    m_all = cscore>=cutoff
    mt_all = np.sort(cand_t_[m_all])
    good_flags = np.zeros(len(centers), bool)
    for i,tc in enumerate(centers):
        lo,hi = tc-win_us//2, tc+win_us//2
        sel=(mt_all>=lo-pad)&(mt_all<hi+pad); mt=mt_all[sel]
        inwin=(mt>=lo)&(mt<hi); n_in=inwin.sum()
        if n_in<min_n or len(mt)<2: continue
        gaps=np.diff(mt); gap_lo=mt[:-1]; gap_hi=mt[1:]
        overlap=(gap_hi>lo)&(gap_lo<hi); rel=gaps[overlap]
        if len(rel)==0: continue
        good_flags[i] = rel.max()<=gap_max_us
    order=np.argsort(centers); tc_s=centers[order]; good_s=good_flags[order]
    ivs=[]; cur_start=None
    for i in range(len(tc_s)):
        if good_s[i]:
            if cur_start is None: cur_start=tc_s[i]-step_us//2
        else:
            if cur_start is not None: ivs.append((cur_start, tc_s[i]-step_us//2)); cur_start=None
    if cur_start is not None: ivs.append((cur_start, tc_s[-1]+step_us//2))
    return ivs

# ---------- CNN mask decides good/bad, no other restriction ----------
auto_ivs_cnn = make_auto_ivs(cand_t, cand_cnn, 0.55, int(3.5e6), 1)
covered_cnn = sum(b-a for a,b in auto_ivs_cnn)/1e6
det_v_cnn = good_mask(det_tmid_auto, auto_ivs_cnn) & ecg_v_at_det
xa_cnn, yb_cnn, ta_cnn = align(det_tmid_auto, det_hr_sm_auto, det_v_cnn, ecg_tmid, ecg_hr_sm, ecg_v_full, lag_us=LAG_US, restrict=RESTRICT)
s_cnn = stats(xa_cnn, yb_cnn)
print(f"\nCNN-mask auto_ivs coverage: {covered_cnn:.0f}s  ({len(auto_ivs_cnn)} intervals)")
print(f"FULLY AUTOMATIC (primary detector + CNN mask) vs ECG hand-picked:")
print(f"  n={s_cnn['n']}  r={s_cnn['r']:.4f}  MAE={s_cnn['mae']:.2f}  bias={s_cnn['bias']:+.2f}")

# ---------- NCC mask decides good/bad, no other restriction (for comparison) ----------
auto_ivs_ncc = make_auto_ivs(ncc_cand_t, ncc_cand_ncc, 0.7, int(2.0e6), 2)
covered_ncc = sum(b-a for a,b in auto_ivs_ncc)/1e6
det_v_ncc = good_mask(det_tmid_auto, auto_ivs_ncc) & ecg_v_at_det
xa_ncc, yb_ncc, ta_ncc = align(det_tmid_auto, det_hr_sm_auto, det_v_ncc, ecg_tmid, ecg_hr_sm, ecg_v_full, lag_us=LAG_US, restrict=RESTRICT)
s_ncc = stats(xa_ncc, yb_ncc)
print(f"\nNCC-mask auto_ivs coverage: {covered_ncc:.0f}s  ({len(auto_ivs_ncc)} intervals)")
print(f"FULLY AUTOMATIC (primary detector + NCC mask) vs ECG hand-picked:")
print(f"  n={s_ncc['n']}  r={s_ncc['r']:.4f}  MAE={s_ncc['mae']:.2f}  bias={s_ncc['bias']:+.2f}")

# ---------- context: no mask at all (raw unfiltered detector output vs ECG hand-picked) ----------
det_v_none = ecg_v_at_det.copy()
xa_none, yb_none, ta_none = align(det_tmid_auto, det_hr_sm_auto, det_v_none, ecg_tmid, ecg_hr_sm, ecg_v_full, lag_us=LAG_US, restrict=RESTRICT)
s_none = stats(xa_none, yb_none)
print(f"\nNO MASK (raw detector, only ECG-side quality applied) vs ECG hand-picked:")
print(f"  n={s_none['n']}  r={s_none['r']:.4f}  MAE={s_none['mae']:.2f}  bias={s_none['bias']:+.2f}")

with open("/tmp/fully_auto_pipeline.pkl","wb") as f:
    pickle.dump(dict(s_cnn=s_cnn, s_ncc=s_ncc, s_none=s_none,
                      xa_cnn=xa_cnn, yb_cnn=yb_cnn, ta_cnn=ta_cnn,
                      xa_ncc=xa_ncc, yb_ncc=yb_ncc, ta_ncc=ta_ncc,
                      xa_none=xa_none, yb_none=yb_none, ta_none=ta_none,
                      auto_ivs_cnn=auto_ivs_cnn, auto_ivs_ncc=auto_ivs_ncc,
                      covered_cnn=covered_cnn, covered_ncc=covered_ncc,
                      det_tmid_auto=det_tmid_auto, det_hr_sm_auto=det_hr_sm_auto,
                      LAG_US=LAG_US, RESTRICT=RESTRICT), f)
print("\nsaved /tmp/fully_auto_pipeline.pkl")

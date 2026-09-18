#!/usr/bin/env python3
"""Two-pass Shannon S1/S2 detector with interval-filtering search-back:
  1) PRIMARY pass: strict/confident detection (thr=0.6, gap window 190-230ms)
  2) For any primary-cycle-to-cycle gap that lands in [SEARCHBACK_LO, SEARCHBACK_HI]
     seconds (looks like a missed beat, not just a genuinely slow cycle),
     re-scan ONLY inside that gap with relaxed settings (lower threshold,
     wider gap window) and splice in anything found.
Then re-run the same 6-way comparison table as before."""
import pickle, sqlite3, sys
import numpy as np
from scipy import signal as sg
from scipy.ndimage import median_filter

SEARCH_LO, SEARCH_HI = 50, 200
SMOOTH_S = 3.0

PRIMARY_THR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
PRIMARY_LO = float(sys.argv[2]) if len(sys.argv) > 2 else 0.19
PRIMARY_HI = float(sys.argv[3]) if len(sys.argv) > 3 else 0.23
RELAX_THR = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
RELAX_LO = float(sys.argv[5]) if len(sys.argv) > 5 else 0.17
RELAX_HI = float(sys.argv[6]) if len(sys.argv) > 6 else 0.25
SB_LO_S = float(sys.argv[7]) if len(sys.argv) > 7 else 1.2   # gap range (s) that
SB_HI_S = float(sys.argv[8]) if len(sys.argv) > 8 else 1.5   # triggers a search-back

with open("/tmp/shannon_restricted_results_thr0.3.pkl", "rb") as f:
    d = pickle.load(f)
ts_sd, sharp_s, fsd_s = d["ts_sd"], d["sharp_s"], d["fsd_s"]


def pick_doublets_in(ts, env, fsd, lo, hi, thr, refract_pk_s=0.08, post_s2_refract_s=0.15):
    cand, _ = sg.find_peaks(env, height=thr, distance=max(1, int(refract_pk_s*fsd)))
    cand_t = ts[cand]
    s1_times, s2_times = [], []
    i = 0; n = len(cand_t)
    while i < n - 1:
        t1 = cand_t[i]
        lo_t, hi_t = t1+int(lo*1e6), t1+int(hi*1e6)
        j = i+1; found = -1
        while j < n and cand_t[j] <= hi_t:
            if cand_t[j] >= lo_t: found = j; break
            j += 1
        if found >= 0:
            s1_times.append(t1); s2_times.append(cand_t[found])
            s2_t = cand_t[found]; k = found+1
            while k < n and cand_t[k] < s2_t + int(post_s2_refract_s*1e6): k += 1
            i = k
        else:
            i += 1
    return np.array(s1_times, dtype="int64"), np.array(s2_times, dtype="int64"), cand_t


# ---- PRIMARY pass, over the whole restricted window ----
s1_p, s2_p, cand_p = pick_doublets_in(ts_sd, sharp_s, fsd_s, PRIMARY_LO, PRIMARY_HI, PRIMARY_THR)
print(f"primary pass (thr={PRIMARY_THR}, gap={PRIMARY_LO*1000:.0f}-{PRIMARY_HI*1000:.0f}ms): {len(s1_p)} cycles")

# ---- interval filtering: find primary-to-primary gaps in [SB_LO_S, SB_HI_S] ----
order = np.argsort(s1_p)
s1_p, s2_p = s1_p[order], s2_p[order]
gaps_s = np.diff(s1_p) / 1e6
flagged = np.where((gaps_s >= SB_LO_S) & (gaps_s <= SB_HI_S))[0]
print(f"flagged {len(flagged)} primary-to-primary gaps in [{SB_LO_S},{SB_HI_S}]s for search-back")

s1_extra, s2_extra = [], []
for idx in flagged:
    w0, w1 = s2_p[idx], s1_p[idx+1]           # search strictly between prev S2 and next S1
    m = (ts_sd >= w0) & (ts_sd <= w1)
    if m.sum() < 5:
        continue
    sub_ts, sub_env = ts_sd[m], sharp_s[m]
    ns1, ns2, _ = pick_doublets_in(sub_ts, sub_env, fsd_s, RELAX_LO, RELAX_HI, RELAX_THR)
    s1_extra.extend(ns1.tolist()); s2_extra.extend(ns2.tolist())

s1_extra = np.array(s1_extra, dtype="int64")
print(f"recovered {len(s1_extra)} extra cycles via search-back")

s1_all = np.sort(np.concatenate([s1_p, s1_extra]))
print(f"total cycles after search-back: {len(s1_all)}  (was {len(s1_p)})")


def clean_hr(tpk):
    if len(tpk) < 4: return np.array([]), np.array([])
    rr = np.diff(tpk)/1e6; hr = 60.0/rr
    tmid = tpk[:-1] + np.diff(tpk)//2
    keep = (hr >= SEARCH_LO) & (hr <= SEARCH_HI)
    med = median_filter(hr, 7, mode="nearest")
    mad = median_filter(np.abs(hr-med), 7, mode="nearest") + 1e-6
    keep &= np.abs(hr-med) <= 5.0*1.4826*mad
    return tmid[keep], hr[keep]


def smooth(tmid, hr, win_s=SMOOTH_S):
    win_us = int(win_s*1e6)
    out = np.empty(len(tmid))
    for i, t in enumerate(tmid):
        sel = (tmid >= t-win_us) & (tmid <= t+win_us)
        out[i] = hr[sel].mean()
    return out


sh_scg_tmid_p, sh_scg_hr_p = clean_hr(s1_p)
sh_scg_tmid_a, sh_scg_hr_a = clean_hr(s1_all)

# ---- ECG: unchanged single-peak Shannon detector, thr=0.6, 0.28s refractory ----
import glob, polars as pl
HIVE = "/tmp/dog-test-ecg/dog-test-ecg-code/hive"
def bandpass(x, fs, lo, hi):
    hi = min(hi, 0.45*fs); b = sg.butter(2, [lo/(fs/2), hi/(fs/2)], btype="band")
    return sg.filtfilt(*b, x - np.mean(x))
def load_window(glob_pat, t0, t1, pad_s=30.0):
    p = glob.glob(glob_pat)[0]; df = pl.read_parquet(p, columns=["ts","c1"])
    ts = df["ts"].to_numpy().astype("int64"); x = df["c1"].to_numpy().astype(float)
    a = np.searchsorted(ts, t0-int(pad_s*1e6)); b = np.searchsorted(ts, t1+int(pad_s*1e6))
    return ts[a:b], x[a:b]
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
def shannon_envelope_decimated(xf, ts, fs, avg_win_s=0.02, gauss_sigma_s=0.01, FSD=200.0):
    xn = xf/(np.max(np.abs(xf))+1e-12); se=-(xn**2)*np.log(xn**2+1e-9)
    se_avg = uniform_filter1d(se, max(1,int(avg_win_s*fs)))
    step = max(1,int(round(fs/FSD))); se_d=se_avg[::step]; ts_d=ts[::step]; fsd=fs/step
    return ts_d, gaussian_filter1d(se_d, sigma=max(1,gauss_sigma_s*fsd)), fsd
from scipy.ndimage import percentile_filter
def sharpen_local(env, fsd, q995_win_s=8.0, q=99.5):
    win = max(3,int(q995_win_s*fsd))|1
    local_q995 = percentile_filter(env, q, size=win, mode="nearest")
    return np.exp(env/np.maximum(local_q995,1e-9))-1.0, local_q995

def load_peaks_and_good(path):
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = '' ORDER BY timestamp")
    pk = np.array([r[0] for r in cur.fetchall()], dtype="int64")
    cur.execute("SELECT DISTINCT timestamp, label FROM events WHERE label != '' ORDER BY timestamp")
    rows = cur.fetchall(); ev_ts = np.array([r[0] for r in rows], dtype="int64")
    ev_lab = [r[1].lower() for r in rows]
    return pk, ev_ts, ev_lab
def good_intervals(ev_ts, ev_lab, span_end):
    ivs=[]; state="bad"; cur_start=None
    for t,lab in zip(ev_ts, ev_lab):
        if "good" in lab and "start" in lab:
            if state!="good": cur_start=t; state="good"
        elif "bad" in lab and "start" in lab:
            if state=="good": ivs.append((cur_start,t)); state="bad"
        elif "bad" in lab and "end" in lab:
            if state!="good": cur_start=t; state="good"
        elif "good" in lab and "end" in lab:
            if state=="good": ivs.append((cur_start,t)); state="bad"
        elif "stop" in lab:
            if state=="good": ivs.append((cur_start,t)); state="bad"
    if state=="good": ivs.append((cur_start,span_end))
    return ivs
def good_fn(ev_ts, ev_lab, span_end):
    ivs = good_intervals(ev_ts, ev_lab, span_end)
    def good(ts):
        g=np.zeros(len(ts),bool)
        for t0,t1 in ivs: g |= (ts>=t0)&(ts<t1)
        return g
    return good, ivs
def beat_hr(peaks):
    rr=np.diff(peaks)/1e6; hr=60.0/rr; tmid=peaks[:-1]+np.diff(peaks)//2
    return tmid, hr

h_ecg_pk, h_ecg_ev_ts, h_ecg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_ecg2.db")
h_scg_pk, h_scg_ev_ts, h_scg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_scg2.db")
span_end = max(h_ecg_pk.max(), h_scg_pk.max())
h_ecg_good,_ = good_fn(h_ecg_ev_ts, h_ecg_ev_lab, span_end)
h_scg_good,_ = good_fn(h_scg_ev_ts, h_scg_ev_lab, span_end)
t0 = np.datetime64("2026-06-26T17:26:30").astype("datetime64[us]").astype(int)
t1 = span_end

ts_e, x_e = load_window(f"{HIVE}/username=ecg_polar/device=Chelten/stream=0/date=*/data_0.parquet", t0, t1)
fs_e = float(1e6/np.median(np.diff(ts_e)))
xf_e = bandpass(x_e, fs_e, 10.0, 100.0)
ts_ed, env_ed, fsd_e = shannon_envelope_decimated(xf_e, ts_e, fs_e)
sharp_e, _ = sharpen_local(env_ed, fsd_e, q995_win_s=8.0)
cand_e,_ = sg.find_peaks(sharp_e, height=0.6, distance=max(1,int(0.28*fsd_e)))
R_e = ts_ed[cand_e]
sh_ecg_tmid, sh_ecg_hr = clean_hr(R_e)

sh_scg_hr_p_s = smooth(sh_scg_tmid_p, sh_scg_hr_p)
sh_scg_hr_a_s = smooth(sh_scg_tmid_a, sh_scg_hr_a)
sh_ecg_hr_s = smooth(sh_ecg_tmid, sh_ecg_hr)
h_ecg_tmid, h_ecg_hr_raw = beat_hr(h_ecg_pk); h_scg_tmid, h_scg_hr_raw = beat_hr(h_scg_pk)
h_ecg_hr = smooth(h_ecg_tmid, h_ecg_hr_raw); h_scg_hr = smooth(h_scg_tmid, h_scg_hr_raw)
h_ecg_v = h_ecg_good(h_ecg_tmid); h_scg_v = h_scg_good(h_scg_tmid)
sh_ecg_v = h_ecg_good(sh_ecg_tmid)
sh_scg_v_p = h_scg_good(sh_scg_tmid_p); sh_scg_v_a = h_scg_good(sh_scg_tmid_a)

def align(tsA,hrA,vA,tsB,hrB,vB,win_us=2_000_000,lag_us=0):
    bts,bhr = tsB[vB],hrB[vB]; A=np.where(vA)[0]; xa,yb,ta=[],[],[]
    for i in A:
        t=tsA[i]+lag_us; sel=(bts>=t-win_us)&(bts<=t+win_us)
        if sel.sum()>=1: xa.append(hrA[i]); yb.append(bhr[sel].mean()); ta.append(tsA[i])
    return np.array(xa), np.array(yb), np.array(ta,dtype="int64")
def stats(a,b):
    dd=a-b
    return dict(n=int(len(a)), bias=float(dd.mean()), mae=float(np.abs(dd).mean()),
                within10=float(np.mean(np.abs(dd)<=10)), r=float(np.corrcoef(a,b)[0,1]) if len(a)>2 else float("nan"))
def in_window(xa,yb,ta):
    inw=(ta>=t0)&(ta<=t1); return xa[inw],yb[inw],stats(xa[inw],yb[inw])
def best_lag(tsA,hrA,vA,sideA,tsB,hrB,vB,sideB,lags):
    if sideA==sideB:
        xa,yb,ta=align(tsA,hrA,vA,tsB,hrB,vB,lag_us=0); return 0.0,in_window(xa,yb,ta)
    best=(0.0,-2)
    for lag_s in lags:
        xa,yb,ta=align(tsA,hrA,vA,tsB,hrB,vB,lag_us=int(lag_s*1e6)); _,_,R=in_window(xa,yb,ta)
        if not np.isnan(R['r']) and R['r']>best[1]: best=(lag_s,R['r'])
    lag_s=best[0]; xa,yb,ta=align(tsA,hrA,vA,tsB,hrB,vB,lag_us=int(lag_s*1e6))
    return lag_s, in_window(xa,yb,ta)

lags = np.arange(-20,20.01,0.5)
print(f"\n--- BEFORE search-back (primary only) ---")
lag,(xa,yb,R) = best_lag(h_ecg_tmid,h_ecg_hr,h_ecg_v,'ecg', sh_scg_tmid_p,sh_scg_hr_p_s,sh_scg_v_p,'scg', lags)
print(f"Hand ECG vs Shannon SCG(primary)   lag={lag:+.1f}s n={R['n']:4d} MAE={R['mae']:.2f} bias={R['bias']:+.2f} r={R['r']:.3f}")
lag,(xa,yb,R) = best_lag(h_scg_tmid,h_scg_hr,h_scg_v,'scg', sh_scg_tmid_p,sh_scg_hr_p_s,sh_scg_v_p,'scg', lags)
print(f"Hand SCG  vs Shannon SCG(primary)  lag={lag:+.1f}s n={R['n']:4d} MAE={R['mae']:.2f} bias={R['bias']:+.2f} r={R['r']:.3f}")

print(f"\n--- AFTER search-back ---")
lag,(xa,yb,R) = best_lag(h_ecg_tmid,h_ecg_hr,h_ecg_v,'ecg', sh_ecg_tmid,sh_ecg_hr_s,sh_ecg_v,'ecg', lags)
print(f"Hand ECG  vs Shannon ECG           lag={lag:+.1f}s n={R['n']:4d} MAE={R['mae']:.2f} bias={R['bias']:+.2f} r={R['r']:.3f}")
lag,(xa,yb,R) = best_lag(h_ecg_tmid,h_ecg_hr,h_ecg_v,'ecg', sh_scg_tmid_a,sh_scg_hr_a_s,sh_scg_v_a,'scg', lags)
print(f"Hand ECG  vs Shannon SCG(+search)  lag={lag:+.1f}s n={R['n']:4d} MAE={R['mae']:.2f} bias={R['bias']:+.2f} r={R['r']:.3f}")
lag,(xa,yb,R) = best_lag(h_scg_tmid,h_scg_hr,h_scg_v,'scg', sh_scg_tmid_a,sh_scg_hr_a_s,sh_scg_v_a,'scg', lags)
print(f"Hand SCG  vs Shannon SCG(+search)  lag={lag:+.1f}s n={R['n']:4d} MAE={R['mae']:.2f} bias={R['bias']:+.2f} r={R['r']:.3f}")
lag,(xa,yb,R) = best_lag(sh_ecg_tmid,sh_ecg_hr_s,sh_ecg_v,'ecg', sh_scg_tmid_a,sh_scg_hr_a_s,sh_scg_v_a,'scg', lags)
print(f"Shannon ECG vs Shannon SCG(+search) lag={lag:+.1f}s n={R['n']:4d} MAE={R['mae']:.2f} bias={R['bias']:+.2f} r={R['r']:.3f}")
lag,(xa,yb,R) = best_lag(h_ecg_tmid,h_ecg_hr,h_ecg_v,'ecg', h_scg_tmid,h_scg_hr,h_scg_v,'scg', lags)
print(f"Hand ECG  vs Hand SCG (reference)  lag={lag:+.1f}s n={R['n']:4d} MAE={R['mae']:.2f} bias={R['bias']:+.2f} r={R['r']:.3f}")

with open("/tmp/shannon_searchback_results.pkl","wb") as f:
    pickle.dump(dict(s1_p=s1_p, s2_p=s2_p, s1_extra=np.array(s1_extra,dtype="int64"),
                      s2_extra=np.array(s2_extra,dtype="int64"), s1_all=s1_all,
                      flagged_gaps=[(s2_p[i], s1_p[i+1]) for i in flagged],
                      t0=t0, t1=t1), f)
print("\nsaved /tmp/shannon_searchback_results.pkl")

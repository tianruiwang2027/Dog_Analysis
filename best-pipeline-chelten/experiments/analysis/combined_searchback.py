import pickle, sqlite3
import numpy as np
from scipy.signal import find_peaks

with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)
with open("/tmp/env_template_wide.pkl","rb") as f:
    tpl = pickle.load(f)
with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)

ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; sharp_s = d["sharp_s"]; fsd_s = d["fsd_s"]
template = tpl["template"]; HALF_N = tpl["HALF_N"]
template_energy = np.sqrt((template**2).sum())

def ncc_score(t):
    i = np.searchsorted(ts_sd, t)
    if i-HALF_N < 0 or i+HALF_N+1 > len(env_sd):
        return np.nan
    s = env_sd[i-HALF_N:i+HALF_N+1]
    s = s - s.mean()
    denom = np.sqrt((s**2).sum()) * template_energy
    if denom < 1e-12: return np.nan
    return float((s*template).sum() / denom)

t0 = int(lab["t0"]); t1 = int(lab["t1"])

# --- good masks (both channels) ---
con = sqlite3.connect("/tmp/annotation_chelten_ecg2.db")
cur = con.cursor()
cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = '' ORDER BY timestamp")
ecg_pk = np.array([r[0] for r in cur.fetchall()], dtype="int64")
cur.execute("SELECT DISTINCT timestamp, label FROM events WHERE label != '' ORDER BY timestamp")
rows = cur.fetchall()
ev_ts = np.array([r[0] for r in rows], dtype="int64")
ev_lab = [r[1].lower() for r in rows]

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

ivs = good_intervals(ev_ts, ev_lab, max(ecg_pk.max(), t1))
def good_mask(ts):
    g = np.zeros(len(ts), bool)
    for a,b in ivs: g |= (ts>=a)&(ts<b)
    return g

con2 = sqlite3.connect("/tmp/annotation_chelten_scg2.db")
cur2 = con2.cursor()
cur2.execute("SELECT DISTINCT timestamp FROM events WHERE label = '' ORDER BY timestamp")
scg_hand_pk = np.array([r[0] for r in cur2.fetchall()], dtype="int64")
cur2.execute("SELECT DISTINCT timestamp, label FROM events WHERE label != '' ORDER BY timestamp")
rows2 = cur2.fetchall()
scg_ev_ts = np.array([r[0] for r in rows2], dtype="int64")
scg_ev_lab = [r[1].lower() for r in rows2]
scg_ivs = good_intervals(scg_ev_ts, scg_ev_lab, max(ecg_pk.max(), t1))
def scg_good_mask(ts):
    g = np.zeros(len(ts), bool)
    for a,b in scg_ivs: g |= (ts>=a)&(ts<b)
    return g

def both_good(ts):
    return good_mask(ts) & scg_good_mask(ts)

ecg_pk_w = ecg_pk[(ecg_pk>=t0)&(ecg_pk<=t1)]
ecg_good = ecg_pk_w[both_good(ecg_pk_w)]
scg_hand_w = scg_hand_pk[(scg_hand_pk>=t0)&(scg_hand_pk<=t1)]
scg_hand_good = scg_hand_w[both_good(scg_hand_w)]

# --- primary pass: thr=0.3, refract=0.5s, then wide-template NCC filter >= 0.5 ---
PRIMARY_THR = 0.3; PRIMARY_REFRACT_S = 0.50; NCC_CUTOFF = 0.5
pk, _ = find_peaks(sharp_s, height=PRIMARY_THR, distance=max(1,int(PRIMARY_REFRACT_S*fsd_s)))
primary_all = ts_sd[pk]
primary_all = primary_all[(primary_all>=t0)&(primary_all<=t1)]
primary_all = primary_all[both_good(primary_all)]
primary_scores = np.array([ncc_score(t) for t in primary_all])
primary = np.sort(primary_all[primary_scores>=NCC_CUTOFF])
print("n primary (post NCC filter):", len(primary))

# --- relaxed candidate pool for search-back: much lower amplitude thr ---
RELAX_THR = 0.12; RELAX_REFRACT_S = 0.35
pk2, _ = find_peaks(sharp_s, height=RELAX_THR, distance=max(1,int(RELAX_REFRACT_S*fsd_s)))
relaxed_all = ts_sd[pk2]

SB_LO_S, SB_HI_S = 1.2, 1.8
gaps = np.diff(primary)/1e6
flagged = np.where((gaps>=SB_LO_S)&(gaps<=SB_HI_S))[0]
print(f"n primary-to-primary gaps in [{SB_LO_S},{SB_HI_S}]s: {len(flagged)}  (of {len(gaps)} total gaps)")

extra = []
for gi in flagged:
    a, b = primary[gi], primary[gi+1]
    # candidates strictly inside, with a little margin from both ends
    margin = int(0.15e6)
    m = (relaxed_all > a+margin) & (relaxed_all < b-margin)
    cand = relaxed_all[m]
    for c in cand:
        sc = ncc_score(c)
        if not np.isnan(sc) and sc >= 0.35:   # relaxed NCC bar, still requires beat-like shape
            extra.append(c)

extra = np.array(sorted(set(extra)), dtype="int64")
print("n recovered extra beats:", len(extra))

combined = np.sort(np.concatenate([primary, extra]))
# enforce refractory to avoid double-counting near-duplicates
keep = [combined[0]]
for t in combined[1:]:
    if t - keep[-1] >= int(0.35e6):
        keep.append(t)
combined = np.array(keep, dtype="int64")
print("n combined (post refractory merge):", len(combined))

with open("/tmp/combined_detector.pkl","wb") as f:
    pickle.dump(dict(primary=primary, extra=extra, combined=combined,
                      ecg_good=ecg_good, scg_hand_good=scg_hand_good,
                      t0=t0, t1=t1), f)

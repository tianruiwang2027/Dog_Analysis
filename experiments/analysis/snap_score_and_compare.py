import pickle, sqlite3
import numpy as np
import torch
import torch.nn as nn

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; fsd_s = d["fsd_s"]

with open("/tmp/snapped_candidates.pkl","rb") as f:
    SN = pickle.load(f)
cand_t = SN["final_t"]  # the snapped candidate positions -- these are now the "detected beat" times

with open("/tmp/cnn_mask_apply.pkl","rb") as f:
    CA = pickle.load(f)
LAG_US = CA["LAG_US"]; RESTRICT = CA["RESTRICT"]

class SCGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=9, padding=4)
        self.pool = nn.MaxPool1d(4)
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(16*10, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x)); x = self.pool(x)
        x = self.relu(self.conv2(x)); x = self.pool(x)
        x = x.flatten(1); x = self.drop(x)
        x = self.relu(self.fc1(x)); x = self.fc2(x)
        return x

with open("/tmp/cnn_train_snap_result.pkl","rb") as f:
    TR = pickle.load(f)
scale = TR["scale"]
net = SCGNet(); net.load_state_dict(torch.load("/tmp/cnn_model_snap.pt")); net.eval()
with open("/tmp/cnn_dataset_snap_relabel.pkl","rb") as f:
    HN = pickle.load(f)
HALF_N = HN["HALF_N"]

def snippet_env(tc):
    i = np.searchsorted(ts_sd, tc)
    if i-HALF_N < 0 or i+HALF_N+1 > len(env_sd): return None
    return env_sd[i-HALF_N:i+HALF_N+1]

def score_all():
    scores = np.full(len(cand_t), np.nan)
    batch=[]; idxs=[]
    def flush():
        if not batch: return
        with torch.no_grad():
            x = torch.tensor(np.array(batch), dtype=torch.float32).unsqueeze(1)
            p = torch.sigmoid(net(x)).numpy().ravel()
        scores[idxs] = p
        batch.clear(); idxs.clear()
    for i,tc in enumerate(cand_t):
        s = snippet_env(tc)
        if s is None: continue
        batch.append(s/scale); idxs.append(i)
        if len(batch)>=512: flush()
    flush()
    return scores

print("scoring whole session with SNAP model...")
scores_snap = score_all()
valid = ~np.isnan(scores_snap)
print(f"scored {valid.sum()}/{len(cand_t)}")

with open("/tmp/full_session_scores_snap.pkl","wb") as f:
    pickle.dump(dict(cand_t=cand_t, scores_snap=scores_snap), f)

# ---- downstream confirm + RR-sanity + HR-correlation pipeline, identical to three_model_r_compare.py ----
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
span_end = max(ecg_pk.max(), cand_t.max())
ecg_ivs = good_intervals(ecg_ev_ts, ecg_ev_lab, span_end)
ecg_tmid, ecg_hr, _ = beat_hr(ecg_pk)
ecg_hr_sm = smooth(ecg_tmid, ecg_hr)
ecg_v_full = good_mask(ecg_tmid, ecg_ivs)

def confirm_and_run(scores, cutoff, max_rr_s=1.5):
    valid = ~np.isnan(scores)
    confirmed = np.sort(cand_t[valid][scores[valid] >= cutoff])
    if len(confirmed) < 20: return None
    tmid, hr_raw, rr = beat_hr(confirmed)
    ok = rr <= max_rr_s
    tmid_ok, hr_ok = tmid[ok], hr_raw[ok]
    if len(tmid_ok) < 10: return None
    hr_sm = smooth(tmid_ok, hr_ok, win_s=3.0)
    ecg_v_at = good_mask(tmid_ok, ecg_ivs)
    xa, yb, ta = align(tmid_ok, hr_sm, ecg_v_at, ecg_tmid, ecg_hr_sm, ecg_v_full, lag_us=LAG_US, restrict=RESTRICT)
    s = stats(xa, yb)
    return dict(cutoff=cutoff, s=s, xa=xa, yb=yb, ta=ta)

print("=== CNN-SCG+ECG (snapped + relabeled) ===")
best = None
for cutoff in np.arange(0.05, 0.96, 0.05):
    r = confirm_and_run(scores_snap, cutoff)
    if r is None: continue
    print(f"  cutoff={cutoff:.2f}  n={r['s']['n']:5d}  r={r['s']['r']:.4f}  MAE={r['s']['mae']:.2f}")
    if best is None or r["s"]["r"] > best["s"]["r"]: best = r
print(f"  BEST: cutoff={best['cutoff']:.2f}  r={best['s']['r']:.4f}  n={best['s']['n']}  MAE={best['s']['mae']:.2f}")

with open("/tmp/snap_r_compare.pkl","wb") as f:
    pickle.dump(dict(best=best, scores_snap=scores_snap, cand_t=cand_t), f)
print("saved /tmp/snap_r_compare.pkl")

# side-by-side vs previous best (unsnapped relabeled model)
with open("/tmp/three_model_r_compare.pkl","rb") as f:
    prev = pickle.load(f)
print("\n=== SUMMARY: best r per model ===")
for name, res in prev.items():
    print(f"  {name}: r={res['s']['r']:.4f}  n={res['s']['n']}  cutoff={res['cutoff']:.2f}")
print(f"  CNN-SCG+ECG (SNAPPED + relabeled): r={best['s']['r']:.4f}  n={best['s']['n']}  cutoff={best['cutoff']:.2f}")

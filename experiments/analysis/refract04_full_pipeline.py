import pickle, sqlite3
import numpy as np
import torch, torch.nn as nn

# ---------- Same full pipeline (main CNN -> S2-recovery chain -> gap-aware reporting),
# unchanged trained models, run on candidates generated with refractory=0.4s instead of
# the default 0.5s. Nothing else about the pipeline changes. ----------

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]

with open("/tmp/refract_0.40_cand.pkl","rb") as f:
    cand_t = pickle.load(f)["cand_t"]
print(f"n primary candidates (refract=0.4s): {len(cand_t)}")

with open("/tmp/cnn_mask_apply.pkl","rb") as f:
    CA = pickle.load(f)
LAG_US = CA["LAG_US"]; RESTRICT = CA["RESTRICT"]

with open("/tmp/three_model_r_compare.pkl","rb") as f:
    prev = pickle.load(f)
CUTOFF_MAIN = prev["CNN-SCG+ECG (envelope, relabeled)"]["cutoff"]

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

class S2Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=9, padding=4)
        self.pool = nn.MaxPool1d(4)
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16*10, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x)); x = self.pool(x)
        x = self.relu(self.conv2(x)); x = self.pool(x)
        x = x.flatten(1); x = self.drop(x)
        x = self.relu(self.fc1(x)); x = self.fc2(x)
        return x

with open("/tmp/cnn_train_relabel_result.pkl","rb") as f:
    TRm = pickle.load(f)
scale_main = TRm["scale"]
net_main = SCGNet(); net_main.load_state_dict(torch.load("/tmp/cnn_model_relabel.pt")); net_main.eval()

with open("/tmp/s2_train_result.pkl","rb") as f:
    TRs = pickle.load(f)
scale_s2 = TRs["scale"]
net_s2 = S2Net(); net_s2.load_state_dict(torch.load("/tmp/s2_model.pt")); net_s2.eval()

HALF_N = 80
def snippet(t):
    i = np.searchsorted(ts_sd, t)
    if i-HALF_N < 0 or i+HALF_N+1 > len(env_sd): return None
    return env_sd[i-HALF_N:i+HALF_N+1]
def score_one(net, scale, t):
    s = snippet(t)
    if s is None: return None
    with torch.no_grad():
        x = torch.tensor((s/scale)[None,None,:], dtype=torch.float32)
        return float(torch.sigmoid(net(x)).item())
def score_all(net, scale, cand_t):
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
        s = snippet(tc)
        if s is None: continue
        batch.append(s/scale); idxs.append(i)
        if len(batch)>=512: flush()
    flush()
    return scores

# ---- step 1: main CNN scores every refract=0.4 candidate ----
scores_main = score_all(net_main, scale_main, cand_t)
valid = ~np.isnan(scores_main)
confirmed0 = set(cand_t[valid][scores_main[valid] >= CUTOFF_MAIN].tolist())
rejected_t = cand_t[valid][scores_main[valid] < CUTOFF_MAIN]
print(f"main CNN confirmed: {len(confirmed0)}   rejected (candidates for S2 chain): {len(rejected_t)}")

# ---- step 2/3: S2-recovery chain, same cutoff as before (0.70) ----
SEARCH_LO_US, SEARCH_HI_US = 100_000, 300_000
MERGE_TOL_US = 50_000
S2_CUTOFF = 0.70

def backward_local_max(tc):
    lo, hi = tc-SEARCH_HI_US, tc-SEARCH_LO_US
    i_lo, i_hi = np.searchsorted(ts_sd, lo), np.searchsorted(ts_sd, hi)
    if i_hi <= i_lo: return None
    seg = env_sd[i_lo:i_hi]
    j = np.argmax(seg)
    return ts_sd[i_lo+j]

n_flagged = 0; n_accepted = 0
recovered = []
for tc in rejected_t:
    p_s2 = score_one(net_s2, scale_s2, tc)
    if p_s2 is None or p_s2 < S2_CUTOFF: continue
    n_flagged += 1
    t_back = backward_local_max(tc)
    if t_back is None: continue
    p_main = score_one(net_main, scale_main, t_back)
    if p_main is None or p_main < CUTOFF_MAIN: continue
    n_accepted += 1
    recovered.append(t_back)
recovered = np.array(recovered, dtype="int64")
print(f"S2-flagged: {n_flagged}   recovered+accepted: {n_accepted}")

merged = set(confirmed0)
for rt in recovered:
    nearby = [c for c in merged if abs(c-rt) < MERGE_TOL_US]
    if not nearby: merged.add(int(rt))
merged_arr = np.array(sorted(merged), dtype="int64")
print(f"final confirmed set (pre-gap-aware): {len(merged_arr)}")

with open("/tmp/refract04_merged.pkl","wb") as f:
    pickle.dump(dict(merged_arr=merged_arr), f)

# ---- step 4: gap-aware reporting, identical logic to before ----
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
def smooth_plain(tmid, hr, win_s=3.0):
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
span_end = max(ecg_pk.max(), merged_arr.max())
ecg_ivs = good_intervals(ecg_ev_ts, ecg_ev_lab, span_end)
ecg_tmid, ecg_hr, _ = beat_hr(ecg_pk)
ecg_hr_sm = smooth_plain(ecg_tmid, ecg_hr)
ecg_v_full = good_mask(ecg_tmid, ecg_ivs)

tmid, hr_raw, rr = beat_hr(merged_arr)
MAX_RR_S = 1.5
ok = rr <= MAX_RR_S
tmid_ok, hr_ok = tmid[ok], hr_raw[ok]

GAP_THRESH_S = 1.5; WIN_S = 3.0
win_us = int(WIN_S*1e6); gap_us = int(GAP_THRESH_S*1e6)
all_gaps_len = np.diff(merged_arr)
has_gap = all_gaps_len >= gap_us
gap_windows = list(zip(merged_arr[:-1][has_gap], merged_arr[1:][has_gap]))
print(f"n detection gaps (>= {GAP_THRESH_S}s): {len(gap_windows)}")

def window_touches_a_gap(t):
    lo, hi = t-win_us, t+win_us
    for gs, ge in gap_windows:
        if ge >= lo and gs <= hi: return True
    return False

hr_sm_new = np.full(len(tmid_ok), np.nan)
n_suppressed = 0
for i, t in enumerate(tmid_ok):
    if window_touches_a_gap(t):
        n_suppressed += 1
        continue
    sel = (tmid_ok>=t-win_us)&(tmid_ok<=t+win_us)
    hr_sm_new[i] = hr_ok[sel].mean()

keep_new = ~np.isnan(hr_sm_new)
ecg_v_at_new = good_mask(tmid_ok[keep_new], ecg_ivs)
xa_new, yb_new, ta_new = align(tmid_ok[keep_new], hr_sm_new[keep_new], ecg_v_at_new, ecg_tmid, ecg_hr_sm, ecg_v_full, lag_us=LAG_US, restrict=RESTRICT)
s_new = stats(xa_new, yb_new)

print(f"\n=== FULL PIPELINE, refract=0.4s ===")
print(f"r={s_new['r']:.4f}  n={s_new['n']}  MAE={s_new['mae']:.2f}  bias={s_new['bias']:.2f}")
print(f"(vs full pipeline refract=0.5s: r=0.9095  n=1285  MAE=2.20)")

with open("/tmp/refract04_full_result.pkl","wb") as f:
    pickle.dump(dict(xa=xa_new, yb=yb_new, ta=ta_new, s=s_new, n_flagged=n_flagged, n_accepted=n_accepted,
                      n_suppressed=n_suppressed, merged_arr=merged_arr), f)
print("saved")

import pickle, sqlite3
import numpy as np
import torch
import torch.nn as nn

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; fsd_s = d["fsd_s"]

with open("/tmp/full_session_scores_all_models.pkl","rb") as f:
    S = pickle.load(f)
cand_t = S["cand_t"]; scores_main = S["scores_relabel"]

with open("/tmp/cnn_mask_apply.pkl","rb") as f:
    CA = pickle.load(f)
LAG_US = CA["LAG_US"]; RESTRICT = CA["RESTRICT"]

with open("/tmp/three_model_r_compare.pkl","rb") as f:
    prev = pickle.load(f)
CUTOFF_MAIN = prev["CNN-SCG+ECG (envelope, relabeled)"]["cutoff"]
print(f"main CNN cutoff: {CUTOFF_MAIN:.2f}  (baseline best: r={prev['CNN-SCG+ECG (envelope, relabeled)']['s']['r']:.4f}  n={prev['CNN-SCG+ECG (envelope, relabeled)']['s']['n']})")

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

HALF_N = 80  # same ±400ms @ fsd_s used throughout

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

# ---- step 1: candidates already confirmed by main CNN ----
valid = ~np.isnan(scores_main)
confirmed0 = set(cand_t[valid][scores_main[valid] >= CUTOFF_MAIN].tolist())
rejected_t = cand_t[valid][scores_main[valid] < CUTOFF_MAIN]
print(f"confirmed by main CNN (step 1): {len(confirmed0)}")
print(f"rejected by main CNN, candidates for the S2 chain: {len(rejected_t)}")

SEARCH_LO_US, SEARCH_HI_US = 100_000, 300_000
MERGE_TOL_US = 50_000

def backward_local_max(tc):
    lo, hi = tc-SEARCH_HI_US, tc-SEARCH_LO_US
    i_lo, i_hi = np.searchsorted(ts_sd, lo), np.searchsorted(ts_sd, hi)
    if i_hi <= i_lo: return None
    seg = env_sd[i_lo:i_hi]
    j = np.argmax(seg)
    return ts_sd[i_lo+j]

def run_chain(S2_CUTOFF):
    n_flagged_s2 = 0; n_recovered_accepted = 0
    recovered = []
    for tc in rejected_t:
        p_s2 = score_one(net_s2, scale_s2, tc)
        if p_s2 is None or p_s2 < S2_CUTOFF: continue
        n_flagged_s2 += 1
        # step 2: backward search for the plausible true S1
        t_back = backward_local_max(tc)
        if t_back is None: continue
        # step 3: re-score the recovered point with the ORIGINAL, well-validated main CNN
        p_main = score_one(net_main, scale_main, t_back)
        if p_main is None or p_main < CUTOFF_MAIN: continue
        n_recovered_accepted += 1
        recovered.append(t_back)
    return n_flagged_s2, n_recovered_accepted, np.array(recovered, dtype="int64")

# ---- downstream RR-sanity + HR-correlation pipeline (same as three_model_r_compare.py) ----
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

def eval_confirmed_set(confirmed, max_rr_s=1.5):
    confirmed = np.sort(confirmed)
    if len(confirmed) < 20: return None
    tmid, hr_raw, rr = beat_hr(confirmed)
    ok = rr <= max_rr_s
    tmid_ok, hr_ok = tmid[ok], hr_raw[ok]
    if len(tmid_ok) < 10: return None
    hr_sm = smooth(tmid_ok, hr_ok, win_s=3.0)
    ecg_v_at = good_mask(tmid_ok, ecg_ivs)
    xa, yb, ta = align(tmid_ok, hr_sm, ecg_v_at, ecg_tmid, ecg_hr_sm, ecg_v_full, lag_us=LAG_US, restrict=RESTRICT)
    return stats(xa, yb)

# baseline: just the main CNN's confirmed set (no chain)
base_stats = eval_confirmed_set(np.array(sorted(confirmed0), dtype="int64"))
print(f"\nsanity check -- main CNN alone via this eval path: r={base_stats['r']:.4f}  n={base_stats['n']}  MAE={base_stats['mae']:.2f}")

print("\n=== sweeping S2 classifier cutoff ===")
results = {}
best = None
for S2_CUTOFF in np.arange(0.1, 0.96, 0.1):
    n_flag, n_acc, recovered = run_chain(S2_CUTOFF)
    merged = set(confirmed0)
    n_dupe = 0
    for rt in recovered:
        # dedupe against existing confirmed set within MERGE_TOL_US
        nearby = [c for c in merged if abs(c-rt) < MERGE_TOL_US]
        if nearby:
            n_dupe += 1
            continue
        merged.add(int(rt))
    merged_arr = np.array(sorted(merged), dtype="int64")
    st = eval_confirmed_set(merged_arr)
    if st is None: continue
    print(f"  S2_cutoff={S2_CUTOFF:.2f}  flagged_S2={n_flag:4d}  recovered+accepted={n_acc:4d}  dupes_skipped={n_dupe:3d}  "
          f"-> n={st['n']:5d}  r={st['r']:.4f}  MAE={st['mae']:.2f}")
    results[round(S2_CUTOFF,2)] = dict(n_flag=n_flag, n_acc=n_acc, n_dupe=n_dupe, s=st, recovered=recovered)
    if best is None or st["r"] > best[1]["s"]["r"]:
        best = (S2_CUTOFF, results[round(S2_CUTOFF,2)])

print(f"\nBEST: S2_cutoff={best[0]:.2f}  r={best[1]['s']['r']:.4f}  n={best[1]['s']['n']}  MAE={best[1]['s']['mae']:.2f}  "
      f"(vs baseline main-CNN-alone r={base_stats['r']:.4f} n={base_stats['n']}, and previous snap-strategy r=0.7968)")

with open("/tmp/s2_chain_results.pkl","wb") as f:
    pickle.dump(dict(results=results, best=best, base_stats=base_stats, confirmed0=np.array(sorted(confirmed0),dtype="int64")), f)
print("saved /tmp/s2_chain_results.pkl")

import pickle
import numpy as np
from scipy import signal as sg
import torch, torch.nn as nn

# ---------- Step 2: primary candidate detection (PRIMARY_THR=0.3, PRIMARY_REFRACT_S=0.50,
# unchanged) -> main CNN (unchanged weights, cutoff=0.25) -> S2-recovery chain (unchanged
# weights, cutoff=0.70) -- the EXACT same trained models and thresholds used for the
# validated 34-min window, now applied to the full-session envelope from step 1. No
# retraining, no threshold changes -- this is purely running the frozen pipeline further. ----------

with open("/tmp/extended_full_envelope.pkl","rb") as f:
    E = pickle.load(f)
ts_sd, env_sd, sharp_s, fsd_s = E["ts_sd"], E["env_sd"], E["sharp_s"], E["fsd_s"]

PRIMARY_THR = 0.3; PRIMARY_REFRACT_S = 0.50
pk_idx, _ = sg.find_peaks(sharp_s, height=PRIMARY_THR, distance=max(1,int(PRIMARY_REFRACT_S*fsd_s)))
cand_t = ts_sd[pk_idx]
print(f"primary candidates (full session): {len(cand_t)}")

with open("/tmp/cnn_mask_apply.pkl","rb") as f:
    CA = pickle.load(f)
LAG_US = CA["LAG_US"]

with open("/tmp/three_model_r_compare.pkl","rb") as f:
    prev = pickle.load(f)
CUTOFF_MAIN = prev["CNN-SCG+ECG (envelope, relabeled)"]["cutoff"]
print(f"CUTOFF_MAIN={CUTOFF_MAIN}  LAG_US={LAG_US}")

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

# ---- step A: main CNN scores every candidate ----
scores_main = score_all(net_main, scale_main, cand_t)
valid = ~np.isnan(scores_main)
confirmed0 = set(cand_t[valid][scores_main[valid] >= CUTOFF_MAIN].tolist())
rejected_t = cand_t[valid][scores_main[valid] < CUTOFF_MAIN]
print(f"main CNN confirmed: {len(confirmed0)}   rejected (candidates for S2 chain): {len(rejected_t)}")

# ---- step B/C: S2-recovery chain, same cutoff as validated run (0.70) ----
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
print(f"final confirmed SCG beat set (pre-gap-aware, full session): {len(merged_arr)}")

with open("/tmp/extended_full_merged.pkl","wb") as f:
    pickle.dump(dict(merged_arr=merged_arr, cand_t=cand_t, n_flagged=n_flagged, n_accepted=n_accepted,
                      LAG_US=LAG_US, CUTOFF_MAIN=CUTOFF_MAIN), f)
print("saved /tmp/extended_full_merged.pkl")

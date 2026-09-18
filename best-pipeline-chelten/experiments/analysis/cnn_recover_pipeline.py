import pickle, sqlite3
import numpy as np
import torch
import torch.nn as nn

# ---------- recovery pipeline: when a candidate is rejected by the binary "is this a
# beat" CNN, ask the new 3-class CNN whether it looks like S2 (the second heart sound).
# If so, search backward in the raw envelope for the real S1 peak the primary detector's
# amplitude threshold missed, and verify THAT candidate with the original binary CNN
# before accepting it -- so we only recover a beat when there's real shape evidence for
# one, never by just guessing a fixed offset. ----------

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; sharp_s = d["sharp_s"]; fsd_s = d["fsd_s"]
ts_s = d["ts_s"]

with open("/tmp/cnn_mask_apply.pkl","rb") as f:
    CA = pickle.load(f)
cand_t = CA["cand_t"]; cand_cnn = CA["cand_cnn"]     # original binary scores, whole session
LAG_US = CA["LAG_US"]; RESTRICT = CA["RESTRICT"]

with open("/tmp/cnn_train_result.pkl","rb") as f:
    TR1 = pickle.load(f)
scale1 = TR1["scale"]

with open("/tmp/cnn3_train_result.pkl","rb") as f:
    TR3 = pickle.load(f)
scale3 = TR3["scale"]

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

class SCGNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=9, padding=4)
        self.pool = nn.MaxPool1d(4)
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(16*10, 32)
        self.fc2 = nn.Linear(32, 3)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x)); x = self.pool(x)
        x = self.relu(self.conv2(x)); x = self.pool(x)
        x = x.flatten(1); x = self.drop(x)
        x = self.relu(self.fc1(x)); x = self.fc2(x)
        return x

net1 = SCGNet(); net1.load_state_dict(torch.load("/tmp/cnn_model.pt")); net1.eval()
net3 = SCGNet3(); net3.load_state_dict(torch.load("/tmp/cnn3_model.pt")); net3.eval()

HALF_N = TR1  # placeholder, real HALF_N below
with open("/tmp/cnn_dataset.pkl","rb") as f:
    HN = pickle.load(f)
HALF_N = HN["HALF_N"]

def score_beat(t):
    """binary is-this-S1 score, using net1"""
    i = np.searchsorted(ts_sd, t)
    if i-HALF_N < 0 or i+HALF_N+1 > len(env_sd): return np.nan
    snip = env_sd[i-HALF_N:i+HALF_N+1] / scale1
    with torch.no_grad():
        x = torch.tensor(snip, dtype=torch.float32).view(1,1,-1)
        return float(torch.sigmoid(net1(x)).item())

def classify3(t):
    """3-class S1/S2/noise probs, using net3"""
    i = np.searchsorted(ts_sd, t)
    if i-HALF_N < 0 or i+HALF_N+1 > len(env_sd): return None
    snip = env_sd[i-HALF_N:i+HALF_N+1] / scale3
    with torch.no_grad():
        x = torch.tensor(snip, dtype=torch.float32).view(1,1,-1)
        return torch.softmax(net3(x), dim=1).numpy().ravel()

CUTOFF = 0.40          # same confirm cutoff used in the old per-beat-confirm pipeline
S2_LO_US, S2_HI_US = 100_000, 300_000   # backward search window for the missing S1
S2_PROB_MIN = 0.5

def recover_S1(t_s2):
    """search backward for a local envelope max in the plausible S1 window, verify with net1"""
    lo, hi = t_s2 - S2_HI_US, t_s2 - S2_LO_US
    i_lo = np.searchsorted(ts_sd, lo); i_hi = np.searchsorted(ts_sd, hi)
    if i_hi <= i_lo: return None
    seg = env_sd[i_lo:i_hi]
    if len(seg) == 0: return None
    j = np.argmax(seg)
    t_rec = ts_sd[i_lo + j]
    s = score_beat(t_rec)
    if s is not None and not np.isnan(s) and s >= CUTOFF:
        return t_rec, s
    return None

print(f"scanning {len(cand_t)} candidates for rejected-but-recoverable S2 cases (cutoff={CUTOFF})...")
rejected_idx = np.where(cand_cnn < CUTOFF)[0]
print(f"  n rejected by original binary CNN: {len(rejected_idx)}")

recovered = []
s2_flagged = 0
for i in rejected_idx:
    t = cand_t[i]
    probs = classify3(t)
    if probs is None: continue
    if probs.argmax() == 1 and probs[1] >= S2_PROB_MIN:   # class 1 = S2
        s2_flagged += 1
        rec = recover_S1(t)
        if rec is not None:
            recovered.append(rec)

print(f"  n flagged as S2 by 3-class net: {s2_flagged}")
print(f"  n successfully recovered (backward search + net1 re-verified): {len(recovered)}")

with open("/tmp/cnn_recover_result.pkl","wb") as f:
    pickle.dump(dict(recovered=recovered, s2_flagged=s2_flagged, cutoff=CUTOFF,
                      s2_lo_us=S2_LO_US, s2_hi_us=S2_HI_US), f)
print("saved /tmp/cnn_recover_result.pkl")

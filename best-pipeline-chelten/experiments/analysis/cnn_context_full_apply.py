import pickle
import numpy as np
import torch
import torch.nn as nn

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; fsd_s = d["fsd_s"]

with open("/tmp/cnn_mask_apply.pkl","rb") as f:
    CA = pickle.load(f)
cand_t = CA["cand_t"]        # ALL session candidates (same primary detector, whole session)
LAG_US = CA["LAG_US"]; RESTRICT = CA["RESTRICT"]

with open("/tmp/cnn_context_dataset.pkl","rb") as f:
    D = pickle.load(f)
HALF_N = D["HALF_N"]; CORE_S = D["CORE_S"]; FLANK_S = D["FLANK_S"]

with open("/tmp/cnn_context_train_result.pkl","rb") as f:
    TR = pickle.load(f)
scale = TR["scale"]; bg_mean = TR["bg_mean"]; bg_std = TR["bg_std"]

class SCGNetContext(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=9, padding=4)
        self.pool = nn.MaxPool1d(4)
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(16*10 + 1, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    def forward(self, x, bgz):
        x = self.relu(self.conv1(x)); x = self.pool(x)
        x = self.relu(self.conv2(x)); x = self.pool(x)
        x = x.flatten(1); x = self.drop(x)
        x = torch.cat([x, bgz], dim=1)
        x = self.relu(self.fc1(x)); x = self.fc2(x)
        return x

net = SCGNetContext(); net.load_state_dict(torch.load("/tmp/cnn_context_model.pt")); net.eval()

def snippet(t):
    i = np.searchsorted(ts_sd, t)
    if i-HALF_N < 0 or i+HALF_N+1 > len(env_sd): return None
    return env_sd[i-HALF_N:i+HALF_N+1].copy()

def bg_rms(tc):
    lo1, hi1 = tc-FLANK_S*1e6, tc-CORE_S*1e6
    lo2, hi2 = tc+CORE_S*1e6,  tc+FLANK_S*1e6
    i1a, i1b = np.searchsorted(ts_sd, lo1), np.searchsorted(ts_sd, hi1)
    i2a, i2b = np.searchsorted(ts_sd, lo2), np.searchsorted(ts_sd, hi2)
    seg = np.concatenate([env_sd[i1a:i1b], env_sd[i2a:i2b]])
    if len(seg) < 20: return np.nan
    return float(np.sqrt(np.mean(seg.astype(np.float64)**2)))

print(f"scoring {len(cand_t)} whole-session candidates with the context model...")
scores = np.full(len(cand_t), np.nan)
BATCH=512
snips_list=[]; bgz_list=[]; idx_list=[]
for i,tc in enumerate(cand_t):
    s = snippet(tc); b = bg_rms(tc)
    if s is None or np.isnan(b): continue
    snips_list.append(s/scale)
    bgz_list.append((np.log1p(b*1e4)-bg_mean)/bg_std)
    idx_list.append(i)
    if len(snips_list)>=BATCH:
        with torch.no_grad():
            x = torch.tensor(np.array(snips_list), dtype=torch.float32).unsqueeze(1)
            bz = torch.tensor(np.array(bgz_list), dtype=torch.float32).unsqueeze(1)
            p = torch.sigmoid(net(x,bz)).numpy().ravel()
        scores[idx_list] = p
        snips_list=[]; bgz_list=[]; idx_list=[]
if snips_list:
    with torch.no_grad():
        x = torch.tensor(np.array(snips_list), dtype=torch.float32).unsqueeze(1)
        bz = torch.tensor(np.array(bgz_list), dtype=torch.float32).unsqueeze(1)
        p = torch.sigmoid(net(x,bz)).numpy().ravel()
    scores[idx_list] = p

valid = ~np.isnan(scores)
print(f"scored {valid.sum()} of {len(cand_t)}")

with open("/tmp/cnn_context_full_scores.pkl","wb") as f:
    pickle.dump(dict(cand_t=cand_t[valid], cand_ctx=scores[valid], LAG_US=LAG_US, RESTRICT=RESTRICT), f)
print("saved /tmp/cnn_context_full_scores.pkl")

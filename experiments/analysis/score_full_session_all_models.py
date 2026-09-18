import pickle
import numpy as np
import torch
import torch.nn as nn

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; ts_s = d["ts_s"]; xf_s = d["xf_s"]; fs_s = d["fs_s"]

with open("/tmp/cnn_mask_apply.pkl","rb") as f:
    CA = pickle.load(f)
cand_t = CA["cand_t"]

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

class SCGNetRaw(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=151, padding=75)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=91, padding=45)
        self.pool1 = nn.MaxPool1d(10)
        self.pool2 = nn.MaxPool1d(10)
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(16*16, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x)); x = self.pool1(x)
        x = self.relu(self.conv2(x)); x = self.pool2(x)
        x = x.flatten(1); x = self.drop(x)
        x = self.relu(self.fc1(x)); x = self.fc2(x)
        return x

# ---- model 2: envelope, SCG+ECG relabeled ----
with open("/tmp/cnn_train_relabel_result.pkl","rb") as f:
    TR2 = pickle.load(f)
scale2 = TR2["scale"]
net2 = SCGNet(); net2.load_state_dict(torch.load("/tmp/cnn_model_relabel.pt")); net2.eval()
with open("/tmp/cnn_dataset_relabel.pkl","rb") as f:
    HN2 = pickle.load(f)
HALF_N2 = HN2["HALF_N"]

def snippet_env(tc):
    i = np.searchsorted(ts_sd, tc)
    if i-HALF_N2 < 0 or i+HALF_N2+1 > len(env_sd): return None
    return env_sd[i-HALF_N2:i+HALF_N2+1]

# ---- model 3: raw signal, SCG+ECG relabeled ----
with open("/tmp/cnn_raw_train_result.pkl","rb") as f:
    TR3 = pickle.load(f)
scale3 = TR3["scale"]
net3 = SCGNetRaw(); net3.load_state_dict(torch.load("/tmp/cnn_model_raw.pt")); net3.eval()
with open("/tmp/cnn_dataset_raw.pkl","rb") as f:
    HN3 = pickle.load(f)
HALF_N3 = HN3["HALF_N"]

def snippet_raw(tc):
    i = np.searchsorted(ts_s, tc)
    if i-HALF_N3 < 0 or i+HALF_N3+1 > len(xf_s): return None
    return xf_s[i-HALF_N3:i+HALF_N3+1]

def score_all(net, scale, snippet_fn, half_n):
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
        s = snippet_fn(tc)
        if s is None: continue
        batch.append(s/scale); idxs.append(i)
        if len(batch)>=512: flush()
    flush()
    return scores

print("scoring model 2 (envelope, relabeled)...")
scores2 = score_all(net2, scale2, snippet_env, HALF_N2)
print("scoring model 3 (raw, relabeled)...")
scores3 = score_all(net3, scale3, snippet_raw, HALF_N3)

valid2 = ~np.isnan(scores2); valid3 = ~np.isnan(scores3)
print(f"model2 scored {valid2.sum()}/{len(cand_t)}   model3 scored {valid3.sum()}/{len(cand_t)}")

with open("/tmp/full_session_scores_all_models.pkl","wb") as f:
    pickle.dump(dict(cand_t=cand_t, scores_scg=CA["cand_cnn"], scores_relabel=scores2, scores_raw=scores3), f)
print("saved /tmp/full_session_scores_all_models.pkl")

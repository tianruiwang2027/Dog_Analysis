import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/cnn_train_result.pkl","rb") as f:
    TR = pickle.load(f)
scale = TR["scale"]
with open("/tmp/cnn_dataset.pkl","rb") as f:
    D = pickle.load(f)
t = D["t"]; label = D["label"]; snips = D["snips"]; fsd_s = D["fsd_s"]; HALF_N = D["HALF_N"]

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

net = SCGNet(); net.load_state_dict(torch.load("/tmp/cnn_model.pt")); net.eval()

# ---------- panel A: the 8 learned first-layer filters (each is a ~75ms waveform template) ----------
w = net.conv1.weight.detach().numpy()  # (8,1,15)
kernel_ms = np.arange(15)/fsd_s*1000
kernel_ms -= kernel_ms.mean()

fig, axs = plt.subplots(2, 5, figsize=(16,6))
for i in range(8):
    ax = axs.flat[i]
    ax.plot(kernel_ms, w[i,0], color="tab:blue", lw=1.8)
    ax.axhline(0, color="0.8", lw=0.6)
    ax.set_title(f"filter {i}", fontsize=9)
    ax.set_xlabel("ms", fontsize=7)
    ax.tick_params(labelsize=7)

# panel: example true-positive and false-positive-ish snippets for context
n = len(t); n_train = TR["n_train"]
test_prob = TR["test_prob"]; y_test = TR["y_test"]; t_test = TR["t_test"]
snips_test = snips[n_train:]

hi_tp_idx = np.argsort(-(test_prob*(y_test==1)))[0]     # confident true positive
example_neg_hi = np.where((y_test==0)&(test_prob>0.5))[0]
hard_fp_idx = example_neg_hi[np.argmax(test_prob[example_neg_hi])] if len(example_neg_hi) else None

x_ms = (np.arange(2*HALF_N+1)-HALF_N)/fsd_s*1000

ax = axs.flat[8]
ax.plot(x_ms, snips_test[hi_tp_idx]/scale, color="tab:green", lw=1.3)
ax.set_title(f"confident TRUE beat\nscore={test_prob[hi_tp_idx]:.2f}", fontsize=9)
ax.set_xlabel("ms", fontsize=7); ax.tick_params(labelsize=7)

ax = axs.flat[9]
if hard_fp_idx is not None:
    ax.plot(x_ms, snips_test[hard_fp_idx]/scale, color="tab:red", lw=1.3)
    ax.set_title(f"hardest FALSE positive\nscore={test_prob[hard_fp_idx]:.2f}\n(not a real beat)", fontsize=9)
ax.set_xlabel("ms", fontsize=7); ax.tick_params(labelsize=7)

fig.suptitle("What the CNN learned: first-layer filters act as ~75ms waveform-shape detectors,\n"
             "the deeper layers combine their outputs into a single beat/not-beat score", fontsize=11)
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_cnn_interpret.png", dpi=135)
print("saved")
print(f"hardest FP in test set: score={test_prob[hard_fp_idx]:.3f}  t={t_test[hard_fp_idx]}")

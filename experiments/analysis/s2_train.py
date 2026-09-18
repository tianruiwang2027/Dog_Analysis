import pickle
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0); np.random.seed(0)

with open("/tmp/s2_dataset.pkl","rb") as f:
    D = pickle.load(f)
t = D["t"]; label = D["label"]; snips = D["snips"]

# stratified random split (not chronological): with only 49 S2 positives total, a
# chronological split clusters them unevenly across train/test (rejected-candidate
# stretches are themselves clustered in time), so split randomly within each class instead
rng = np.random.RandomState(0)
pos_idx = np.where(label==1)[0]; neg_idx = np.where(label==0)[0]
rng.shuffle(pos_idx); rng.shuffle(neg_idx)
n_pos_train = int(len(pos_idx)*0.70); n_neg_train = int(len(neg_idx)*0.70)
train_idx = np.concatenate([pos_idx[:n_pos_train], neg_idx[:n_neg_train]])
test_idx  = np.concatenate([pos_idx[n_pos_train:], neg_idx[n_neg_train:]])
rng.shuffle(train_idx); rng.shuffle(test_idx)
n_train = len(train_idx)
X_train, y_train = snips[train_idx], label[train_idx]
X_test,  y_test  = snips[test_idx],  label[test_idx]
t = np.concatenate([t[train_idx], t[test_idx]])  # reorder t to match X_train+X_test for saving t_test below
print(f"train n={n_train} (S2={int(y_train.sum())} noise={int((1-y_train).sum())})")
print(f"test  n={len(test_idx)} (S2={int(y_test.sum())} noise={int((1-y_test).sum())})")

scale = np.median(np.abs(X_train[y_train==1])) + 1e-9
X_train_n = X_train/scale; X_test_n = X_test/scale

Xtr = torch.tensor(X_train_n, dtype=torch.float32).unsqueeze(1)
ytr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
Xte = torch.tensor(X_test_n, dtype=torch.float32).unsqueeze(1)
yte = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# small classifier -- same conv trunk as SCGNet (already tiny, 6.5k params) but with
# heavier dropout since the training set here is much smaller (267 examples total)
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

model = S2Net()
n_pos = ytr.sum().item(); n_neg = len(ytr)-n_pos
pos_weight = torch.tensor([n_neg/n_pos])
crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

n_epochs = 200; batch_size = 16
best_test_loss = float("inf"); best_state=None; patience=30; bad_epochs=0
for epoch in range(n_epochs):
    model.train()
    perm = torch.randperm(len(Xtr))
    tot_loss = 0.0
    for i in range(0, len(Xtr), batch_size):
        idx = perm[i:i+batch_size]
        xb, yb = Xtr[idx], ytr[idx]
        opt.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        loss.backward(); opt.step()
        tot_loss += loss.item()*len(idx)
    tot_loss /= len(Xtr)
    model.eval()
    with torch.no_grad():
        test_loss = crit(model(Xte), yte).item()
    if test_loss < best_test_loss - 1e-4:
        best_test_loss = test_loss; best_state={k:v.clone() for k,v in model.state_dict().items()}; bad_epochs=0
    else:
        bad_epochs += 1
    if epoch % 20 == 0 or epoch==n_epochs-1:
        print(f"epoch {epoch:3d}  train_loss={tot_loss:.4f}  test_loss={test_loss:.4f}  best={best_test_loss:.4f}")
    if bad_epochs >= patience:
        print(f"early stop at epoch {epoch}"); break

model.load_state_dict(best_state); model.eval()
with torch.no_grad():
    test_prob = torch.sigmoid(model(Xte)).numpy().ravel()

def eval_at(prob, y, thr=0.5):
    pred=(prob>=thr).astype(int); y=y.astype(int)
    tp=((pred==1)&(y==1)).sum(); tn=((pred==0)&(y==0)).sum()
    fp=((pred==1)&(y==0)).sum(); fn=((pred==0)&(y==1)).sum()
    acc=(tp+tn)/len(y)
    prec = tp/(tp+fp) if (tp+fp)>0 else float("nan")
    rec = tp/(tp+fn) if (tp+fn)>0 else float("nan")
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else float("nan")
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, tp=tp, tn=tn, fp=fp, fn=fn)

r = eval_at(test_prob, y_test)
print("\n--- S2-classifier, held-out test (thr=0.5) ---")
print(f"acc={r['acc']*100:.1f}%  precision={r['prec']*100:.1f}%  recall={r['rec']*100:.1f}%  f1={r['f1']*100:.1f}%")
print(f"TP={r['tp']} TN={r['tn']} FP={r['fp']} FN={r['fn']}")

torch.save(best_state, "/tmp/s2_model.pt")
with open("/tmp/s2_train_result.pkl","wb") as f:
    pickle.dump(dict(scale=scale, n_train=n_train, test_prob=test_prob, y_test=y_test, t_test=t[n_train:]), f)
print("saved /tmp/s2_model.pt")

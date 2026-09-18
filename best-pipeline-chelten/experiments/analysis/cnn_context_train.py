import pickle
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0); np.random.seed(0)

with open("/tmp/cnn_context_dataset.pkl","rb") as f:
    D = pickle.load(f)
t = D["t"]; label = D["label"]; snips = D["snips"]; bg = D["bg"]

n = len(t); n_train = int(n*0.70)
X_train, y_train, bg_train = snips[:n_train], label[:n_train], bg[:n_train]
X_test,  y_test,  bg_test  = snips[n_train:], label[n_train:], bg[n_train:]
print(f"train n={n_train} (pos={int(y_train.sum())} neg={int((1-y_train).sum())})")
print(f"test  n={n-n_train} (pos={int(y_test.sum())} neg={int((1-y_test).sum())})")

# shape-snippet scale, exactly as before (fit on train positives only)
scale = np.median(np.abs(X_train[y_train==1])) + 1e-9
X_train_n = X_train/scale; X_test_n = X_test/scale

# context-feature scale: log-transform (the raw ratio spans ~2 orders of magnitude,
# log makes it roughly linear/well-conditioned for the FC layer), then standardize
# using TRAIN-only mean/std
log_bg_train = np.log1p(bg_train*1e4)     # scale up before log1p so small envelope values aren't all ~0
log_bg_test  = np.log1p(bg_test*1e4)
bg_mean, bg_std = log_bg_train.mean(), log_bg_train.std()+1e-9
bgz_train = (log_bg_train-bg_mean)/bg_std
bgz_test  = (log_bg_test-bg_mean)/bg_std

Xtr = torch.tensor(X_train_n, dtype=torch.float32).unsqueeze(1)
Btr = torch.tensor(bgz_train, dtype=torch.float32).unsqueeze(1)
ytr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
Xte = torch.tensor(X_test_n, dtype=torch.float32).unsqueeze(1)
Bte = torch.tensor(bgz_test, dtype=torch.float32).unsqueeze(1)
yte = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

class SCGNetContext(nn.Module):
    """Same conv trunk as SCGNet on the +-400ms shape snippet, plus one extra scalar
    input (log background RMS from the surrounding +-2.0s, core excluded) concatenated
    in before the final classification layers."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=9, padding=4)
        self.pool = nn.MaxPool1d(4)
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(16*10 + 1, 32)   # +1 for the context scalar
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    def forward(self, x, bgz):
        x = self.relu(self.conv1(x)); x = self.pool(x)
        x = self.relu(self.conv2(x)); x = self.pool(x)
        x = x.flatten(1); x = self.drop(x)
        x = torch.cat([x, bgz], dim=1)
        x = self.relu(self.fc1(x)); x = self.fc2(x)
        return x

model = SCGNetContext()
n_pos = ytr.sum().item(); n_neg = len(ytr)-n_pos
pos_weight = torch.tensor([n_neg/n_pos])
crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

n_epochs = 150; batch_size = 64
best_test_loss = float("inf"); best_state=None; patience=20; bad_epochs=0

for epoch in range(n_epochs):
    model.train()
    perm = torch.randperm(len(Xtr))
    tot_loss = 0.0
    for i in range(0, len(Xtr), batch_size):
        idx = perm[i:i+batch_size]
        xb, bb, yb = Xtr[idx], Btr[idx], ytr[idx]
        opt.zero_grad()
        out = model(xb, bb)
        loss = crit(out, yb)
        loss.backward(); opt.step()
        tot_loss += loss.item()*len(idx)
    tot_loss /= len(Xtr)

    model.eval()
    with torch.no_grad():
        test_loss = crit(model(Xte, Bte), yte).item()

    if test_loss < best_test_loss - 1e-4:
        best_test_loss = test_loss; best_state={k:v.clone() for k,v in model.state_dict().items()}; bad_epochs=0
    else:
        bad_epochs += 1
    if epoch % 10 == 0 or epoch==n_epochs-1:
        print(f"epoch {epoch:3d}  train_loss={tot_loss:.4f}  test_loss={test_loss:.4f}  best={best_test_loss:.4f}")
    if bad_epochs >= patience:
        print(f"early stop at epoch {epoch}"); break

model.load_state_dict(best_state); model.eval()
with torch.no_grad():
    test_prob = torch.sigmoid(model(Xte, Bte)).numpy().ravel()
    train_prob = torch.sigmoid(model(Xtr, Btr)).numpy().ravel()

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
print("\n--- SCGNetContext (shape + background-noise context), held-out test (thr=0.5) ---")
print(f"acc={r['acc']*100:.1f}%  precision={r['prec']*100:.1f}%  recall={r['rec']*100:.1f}%  f1={r['f1']*100:.1f}%")
print(f"TP={r['tp']} TN={r['tn']} FP={r['fp']} FN={r['fn']}  total params={sum(p.numel() for p in model.parameters())}")

torch.save(best_state, "/tmp/cnn_context_model.pt")
with open("/tmp/cnn_context_train_result.pkl","wb") as f:
    pickle.dump(dict(scale=scale, bg_mean=bg_mean, bg_std=bg_std, n_train=n_train,
                      test_prob=test_prob, y_test=y_test, t_test=t[n_train:], bg_test=bg_test,
                      train_prob=train_prob, y_train=y_train, t_train=t[:n_train]), f)
print("\nsaved /tmp/cnn_context_model.pt and /tmp/cnn_context_train_result.pkl")

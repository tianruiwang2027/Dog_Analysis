import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0); np.random.seed(0)

with open("/tmp/cnn_dataset.pkl","rb") as f:
    D = pickle.load(f)
t = D["t"]; label = D["label"]; snips = D["snips"]

with open("/tmp/cnn_fooling_negs.pkl","rb") as f:
    FN = pickle.load(f)
fool = FN["fool"]   # label==0 AND original model already scores it >=0.4 -- the genuine hard cases

n = len(t); n_train = int(n*0.70)
X_train, y_train, hard_train = snips[:n_train], label[:n_train], fool[:n_train]
X_test,  y_test,  hard_test  = snips[n_train:], label[n_train:], fool[n_train:]
print(f"train n={n_train} (pos={int(y_train.sum())} neg={int((1-y_train).sum())} fooling_hard_neg={int(hard_train.sum())})")
print(f"test  n={n-n_train} (pos={int(y_test.sum())} neg={int((1-y_test).sum())} fooling_hard_neg={int(hard_test.sum())})")

scale = np.median(np.abs(X_train[y_train==1])) + 1e-9
X_train_n = X_train/scale; X_test_n = X_test/scale

Xtr = torch.tensor(X_train_n, dtype=torch.float32).unsqueeze(1)
ytr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
htr = torch.tensor(hard_train.astype(np.float32))
Xte = torch.tensor(X_test_n, dtype=torch.float32).unsqueeze(1)
yte = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
hte = torch.tensor(hard_test.astype(np.float32))

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

n_pos = ytr.sum().item(); n_neg = len(ytr)-n_pos
base_neg_w = n_pos/n_neg
HARD_MULT = 2.0
w_train = torch.where(ytr.squeeze(1)==1, torch.ones(len(ytr)),
                 torch.where(htr==1, torch.full((len(ytr),), base_neg_w*HARD_MULT),
                             torch.full((len(ytr),), base_neg_w)))
# matching per-sample weights for test loss (so early-stopping monitors the SAME
# class-balanced + hard-emphasis objective we're actually optimizing, just on held-out data)
w_test = torch.where(yte.squeeze(1)==1, torch.ones(len(yte)),
                 torch.where(hte==1, torch.full((len(yte),), base_neg_w*HARD_MULT),
                             torch.full((len(yte),), base_neg_w)))

model = SCGNet()
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

n_epochs = 150; batch_size = 64
best_test_loss = float("inf"); best_state=None; patience=20; bad_epochs=0

for epoch in range(n_epochs):
    model.train()
    perm = torch.randperm(len(Xtr))
    tot_loss = 0.0
    for i in range(0, len(Xtr), batch_size):
        idx = perm[i:i+batch_size]
        xb, yb, wb = Xtr[idx], ytr[idx], w_train[idx]
        opt.zero_grad()
        out = model(xb)
        loss_raw = F.binary_cross_entropy_with_logits(out, yb, reduction="none").squeeze(1)
        loss = (loss_raw*wb).sum()/wb.sum()
        loss.backward(); opt.step()
        tot_loss += loss.item()*len(idx)
    tot_loss /= len(Xtr)

    model.eval()
    with torch.no_grad():
        test_out = model(Xte)
        test_loss_raw = F.binary_cross_entropy_with_logits(test_out, yte, reduction="none").squeeze(1)
        test_loss = ((test_loss_raw*w_test).sum()/w_test.sum()).item()

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
print("\n--- fine-tuned-v2 model, held-out test set (thr=0.5) ---")
print(f"acc={r['acc']*100:.1f}%  precision={r['prec']*100:.1f}%  recall={r['rec']*100:.1f}%  f1={r['f1']*100:.1f}%")
print(f"TP={r['tp']} TN={r['tn']} FP={r['fp']} FN={r['fn']}")

if hard_test.sum()>0:
    hp = test_prob[hard_test.astype(bool)]
    print(f"\nfooling hard negatives in test set: n={len(hp)}  mean score (fine-tuned)={hp.mean():.3f}  frac still>=0.4: {(hp>=0.4).mean()*100:.1f}%")

torch.save(best_state, "/tmp/cnn_model_finetuned_v2b.pt")
with open("/tmp/cnn_finetune_v2b_result.pkl","wb") as f:
    pickle.dump(dict(scale=scale, n_train=n_train, test_prob=test_prob, y_test=y_test,
                      t_test=t[n_train:], hard_test=hard_test, HARD_MULT=HARD_MULT), f)
print("\nsaved /tmp/cnn_model_finetuned_v2b.pt and /tmp/cnn_finetune_v2b_result.pkl")

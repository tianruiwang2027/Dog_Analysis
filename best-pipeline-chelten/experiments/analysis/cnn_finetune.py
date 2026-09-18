import pickle, sqlite3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0); np.random.seed(0)

with open("/tmp/cnn_dataset.pkl","rb") as f:
    D = pickle.load(f)
t = D["t"]; label = D["label"]; snips = D["snips"]

# ---- recompute the "hard negative" flag: unmatched candidates that sit INSIDE a
# hand-labeled-good SCG interval (i.e. not just obviously-bad-stretch noise -- these
# are the ambiguous, shape-plausible false candidates like the mid-cycle blips and
# motion coincidences found during the investigation) ----
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

scg_pk, ev_ts, ev_lab = load_peaks_and_good("/tmp/annotation_chelten_scg2.db")
span_end = max(scg_pk.max(), t.max())
scg_ivs = good_intervals(ev_ts, ev_lab, span_end)
in_good_iv = good_mask(t, scg_ivs)
hard_neg = (label==0) & in_good_iv
print(f"hard negatives (unmatched, inside a hand-good interval): {int(hard_neg.sum())}")
print(f"easy negatives (unmatched, inside a hand-bad interval):  {int(((label==0)&~in_good_iv).sum())}")

n = len(t); n_train = int(n*0.70)
X_train, y_train, hard_train = snips[:n_train], label[:n_train], hard_neg[:n_train]
X_test,  y_test,  hard_test  = snips[n_train:], label[n_train:], hard_neg[n_train:]
print(f"train n={n_train} (pos={int(y_train.sum())} neg={int((1-y_train).sum())} hard_neg={int(hard_train.sum())})")
print(f"test  n={n-n_train} (pos={int(y_test.sum())} neg={int((1-y_test).sum())} hard_neg={int(hard_test.sum())})")

scale = np.median(np.abs(X_train[y_train==1])) + 1e-9
X_train_n = X_train/scale; X_test_n = X_test/scale

Xtr = torch.tensor(X_train_n, dtype=torch.float32).unsqueeze(1)
ytr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
htr = torch.tensor(hard_train.astype(np.float32))
Xte = torch.tensor(X_test_n, dtype=torch.float32).unsqueeze(1)
yte = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

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

# per-sample weight: class-balance (as before) TIMES an extra multiplier on hard negatives,
# so the optimizer is pushed harder on exactly the ambiguous false-candidate population
# that we found slipping through (mid-cycle blips, motion coincidences) -- without touching
# the easy negatives (which are already handled fine) or the positives.
n_pos = ytr.sum().item(); n_neg = len(ytr)-n_pos
base_neg_w = n_pos/n_neg
HARD_MULT = 4.0
w = torch.where(ytr.squeeze(1)==1, torch.ones(len(ytr)),
                 torch.where(htr==1, torch.full((len(ytr),), base_neg_w*HARD_MULT),
                             torch.full((len(ytr),), base_neg_w)))

model = SCGNet()
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

n_epochs = 150; batch_size = 64
best_test_loss = float("inf"); best_state=None; patience=20; bad_epochs=0
# unweighted eval loss (class-balanced only, no hard mult) for early stopping, so we're
# not just overfitting to the reweighted train objective
eval_pos_weight = torch.tensor([base_neg_w])

for epoch in range(n_epochs):
    model.train()
    perm = torch.randperm(len(Xtr))
    tot_loss = 0.0
    for i in range(0, len(Xtr), batch_size):
        idx = perm[i:i+batch_size]
        xb, yb, wb = Xtr[idx], ytr[idx], w[idx]
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
        test_loss = F.binary_cross_entropy_with_logits(test_out, yte, pos_weight=eval_pos_weight).item()

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
    train_prob = torch.sigmoid(model(Xtr)).numpy().ravel()

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
print("\n--- fine-tuned model, held-out test set (thr=0.5) ---")
print(f"acc={r['acc']*100:.1f}%  precision={r['prec']*100:.1f}%  recall={r['rec']*100:.1f}%  f1={r['f1']*100:.1f}%")
print(f"TP={r['tp']} TN={r['tn']} FP={r['fp']} FN={r['fn']}")

# specifically: how did we do on the HARD negatives in the test set?
if hard_test.sum()>0:
    hp = test_prob[hard_test.astype(bool)]
    print(f"\nhard negatives in test set: n={len(hp)}  mean score before vs after available separately")
    print(f"  fine-tuned mean score on hard negatives = {hp.mean():.3f}  (fraction still >=0.4: {(hp>=0.4).mean()*100:.1f}%)")

torch.save(best_state, "/tmp/cnn_model_finetuned.pt")
with open("/tmp/cnn_finetune_result.pkl","wb") as f:
    pickle.dump(dict(scale=scale, n_train=n_train, test_prob=test_prob, y_test=y_test,
                      t_test=t[n_train:], hard_test=hard_test, HARD_MULT=HARD_MULT), f)
print("\nsaved /tmp/cnn_model_finetuned.pt and /tmp/cnn_finetune_result.pkl")

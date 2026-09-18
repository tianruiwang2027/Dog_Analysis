import pickle
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)

with open("/tmp/cnn3_dataset.pkl","rb") as f:
    D = pickle.load(f)
t = D["t"]; cls = D["cls"]; snips = D["snips"]

n = len(t)
n_train = int(n*0.70)
X_train, y_train = snips[:n_train], cls[:n_train]
X_test,  y_test  = snips[n_train:], cls[n_train:]
print(f"train n={n_train}  S1={np.sum(y_train==0)} S2={np.sum(y_train==1)} noise={np.sum(y_train==2)}")
print(f"test  n={n-n_train}  S1={np.sum(y_test==0)} S2={np.sum(y_test==1)} noise={np.sum(y_test==2)}")

scale = np.median(np.abs(X_train[y_train==0])) + 1e-9
X_train_n = X_train / scale
X_test_n = X_test / scale

Xtr = torch.tensor(X_train_n, dtype=torch.float32).unsqueeze(1)
ytr = torch.tensor(y_train, dtype=torch.long)
Xte = torch.tensor(X_test_n, dtype=torch.float32).unsqueeze(1)
yte = torch.tensor(y_test, dtype=torch.long)

class SCGNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=9, padding=4)
        self.pool = nn.MaxPool1d(4)
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(16*10, 32)
        self.fc2 = nn.Linear(32, 3)   # S1 / S2 / noise
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x)); x = self.pool(x)
        x = self.relu(self.conv2(x)); x = self.pool(x)
        x = x.flatten(1); x = self.drop(x)
        x = self.relu(self.fc1(x)); x = self.fc2(x)
        return x

model = SCGNet3()
counts = torch.tensor([np.sum(y_train==c) for c in range(3)], dtype=torch.float32)
class_weight = counts.sum() / (3*counts)
crit = nn.CrossEntropyLoss(weight=class_weight)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

n_epochs = 200
batch_size = 64
best_test_loss = float("inf"); best_state = None; patience = 25; bad_epochs = 0

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
        best_test_loss = test_loss; best_state = {k:v.clone() for k,v in model.state_dict().items()}; bad_epochs=0
    else:
        bad_epochs += 1
    if epoch % 20 == 0 or epoch==n_epochs-1:
        print(f"epoch {epoch:3d}  train_loss={tot_loss:.4f}  test_loss={test_loss:.4f}  best={best_test_loss:.4f}")
    if bad_epochs >= patience:
        print(f"early stop at epoch {epoch}"); break

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    test_logits = model(Xte)
    test_prob = torch.softmax(test_logits, dim=1).numpy()
    test_pred = test_prob.argmax(axis=1)

y_test_np = y_test
acc = (test_pred == y_test_np).mean()
print(f"\noverall test accuracy: {acc*100:.1f}%")
names = ["S1","S2","noise"]
cm = np.zeros((3,3), dtype=int)
for a,p in zip(y_test_np, test_pred):
    cm[a,p]+=1
print("confusion matrix (rows=true, cols=pred):")
print("        " + "  ".join(f"{n:>6s}" for n in names))
for i,n_ in enumerate(names):
    print(f"{n_:>6s}  " + "  ".join(f"{cm[i,j]:6d}" for j in range(3)))
for i,n_ in enumerate(names):
    tot = cm[i].sum()
    print(f"  {n_} recall: {cm[i,i]/tot*100:.1f}%  (n={tot})")

torch.save(best_state, "/tmp/cnn3_model.pt")
with open("/tmp/cnn3_train_result.pkl","wb") as f:
    pickle.dump(dict(scale=scale, n_train=n_train, test_prob=test_prob, y_test=y_test_np,
                      t_test=t[n_train:], cm=cm), f)
print("\nsaved /tmp/cnn3_model.pt and /tmp/cnn3_train_result.pkl")

import pickle, sqlite3
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; fsd_s = d["fsd_s"]

with open("/tmp/full_session_scores_all_models.pkl","rb") as f:
    S = pickle.load(f)
cand_t = S["cand_t"]; scores_main = S["scores_relabel"]

with open("/tmp/cnn_mask_apply.pkl","rb") as f:
    CA = pickle.load(f)
LAG_US = CA["LAG_US"]; RESTRICT = CA["RESTRICT"]

with open("/tmp/three_model_r_compare.pkl","rb") as f:
    prev = pickle.load(f)
CUTOFF_MAIN = prev["CNN-SCG+ECG (envelope, relabeled)"]["cutoff"]

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

valid = ~np.isnan(scores_main)
confirmed0 = set(cand_t[valid][scores_main[valid] >= CUTOFF_MAIN].tolist())
rejected_t = cand_t[valid][scores_main[valid] < CUTOFF_MAIN]

SEARCH_LO_US, SEARCH_HI_US = 100_000, 300_000
MERGE_TOL_US = 50_000
def backward_local_max(tc):
    lo, hi = tc-SEARCH_HI_US, tc-SEARCH_LO_US
    i_lo, i_hi = np.searchsorted(ts_sd, lo), np.searchsorted(ts_sd, hi)
    if i_hi <= i_lo: return None
    seg = env_sd[i_lo:i_hi]
    j = np.argmax(seg)
    return ts_sd[i_lo+j]

S2_CUTOFF_FINAL = 0.70
examples = []  # (orig_t, p_s2, recovered_t, p_main, accepted)
recovered_accepted = []
for tc in rejected_t:
    p_s2 = score_one(net_s2, scale_s2, tc)
    if p_s2 is None or p_s2 < S2_CUTOFF_FINAL: continue
    t_back = backward_local_max(tc)
    if t_back is None: continue
    p_main = score_one(net_main, scale_main, t_back)
    accepted = (p_main is not None and p_main >= CUTOFF_MAIN)
    examples.append((tc, p_s2, t_back, p_main, accepted))
    if accepted: recovered_accepted.append(t_back)

recovered_accepted = np.array(recovered_accepted, dtype="int64")
merged = set(confirmed0)
for rt in recovered_accepted:
    nearby = [c for c in merged if abs(c-rt) < MERGE_TOL_US]
    if not nearby: merged.add(int(rt))
merged_arr = np.array(sorted(merged), dtype="int64")
print(f"chain: {len(examples)} flagged S2 @ cutoff {S2_CUTOFF_FINAL}, {len(recovered_accepted)} accepted after main-CNN re-score")
print(f"final confirmed set: {len(merged_arr)}  (vs {len(confirmed0)} from main CNN alone)")

# ---- downstream pipeline, keep xa/yb/ta this time ----
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
def beat_hr(peaks):
    rr = np.diff(peaks)/1e6; hr = 60.0/rr
    tmid = peaks[:-1] + np.diff(peaks)//2
    return tmid, hr, rr
def smooth(tmid, hr, win_s=3.0):
    win_us2 = int(win_s*1e6); out = np.empty(len(tmid))
    for i,t in enumerate(tmid):
        sel = (tmid>=t-win_us2)&(tmid<=t+win_us2); out[i]=hr[sel].mean()
    return out
def stats(a,b):
    dd=a-b
    return dict(n=int(len(a)), bias=float(dd.mean()), mae=float(np.abs(dd).mean()),
                r=float(np.corrcoef(a,b)[0,1]) if len(a)>2 else float("nan"))
def align(tsA, hrA, vA, tsB, hrB, vB, lag_us=0, win_us=2_000_000, restrict=None):
    bts, bhr = tsB[vB], hrB[vB]
    A_ = np.where(vA)[0]
    xa, yb, ta = [], [], []
    for i in A_:
        traw = tsA[i]
        if restrict is not None and not (restrict[0] <= traw <= restrict[1]): continue
        t = traw + lag_us
        sel = (bts>=t-win_us)&(bts<=t+win_us)
        if sel.sum()>=1:
            xa.append(hrA[i]); yb.append(bhr[sel].mean()); ta.append(traw)
    return np.array(xa), np.array(yb), np.array(ta, dtype="int64")

ecg_pk, ecg_ev_ts, ecg_ev_lab = load_peaks_and_good("/tmp/annotation_chelten_ecg2.db")
span_end = max(ecg_pk.max(), cand_t.max())
ecg_ivs = good_intervals(ecg_ev_ts, ecg_ev_lab, span_end)
ecg_tmid, ecg_hr, _ = beat_hr(ecg_pk)
ecg_hr_sm = smooth(ecg_tmid, ecg_hr)
ecg_v_full = good_mask(ecg_tmid, ecg_ivs)

tmid, hr_raw, rr = beat_hr(merged_arr)
ok = rr <= 1.5
tmid_ok, hr_ok = tmid[ok], hr_raw[ok]
hr_sm = smooth(tmid_ok, hr_ok, win_s=3.0)
ecg_v_at = good_mask(tmid_ok, ecg_ivs)
xa, yb, ta = align(tmid_ok, hr_sm, ecg_v_at, ecg_tmid, ecg_hr_sm, ecg_v_full, lag_us=LAG_US, restrict=RESTRICT)
st = stats(xa, yb)
print(f"FINAL: r={st['r']:.4f}  n={st['n']}  MAE={st['mae']:.2f}  bias={st['bias']:.2f}")

with open("/tmp/s2_chain_final.pkl","wb") as f:
    pickle.dump(dict(xa=xa, yb=yb, ta=ta, s=st, examples=examples, merged_arr=merged_arr,
                      confirmed0=np.array(sorted(confirmed0),dtype="int64"), S2_CUTOFF_FINAL=S2_CUTOFF_FINAL), f)

# =================== FIGURE ===================
fig = plt.figure(figsize=(13, 11))
gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 0.9, 0.9])

# panel 1: updated bar chart across all approaches
ax0 = fig.add_subplot(gs[0, :])
names = ["CNN-SCG\n(original)", "CNN-SCG+ECG\n(relabeled)", "CNN-raw\n(relabeled)",
         "snap strategy\n(failed)", "3-step chain\n(S2-classifier)"]
rs = [prev["CNN-SCG (original)"]["s"]["r"],
      prev["CNN-SCG+ECG (envelope, relabeled)"]["s"]["r"],
      prev["CNN-raw (raw signal, relabeled)"]["s"]["r"],
      0.7968, st["r"]]
ns = [prev["CNN-SCG (original)"]["s"]["n"],
      prev["CNN-SCG+ECG (envelope, relabeled)"]["s"]["n"],
      prev["CNN-raw (raw signal, relabeled)"]["s"]["n"],
      1533, st["n"]]
colors = ["tab:blue","tab:green","tab:orange","tab:red","tab:purple"]
bars = ax0.bar(names, rs, color=colors)
for b, r, n in zip(bars, rs, ns):
    ax0.text(b.get_x()+b.get_width()/2, r+0.01, f"r={r:.3f}\nn={n}", ha="center", va="bottom", fontsize=9)
ax0.set_ylabel("best downstream r (SCG HR vs ECG HR)")
ax0.set_ylim(0, 1.0)
ax0.axhline(prev["CNN-SCG+ECG (envelope, relabeled)"]["s"]["r"], color="green", ls="--", lw=1, alpha=0.5)
ax0.set_title("3-step chain (S2-classifier -> backward search -> re-validate with main CNN)\n"
              f"recovers {len(recovered_accepted)}/{len(rejected_t)} rejected candidates as genuine beats", fontsize=11)

# panels 2-4: example chain recoveries (rejected candidate -> flagged S2 -> recovered S1 -> accepted)
accepted_examples = [e for e in examples if e[4]]
example_idx = np.linspace(0, len(accepted_examples)-1, 3).astype(int)
for k, ei in enumerate(example_idx):
    orig_t, p_s2, t_back, p_main, accepted = accepted_examples[ei]
    ax = fig.add_subplot(gs[1, k])
    win = 700_000
    lo, hi = t_back - win, t_back + win
    i_lo, i_hi = np.searchsorted(ts_sd, lo), np.searchsorted(ts_sd, hi)
    tt = (ts_sd[i_lo:i_hi] - t_back) / 1000.0
    ax.plot(tt, env_sd[i_lo:i_hi], color="black", lw=1)
    ax.axvline(0, color="tab:purple", lw=2, label=f"recovered S1\nmain CNN p={p_main:.2f}")
    ax.axvline((orig_t-t_back)/1000.0, color="tab:gray", ls="--", lw=2, label=f"rejected candidate (S2)\nS2-clf p={p_s2:.2f}")
    ax.set_title(f"recovered example {k+1}", fontsize=9)
    ax.set_xlabel("ms")
    ax.legend(fontsize=6.5, loc="upper right")

# panel 5: r scatter
ax4 = fig.add_subplot(gs[2, 0])
lo_v, hi_v = min(xa.min(), yb.min()), max(xa.max(), yb.max())
ax4.scatter(yb, xa, s=6, alpha=0.35, color="tab:purple")
ax4.plot([lo_v,hi_v],[lo_v,hi_v], color="black", lw=1, ls="--")
ax4.set_xlabel("ECG HR (bpm)"); ax4.set_ylabel("SCG HR, 3-step chain (bpm)")
ax4.set_title(f"r={st['r']:.3f}  n={st['n']}  MAE={st['mae']:.2f}bpm", fontsize=9)

# panel 6: HR over time
ax5 = fig.add_subplot(gs[2, 1:])
t_hr = (ta - ta.min())/1e6/60.0
ax5.plot(t_hr, yb, color="black", lw=0.8, label="ECG HR", alpha=0.7)
ax5.plot(t_hr, xa, color="tab:purple", lw=0.8, label="SCG HR (3-step chain)", alpha=0.7)
ax5.set_xlabel("time (min)"); ax5.set_ylabel("HR (bpm)")
ax5.legend(fontsize=8)
ax5.set_title("HR over time", fontsize=9)

plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_s2_chain_result.png", dpi=130)
print("saved chelten_s2_chain_result.png")

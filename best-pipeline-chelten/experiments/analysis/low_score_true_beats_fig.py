import pickle, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_s = d["ts_s"]; xf_s = d["xf_s"]
with open("/tmp/full_session_scores_all_models.pkl","rb") as f:
    S = pickle.load(f)
cand_t = S["cand_t"]; scores = S["scores_relabel"]

def tsec(hh,mm,ss,ms=0):
    return datetime.datetime(2026,6,26,hh,mm,ss,ms*1000, tzinfo=datetime.timezone.utc).timestamp()*1e6
w0=tsec(17,54,3); w1=tsec(17,54,41)
sel=(cand_t>=w0)&(cand_t<w1)
ct = cand_t[sel]; cs = scores[sel]

low_idx = np.argsort(cs)[:4]     # 4 lowest-scoring (but real, per ECG/SCG count match) candidates
high_idx = np.argsort(-cs)[:2]   # 2 highest-scoring for contrast

fig, axs = plt.subplots(2,3, figsize=(15,7))
WIN_MS = 500
for k,i in enumerate(list(low_idx)+list(high_idx)):
    ax = axs.flat[k]
    tc = ct[i]; sc = cs[i]
    wsel = (ts_s>=tc-WIN_MS*1e3)&(ts_s<=tc+WIN_MS*1e3)
    color = "tab:red" if sc<0.25 else "tab:green"
    ax.plot((ts_s[wsel]-tc)/1e3, xf_s[wsel], color=color, lw=1.1)
    ax.axvline(0, color="0.3", lw=0.7, ls=":")
    tag = "REJECTED (score<0.25)" if sc<0.25 else "confirmed"
    ax.set_title(f"score={sc:.3f}  {tag}", fontsize=9.5)
    ax.set_xlabel("ms")

fig.suptitle("17:54:05-39 window: candidate count exactly matches ECG (36=36) -- these are real beats,\n"
             "but several score far below cutoff=0.25 (top row) vs confidently-scored real beats (bottom row)", fontsize=11)
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_low_score_true_beats.png", dpi=135)
print("saved")

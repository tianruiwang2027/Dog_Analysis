import pickle, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; fsd_s = d["fsd_s"]
with open("/tmp/full_session_scores_all_models.pkl","rb") as f:
    S = pickle.load(f)
cand_t = S["cand_t"]; scores = S["scores_relabel"]
with open("/tmp/cnn_train_relabel_result.pkl","rb") as f:
    TR = pickle.load(f)
scale = TR["scale"]
with open("/tmp/cnn_dataset_relabel.pkl","rb") as f:
    HN = pickle.load(f)
HALF_N = HN["HALF_N"]

def tsec(hh,mm,ss,ms=0):
    return datetime.datetime(2026,6,26,hh,mm,ss,ms*1000, tzinfo=datetime.timezone.utc).timestamp()*1e6
w0=tsec(17,54,3); w1=tsec(17,54,41)
sel=(cand_t>=w0)&(cand_t<w1)
ct = cand_t[sel]; cs = scores[sel]

def env_snippet(tc):
    i = np.searchsorted(ts_sd, tc)
    if i-HALF_N<0 or i+HALF_N+1>len(env_sd): return None
    return env_sd[i-HALF_N:i+HALF_N+1]

low_idx = np.where(cs<0.3)[0][:4]
high_idx = np.where(cs>0.9)[0][:2]

x_ms = (np.arange(2*HALF_N+1)-HALF_N)/fsd_s*1000
fig, axs = plt.subplots(2,3, figsize=(15,7))
for k,i in enumerate(list(low_idx)+list(high_idx)):
    ax = axs.flat[k]
    snip = env_snippet(ct[i]) / scale
    color = "tab:red" if cs[i]<0.3 else "tab:green"
    ax.plot(x_ms, snip, color=color, lw=1.4)
    ax.axvline(0, color="0.3", lw=0.7, ls=":")
    tag = "REJECTED" if cs[i]<0.3 else "confirmed"
    ax.set_title(f"score={cs[i]:.3f}  {tag}", fontsize=9.5)
    ax.set_xlabel("ms")

fig.suptitle("Same 6 candidates, but this time showing the ACTUAL envelope input the CNN sees\n(not the raw waveform) -- t=17:54:xx window, candidate count exactly matches ECG", fontsize=11)
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_low_score_envelope_view.png", dpi=135)
print("saved")

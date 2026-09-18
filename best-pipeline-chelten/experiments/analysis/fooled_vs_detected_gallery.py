import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_s = d["ts_s"]; xf_s = d["xf_s"]

with open("/tmp/cnn_dataset.pkl","rb") as f:
    D = pickle.load(f)
t = D["t"]; label = D["label"]
with open("/tmp/cnn_fooling_negs.pkl","rb") as f:
    FN = pickle.load(f)
fool = FN["fool"]; prob_all = FN["prob_all_orig"]

# ---- pick 10 diverse fooled (false-positive) candidates, spread across confidence range ----
fool_idx = np.where(fool)[0]
order_f = np.argsort(-prob_all[fool_idx])
picks_f = fool_idx[order_f[:: max(1, len(order_f)//10)][:10]]

# ---- pick 10 diverse confidently-correct true-positive candidates ----
pos_idx = np.where((label==1) & (prob_all>=0.8))[0]
order_p = np.argsort(-prob_all[pos_idx])
picks_p = pos_idx[order_p[:: max(1, len(order_p)//10)][:10]]

WIN_S = 0.5   # +-500ms raw waveform window around each candidate

def plot_grid(picks, title, fname, color):
    fig, axs = plt.subplots(2, 5, figsize=(18, 6.5))
    for k, idx in enumerate(picks):
        ax = axs.flat[k]
        tc = t[idx]; sc = prob_all[idx]
        sel = (ts_s>=tc-WIN_S*1e6)&(ts_s<=tc+WIN_S*1e6)
        ax.plot((ts_s[sel]-tc)/1e6*1000, xf_s[sel], color=color, lw=0.9)
        ax.axvline(0, color="0.3", lw=0.8, ls=":")
        ax.set_title(f"score={sc:.2f}", fontsize=9)
        ax.set_xlabel("ms", fontsize=7.5)
        ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(f"/tmp/dog-test-ecg/dog-test-ecg-code/figs/{fname}", dpi=130)
    print("saved", fname)

plot_grid(picks_f, "10 waveforms that FOOLED the CNN\n(not real beats -- >150ms/no match to any hand click -- but scored >=0.4, confirmed)",
           "chelten_fooled_gallery.png", "tab:red")
plot_grid(picks_p, "10 waveforms CORRECTLY detected by the CNN\n(genuine hand-clicked beats, confidently confirmed)",
           "chelten_detected_gallery.png", "tab:green")

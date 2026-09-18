import pickle
import numpy as np

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)

ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; fsd_s = d["fsd_s"]
tp_times = lab["tp_times"]; fp_times = lab["fp_times"]

HALF_S = 0.40
HALF_N = int(HALF_S*fsd_s)

def snippet(t):
    i = np.searchsorted(ts_sd, t)
    if i-HALF_N < 0 or i+HALF_N+1 > len(env_sd):
        return None
    return env_sd[i-HALF_N:i+HALF_N+1].copy()

order = np.argsort(tp_times)
tp_sorted = tp_times[order]
n_train = len(tp_sorted)//2
train_tp = tp_sorted[:n_train]
test_tp = tp_sorted[n_train:]

snips = [snippet(t) for t in train_tp]
snips = np.array([s for s in snips if s is not None])
snips_norm = snips / (snips.max(axis=1, keepdims=True)+1e-12)
template = snips_norm.mean(axis=0)
template = template - template.mean()
template_energy = np.sqrt((template**2).sum())

with open("/tmp/env_template_wide.pkl","wb") as f:
    pickle.dump(dict(template=template, HALF_N=HALF_N, fsd_s=fsd_s), f)

t_ms = (np.arange(len(template))-HALF_N)/fsd_s*1000
# print the shape to look for a secondary lobe
for tm, v in zip(t_ms, template):
    if -260 <= tm <= 260 and abs(tm)%20 < (1000/fsd_s):
        print(f"{tm:7.1f}ms  {v:+.4f}")

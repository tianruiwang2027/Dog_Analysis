import pickle
import numpy as np

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_s = d["ts_s"]; xf_s = d["xf_s"]; fs_s = d["fs_s"]
print(f"raw signal sample rate: {fs_s} Hz")

with open("/tmp/cnn_dataset_relabel.pkl","rb") as f:
    D = pickle.load(f)
t = D["t"]; label = D["label"]

HALF_S = 0.40
HALF_N_RAW = int(HALF_S*fs_s)
print(f"raw snippet half-window: {HALF_N_RAW} samples ({HALF_S*1000:.0f}ms @ {fs_s:.0f}Hz) -> total length {2*HALF_N_RAW+1}")

def snippet_raw(tc):
    i = np.searchsorted(ts_s, tc)
    if i-HALF_N_RAW < 0 or i+HALF_N_RAW+1 > len(xf_s): return None
    return xf_s[i-HALF_N_RAW:i+HALF_N_RAW+1].copy()

snips_raw = []
keep_t = []; keep_label = []
for tc, lab in zip(t, label):
    s = snippet_raw(tc)
    if s is None: continue
    snips_raw.append(s); keep_t.append(tc); keep_label.append(lab)

snips_raw = np.array(snips_raw, dtype=np.float32)
keep_t = np.array(keep_t, dtype="int64")
keep_label = np.array(keep_label, dtype=np.float32)
print(f"final raw dataset: n={len(keep_t)}  positive={int(keep_label.sum())}  negative={int((1-keep_label).sum())}")

with open("/tmp/cnn_dataset_raw.pkl","wb") as f:
    pickle.dump(dict(t=keep_t, label=keep_label, snips=snips_raw, HALF_N=HALF_N_RAW, fs_s=fs_s), f)
print("saved /tmp/cnn_dataset_raw.pkl")

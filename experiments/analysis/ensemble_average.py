import pickle, sqlite3
import numpy as np

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_s = d["ts_s"]; x_s = d["x_s"]; xf_s = d["xf_s"]; fs_s = d["fs_s"]
ts_sd = d["ts_sd"]; env_sd = d["env_sd"]; fsd_s = d["fsd_s"]

def load_peaks(path):
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = '' ORDER BY timestamp")
    return np.array([r[0] for r in cur.fetchall()], dtype="int64")
scg_pk = load_peaks("/tmp/annotation_chelten_scg2.db")

with open("/tmp/valid_regions.pkl","rb") as f:
    V = pickle.load(f)
valid_ivs = V["valid_ivs"]

def valid_mask(ts):
    g = np.zeros(len(ts), bool)
    for a,b in valid_ivs: g |= (ts>=a)&(ts<b)
    return g

scg_valid = scg_pk[valid_mask(scg_pk)]
print(f"n hand-clicked SCG peaks total: {len(scg_pk)}")
print(f"n within valid regions (used for ensemble average): {len(scg_valid)}")

HALF_MS = 450
HALF_N_hi = int(HALF_MS/1000*fs_s)     # for raw/bandpassed (2000Hz)
HALF_N_lo = int(HALF_MS/1000*fsd_s)    # for envelope (200Hz)

def stack_snippets(ts_arr, sig_arr, peaks, half_n):
    rows = []
    for t in peaks:
        i = np.searchsorted(ts_arr, t)
        if i-half_n < 0 or i+half_n+1 > len(sig_arr): continue
        rows.append(sig_arr[i-half_n:i+half_n+1])
    return np.array(rows)

raw_stack = stack_snippets(ts_s, x_s, scg_valid, HALF_N_hi)
bp_stack  = stack_snippets(ts_s, xf_s, scg_valid, HALF_N_hi)
env_stack = stack_snippets(ts_sd, env_sd, scg_valid, HALF_N_lo)

print(f"raw_stack: {raw_stack.shape}   bp_stack: {bp_stack.shape}   env_stack: {env_stack.shape}")

t_ms_hi = (np.arange(raw_stack.shape[1]) - HALF_N_hi) / fs_s * 1000
t_ms_lo = (np.arange(env_stack.shape[1]) - HALF_N_lo) / fsd_s * 1000

with open("/tmp/ensemble_average.pkl","wb") as f:
    pickle.dump(dict(raw_stack=raw_stack, bp_stack=bp_stack, env_stack=env_stack,
                      t_ms_hi=t_ms_hi, t_ms_lo=t_ms_lo, n_used=len(scg_valid), n_total=len(scg_pk)), f)
print("saved /tmp/ensemble_average.pkl")

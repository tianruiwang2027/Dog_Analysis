import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/ensemble_average.pkl","rb") as f:
    E = pickle.load(f)

raw_stack = E["raw_stack"]; bp_stack = E["bp_stack"]; env_stack = E["env_stack"]
t_ms_hi = E["t_ms_hi"]; t_ms_lo = E["t_ms_lo"]
n_used = E["n_used"]; n_total = E["n_total"]

def mean_band(stack):
    mu = stack.mean(axis=0)
    sd = stack.std(axis=0)
    return mu, sd

raw_mu, raw_sd = mean_band(raw_stack)
bp_mu, bp_sd = mean_band(bp_stack)
env_mu, env_sd = mean_band(env_stack)

fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=False)

# Panel A: raw
axs[0].plot(t_ms_hi, raw_mu, color="tab:blue", lw=1.3)
axs[0].fill_between(t_ms_hi, raw_mu-raw_sd, raw_mu+raw_sd, color="tab:blue", alpha=0.2, label="±1 SD across beats")
axs[0].axvline(0, color="black", lw=0.7, ls=":")
axs[0].set_title(f"A. Raw SCG, ensemble average (n={raw_stack.shape[0]} beats)")
axs[0].set_ylabel("raw amplitude (a.u.)")
axs[0].legend(fontsize=8, loc="upper right")

# Panel B: bandpassed
axs[1].plot(t_ms_hi, bp_mu, color="tab:green", lw=1.3)
axs[1].fill_between(t_ms_hi, bp_mu-bp_sd, bp_mu+bp_sd, color="tab:green", alpha=0.2, label="±1 SD across beats")
axs[1].axvline(0, color="black", lw=0.7, ls=":", label="hand-clicked peak (t=0)")
axs[1].set_title("B. Bandpassed SCG, ensemble average")
axs[1].set_ylabel("bandpassed amplitude (a.u.)")
axs[1].legend(fontsize=8, loc="upper right")

# Panel C: Shannon envelope
axs[2].plot(t_ms_lo, env_mu, color="tab:red", lw=1.6)
axs[2].fill_between(t_ms_lo, env_mu-env_sd, env_mu+env_sd, color="tab:red", alpha=0.2, label="±1 SD across beats")
axs[2].axvline(0, color="black", lw=0.7, ls=":")
axs[2].set_title("C. Shannon envelope, ensemble average — expected S1/S2 structure")
axs[2].set_xlabel("time relative to hand-clicked peak (ms)")
axs[2].set_ylabel("envelope amplitude (a.u.)")
axs[2].legend(fontsize=8, loc="upper right")

for ax in axs:
    ax.set_xlim(-450, 450)
    ax.grid(alpha=0.25)

fig.suptitle(f"Chelten SCG: ensemble-averaged cardiac cycle\n"
             f"({n_used} of {n_total} hand-clicked peaks within valid regions; "
             f"{raw_stack.shape[0]} retained after edge-trimming)", fontsize=12)
plt.tight_layout(rect=[0,0,1,0.94])
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_ensemble_average.png", dpi=130)
print("saved")

# Print some quick numeric diagnostics on the envelope shape (peak locations)
from scipy.signal import find_peaks
pk, props = find_peaks(env_mu, prominence=env_mu.max()*0.02)
pk_t = t_ms_lo[pk]
pk_h = env_mu[pk]
order = np.argsort(-pk_h)
print("Top envelope peaks (time_ms, height):")
for i in order[:6]:
    print(f"  t={pk_t[i]:+7.1f} ms   h={pk_h[i]:.4f}")

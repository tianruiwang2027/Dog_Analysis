import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# re-run both sweeps with denser cutoffs, reuse helpers by importing via exec of the eval script's functions
exec(open("/tmp/dog-test-ecg/dog-test-ecg-code/analysis/cnn_context_confirm_eval.py").read().split('print("=== ORIGINAL')[0])

cutoffs_old = np.arange(0.20, 0.85, 0.05)
cutoffs_new = np.arange(0.02, 0.85, 0.04)

rs_old = []
for c in cutoffs_old:
    r = confirm_and_run(cand_cnn_old, c, 1.5)
    rs_old.append(r["s"]["r"] if r else np.nan)
rs_new = []
for c in cutoffs_new:
    r = confirm_and_run(cand_ctx, c, 1.5)
    rs_new.append(r["s"]["r"] if r else np.nan)

fig, ax = plt.subplots(figsize=(9,6))
ax.plot(cutoffs_old, rs_old, "o-", color="tab:gray", label="shape only")
ax.plot(cutoffs_new, rs_new, "o-", color="tab:purple", label="shape + background context")
ax.set_xlabel("confirm cutoff")
ax.set_ylabel("downstream r (vs ECG hand-picked HR)")
ax.set_title("Full pipeline: downstream r vs cutoff\nAUC improved (0.736->0.80), but the end-to-end HR correlation doesn't beat\nthe shape-only model's best operating point (cutoff=0.70, r=0.833)")
ax.legend(fontsize=10)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_context_r_vs_cutoff.png", dpi=135)
print("saved")

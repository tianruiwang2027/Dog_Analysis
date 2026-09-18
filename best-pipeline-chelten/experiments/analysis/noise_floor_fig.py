import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

with open("/tmp/remove_badinterval_result.pkl","rb") as f:
    R = pickle.load(f)
xa=R["xa_2"]; yb=R["yb_2"]
err = xa - yb

fig, axs = plt.subplots(1,2, figsize=(12,4.8))

axs[0].hist(err, bins=40, color="tab:blue", alpha=0.7, density=True)
mu, sigma = err.mean(), err.std()
xs = np.linspace(err.min(), err.max(), 200)
axs[0].plot(xs, norm.pdf(xs, mu, sigma), color="black", lw=1.5, label=f"Gaussian fit\n(mean={mu:+.2f}, std={sigma:.2f})")
axs[0].set_xlabel("SCG HR - ECG HR (bpm)")
axs[0].set_title("A. Residual error distribution after both cleanup stages\nnearly perfectly Gaussian -- skew=0.03, excess kurtosis=-0.06")
axs[0].legend(fontsize=8)

tests = ["HR level\n(ECG)", "HR level\n(SCG)", "|dHR/dt|\n(rate of change)", "local click-\ntiming jitter", "beats-in-\nwindow count"]
corrs = [-0.017, -0.024, 0.050, 0.078, -0.033]
colors = ["tab:gray" if abs(c)<0.1 else "tab:orange" for c in corrs]
axs[1].barh(tests, corrs, color=colors)
axs[1].axvline(0, color="black", lw=0.7)
axs[1].set_xlim(-0.3,0.3)
axs[1].set_xlabel("correlation with |residual error|")
axs[1].set_title("B. None of the usual suspects explain it\n(all correlations near zero)")

plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_noise_floor.png", dpi=130)
print("saved")

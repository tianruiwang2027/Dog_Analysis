import pickle, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
with open("/tmp/illustrate_15missed.pkl","rb") as f:
    I = pickle.load(f)
with open("/tmp/combined_detector.pkl","rb") as f:
    c = pickle.load(f)
with open("/tmp/singles_labels.pkl","rb") as f:
    lab = pickle.load(f)
with open("/tmp/env_template_wide.pkl","rb") as f:
    tpl = pickle.load(f)

ts_s = d["ts_s"]; x_s = d["x_s"]; xf_s = d["xf_s"]
ts_sd = d["ts_sd"]; sharp_s = d["sharp_s"]; env_sd = d["env_sd"]
template = tpl["template"]; HALF_N = tpl["HALF_N"]
template_energy = np.sqrt((template**2).sum())
def ncc_score(t):
    i = np.searchsorted(ts_sd, t)
    if i-HALF_N < 0 or i+HALF_N+1 > len(env_sd): return np.nan
    s = env_sd[i-HALF_N:i+HALF_N+1]; s = s - s.mean()
    denom = np.sqrt((s**2).sum()) * template_energy
    if denom < 1e-12: return np.nan
    return float((s*template).sum() / denom)

det = np.sort(c["primary"])
det_set = set(det.tolist())
singles_good = np.sort(lab["singles_good"])

ta = I["ta"]; bad_idx = I["bad_idx"]
pts = np.sort(ta[bad_idx])

UTC = datetime.timezone.utc
def fmt(us):
    return datetime.datetime.fromtimestamp(us/1e6, tz=UTC).strftime("%H:%M:%S.%f")[:-3]

WIN_US = int(2.5e6)
n = len(pts)
fig, axs = plt.subplots(n, 3, figsize=(15, 2.0*n))

for row, t_center in enumerate(pts):
    lo = np.searchsorted(ts_s, t_center-WIN_US); hi = np.searchsorted(ts_s, t_center+WIN_US)
    t_ms = (ts_s[lo:hi]-t_center)/1e3
    lo2 = np.searchsorted(ts_sd, t_center-WIN_US); hi2 = np.searchsorted(ts_sd, t_center+WIN_US)
    t_ms2 = (ts_sd[lo2:hi2]-t_center)/1e3

    det_in = det[(det>=t_center-WIN_US)&(det<=t_center+WIN_US)]
    cands = singles_good[(singles_good>=t_center-WIN_US)&(singles_good<=t_center+WIN_US)]
    removed = np.array([t for t in cands if t not in det_set])
    has_removed = len(removed) > 0
    tag = "TEMPLATE-FILTER REMOVED A CANDIDATE HERE" if has_removed else "no candidate existed above amplitude threshold (genuine dropout)"

    ax = axs[row,0]
    ax.plot(t_ms, x_s[lo:hi], color="tab:gray", lw=0.5)
    ax.set_ylabel(fmt(t_center)+"\n"+("[filter-removed]" if has_removed else "[genuine gap]"), fontsize=7.5, rotation=0, ha="right", va="center")
    if row==0: ax.set_title("RAW", fontsize=10)
    for dtb in det_in: ax.axvline((dtb-t_center)/1e3, color="tab:green", lw=0.7, alpha=0.6)

    ax = axs[row,1]
    ax.plot(t_ms, xf_s[lo:hi], color="tab:purple", lw=0.5)
    if row==0: ax.set_title("BANDPASSED 10-100Hz", fontsize=10)
    for dtb in det_in: ax.axvline((dtb-t_center)/1e3, color="tab:green", lw=0.7, alpha=0.6)

    ax = axs[row,2]
    ax.plot(t_ms2, sharp_s[lo2:hi2], color="tab:blue", lw=0.9)
    ax.axhline(0.3, color="gray", ls="--", lw=0.8)
    for dtb in det_in: ax.axvline((dtb-t_center)/1e3, color="tab:green", lw=0.7, alpha=0.6, label="_nolegend_")
    for tr in removed:
        ax.plot((tr-t_center)/1e3, 0.32, "rx", ms=9, mew=2)
    if row==0: ax.set_title("SHARPENED ENVELOPE (green=kept beat, red X=candidate the\ntemplate filter threw out as NCC<0.5)", fontsize=9.5)

    for ax in axs[row]:
        ax.set_xlim(-WIN_US/1e3, WIN_US/1e3)
        ax.tick_params(labelsize=7)

for ax in axs[-1]:
    ax.set_xlabel("ms from the point's timestamp", fontsize=8)

fig.suptitle("The 15 missed-beat points, split by cause: genuine signal dropout vs. template-filter over-rejection", fontsize=13)
plt.tight_layout(rect=[0,0,1,0.97])
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_15points_raw_bp_env_v2.png"
plt.savefig(out, dpi=120)
print("->", out)

n_removed_group = sum(1 for t_center in pts if len(np.array([t for t in singles_good[(singles_good>=t_center-WIN_US)&(singles_good<=t_center+WIN_US)] if t not in det_set]))>0)
print(f"points where the template filter threw out a real candidate: {n_removed_group} / {n}")
print(f"points that are genuine gaps (no candidate ever existed):    {n-n_removed_group} / {n}")

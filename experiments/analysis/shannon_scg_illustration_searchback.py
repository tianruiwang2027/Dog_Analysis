#!/usr/bin/env python3
"""Illustrate the two-pass pipeline: primary (thr=0.3, gap 170-250ms) in
red/purple, plus search-back recoveries (relaxed thr=0.2, gap 150-280ms,
only inside flagged 1.2-1.5s primary-to-primary gaps) in orange."""
import pickle, sqlite3
import numpy as np
from scipy import signal as sg
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import datetime

UTC = datetime.timezone.utc
PRIMARY_THR = 0.3
RELAX_THR = 0.2

def dn(us):
    return mdates.date2num([datetime.datetime.fromtimestamp(t/1e6, tz=UTC) for t in np.atleast_1d(us)])

with open("/tmp/shannon_restricted_results_thr0.3_gap0.17-0.25.pkl","rb") as f:
    d = pickle.load(f)
ts_s, x_s, xf_s, fs_s = d["ts_s"], d["x_s"], d["xf_s"], d["fs_s"]
ts_sd, env_sd, sharp_s, q995_s, fsd_s = d["ts_sd"], d["env_sd"], d["sharp_s"], d["q995_s"], d["fsd_s"]

with open("/tmp/shannon_searchback_results.pkl","rb") as f:
    sb = pickle.load(f)
s1_p, s2_p = sb["s1_p"], sb["s2_p"]
s1_extra, s2_extra = sb["s1_extra"], sb["s2_extra"]
flagged_gaps = sb["flagged_gaps"]

con = sqlite3.connect("/tmp/annotation_chelten_scg2.db")
cur = con.cursor()
cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = '' ORDER BY timestamp")
h_scg_pk = np.array([r[0] for r in cur.fetchall()], dtype="int64")

w0 = np.datetime64("2026-06-26T17:29:10").astype("datetime64[us]").astype(int)
w1 = np.datetime64("2026-06-26T17:29:40").astype("datetime64[us]").astype(int)

mR = (ts_s >= w0) & (ts_s <= w1)
mD = (ts_sd >= w0) & (ts_sd <= w1)

# candidates at the PRIMARY threshold (for gray x's)
cand_p_all, _ = sg.find_peaks(sharp_s, height=PRIMARY_THR, distance=max(1, int(0.08*fsd_s)))
cand_p_t = ts_sd[cand_p_all]
cand_w = cand_p_t[(cand_p_t >= w0) & (cand_p_t <= w1)]

s1p_w = s1_p[(s1_p >= w0) & (s1_p <= w1)]
s2p_w = s2_p[(s2_p >= w0) & (s2_p <= w1)]
s1e_w = s1_extra[(s1_extra >= w0) & (s1_extra <= w1)]
s2e_w = s2_extra[(s2_extra >= w0) & (s2_extra <= w1)]
hand_w = h_scg_pk[(h_scg_pk >= w0) & (h_scg_pk <= w1)]
gaps_w = [(a, b) for a, b in flagged_gaps if b >= w0 and a <= w1]

accepted_primary = set(s1p_w.tolist()) | set(s2p_w.tolist())
accepted_extra = set(s1e_w.tolist()) | set(s2e_w.tolist())
rejected_w = np.array([t for t in cand_w if t not in accepted_primary and t not in accepted_extra])

def env_at(ts_query, ts_arr, env_arr):
    idx = np.searchsorted(ts_arr, ts_query)
    idx = np.clip(idx, 0, len(env_arr)-1)
    return env_arr[idx]

fig, axs = plt.subplots(4, 1, figsize=(16, 11), sharex=True,
                         gridspec_kw={"height_ratios":[1,1,1.4,0.35]})

axs[0].plot(dn(ts_s[mR]), x_s[mR], color="0.4", lw=0.6)
axs[0].set_ylabel("raw SCG"); axs[0].set_title("1) Raw SCG signal")
axs[0].grid(True, alpha=0.3)

axs[1].plot(dn(ts_s[mR]), xf_s[mR], color="#1f77b4", lw=0.7)
axs[1].set_ylabel("bandpassed"); axs[1].set_title("2) Bandpassed 10-100Hz")
axs[1].grid(True, alpha=0.3)

axs[2].plot(dn(ts_sd[mD]), sharp_s[mD], color="#2ca02c", lw=1.2, label="sharpened envelope")
axs[2].axhline(PRIMARY_THR, color="0.4", ls="--", lw=1, label=f"primary threshold ({PRIMARY_THR})")
axs[2].axhline(RELAX_THR, color="darkorange", ls=":", lw=1.2, label=f"search-back threshold ({RELAX_THR})")
for a, b in gaps_w:
    axs[2].axvspan(dn(a)[0], dn(b)[0], color="orange", alpha=0.12)
if len(rejected_w):
    axs[2].scatter(dn(rejected_w), env_at(rejected_w, ts_sd, sharp_s), s=40, color="0.6",
                    marker="x", label="candidate (rejected)", zorder=4)
axs[2].scatter(dn(s1p_w), env_at(s1p_w, ts_sd, sharp_s), s=90, color="#d62728",
                marker="o", edgecolor="k", label="primary S1", zorder=5)
axs[2].scatter(dn(s2p_w), env_at(s2p_w, ts_sd, sharp_s), s=90, color="#9467bd",
                marker="^", edgecolor="k", label="primary S2", zorder=5)
axs[2].scatter(dn(s1e_w), env_at(s1e_w, ts_sd, sharp_s), s=110, color="darkorange",
                marker="o", edgecolor="k", label="search-back S1", zorder=6)
axs[2].scatter(dn(s2e_w), env_at(s2e_w, ts_sd, sharp_s), s=110, color="gold",
                marker="^", edgecolor="k", label="search-back S2", zorder=6)
for a, b in zip(s1p_w, s2p_w):
    axs[2].plot(dn([a, b]), [env_at(a, ts_sd, sharp_s), env_at(b, ts_sd, sharp_s)], color="k", lw=1, alpha=0.5, zorder=3)
for a, b in zip(s1e_w, s2e_w):
    axs[2].plot(dn([a, b]), [env_at(a, ts_sd, sharp_s), env_at(b, ts_sd, sharp_s)], color="darkorange", lw=1.3, alpha=0.7, zorder=3)
axs[2].set_ylabel("sharpened\n(local q995 + exp)")
axs[2].set_title("3) Primary pass (thr=0.3, 170-250ms, red/purple) + search-back in flagged 1.2-1.5s gaps (thr=0.2, 150-280ms, orange -- shaded bands)")
axs[2].legend(loc="upper right", fontsize=7, ncol=3)
axs[2].grid(True, alpha=0.3)

axs[3].vlines(dn(hand_w), 0, 1, color="#145214", lw=1.5)
axs[3].vlines(dn(s1p_w), 0, 1, color="#d62728", lw=1.5, linestyle=":")
axs[3].vlines(dn(s1e_w), 0, 1, color="darkorange", lw=2.0, linestyle="-")
axs[3].set_ylim(0, 1); axs[3].set_yticks([])
axs[3].set_title("4) hand-clicked (green) vs primary-accepted (red dotted) vs search-back-recovered (orange solid)", fontsize=10)
axs[3].xaxis.set_major_formatter(DateFormatter("%H:%M:%S", tz=UTC))
axs[3].set_xlim(dn(w0)[0], dn(w1)[0])

fig.suptitle("[Chelten] SCG Shannon pipeline with interval-filtering search-back -- 17:29:10-17:29:40", fontsize=13)
fig.tight_layout()
out = "/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_shannon_scg_searchback_illustration.png"
fig.savefig(out, dpi=150)
print("->", out)
print(f"in window: {len(cand_w)} candidates, {len(s1p_w)} primary cycles, {len(s1e_w)} search-back recoveries, "
      f"{len(rejected_w)} still rejected, {len(hand_w)} hand clicks, {len(gaps_w)} flagged gaps")

import pickle, sqlite3, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/cnn_recover_eval.pkl","rb") as f:
    R = pickle.load(f)
res_new = R["res_new"]
confirmed_new = R["confirmed_new"]
rec_t = R["rec_t"]

with open("/tmp/cnn_mask_apply.pkl","rb") as f:
    CA = pickle.load(f)
cand_t = CA["cand_t"]; cand_cnn = CA["cand_cnn"]

with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_s = d["ts_s"]; xf_s = d["xf_s"]

with open("/tmp/final_4way_compare.pkl","rb") as f:
    D = pickle.load(f)
ecg_tmid = D["ecg_tmid"]; ecg_hr_sm = D["ecg_hr_sm"]; ecg_v = D["ecg_v"]

def load_peaks(path):
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = ''")
    return np.array([r[0] for r in cur.fetchall()], dtype="int64")
scg_pk = load_peaks("/tmp/annotation_chelten_scg2.db")
ecg_pk = load_peaks("/tmp/annotation_chelten_ecg2.db")

rec_set = set(rec_t.tolist())

def tsec(hh,mm,ss):
    return datetime.datetime(2026,6,26,hh,mm,ss, tzinfo=datetime.timezone.utc).timestamp()*1e6

clusters = [
    ("1: ~0-55s dip below",        tsec(17,26,31), tsec(17,27,26)),
    ("2: ~150-290s scatter",       tsec(17,29, 1), tsec(17,31,11)),
    ("3: ~575-655s above",         tsec(17,36, 6), tsec(17,37,26)),
    ("4: ~745-805s above",         tsec(17,38,56), tsec(17,39,56)),
    ("5: ~1005-1050s isolated low",tsec(17,43,16), tsec(17,44, 1)),
]

fig, axs = plt.subplots(5, 2, figsize=(15, 19))

for row, (label, t0w, t1w) in enumerate(clusters):
    axA, axB = axs[row]
    pad = 6_000_000
    # Panel A: zoomed HR comparison
    esel = (ecg_tmid>=t0w-pad)&(ecg_tmid<t1w+pad)&ecg_v
    axA.plot((ecg_tmid[esel]-t0w)/1e6, ecg_hr_sm[esel], "-", lw=1.4, color="tab:blue", alpha=0.85, label="ECG")
    ssel = (res_new["ta"]>=t0w-pad)&(res_new["ta"]<t1w+pad)
    axA.plot((res_new["ta"][ssel]-t0w)/1e6, res_new["xa"][ssel], ".", ms=6, color="tab:purple", alpha=0.85, label="SCG (new algo)")
    axA.axvspan(0, (t1w-t0w)/1e6, color="0.9", zorder=0)
    axA.set_title(f"{label}\nHR comparison", fontsize=10)
    axA.legend(fontsize=7, loc="best")
    axA.set_ylabel("HR (bpm)")

    # Panel B: waveform diagnostic
    wsel = (ts_s>=t0w)&(ts_s<t1w)
    axB.plot((ts_s[wsel]-t0w)/1e6, xf_s[wsel], color="black", lw=0.5)
    for c in scg_pk[(scg_pk>=t0w)&(scg_pk<t1w)]:
        axB.axvline((c-t0w)/1e6, color="green", lw=1.0, alpha=0.6)
    csel = (cand_t>=t0w)&(cand_t<t1w)
    for t,s in zip(cand_t[csel], cand_cnn[csel]):
        if t in rec_set:
            color="tab:orange"
        elif s>=0.40:
            color="tab:purple"
        else:
            color="red"
        axB.axvline((t-t0w)/1e6, color=color, lw=0.8, alpha=0.5, ymin=0, ymax=0.15)
    axB.set_title("waveform: green=hand click, purple=confirmed, orange=S2-recovered, red=rejected", fontsize=8.5)
    axB.set_xlabel("s")

plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_remaining_clusters_diag.png", dpi=125)
print("saved")

# print quick numeric summary per cluster too
for label, t0w, t1w in clusters:
    hsel = (scg_pk>=t0w)&(scg_pk<t1w)
    esel2 = (ecg_pk>=t0w-D["LAG_US"])&(ecg_pk<t1w-D["LAG_US"])
    csel = (cand_t>=t0w)&(cand_t<t1w)
    n_conf = ((cand_cnn[csel]>=0.40)).sum()
    n_rec = sum(1 for t in cand_t[csel] if t in rec_set)
    print(f"{label:30s} n_hand={hsel.sum():4d} n_ecg={esel2.sum():4d} n_cand={csel.sum():4d} n_confirmed={n_conf:4d} n_recovered={n_rec:3d}")

import pickle, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/mask_v5_flatness.pkl","rb") as f:
    M = pickle.load(f)
with open("/tmp/mask_v5_apply.pkl","rb") as f:
    V = pickle.load(f)
with open("/tmp/shannon_restricted_results_thr0.3.pkl","rb") as f:
    d = pickle.load(f)
ts_s = d["ts_s"]; x_s = d["x_s"]; xf_s = d["xf_s"]

tc=M["tc"]; rms=M["rms"]; lab=M["label"]; THR=V["THR"]
good_rms = rms[lab=="good"]; bad_rms = rms[lab=="bad"]

fig = plt.figure(figsize=(13,10))
gs = fig.add_gridspec(3,2, height_ratios=[1,1,0.9])

# Panel A: RMS histogram
axA = fig.add_subplot(gs[0,:])
bins = np.logspace(np.log10(1), np.log10(500), 60)
axA.hist(good_rms, bins=bins, color="tab:green", alpha=0.5, label=f"hand GOOD (n={len(good_rms)})")
axA.hist(bad_rms, bins=bins, color="tab:red", alpha=0.5, label=f"hand BAD (n={len(bad_rms)})")
axA.axvline(THR, color="black", ls="--", lw=1.5, label=f"threshold={THR:.0f}")
axA.set_xscale("log")
axA.set_xlabel("RMS of respiration-removed (10-100Hz bandpassed) signal in 3s window")
axA.set_title("A. 'Flatness' = RMS after removing respiration.  Threshold catches the LOUD half of bad data...\n"
               "...but ~half of hand-BAD windows are themselves quiet/flat (signal DROPOUTS, not motion) and slip through")
axA.legend(fontsize=8)

# Panel B: example -- genuinely clean flat-between-beats GOOD window
axB = fig.add_subplot(gs[1,0])
t_good = datetime.datetime(2026,6,26,17,35,0, tzinfo=datetime.timezone.utc).timestamp()*1e6
i0 = np.searchsorted(ts_s, t_good-1.5e6); i1 = np.searchsorted(ts_s, t_good+1.5e6)
axB.plot((ts_s[i0:i1]-t_good)/1e3, xf_s[i0:i1], color="tab:purple", lw=0.8)
axB.set_title("B. Example GOOD window: flat baseline + two clean pulses\n(low RMS -- correctly passes)")
axB.set_xlabel("ms")

# Panel C: example -- dropout BAD window that also LOOKS flat
t_bad = datetime.datetime(2026,6,26,17,43,33, tzinfo=datetime.timezone.utc).timestamp()*1e6
i0 = np.searchsorted(ts_s, t_bad-1.5e6); i1 = np.searchsorted(ts_s, t_bad+1.5e6)
axC = fig.add_subplot(gs[1,1])
axC.plot((ts_s[i0:i1]-t_bad)/1e3, xf_s[i0:i1], color="tab:red", lw=0.8)
axC.set_title("C. Example BAD window (near 17:43:33): weak/attenuated signal\n(poor contact, not motion) -- also LOOKS flat by RMS,\nso it incorrectly passes the flatness test")
axC.set_xlabel("ms")

ylim = (min(axB.get_ylim()[0],axC.get_ylim()[0]), max(axB.get_ylim()[1],axC.get_ylim()[1]))
axB.set_ylim(ylim); axC.set_ylim(ylim)

# Panel D: timeline ribbon, hand vs auto(flatness)
axD = fig.add_subplot(gs[2,:])
scg_tmid = V["scg_tmid"]; scg_v_hand = V["scg_v_hand"]; scg_v_auto = V["scg_v_auto"]
t0 = scg_tmid.min()
th = (scg_tmid-t0)/1e6/60
order = np.argsort(scg_tmid)
th_s = th[order]; hv = scg_v_hand[order].astype(int); av = scg_v_auto[order].astype(int)
axD.fill_between(th_s, 1.0, 1.9, where=hv.astype(bool), color="tab:green", step="mid", alpha=0.8)
axD.fill_between(th_s, 1.0, 1.9, where=~hv.astype(bool), color="tab:red", step="mid", alpha=0.4)
axD.fill_between(th_s, 0.0, 0.9, where=av.astype(bool), color="tab:green", step="mid", alpha=0.8)
axD.fill_between(th_s, 0.0, 0.9, where=~av.astype(bool), color="tab:red", step="mid", alpha=0.4)
axD.set_yticks([0.45,1.45]); axD.set_yticklabels(["FLATNESS mask","HAND label"])
axD.set_xlabel("minutes from start of comparison window")
axD.set_title(f"D. Hand label vs flatness-only mask (thr={THR:.0f})  --  beat-level agreement 96.9%, "
              f"but r drops {V['s_h']['r']:.3f} -> {V['s_a']['r']:.3f} because the disagreements are dropouts, not motion")

plt.tight_layout()
plt.savefig("/tmp/dog-test-ecg/dog-test-ecg-code/figs/chelten_flatness_mask.png", dpi=130)
print("saved")

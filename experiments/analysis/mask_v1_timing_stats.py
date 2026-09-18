import pickle
import numpy as np

with open("/tmp/mask_v1_good_samples.pkl","rb") as f:
    G = pickle.load(f)
results = G["results"]

# For each good beat, the click t=0 should sit near the START of S1's first peak.
# classify peaks into "S1 region" (-50 to 120ms) and "S2 region" (120 to 320ms)
n_s1_peaks = []
n_s2_peaks = []
s1_to_s2_gaps = []
s1_amp = []
s2_amp = []
noise_amp = []  # peaks OUTSIDE both S1 and S2 windows (should be quiet)

for r in results:
    pk_t, pk_h = r["pk_t"], r["pk_h"]
    s1_mask = (pk_t>=-50)&(pk_t<=120)
    s2_mask = (pk_t>120)&(pk_t<=320)
    other_mask = ~(s1_mask|s2_mask) & (pk_t>=-300)&(pk_t<=450)
    n_s1_peaks.append(s1_mask.sum())
    n_s2_peaks.append(s2_mask.sum())
    if s1_mask.sum()>0: s1_amp.append(pk_h[s1_mask].max())
    if s2_mask.sum()>0: s2_amp.append(pk_h[s2_mask].max())
    if other_mask.sum()>0: noise_amp.append(pk_h[other_mask].max())
    if s1_mask.sum()>0 and s2_mask.sum()>0:
        t_s1 = pk_t[s1_mask][np.argmax(pk_h[s1_mask])]
        t_s2 = pk_t[s2_mask][np.argmax(pk_h[s2_mask])]
        s1_to_s2_gaps.append(t_s2-t_s1)

n_s1_peaks = np.array(n_s1_peaks); n_s2_peaks = np.array(n_s2_peaks)
print("distribution of #peaks found in S1 window (-50 to 120ms):")
for v in sorted(set(n_s1_peaks.tolist())):
    print(f"  {v} peaks: {100*(n_s1_peaks==v).mean():.1f}%  (n={np.sum(n_s1_peaks==v)})")
print("\ndistribution of #peaks found in S2 window (120 to 320ms):")
for v in sorted(set(n_s2_peaks.tolist())):
    print(f"  {v} peaks: {100*(n_s2_peaks==v).mean():.1f}%  (n={np.sum(n_s2_peaks==v)})")

s1_to_s2_gaps = np.array(s1_to_s2_gaps)
print(f"\nS1-to-S2 peak gap (n={len(s1_to_s2_gaps)}): mean={s1_to_s2_gaps.mean():.0f}ms median={np.median(s1_to_s2_gaps):.0f}ms std={s1_to_s2_gaps.std():.0f}ms")
print("percentiles:", np.percentile(s1_to_s2_gaps,[5,25,50,75,95]))

s1_amp=np.array(s1_amp); s2_amp=np.array(s2_amp); noise_amp=np.array(noise_amp)
print(f"\nS1 amplitude (envelope): median={np.median(s1_amp):.3f}  p10-p90=[{np.percentile(s1_amp,10):.3f},{np.percentile(s1_amp,90):.3f}]")
print(f"S2 amplitude (envelope): median={np.median(s2_amp):.3f}  p10-p90=[{np.percentile(s2_amp,10):.3f},{np.percentile(s2_amp,90):.3f}]")
print(f"S2/S1 ratio: median={np.median(s2_amp)/np.median(s1_amp):.2f}")
if len(noise_amp):
    print(f"'noise' peaks outside S1/S2 windows (n={len(noise_amp)}, {100*len(noise_amp)/len(results):.0f}% of beats had one): median={np.median(noise_amp):.3f}")
    print(f"noise/S1 amplitude ratio: median={np.median(noise_amp)/np.median(s1_amp):.2f}")

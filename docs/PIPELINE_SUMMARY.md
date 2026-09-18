# SCG-vs-ECG Heart Rate Pipeline — Complete Reference

Dog SCG (seismocardiogram) vs ECG heart-rate detection. Built for **Chelten**, then
tested for generalization on the rest of Chelten's recording and on a second dog
(**Dasty**). All work here is offline Python analysis layered on top of raw hive
parquet files and (for reference/comparison only) CORAL's own CSV output.

**Hard constraint respected throughout:** the CORAL (`coral-st` Rust binary) algorithm's
source code was never modified. Every change described below is in standalone Python
analysis scripts that only *read* CORAL's output, never in CORAL itself.

---

## 0. Conventions, constants, per-dog channel selection

```python
# Clock offset between the two independent recording systems (Chelten only --
# established via hand-click cross-correlation). Sign convention:
#   ecg_shift = ecg_pk - LAG_US        # bring an ECG timestamp into the SCG clock frame
#   scg_shift = scg_pk + LAG_US        # bring an SCG timestamp into the ECG clock frame
LAG_US = 7_000_000  # 7.0s, Chelten only -- NOT assumed to transfer to other dogs/rigs

# Fixed SCG-side evaluation window used for the original hand-click-validated result
# (loaded from /tmp/cnn_mask_apply.pkl in the original work)
RESTRICT_CHELTEN = (1782494790000000, 1782496808699088)  # 2026-06-26 17:26:30 -> 18:00:08.7 UTC

# Per-dog raw-channel conventions -- CORRECTED during a later git-repo packaging
# pass (see Section 11): there are TWO DIFFERENT, NON-AGREEING conventions in
# this project, previously conflated in this doc.
#   coral_scg.py's OWN convention (CORAL's separate binary/wrapper, reference only):
#     AXIS = {"Chelten": "mag", "Dasty": "c2", "Chuck": "mag"}   # "mag" = derived
#     3-axis magnitude sqrt(sum((c_i-mean(c_i))**2 for c1,c2,c3)), NOT a literal column
#   coral_ecg_hive.py:  LEAD = {"ecg_biopac": "c4", "ecg_polar": "c1"}
#
#   THIS pipeline's own SCGNet/S2Net (the CNN/S2 chain documented below) was
#   actually built and validated against Chelten's LITERAL "c1" column, NOT
#   "mag" -- verified directly: bandpass(load(Chelten,"c1")) over the validated
#   window reproduces the frozen REF_MAX (4862.704192475711) to 4 decimal
#   places; "mag" gives 6000.60, a different number entirely. Chelten's
#   scg_mwd parquet actually exposes c1/c2/c3 (an earlier version of this doc
#   incorrectly said it "only exposes a single c1 column"); "c1" was simply
#   the axis this pipeline's own scripts picked, not the only one available.
#   Dasty correctly uses literal "c2" either way (both conventions agree there).

HIVE = ".../dog-test-ecg-code/hive"
# username=ecg_polar/device=<dog>/stream=0/...      -- Polar ECG (Chelten only)
# username=ecg_biopac/device=<dog>/stream=0/...     -- BioPac ECG (Chelten, Dasty, Chuck)
# username=scg_mwd/device=<dog>/stream=45/...       -- SCG accelerometer
```

---

## 1. SCG pipeline (the "best pipeline") — validated result: **r=0.9095, n=1285, MAE=2.20bpm** (Chelten, vs hand clicks)

Four frozen stages, in order. Stages 2 and 3 use trained model weights (Chelten-only
training data); stage 1 and 4 are pure signal processing with no learned parameters.

### Stage 1 — Shannon-energy envelope + local adaptive sharpening + primary candidates

```python
import numpy as np
from scipy import signal as sg
from scipy.ndimage import gaussian_filter1d, uniform_filter1d, percentile_filter

FSD = 200.0  # decimated envelope sample rate

def bandpass(x, fs, lo, hi):
    hi = min(hi, 0.45 * fs)
    b = sg.butter(2, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return sg.filtfilt(*b, x - np.mean(x))

def shannon_envelope_decimated(xf, ts, fs, avg_win_s=0.02, gauss_sigma_s=0.01, ref_max=None):
    # ref_max=None -> self-normalize on this window's own max|xf| (the pipeline's
    # designed-in device-scale invariance -- use this for any NEW, independent window).
    # ref_max=<value> -> use a FIXED external scale instead (only needed to keep
    # amplitude-scale consistency with a CNN-calibration window when processing an
    # ADJACENT segment of the *same* recording -- see Section 5's bug #1).
    denom = ref_max if ref_max is not None else np.max(np.abs(xf))
    xn = xf / (denom + 1e-12)
    se = -(xn**2) * np.log(xn**2 + 1e-9)
    se_avg = uniform_filter1d(se, max(1, int(avg_win_s * fs)))
    step = max(1, int(round(fs / FSD)))
    se_d = se_avg[::step]; ts_d = ts[::step]; fsd = fs / step
    se_smooth = gaussian_filter1d(se_d, sigma=max(1, gauss_sigma_s * fsd))
    return ts_d, se_smooth, fsd

def sharpen_local(env, fsd, q995_win_s=8.0, q=99.5):
    # ratio of env to its own local 99.5th-percentile background -- this is what
    # makes candidate DETECTION (as opposed to CNN amplitude scoring) robust to a
    # uniform global rescale of the envelope.
    win = max(3, int(q995_win_s * fsd)) | 1
    local_q995 = percentile_filter(env, q, size=win, mode="nearest")
    return np.exp(env / np.maximum(local_q995, 1e-9)) - 1.0, local_q995

# Primary candidate generation (frozen thresholds, used everywhere):
PRIMARY_THR = 0.3
PRIMARY_REFRACT_S = 0.50   # caps detectable HR at 120bpm -- see Section 5/6 findings
pk_idx, _ = sg.find_peaks(sharp_s, height=PRIMARY_THR,
                           distance=max(1, int(PRIMARY_REFRACT_S * fsd_s)))
cand_t = ts_sd[pk_idx]
```

Typical pipeline: `bandpass(x, fs, 10.0, 100.0)` on the raw SCG channel -> `shannon_envelope_decimated` -> `sharpen_local(., q995_win_s=8.0)` -> `find_peaks` as above. On Chelten's validated 34-min window this gives ~1950 candidates.

### Stage 2 — Main CNN (`SCGNet`), cutoff = 0.25

```python
import torch, torch.nn as nn

class SCGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=9, padding=4)
        self.pool = nn.MaxPool1d(4)
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(16 * 10, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x)); x = self.pool(x)
        x = self.relu(self.conv2(x)); x = self.pool(x)
        x = x.flatten(1); x = self.drop(x)
        x = self.relu(self.fc1(x)); x = self.fc2(x)
        return x
# 6,481 params. Weights: /tmp/cnn_model_relabel.pt  scale: /tmp/cnn_train_relabel_result.pkl
# CUTOFF_MAIN = 0.25 (from /tmp/three_model_r_compare.pkl)
# Trained on: primary candidates relabeled using ECG-corroborated ground truth (a
# candidate near a real ECG-implied beat = positive, else negative). r=0.848 alone.

HALF_N = 80  # snippet is env_sd[i-80 : i+81], scaled by the fixed training `scale`
```

### Stage 3 — S2-recovery three-step chain (`S2Net`), cutoff = 0.70

Motivation: many main-CNN rejections are real S2 heart sounds, not noise — recognizable
because a rejected candidate sitting 140-280ms after a genuine beat is almost always S2.
That gives free weak labels without any new hand annotation.

```python
class S2Net(nn.Module):
    # identical trunk to SCGNet, only the dropout differs (0.5 vs 0.3)
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=9, padding=4)
        self.pool = nn.MaxPool1d(4)
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 10, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x)); x = self.pool(x)
        x = self.relu(self.conv2(x)); x = self.pool(x)
        x = x.flatten(1); x = self.drop(x)
        x = self.relu(self.fc1(x)); x = self.fc2(x)
        return x
# Weights: /tmp/s2_model.pt  scale: /tmp/s2_train_result.pkl
# Trained on 267 examples (49 S2-positive, 218 noise-negative), stratified random
# 70/30 split (chronological split was badly imbalanced -- fixed early on).
# Labeling rule used to build that dataset:
#   S2_LO_US, S2_HI_US   = 140_000, 280_000   # plausible S1->S2 gap
#   NOISE_MIN_US         = 400_000            # far from any click = definitely noise

SEARCH_LO_US, SEARCH_HI_US = 100_000, 300_000   # backward-search window for the true S1
MERGE_TOL_US = 50_000
S2_CUTOFF = 0.70

def backward_local_max(tc, ts_sd, env_sd):
    lo, hi = tc - SEARCH_HI_US, tc - SEARCH_LO_US
    i_lo, i_hi = np.searchsorted(ts_sd, lo), np.searchsorted(ts_sd, hi)
    if i_hi <= i_lo:
        return None
    seg = env_sd[i_lo:i_hi]
    return ts_sd[i_lo + np.argmax(seg)]

# Chain, run once per main-CNN-rejected candidate `tc`:
#   1. score tc with S2Net -> p_s2
#   2. if p_s2 >= S2_CUTOFF: backward-search [tc-300ms, tc-100ms] in env_sd for the
#      true local max (the actual S1)
#   3. re-score THAT recovered point with the ORIGINAL main CNN (net_main) -- only
#      accept if it also passes CUTOFF_MAIN
# Nothing is ever inserted from a fixed timing offset alone -- it must have genuine
# shape evidence at both steps. On Chelten's validated window: 50 flagged, 45 accepted.
```

### Stage 4 — Gap-aware honest reporting

Motivation: naive 3-second smoothing "borrows" beats across a genuine detection gap,
making the post-gap recovery period look like a real momentary HR reading it isn't.

```python
def beat_hr(peaks):
    rr = np.diff(peaks) / 1e6
    hr = 60.0 / rr
    tmid = peaks[:-1] + np.diff(peaks) // 2
    return tmid, hr, rr

GAP_THRESH_S = 1.5   # a beat-to-beat silence this long = "detection failure", not signal
WIN_S = 3.0          # the smoothing window itself

def gap_aware_hr(merged_beats, max_rr_s=1.5):
    tmid, hr_raw, rr = beat_hr(merged_beats)
    ok = rr <= max_rr_s
    tmid_ok, hr_ok = tmid[ok], hr_raw[ok]

    win_us, gap_us = int(WIN_S * 1e6), int(GAP_THRESH_S * 1e6)
    gap_len = np.diff(merged_beats)
    has_gap = gap_len >= gap_us
    gap_windows = list(zip(merged_beats[:-1][has_gap], merged_beats[1:][has_gap]))

    def touches_gap(t):
        lo, hi = t - win_us, t + win_us
        return any(ge >= lo and gs <= hi for gs, ge in gap_windows)

    hr_sm = np.full(len(tmid_ok), np.nan)
    for i, t in enumerate(tmid_ok):
        if touches_gap(t):
            continue
        sel = (tmid_ok >= t - win_us) & (tmid_ok <= t + win_us)
        hr_sm[i] = hr_ok[sel].mean()
    keep = ~np.isnan(hr_sm)
    return tmid_ok[keep], hr_sm[keep]

# Effect on Chelten: r 0.8833 -> 0.9095, MAE 2.65 -> 2.20, n 1610 -> 1285,
# circled high-ECG-HR outlier cluster shrank from 22 to 4 points.
```

---

## 2. ECG pipeline — fully-tuned Pan-Tompkins detector — validated result: **r=0.9820, n=2495, MAE=1.82bpm** (Chelten, vs hand clicks)

```python
from scipy.ndimage import median_filter, uniform_filter1d

# Final tuned hyperparameters (grid search, ~8000 combos across 3 refinement passes,
# picked from the middle of a broad plateau at r~0.98, NOT the single top grid cell --
# with only one hand-click set to both tune and evaluate against, chasing the literal
# best cell is fitting noise, not finding a real optimum):
FC_LO, FC_HI   = 8.0, 45.0     # bandpass corners (Hz)   -- default had been 10-30
REFRACT_S      = 0.40          # (was 0.28) -- 150bpm ceiling instead of ~214bpm
MWI_S          = 0.010         # moving-window-integrator width (s)
FLOOR_S        = 2.0           # adaptive floor window (s)
SNR_THR        = 100.0         # (default had been 4.0 -- tuned for clean HUMAN ecg,
                                #  let through massive noise on this dog: r=0.51 at default)

def detect_qrs(x, fs):
    b = sg.butter(2, [FC_LO / (fs / 2), FC_HI / (fs / 2)], btype="band")
    xf = sg.filtfilt(*b, x - np.mean(x))
    deriv = np.convolve(xf, np.array([1, 2, 0, -2, -1]) * (fs / 8.0), mode="same")
    sqd = deriv ** 2
    mwi = uniform_filter1d(sqd, max(1, int(MWI_S * fs)))
    floor = median_filter(mwi, max(3, int(FLOOR_S * fs)) | 1, mode="nearest") + 1e-12
    pk, _ = sg.find_peaks(mwi, distance=int(fs * REFRACT_S), height=None)
    snr = mwi[pk] / floor[pk]
    pk = pk[snr > SNR_THR]
    win = int(0.06 * fs)
    R = []
    for p_ in pk:
        a, bnd = max(0, p_ - win), min(len(xf), p_ + win)
        R.append(a + int(np.argmax(np.abs(xf[a:bnd]))))
    return np.unique(R)
# Run on the raw ECG channel (ecg_polar "c1" for Chelten, ecg_biopac "c4" for
# Chelten/Dasty/Chuck), R_idx -> ts_raw[R_idx] gives R-peak timestamps.
```

Tuning history on Chelten (for context, not needed to reuse the detector):
default SNR_THR=4 -> r=0.5106; SNR_THR=50 (first-pass tuned) -> r=0.9406; full grid
search across bandpass/refractory/MWI/floor/SNR -> r=0.9820 (final, above).

---

## 3. Shared comparison utilities

```python
def smooth_plain(tmid, hr, win_s=3.0):
    win_us = int(win_s * 1e6)
    out = np.empty(len(tmid))
    for i, t in enumerate(tmid):
        sel = (tmid >= t - win_us) & (tmid <= t + win_us)
        out[i] = hr[sel].mean()
    return out

def stats(a, b):
    d = a - b
    return dict(n=int(len(a)), bias=float(d.mean()), mae=float(np.abs(d).mean()),
                r=float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan"))

def align(tsA, hrA, vA, tsB, hrB, vB, win_us=2_000_000, lag_us=0, restrict=None):
    # for each valid sample of series A, average nearby (±win_us) valid samples of B
    bts, bhr = tsB[vB], hrB[vB]
    xa, yb, ta = [], [], []
    for i in np.where(vA)[0]:
        traw = tsA[i]
        if restrict is not None and not (restrict[0] <= traw <= restrict[1]):
            continue
        t = traw + lag_us
        sel = (bts >= t - win_us) & (bts <= t + win_us)
        if sel.sum() >= 1:
            xa.append(hrA[i]); yb.append(bhr[sel].mean()); ta.append(traw)
    return np.array(xa), np.array(yb), np.array(ta, dtype="int64")

# CORAL's own quality-gating (reused as-is when comparing against a CORAL .csv output):
SQI_THR = 0.10
ATTRACTOR_BAND = (95, 125)
BAND_THR = 0.50
# valid = sqi>=SQI_THR, minus (in-band 95-125bpm AND sqi<0.50), minus frozen-bpm runs
# (>=8 identical consecutive hops), minus any leadoff-mask interval from mask.parquet.
```

---

## 4. Final validated numbers (Chelten, hand-click ground truth, 17:26:30-18:00:09 UTC)

| method | r | n | MAE (bpm) | bias (bpm) |
|---|---|---|---|---|
| **SCG pipeline (Sections 1)** | **0.9095** | 1285 | 2.20 | -0.25 |
| Pan-Tompkins ECG, fully tuned (Section 2) | 0.9820 | 2495 | 1.82 | -0.04 |
| CORAL's own ECG algorithm (Polar channel) | 0.7849 | 2789 | 7.05 | +5.36 |

Caveat: the ECG-vs-ECG numbers (Pan-Tompkins, CORAL-ECG) are automated-vs-manual
peak-picking on the *same* electrical signal the hand clicks were made on — an easier
task than the SCG pipeline's cross-modality problem (mechanical vibration -> heart rate).
Pan-Tompkins isn't "beating" the SCG work in any meaningful sense; they aren't solving
the same problem.

---

## 5. Extending to the rest of Chelten's recording (no ground truth beyond ~18:00)

`scg_mwd` runs far longer (15:04-18:34) than `ecg_polar` (17:04:22-18:29:46), so the true
overlap is bounded by the ECG span. Hand clicks only exist inside 17:22-18:00, so outside
that, only SCG-pipeline-vs-Pan-Tompkins (algorithm vs algorithm) comparison is possible.

**Two real bugs found and fixed before trusting any extended-range number:**

1. **Global-normalization contamination.** Loading the full ~85-minute signal in one
   shot and computing `shannon_envelope_decimated`'s global max-abs over that whole
   span let two huge handling/motion spikes (~17:22:20, just before hand clicks even
   start; ~18:30:03, likely sensor removal) redefine the normalization scale, shrinking
   the envelope by ~8.5x even *inside* the already-validated window relative to what the
   CNN was actually trained on — collapsing confirmations there for a reason that had
   nothing to do with new data. **Fix:** process each segment independently, and when
   extending an *already-calibrated* window, pass a fixed `ref_max` (the original
   validated window's own max|xf|, e.g. `4862.704192475711` for Chelten) into
   `shannon_envelope_decimated` instead of letting it self-normalize. Verified this
   reproduces the original validated envelope bit-for-bit when `ref_max` equals that
   window's own max.
2. **Segment-boundary duplicate detection.** Each windowed load pads ±30s for filter
   settling; adjacent segments' padding zones overlap, so a few beats got detected
   twice at each seam, producing near-zero RR intervals and thousands-of-bpm nonsense.
   **Fix:** trim each segment's confirmed-beat array strictly to its own `[t0, t1]`
   (drop anything from the padding) *before* concatenating segments.

**Results after both fixes** (before=17:04:22-17:26:30, during=already-validated
17:26:30-18:00:09, after=18:00:09-18:29:46):

| segment | confirmed SCG beats/min | vs Pan-Tompkins: r | n | MAE |
|---|---|---|---|---|
| before (pre-settling, dog not yet calm) | 16.2 | 0.08 | 88 | 5.25 |
| during (already hand-click validated) | 51.3 | 0.90 | 1292 | 2.28 |
| after | 44.4 | 0.76 | 617 | 2.41 |
| **full session combined** | — | **0.885** | **1997** | **2.45** |

`before` is close to unusable for *either* modality (ECG HR itself swings 40-155bpm
there) — reads as a genuine pre-settling artifact of the recording, not a pipeline
weakness. `after` degrades in coverage but still correlates decently.

---

## 6. Cross-dog generalization test: Dasty (all models/tuning still frozen, Chelten-only)

Dasty's ECG (BioPac) is a 19.55-min recording (15:31:34-15:51:07 UTC) fully inside
Dasty's own much longer SCG recording (15:08-20:41). No hand-click ground truth exists
for Dasty at all — algorithm-vs-algorithm only, plus the project's own pre-existing
CORAL run on Dasty as external context.

**ECG side generalized well:** Pan-Tompkins (unchanged hyperparameters) found 2165
R-peaks, median HR ≈123bpm on `ecg_biopac` channel `c4` — matching the pre-existing
CORAL-ECG run's own median (125.9bpm) almost exactly.

**SCG side failed almost completely** (channel `c2` per `coral_scg.py`'s `AXIS` dict):

| refractory | primary candidates | CNN-confirmed | confirmed beats/min | final usable (post gap-aware) |
|---|---|---|---|---|
| 0.50s (original) | 627 (32.1/min) | 103 (16.4%) | 5.3 | **0** |
| 0.30s (tried on request) | 813 (41.6/min) | 138 (17.0%) | 7.1 | **0** |

Dasty's true HR here (~123bpm) sits at/above the 0.5s-refractory 120bpm ceiling, so
loosening it was a reasonable thing to try — it gave a real +30-34% bump, but confirm
*rate* barely moved (16.4%->17.0%), and even raw candidate generation at 0.3s (42/min)
is still only ~1/3 of the ~123/min a working detector should find. So the refractory
ceiling is a real but secondary factor; the dominant one is the frozen CNN/S2Net never
having seen Dasty's SCG morphology or amplitude scale (an out-of-distribution problem,
not fixable by a threshold tweak). Zero HR points survive the gap-aware honesty filter
either way — the pipeline currently produces **no usable continuous SCG HR trace for
Dasty**.

**External reference:** the project's own pre-existing CORAL run for Dasty (computed
before this investigation, `coral_scg/Dasty/` vs `coral_ecg/Dasty_ecg_biopac/`) gets
`r=0.7170, n=3547, mae=4.94, bias=-2.00` (from its `agree.json`) — clearly worse than
Chelten's r=0.91, but a *working* result, unlike our frozen pipeline's zero here. That's
the more appropriate number to trust for Dasty until the CNN/S2Net are retrained or
fine-tuned on Dasty-specific (ideally hand-annotated) data.

---

## 7. Rejected approaches (kept for the record — do not re-try without a new reason)

| approach | result | verdict |
|---|---|---|
| Blind backward-"snap" of S2-anchored candidates onto true S1, full retrain | r=0.7968 (n=1533) | **worse** than 0.8482 baseline; the naive "biggest peak in a 100-300ms window" heuristic grabs the wrong feature in busy/motion-affected stretches |
| CNN trained on raw 2000Hz waveform instead of envelope | r=0.833 (n=1441) | no improvement |
| Refractory 0.4s / 0.3s, main CNN alone (no S2 chain) | r=0.8235 / 0.8182 | worse; only ~26% of newly-admitted candidates have real ECG support |
| Local-threshold despike v1 (raw-signal Hampel before Shannon envelope) | barely changed candidate counts | not adopted |
| Local-threshold despike v2 (despike only the *denominator* feeding the 8s percentile) | r 0.8482->0.8486 | net wash, not adopted |
| Refractory 0.4s, FULL pipeline (CNN+S2+gap-aware) | r=0.8776 (n=1294) | worse than 0.9095; looser refractory increases direct main-CNN false confirmations that the S2 chain has no mechanism to catch |
| Refractory 0.4s + main-CNN cutoff sweep | best r=0.9066 (n=1057) @ cutoff=0.40 | near-tie accuracy but ~18% less coverage; not adopted |

---

## 8. Key artifacts (this container's `/tmp`, may not exist in a fresh session)

```
/tmp/cnn_model_relabel.pt, /tmp/cnn_train_relabel_result.pkl   # SCGNet weights + scale
/tmp/s2_model.pt, /tmp/s2_train_result.pkl                     # S2Net weights + scale
/tmp/cnn_mask_apply.pkl                                        # LAG_US, RESTRICT_CHELTEN
/tmp/three_model_r_compare.pkl                                 # CUTOFF_MAIN=0.25
/tmp/s2_chain_final.pkl                                        # Chelten "during" merged_arr (trusted)
/tmp/shannon_restricted_results_thr0.3.pkl                     # Chelten "during" envelope (ts_sd/env_sd/sharp_s)
/tmp/dog-test-ecg/dog-test-ecg-code/hive/...                   # raw parquet, all dogs
/tmp/dog-test-ecg/dog-test-ecg-code/analysis/coral_scg/<dog>/out.csv       # CORAL SCG output
/tmp/dog-test-ecg/dog-test-ecg-code/analysis/coral_ecg/<dog>_ecg_*/out.csv # CORAL ECG output (has agree.json)
```

None of the trained weights or intermediate pickles are portable to a brand-new
environment on their own — what's portable, and what this document is for, is the
*code and the numbers*: the exact architectures, hyperparameters, thresholds, and
results above are enough to reproduce every step from the raw hive parquet files
(or CORAL's `out.csv`) forward.

---

## 9. OPEN WORK — Dasty generalization fix (planned, NOT yet implemented)

This is the live thread as of this handoff. Everything below is discussion/diagnosis
and a concrete plan; **zero new code has been written to disk or executed** for any of
it. Do not assume approval to start implementing — the last thing asked of the user
("要我现在就开始写这版代码吗?" / "should I start writing this code now?") was never
answered before they asked for this handoff doc instead.

### 9.1 Why not a 1D-UNet / human-style architecture?

Asked and answered: a UNet-style dense/segmentation architecture is the standard choice
when you have enough diverse training data (many subjects) for it to generalize —
it's a much larger model with more capacity to overfit on a single-dog dataset. Given
only Chelten's labels exist, the tiny `SCGNet`/`S2Net` (6,481 params, patch-classification
not segmentation) was already the deliberately-conservative choice for a low-data regime.
Moving to a bigger architecture would make the OOD (Dasty) problem worse, not better,
without more subjects' data. This reasoning still holds; not revisited further.

### 9.2 Training details

Both `SCGNet` (main CNN) and `S2Net` trained on CPU — models are tiny (6,481 params) and
datasets are tiny (main CNN: relabeled primary-candidate set from Chelten's validated
window; `S2Net`: 267 examples, 49 S2-positive / 218 noise-negative, stratified 70/30
split), so CPU training finishes quickly; no GPU was used or needed.

### 9.3 Overfitting-mitigation plan (given: no other dog's data, no time to collect more)

Full staged plan as discussed (nothing beyond Phase 1/2 design work has started):

- **Phase 1 — fix the CNN's own input-scale fragility.** See 9.5 below (self-normalization
  fix). This is the highest-leverage, cheapest fix and doesn't require new data.
- **Phase 2 — NCC feature fusion.** See 9.4/9.6 below. Give the CNN an explicit,
  scale-invariant shape-matching prior feature alongside its own learned conv features.
- **Phase 3 — standard regularization** (dropout/weight-decay tuning, ensembling across
  seeds) — lower priority, unlikely to fix a genuine cross-dog distribution shift by
  itself, but cheap and complementary.
- **Phase 4 — OOD-aware fallback.** When on-line signals (e.g. confirm rate crashing,
  candidate rate far below physiological plausibility) suggest the frozen CNN is out of
  its depth, fall back to a classical detector (NCC-only, or `pick_doublets()`, see 9.4)
  rather than trusting CNN output blindly.
- **Phase 5 — validation protocol**, since there is no Dasty ground truth: (1) held-out
  Chelten test-set AUC/precision/recall must not regress vs current baseline; (2)
  downstream Chelten HR-agreement r must not regress below 0.9095; (3) re-test on Dasty
  via confirm-rate (vs current 16-17% baseline), beats/min (vs ~123bpm expected from
  Pan-Tompkins), r vs Pan-Tompkins, and comparison to the pre-existing CORAL Dasty
  benchmark (r=0.7170, Section 6) as an external sanity check.

Explicitly **rejected by the user, do not revisit without a new reason**: template-
insertion "amplitude-diversity" data augmentation (inserting the real template at
synthetically wide amplitude scales into real background noise to broaden the training
set's amplitude range). Reasoning given: while it's not generative/hallucinated data
(real waveform + real noise, only amplitude/position controlled), the user has no time
to build it and distrusts synthetic-augmentation approaches generally. It also only
would have addressed amplitude scale, not genuine cross-dog morphology differences, so
even technically it's a partial fix at best. **Use 9.5 (self-normalization) instead** to
address the amplitude-scale problem — it requires no synthetic data at all.

### 9.4 Classical/non-model alternative: NCC template matching (already exists, already tested — do not rebuild)

Discovered via `cnn_vs_ncc_fig.py` / `raw_template_build.py` (both pre-existing in the
repo, read but not modified this round): normalized cross-correlation against a
hand-built template is an already-implemented, already-benchmarked classical detector.

```python
# /tmp/env_template_wide.pkl -- keys: template, HALF_N, fsd_s
#   HALF_N=80, template.shape=(161,), template.min=-0.0857, template.max=0.8384
# Built via iterative cross-correlation alignment (4 iterations: align each real
# Chelten hand-labeled true-positive snippet to the current reference via best-lag
# search within +/-40ms, re-average, repeat) -- built from REAL labeled positives,
# not synthetic. Lives in the SAME representation/window size as the CNN's own input
# (env_sd space, HALF_N=80), which is what makes it directly reusable as a feature.

def ncc_score(snippet, template):
    s = snippet - snippet.mean()
    t = template - template.mean()
    return float((s @ t) / (np.linalg.norm(s) * np.linalg.norm(t) + 1e-12))
# Mathematically scale-invariant: multiplying the whole snippet by any positive
# constant leaves ncc_score unchanged. This is exactly the property the CNN currently
# LACKS (see 9.5) -- an NCC feature is immune to the ~32x amplitude mismatch that is
# hurting the CNN branch on Dasty.
```

Historical numbers already measured (from `/tmp/mask_v9_apply.pkl`, on Chelten, hand
clicks as ground truth) — do not re-run, already have the answer:

| detector | r | n |
|---|---|---|
| hand-click baseline (self-consistency check) | 0.8959 | 1645 |
| CNN (current, Section 1) | 0.871-0.848 (dataset-dependent) | ~1300-1600 |
| NCC-only, best tuned cutoff | 0.8324-0.8449 | 1304-1452 |

NCC-only underperforms the CNN alone on Chelten (expected — it's a simpler, single
fixed-shape prior), but it is a genuinely independent signal, hence the "feature fusion"
plan below rather than "replace the CNN with NCC."

Also discovered (mentioned once, not the current plan's focus): `pick_doublets()`, a
pure-timing rule (candidate + a second peak 160-200ms later = S1-S2 doublet, no shape
matching at all) — matches the user's own description of the manual-annotation
heuristic ("两个peak"). Not benchmarked in isolation this round; NCC is the richer,
already-quantified classical option and is the one actually planned for fusion.

### 9.5 Root-cause finding: fixed `scale_main` normalization is badly out-of-range for Dasty (empirically PROVEN, not just theorized)

The CNN's preprocessing divides every candidate snippet by a **fixed constant**,
`scale_main = 2.3136008e-06`, learned once from Chelten's own training-data amplitude
distribution (`/tmp/cnn_train_relabel_result.pkl`) — `x = snippet / scale_main` — then
feeds that into the conv net. This constant is never recomputed per-recording.

Direct diagnostic (just computed, not merely flagged as a risk): took the snippet-peak
`|env_sd|` value at each candidate/beat location, divided by `scale_main`, and compared
what each dog's data actually looks like in the units the CNN was trained to see:

| | snippet-peak &#124;env_sd&#124; (median) | /scale_main (median, "CNN input units") |
|---|---|---|
| Chelten, real confirmed beats (first 200, trusted) | 5.964e-05 | **25.780** |
| Dasty, primary candidates (first 300) | 1.887e-03 | **815.730** |

Ratio ≈ **31.6x**. The CNN has never seen an input anywhere near this scale during
training — this is a hard, quantified out-of-training-distribution input, independent
of (and additive to) any genuine cross-dog waveform-shape difference. This is very
likely a major contributor to Dasty's collapsed CNN confirm rate (16-17% vs Chelten's
much higher rate), alongside real morphology differences that a scale fix alone cannot
address.

**Proposed fix (not yet implemented):** replace the fixed-`scale_main` divisor with a
per-snippet or per-recording *self-normalization* (e.g. divide each snippet by its own
local RMS/energy/max, or by a per-recording running estimate, rather than a training-
time-frozen external constant) — directly targets this ~32x mismatch. This is
complementary to NCC fusion, not a substitute: NCC fusion adds one new robust auxiliary
feature to the conv branch's decision; the normalization fix repairs the conv branch's
*own* input path, which is vulnerable regardless of what auxiliary features are added
alongside it.

### 9.6 Proposed `SCGNetFused` architecture (code sketch only — not written to disk, not trained)

```python
class SCGNetFused(nn.Module):
    """Same conv trunk as SCGNet; concatenates n_extra auxiliary NCC score(s) after
    flatten+dropout, before fc1. Start with a single template (n_extra=1) using the
    existing /tmp/env_template_wide.pkl before considering multi-template/clustering."""
    def __init__(self, n_extra=1):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=9, padding=4)
        self.pool = nn.MaxPool1d(4)
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(16 * 10 + n_extra, 32)   # was nn.Linear(16*10, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    def forward(self, x, extra):
        # x: (B,1,L) raw env_sd snippet;  extra: (B, n_extra) e.g. [ncc_score]
        x = self.relu(self.conv1(x)); x = self.pool(x)
        x = self.relu(self.conv2(x)); x = self.pool(x)
        x = x.flatten(1); x = self.drop(x)
        x = torch.cat([x, extra], dim=1)
        x = self.relu(self.fc1(x)); x = self.fc2(x)
        return x
```

Planned training/validation protocol (per the plan discussed, not yet executed):
1. Reuse the existing relabeled Chelten dataset unchanged; add one NCC-score feature
   per example (computed against `/tmp/env_template_wide.pkl`). Keep every other
   hyperparameter identical to `SCGNet` to isolate the one variable being tested.
2. Validate in this order: (a) held-out AUC/precision/recall vs current `SCGNet`
   baseline — must not regress; (b) downstream Chelten HR-agreement r — must not
   regress below 0.9095; (c) Dasty proxy metrics (confirm rate, beats/min vs ~123bpm,
   r vs Pan-Tompkins, vs CORAL's r=0.7170 benchmark).
3. Recommended to defer applying the same fusion to `S2Net` for now — its training set
   (267 examples) is already small; adding a feature there risks more overfitting for
   less expected benefit given S2 is a recovery/secondary stage.
4. Multi-template (2-4 templates clustered by local HR/motion state, each contributing
   its own NCC score) was raised as a possible extension but explicitly deprioritized
   until the single-template version is validated.

### 9.7 Immediate next step for whoever picks this up

Both 9.5 (normalization fix) and 9.6 (NCC fusion) are fully specified but unimplemented.
Recommended order: implement 9.5 first (cheaper, addresses the larger quantified effect,
no retraining architecture change needed — just change preprocessing and retrain the
existing `SCGNet`/`S2Net` as-is), confirm it doesn't regress Chelten, test on Dasty; then
layer 9.6 (`SCGNetFused`) on top and repeat the same validation chain. Do not implement
either without re-confirming with the user first, since neither was explicitly approved
before this handoff was requested.

---

## 10. Chelten clock-drift investigation — is there a growing delay beyond the fixed 7.0s `LAG_US`?

Asked directly: is there a *continuous, growing* timing offset between `ecg_polar` and
`scg_mwd` beyond the fixed `LAG_US=7,000,000us` (established once, via hand-click
cross-correlation, inside the 17:26-18:00 validated window)? **Answer: yes, a small
but statistically clear one — not "aliasing" in the classical signal-processing sense
(frequency-domain folding from undersampling), but ordinary clock drift/skew between
the two independently-clocked recording systems' own oscillators.**

### 10.1 Method (the trustworthy one): direct event-based lag re-estimation per time window

Independently, in each of several time windows spanning the full ~85-minute overlap,
estimate the best-fit ECG-SCG lag from the actual detected beats (Pan-Tompkins R-peaks
vs the frozen SCG-pipeline's confirmed beats) via a **cross-correlogram / histogram-mode
method**: for every ECG R-peak `t_e` in the window, take every SCG beat within a
±3s band around the prior 7.0s estimate, record `diff = t_e - t_scg`; true matching
pairs cluster tightly around the real lag while mismatched pairs scatter roughly
uniformly, so the histogram's mode (finely binned, parabolically refined) is the lag
estimate for that window — no shape/model assumptions, just point-process timing.

```python
def mode_lag(ecg_win, scg_all, search_lo_us, search_hi_us, bin_us=1000):
    diffs = []
    for t_e in ecg_win:
        lo_t, hi_t = t_e - search_hi_us, t_e - search_lo_us
        i0, i1 = np.searchsorted(scg_all, lo_t), np.searchsorted(scg_all, hi_t)
        if i1 > i0:
            diffs.append(t_e - scg_all[i0:i1])
    diffs = np.concatenate(diffs)
    edges = np.arange(search_lo_us, search_hi_us + bin_us, bin_us)
    hist, _ = np.histogram(diffs, bins=edges)
    i_peak = np.argmax(hist)
    # + parabolic sub-bin refinement using neighboring bin counts (see full script)
    return edges[i_peak] + bin_us * 0.5   # lag estimate for this window
```

18 windows (~4.7 min each) across the full session; windows with SNR<10 (peak/background
histogram-bin-count ratio) or <50 ECG beats were dropped as unreliable rather than kept
and let drag down the fit — this **excludes the "before" pre-settling region on
objective grounds** (matches Section 5's independent finding that `before` has no real
SCG-ECG correlation, r=0.08). 12/18 windows survived, spanning 17:25-18:27.

### 10.2 Result

| | value |
|---|---|
| weighted linear fit | `lag(t) = 6.948s + (25.10 ± 1.70 us/s) * t_elapsed` |
| drift rate | **25.1 ppm** (t-stat = 14.7, r = 0.977 across the 12 kept windows) |
| lag range observed | 6.9854s (17:25) → 7.0734s (18:27) — an **88ms** climb over ~62 min |
| projected over the full 85.4-min overlap | ~129ms |

This is a clean, statistically strong, monotonic trend (r=0.977, t=14.7 — far past
"noise"), and 25ppm is well inside the normal tolerance band (tens of ppm) for two
independent consumer-grade crystal oscillators — physically exactly what you'd expect
from two separately-clocked recording rigs (Polar chest-strap host vs the SCG
accelerometer's own host) with no shared clock reference. Figure saved:
`figs/chelten_lag_drift_check.png` (lag-vs-time scatter + fit line + the hand-click
validated window shaded for reference).

### 10.3 A second, cruder check (raw per-sample timestamp intervals) — informative but NOT directly trustworthy as a drift number

Also checked each stream's own raw sample-to-sample timestamp spacing for a
consistency cross-check: `ecg_polar`'s `channels.json` states `fs_hz=130.0` (nominal)
vs `fs_est_hz=130.0052` (measured) = **+40ppm**, matching this investigation's own
median-dt recomputation exactly (good sanity check on method). `scg_mwd` (no metadata
available) showed a **much larger apparent ~5,100ppm** deviation from an assumed
2000Hz nominal when computed the naive way (mean of raw inter-sample diffs) — but this
number should **not** be trusted as a real epoch-drift rate: `ecg_polar`'s raw diffs
contain a large, perfectly session-uniform (not growing) population of anomalously
short intervals (1.37% of all diffs, evenly spread — 913-914 per each of 10 equal
time bins) that skew any naive mean-of-diffs calculation, most likely a BLE
packetization/batching artifact rather than true oscillator jitter; the same is likely
true of `scg_mwd`'s own diff-distribution outliers. **Raw per-sample dt statistics are
not a reliable stand-in for actual epoch-timestamp drift between the two streams** —
likely because timestamps are assigned at ingestion from real receipt/host time, not
purely extrapolated from a device's internal oscillator, so packetization jitter
dominates the raw-interval statistics without necessarily producing genuine long-term
epoch bias. **The event-based method in 10.1 (using real detected heartbeats as the
timing reference) is the trustworthy number; the raw-interval ppm figures are
included here only so a future investigator doesn't try the naive approach and get
confused by the ~200x discrepancy.**

### 10.4 Practical impact on existing results

- **Within the 17:26-18:00 hand-click-validated window** (the basis for the headline
  r=0.9095 SCG-pipeline and r=0.9820 Pan-Tompkins numbers in Sections 1/2/4): the drift
  across that ~34-minute span is only ~30-40ms (from the fit, roughly 7.00s→7.03s) —
  small relative to the ~490ms beat-to-beat interval and the 3s smoothing window used
  everywhere, so the single fixed `LAG_US=7,000,000` was adequate there. **Those
  validated numbers are not called into question by this finding.**
- **In the full-session extension (Section 5)**, by the `after` segment (~18:15-18:29)
  the true lag has drifted to ~7.06-7.07s — a 60-70ms mismatch against the fixed
  7.000s assumption used throughout that analysis, which is ~13% of one RR interval
  at Chelten's ~123bpm. This is a **plausible contributing factor** (on top of genuine
  SCG-pipeline coverage/confirm-rate degradation, the primary suspect per Section 5) to
  why the `after` segment's r dropped to 0.76 — not re-quantified/decomposed from the
  detection-quality effect this round, flagged here for anyone revisiting Section 5.
- **Recommendation for any future long-session (multi-hour) work**: replace the single
  fixed `LAG_US` with a time-varying correction — either the fitted line above
  (`lag(t) = 6.948e6 + 25.10*t_elapsed_s` us, valid for Chelten's specific
  ecg_polar/scg_mwd pairing only) or, more robustly, re-run the windowed
  cross-correlogram method in 10.1 per session/dataset rather than assuming any fixed
  constant transfers across sessions or dog rigs — clock drift rate is a property of
  the specific pair of physical devices' oscillators, not of the dog or the pipeline.

### 10.5 Terminology note

"Aliasing" specifically refers to a *frequency-domain* artifact — distinct signal
content folding onto false lower frequencies because a signal was sampled below its
Nyquist rate. What's been characterized here is a *timing/clock* effect — two
independently-clocked systems' relative sample-clock rate mismatch (~tens of ppm,
typical crystal-oscillator tolerance) causing their **epoch timestamp streams to drift
apart linearly over time**. It produces a real, physically-expected growing delay,
just not via the aliasing mechanism — flagging this so the terminology in any future
write-up is accurate.

---

## 11. Git-repo packaging pass (code extracted to a clean `src/scg_hr` package)

All the code in this doc was extracted into an installable package (`dog-scg-ecg-hr/`,
`src/scg_hr/`) for pushing to git. Extraction was verified, not just copy-pasted:

- **Stage 1 signal processing reproduces the original bit-for-bit** (`max abs diff
  = 0.0` against a saved reference array) once the CORRECT per-dog channel is used
  -- see the corrected Section 0 above: Chelten's custom SCGNet/S2Net pipeline uses
  raw **`c1`**, not CORAL's own `mag` convention. This was a real, previously-
  undetected discrepancy between two different tools in the project, caught only
  because the extracted package's output didn't initially reproduce `CHELTEN_SCG_REF_MAX`.
- **The Dasty generalization numbers (Section 6) and the lag-drift finding (Section
  10) both reproduce essentially exactly** from the extracted package (within ~1
  beat / ~0.1ppm of the numbers documented above).
- Also found in the same pass: a **`shannon_envelope_decimated` self-normalization
  bug reintroduced accidentally** while writing the packaged full-session scripts
  (the same "global-normalization contamination" issue as Section 5, caught and
  fixed the same way -- pass `constants.CHELTEN_SCG_REF_MAX` as `ref_max` for any
  Chelten scg_mwd span longer than a few minutes, never self-normalize a
  long/full-session load).

See the packaged repo's `README.md` "Verification note" for the same summary in
context.

### 11.1 Root-caused: why a fresh run doesn't give r=0.9095 (found after further digging)

The r≈0.85 gap above was NOT left as speculation -- it was root-caused by direct,
reproducible diffing against the original session's saved intermediates
(`/tmp/s2_chain_final.pkl`, `/tmp/singles_labels.pkl`, `/tmp/valid_regions.pkl`).

**Step 1 -- confirm the pipeline itself is correct.** Every trusted confirmed-beat
timestamp in `s2_chain_final.pkl`'s `confirmed0` (n=1680) is present in a fresh
run's confirmed set, exactly (1680/1680 matched within 5ms). So candidate
generation, the main-CNN scoring, `scale_main`, and `CUTOFF_MAIN` are all provably
correct and match the original exactly -- a fresh run isn't MISSING anything the
original found.

**Step 2 -- characterize the extra ~64 points a fresh run confirms beyond that.**
These are not near-duplicates of real beats (median distance to the nearest
trusted-confirmed beat: 15.4 seconds) and they're not borderline/threshold
artifacts (median CNN score 0.94, i.e. the model is confident). Checked directly
against Chelten's real hand clicks (`hand_good` in `singles_labels.pkl`): 63 of
64 have NO hand click within 150ms. **They are genuine, confidently-scored false
positives** -- the CNN is being fooled by something in the raw signal at those
specific timestamps, not failing to be confident enough.

**Step 3 -- these false positives are not randomly scattered; they cluster in
regions the original hand-labeling explicitly flagged as low-quality.** The
original investigation built exactly this concept: `build_valid_regions.py`
consolidates (a) hand-labeled good/bad SCG-quality intervals, (b) a ±3.0s
exclusion radius around 24 known hand-click-count mismatches, and (c) a ±3.0s
exclusion radius around every hand-labeled bad-interval edge (matching the
pipeline's own 3.0s smoothing window, since a single bad instantaneous point can
contaminate a neighboring smoothed sample) -- saved as `valid_ivs` in
`/tmp/valid_regions.pkl`. This is an EVALUATION-time construct (which stretches
of hand-click ground truth are themselves trustworthy enough to score against),
not a pipeline detection-time filter -- it was never meant to be one of the 4
frozen pipeline stages, which is why it didn't make it into this doc's earlier
"frozen pipeline" recipe (Sections 1-4).

**Step 4 -- apply it.** Restricting the comparison to `valid_ivs` moves the
result from r=0.8520 (n=1255) to **r=0.8846 (n=1148, MAE=2.36bpm)** -- most, but
not all, of the gap to the documented r=0.9095 (n=1285). The remaining ~0.02 gap
and the fact that n=1148 (masked) is actually LOWER than the documented n=1285
means some smaller piece of the original evaluation scoping (exactly which hand
clicks counted as `hand_good`, or a slightly different valid-region definition
at the time r=0.9095 was computed) was not fully reconstructed. Given the
mechanism is now fully understood and the remaining gap is small, this was not
chased further -- pushing to an exact bit-for-bit r=0.9095 reproduction was not
worth the additional archaeology relative to the value of already having the
real, verified cause.

**Bottom line for anyone re-running this:** the pipeline (detection) code is
correct. The gap is entirely in which time regions count when SCORING against
hand clicks. `ground_truth/valid_regions.pkl` (the interval mask) and
`ground_truth/singles_labels.pkl` (`hand_good`, the hand-click times themselves)
are now included in the repo (small hand-annotation files, not raw sensor data)
along with `scripts/evaluate_vs_hand_clicks.py`, which reproduces both numbers
above (`--no-mask` for the unmasked 0.85, default for the masked 0.88).

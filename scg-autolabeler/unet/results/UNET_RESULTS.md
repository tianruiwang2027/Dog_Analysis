# 1D U-Net — First Real Attempt, Results

## What this is

The 1D U-Net approach discussed in conversation, actually trained and evaluated on real
Chelten data for the first time. Unlike everything else in this project, this doesn't
use Stage-1 (the Shannon-energy candidate detector) at all — it takes the continuous
SCG envelope directly and predicts, for every single sample, whether it's near a real
peak. Code: `teacher/unet1d.py` (the network itself) and `run_unet.py` (training +
evaluation on Chelten).

## Two real problems found and fixed along the way

**Bug #1 — BatchNorm collapse.** The first run trained fine (loss dropped steadily) but
then predicted essentially nothing at evaluation time. The cause: the SCG envelope's raw
amplitude is tiny (mean ~0.00005, max ~0.08), and feeding that scale straight into the
network left the very first BatchNorm layer's running variance collapsing to ~1e-7 —
numerically fine during small windowed training batches, but badly miscalibrated the
moment inference ran over a much longer, differently-shaped continuous sequence. Verified
directly: in training mode (using live batch statistics) the model separated real clicks
from background just fine (mean probability 0.47 vs 0.16); in eval mode (using the
broken running statistics) everything collapsed to near-zero. Fixed by z-scoring the
envelope before it reaches the network, using only the training portion's own mean/std
(never test's, to avoid leaking test statistics).

**Methodology issue — threshold tuning on the test set.** After the fix, sweeping the
decision threshold directly against the held-out test set showed a big, suspicious swing
(F1 jumping from 0.25 at threshold 0.5 to 0.66 at threshold 0.3). Picking a threshold by
looking at test performance is a real form of leakage — the same mistake the earlier
repeated-trials work was specifically designed to avoid. Fixed by carving an internal
validation slice (the last 5 of the 30 labeled training minutes) that the true held-out
test span never touches; every epoch, the model's decision threshold is chosen on this
validation slice via a small grid search, and the single best (epoch, threshold)
combination — by validation F1 — is used for exactly one evaluation on the real held-out
test data at the very end.

## The result — and a genuinely important wrinkle

At first glance, the U-Net looks worse than your existing best approach:

| | precision | recall | F1 |
|---|---|---|---|
| MAE-pretrained + 30-min supervised (candidate classifier) | 0.803 | 0.911 | 0.854 |
| **1D U-Net (this experiment)** | **0.596** | **0.851** | **0.701** |

But those numbers are not actually measuring the same thing, and the difference matters.
The candidate-classifier's "recall" is only ever computed over the real clicks that
Stage-1 happened to propose a candidate for in the first place — clicks Stage-1 missed
entirely are simply invisible to that metric, never counted as a miss. Checking this
directly on the held-out window: of the 309 real hand-clicked peaks, Stage-1 only ever
proposes a candidate for **157 of them (50.8%)** — the rest can never be recovered by
any classifier built on top of Stage-1, no matter how good it is. The U-Net has no such
ceiling, because it works directly on the raw signal — its 0.851 recall is against all
309 real clicks, unconditionally.

Putting both methods on the same, genuinely fair basis — recall against every real click
in the window, not just the ones Stage-1 bothered to propose — flips the story:

| | precision | recall (vs. ALL real clicks) | F1 (overall) |
|---|---|---|---|
| From-scratch + 30-min supervised | 0.797 | 0.382 | 0.516 |
| MAE-pretrained + 30-min supervised | 0.803 | 0.463 | 0.587 |
| **1D U-Net** | **0.596** | **0.851** | **0.701** |

On this basis, the U-Net actually has the best overall F1 of anything built in this
project so far — not because its predictions are cleaner (its precision is genuinely
the worst of the three, more false alarms per detection), but because it isn't capped by
Stage-1's candidate-generation ceiling. This is a real, useful finding independent of
whether the U-Net's precision can be improved: **Stage-1 itself, not the classifier
sitting on top of it, has been the main ceiling on recall this whole project.**

## Honest caveats

- Precision (0.596) is a real weakness — the U-Net produces meaningfully more false
  positives per true detection than the candidate classifiers. Likely fixable with more
  training data, more epochs (the training loss hadn't fully plateaued at 40 epochs),
  or a stricter post-processing rule.
- The validation slice used to pick the epoch/threshold is small (5 minutes, 296 real
  clicks), and its own F1 swung noisily between 0.2 and 0.95 across epochs — a sign that
  a single number chosen from one small validation slice deserves the same skepticism the
  earlier single-split result got, before this project's repeated-trials evaluation.
  Repeating this with multiple validation/test windows (the same idea as
  `REPEATED_TRIALS_RESULTS.md`) would be the natural next check.
- Trained completely from scratch — no reuse of `mae_pretrained.pt`. Given how much the
  MAE encoder helped the candidate classifier's few-label regime, transferring it into
  the U-Net (it would need an adapter, since the MAE is a patch/token transformer and the
  U-Net is a conv stack) is a promising next step, not yet tried.

## Files

- `unet1d.py` — the standalone `UNet1D` module and `build_peak_proximity_mask`.
- `unet_train.py` — shared train/validation-based epoch/threshold selection + one
  held-out evaluation, reused by `run_unet.py`.
- `run_unet.py` — thin wrapper that runs the from-scratch training above.
- `unet_best.pt` — the selected model weights (epoch 38/40, threshold 0.25).
- `results/unet_results.json` — full per-epoch history plus the final held-out result.
- `results/unet_comparison.png` — the two-panel comparison plot above.

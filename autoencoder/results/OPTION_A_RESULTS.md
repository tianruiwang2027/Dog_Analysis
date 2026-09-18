# Option A (Foundation-Model Pretraining) — Results

What this is: `DESIGN_PROPOSAL.md`'s Option A (`teacher/models.py:JointMaskedAutoencoder`,
the npj Cardiovascular Health / Eko Health masked-autoencoder recipe applied to ECG+SCG)
was architecturally complete but never trained or wired into anything. This run
implements the missing training loop (`teacher/pretrain_mae.py`), wires the pretrained
encoder into a `ShapeMILModel`-compatible trunk, and runs the actual comparison this
was for: **does pretraining on unlabeled multi-dog/human data help when you only have
~30 real minutes of hand-labeled Chelten data to fine-tune on?**

Code: `teacher/pretrain_mae.py` (MAE pretraining + `MAEShapeEncoder`/`ShapeMILModelMAE`),
`teacher/finetune_chelten.py` (real supervised fine-tune on hand clicks),
`run_option_a.py` (orchestrates all of it).

## Setup

**Pretraining pool** (Chelten excluded, exactly as everywhere else in this codebase):
Chuck + Dasty (non-invasive dogs, 30.6 min), 15 epicardial dog/porcine recordings
(10.4 min — 3 of the original 18 `EPI_FILES` were skipped, same as the ShapeMIL baseline
run: too few Pan-Tompkins-detected beats on their LBBB-intervention recordings), 4 human
CEBS recordings (19.5 min). **12,564 sliding SCG+ECG windows** (161 samples each,
matching the existing candidate-snippet convention exactly — see `pretrain_mae.py`'s
module docstring for why) pooled across all of it.

**JointMaskedAutoencoder**: d_model=64, 4 encoder layers, patch_len=23,
7 patches/channel (7 SCG + 7 ECG tokens per window), mask_ratio=0.5. Trained 12 epochs,
reconstruction MSE 0.143 → 0.07ish (both channels self-normalized per window first).

**Real labels**: both `Annotation Chelten SCG` (2,011 hand-clicked SCG peaks) and
`Annotation Chelten` (5,166 hand-verified ECG beats) cover the same **37.6-minute**
window. Split at 30 minutes: **1,771 Stage-1 candidates / 888 real positives** for
fine-tuning, **405 candidates / 157 real positives** in the held-out ~7.6 minutes for
grading. Both files are in their own recording's native clock (confirmed by matching
existing code's use of `cand_t_native` for grading), so no clock alignment was needed
for the supervised experiments at all.

## Headline result — does Option A pretraining help the 30-minute fine-tune?

| | precision | recall | F1 |
|---|---|---|---|
| From-scratch trunk, 30-min supervised fine-tune | 0.797 | 0.752 | 0.774 |
| **MAE-pretrained trunk, 30-min supervised fine-tune** | **0.803** | **0.911** | **0.854** |

Same real labels, same held-out grading window, same fine-tuning recipe — only the
trunk differs. Pretraining recovers **21 of the 39 real peaks the from-scratch model
missed** (39 false negatives → 14) at essentially no precision cost (0.797 → 0.803).
This is the actual npj Cardiovascular Health / Eko Health thesis, holding up on real
ECG+SCG data with genuinely small (30-minute) labeled fine-tuning: **yes, it helps.**

## Secondary result — does it help under pure weak supervision too (no real labels at all)?

| | precision | recall_conditional | recall_overall | ceiling |
|---|---|---|---|---|
| ShapeMIL baseline (from-scratch trunk, fresh run) | 0.804 | 0.947 | 0.495 | 0.523 |
| MAE-pretrained trunk, same weak MIL objective | 0.787 | 0.933 | 0.488 | 0.523 |

No — under the existing fully-blind weak-supervision recipe (ECG-timing-only pseudo-labels,
no hand clicks anywhere), the MAE-pretrained trunk is statistically indistinguishable
from (very slightly behind) training from scratch. This makes sense: weak supervision
already has plenty of pseudo-labeled examples (thousands of candidates across the pool),
so there's little "label scarcity" for pretraining to compensate for — pretraining's
value shows up specifically in the **few-real-label regime**, exactly where the
supervised experiment above tested it.

**Bonus finding, unplanned**: the MAE-pretrained trunk trains roughly **5x faster**
under the weak-supervision objective than the from-scratch `CandidateTokenEncoder`
(168s vs ~900s+ for 20 epochs over the same pool, both on this machine's 2 CPU cores) —
the patch-based encoder attends over 14 tokens per candidate instead of 161 raw samples,
an ~130x reduction in attention cost per candidate. Worth keeping in mind independent of
the accuracy question if training wall-clock ever matters (e.g. iterating on a new dog).

## Honest caveats

- The held-out grading window is only ~7.6 minutes (157 real positive clicks) — enough
  to see a real, fairly large effect (21 more true positives), not enough for tight
  confidence intervals. Worth confirming on Dasty or a future dog with its own hand
  clicks before treating the gap as precisely "21 beats," though the direction (MAE
  pretraining helps the few-label regime) is the more important, more likely-to-replicate
  finding than the exact numbers.
- Supervised precision/recall (per-Stage-1-candidate) and the weak-supervision
  recall_conditional/ceiling framing are different metrics computed differently — see
  `option_a_comparison.png`'s right panel, which keeps them visually separate rather
  than implying they're on the same scale.
- Pretraining window length (161 samples, ~0.8s) is shorter than DESIGN_PROPOSAL's
  original 4-second suggestion, chosen instead to exactly match the existing
  161-sample candidate-snippet convention so the pretrained encoder plugs into
  `ShapeMILModel`'s downstream usage with no resizing/pooling mismatch. A longer
  pretraining window (spanning several heartbeats, letting positional embeddings learn
  cross-beat periodicity the way the original npj paper reports) is the natural next
  experiment if this direction is pursued further, at the cost of no longer being a
  drop-in trunk for the existing candidate-snippet pipeline without an adapter.
- This machine has 2 CPU cores; the full run (pretraining + weak-supervision comparison
  + both fine-tune experiments) took ~9 minutes end to end, separate from the ~17
  minutes the from-scratch ShapeMIL baseline took on its own.

## Files

- `teacher/pretrain_mae.py` — MAE pretraining loop, `MAEShapeEncoder` (ECG tokens masked
  out at inference so shape scoring stays strictly shape-only, matching `shape_mil.py`'s
  design), `ShapeMILModelMAE`, `train_shape_mil_given_model`.
- `teacher/finetune_chelten.py` — loads both annotation files, splits Chelten's labeled
  window at 30 minutes, supervised fine-tune + held-out grading.
- `run_option_a.py` — runs everything above end to end; `option_a_results.json` is its
  raw output.
- `mae_pretrained.pt` — the pretrained `JointMaskedAutoencoder` weights.
- `option_a_comparison.png` — the two-panel comparison plot.

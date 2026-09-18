# Repeated-Trials Evaluation — Does MAE Pretraining Actually Help?

## Why this run happened

The original Option A result (MAE-pretrained trunk: F1 0.854 vs. from-scratch: F1 0.774) came from **one** train/test split — the last 7.6 minutes of Chelten's labeled recording held out, the first 30 minutes used to fine-tune. That's one data point. You asked to see how it performs "repeated many times using different test segments," to find out whether that result was real or just a lucky split.

I checked whether Dasty could supply a second real dog for this — it can't. Dasty's SCG annotation file ("Annotation Dasty SCG") turned out to have **zero hand-clicked SCG peaks** in it (only 38 ECG clicks over 5.6 minutes, no SCG ground truth at all). So "different test segments" here means different random time windows drawn from Chelten's one labeled recording, not different dogs.

## What this run did

Chelten has 37.6 minutes of recording where both the SCG peaks and the ECG beats were hand-labeled (2,011 real clicked SCG peaks total). For 15 separate trials, I:

1. Picked a random contiguous 7.5-minute stretch as the **held-out test window** (required at least 15 real clicks in it, otherwise redrew).
2. Trained a **fresh** from-scratch model and a **fresh** MAE-pretrained model (same pretrained starting point every time, but each gets its own new fine-tuning) on everything **outside** that window.
3. Graded both models only on the held-out window, and compared them.

Same test window used for both models within a trial, so each trial is a fair, paired comparison — both models faced exactly the same segment of real data, just with different starting points.

As a quick reminder since you asked about this recently: **precision** is "of the peaks the model flagged, what fraction were real" (false alarms hurt this), **recall** is "of the real peaks that existed, what fraction did the model find" (missed peaks hurt this), and **F1** is a single number that averages the two, so a model can't score well by only being good at one and ignoring the other.

## Result: no, not reliably

| | mean F1 across 15 trials | std | range |
|---|---|---|---|
| From-scratch | **0.902** | ±0.050 | 0.801 – 0.966 |
| MAE-pretrained | **0.880** | ±0.091 | 0.588 – 0.973 |

Head-to-head on the same 15 windows: **the MAE-pretrained model won 7 times, the from-scratch model won 8 times.** The average difference (MAE − scratch) was **−0.022 F1**, with a spread of ±0.095 — meaning the "difference" is smaller than the noise between trials. That's essentially a coin flip, not a real advantage in either direction.

This is a genuinely different conclusion from the single-split result. That original 30-min/7.6-min split happened to be one where the MAE-pretrained model did unusually well — it wasn't wrong, exactly, but it wasn't representative either. This is exactly the kind of thing repeated trials are for: catching when one split got lucky.

One more thing worth flagging honestly: in trial 3, the MAE-pretrained model had a bad episode — its F1 dropped to 0.588 (recall collapsed to 0.42, meaning it missed well over half the real peaks in that window), while the from-scratch model handled the same window fine (F1 0.943). The from-scratch model didn't have any comparably bad trial. That's a mild instability signal for the MAE-pretrained fine-tuning, not just "roughly equal" — it's one plausible source of MAE's higher score variance (±0.091 vs. ±0.050).

## Caveat on what "15 different segments" really means here

These 15 windows all come from the *same* 37.6-minute Chelten recording, and their training sets overlap heavily (each trial's "training data" is just "everything except this one window," so most of the recording is reused across all 15 trials). This tells you the result is **not sensitive to which 7.5 minutes you happen to hold out from Chelten** — a real and useful thing to know — but it is not the same as testing on a second, independent dog. If another dog gets real hand-labeled SCG clicks in the future, that would be the next meaningful test of whether pretraining generalizes.

## Bottom line

Across 15 different real held-out segments, MAE-pretraining and training from scratch perform statistically indistinguishably on Chelten (and the pretrained model is occasionally less stable). The original headline number (0.854 vs. 0.774) was a real result on that one split, but not a reliable general finding — treat it as "we don't have solid evidence pretraining helps here yet," rather than "pretraining helps."

## Files

- `run_repeated_trials.py` — the trial loop (with checkpoint/resume, since this run had to survive a couple of sandbox restarts partway through).
- `repeated_trials_results.json` — full per-trial results + summary statistics.
- `repeated_trials_comparison.png` — the plot: per-trial F1 bars, box-plot spread, and paired differences.

"""
Frozen constants for the SCG/ECG heart-rate pipeline.

Every value here was tuned/validated against Chelten's hand-click-labeled window
and is treated as FROZEN -- changing any of these numbers changes the pipeline's
behavior and should be done deliberately, with re-validation against
docs/PIPELINE_SUMMARY.md's headline numbers (r=0.9095 SCG, r=0.9820 ECG).

See docs/PIPELINE_SUMMARY.md for the full narrative and validation history behind
each constant.
"""

# ---------------------------------------------------------------------------
# Clock offset between Chelten's two independent recording systems (ecg_polar vs
# scg_mwd), established via hand-click cross-correlation inside RESTRICT_CHELTEN.
# NOT assumed to transfer to other dogs/rigs -- each dog/device pairing has its
# own clock offset (and its own drift rate -- see docs/PIPELINE_SUMMARY.md Section
# 10 for Chelten's measured ~25ppm clock-drift rate on top of this fixed value).
#
# Sign convention:
#   ecg_shift = ecg_pk - LAG_US        # bring an ECG timestamp into the SCG clock frame
#   scg_shift = scg_pk + LAG_US        # bring an SCG timestamp into the ECG clock frame
LAG_US = 7_000_000  # 7.0s, CHELTEN ONLY

# Chelten-only measured clock-drift correction on top of LAG_US (see
# docs/PIPELINE_SUMMARY.md Section 10). lag(t) = LAG_US_INTERCEPT + LAG_DRIFT_US_PER_S *
# t_elapsed_s, where t_elapsed_s is seconds since ecg_polar recording start. Only
# use this for long (>~30min) Chelten sessions; within the original hand-click
# window a fixed LAG_US is fine (drift there is <40ms).
LAG_US_INTERCEPT = 6_948_000  # us, weighted-linear-fit intercept
LAG_DRIFT_US_PER_S = 25.10    # us of extra lag per second of elapsed recording time (~25.1 ppm)

# Fixed SCG-side evaluation window used for the original hand-click-validated
# result (r=0.9095 SCG-pipeline, r=0.9820 Pan-Tompkins ECG vs hand clicks).
RESTRICT_CHELTEN = (1782494790000000, 1782496808699088)  # 2026-06-26 17:26:30 -> 18:00:08.7 UTC

# Per-dog raw-channel conventions.
#
# IMPORTANT -- there are TWO independent conventions in this project and they
# do NOT agree for Chelten:
#   - CORAL's own separate coral_scg.py wrapper uses AXIS below (a derived
#     3-axis magnitude, "mag", for Chelten/Chuck; literal "c2" for Dasty).
#   - THIS repo's custom SCGNet/S2Net pipeline was built and validated (the
#     r=0.9095 headline number) against Chelten's literal "c1" column, NOT
#     "mag" -- verified directly: bandpass(load(Chelten,"c1"))'s max|xf| over
#     the original hand-click-validated window equals CHELTEN_SCG_REF_MAX
#     below to 4 decimal places; "mag" does not (6000.60 vs 4862.70). Use
#     SCG_CHANNEL, not AXIS, when loading data for scg_pipeline/SCGNet.
AXIS = {"Chelten": "mag", "Dasty": "c2", "Chuck": "mag"}         # CORAL's own convention (coral_scg.py) -- reference only
SCG_CHANNEL = {"Chelten": "c1", "Dasty": "c2"}                    # THIS pipeline's validated convention (Chuck: unvalidated, no prior work)
LEAD = {"ecg_biopac": "c4", "ecg_polar": "c1"}             # ecg lead column per stream

# CRITICAL: exact max|bandpassed signal| of Chelten's original hand-click-validated
# 34-minute window (17:26:30-18:00:08.7 UTC) -- the amplitude scale the frozen
# SCGNet/S2Net were actually calibrated against. Pass this as shannon_envelope_
# decimated()'s ref_max whenever processing ANY Chelten scg_mwd span longer than
# a few minutes (and always when processing the full session in one shot).
# Self-normalizing (ref_max=None) over a long/full-session span lets rare large
# motion/handling spikes elsewhere in the recording redefine the normalization
# scale and silently shrink the envelope the CNN sees -- this was a real bug,
# found and fixed during the original investigation (see docs/PIPELINE_SUMMARY.md
# Section 5, "global-normalization contamination"). Self-normalization is fine
# for a short window (a few minutes) that doesn't contain an outlier spike, and
# is what's used for OTHER dogs (Dasty) since there's no calibrated reference to
# stay consistent with there.
CHELTEN_SCG_REF_MAX = 4862.704192475711

# ---------------------------------------------------------------------------
# Stage 1: envelope + primary candidate generation
FSD = 200.0                 # decimated envelope sample rate (Hz)
PAD_S = 30.0                # padding (s) for filter settling on windowed loads
PRIMARY_THR = 0.3           # sharpen_local() threshold for primary candidate peaks
PRIMARY_REFRACT_S = 0.50    # caps detectable HR at 120bpm -- loosen for fast-HR dogs (see Dasty findings)

# ---------------------------------------------------------------------------
# Stage 2/3: CNN + S2-recovery chain
CUTOFF_MAIN = 0.25          # SCGNet confirm threshold
S2_CUTOFF = 0.70             # S2Net flag threshold
SEARCH_LO_US, SEARCH_HI_US = 100_000, 300_000   # S2 backward-search window for the true S1
MERGE_TOL_US = 50_000
HALF_N = 80                  # snippet half-width: env_sd[i-80 : i+81]

# ---------------------------------------------------------------------------
# Stage 4: gap-aware honest reporting
GAP_THRESH_S = 1.5   # a beat-to-beat silence this long = "detection failure", not signal
WIN_S = 3.0           # HR smoothing window

# ---------------------------------------------------------------------------
# Fully-tuned Pan-Tompkins ECG detector (grid-searched, ~8000 combos, picked from
# the middle of a broad r~0.98 plateau -- see docs/PIPELINE_SUMMARY.md Section 2)
PT_FC_LO, PT_FC_HI = 8.0, 45.0
PT_REFRACT_S = 0.40
PT_MWI_S = 0.010
PT_FLOOR_S = 2.0
PT_SNR_THR = 100.0

# ---------------------------------------------------------------------------
# CORAL's own quality-gating (reused as-is when comparing against a CORAL .csv output)
CORAL_SQI_THR = 0.10
CORAL_ATTRACTOR_BAND = (95, 125)
CORAL_BAND_THR = 0.50

import pickle, sqlite3
import numpy as np

# ---------- Relabel the candidate dataset using ECG as a second corroborating signal.
# Original rule: match candidate to nearest SCG hand click within 150ms -> positive, else negative.
# Problem (discovered this session): SCG hand clicks are frequently ABSENT during stretches you
# labeled "bad", even when ECG clearly shows a real heartbeat there -- so many "negatives" are
# actually mislabeled true beats.
#
# New rule (uses the CORRECTED ecg_shift = ecg_pk - 7.000s alignment, verified this session):
#   1. matched to an SCG click within 150ms -> POSITIVE (unchanged, most reliable evidence)
#   2. not matched to SCG, but matched to an ECG click within a TIGHT 50ms window -> POSITIVE
#      ("ECG-rescued": known real beats match ECG within 50ms 96% of the time, vs only 10% by
#      chance, so this is a high-precision rescue, not a guess)
#   3. not matched to SCG, and nearest ECG click is >250ms away -> NEGATIVE (clean, confident negative)
#   4. everything else (not SCG-matched, nearest ECG between 50-250ms) -> AMBIGUOUS, dropped from
#      the training set entirely rather than guessing (at that distance chance-level ECG proximity
#      is too high to trust either way) ----------

with open("/tmp/cnn_dataset.pkl","rb") as f:
    D = pickle.load(f)
t = D["t"]; label_old = D["label"]; snips = D["snips"]; HALF_N = D["HALF_N"]; fsd_s = D["fsd_s"]

def load_peaks(path):
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = ''")
    return np.array([r[0] for r in cur.fetchall()], dtype="int64")
ecg_pk = load_peaks("/tmp/annotation_chelten_ecg2.db")
LAG_US = 7_000_000
ecg_shift = ecg_pk - LAG_US   # CORRECTED sign, verified this session

RESCUE_US = 50_000
CLEAN_NEG_US = 250_000

d_ecg = np.array([np.min(np.abs(ecg_shift-tc)) for tc in t])

label_new = np.full(len(t), -1, dtype=int)   # -1 = drop (ambiguous)
n_rescued = 0; n_clean_neg = 0; n_kept_pos = 0; n_dropped = 0
for i in range(len(t)):
    if label_old[i] == 1:
        label_new[i] = 1; n_kept_pos += 1
    elif d_ecg[i] <= RESCUE_US:
        label_new[i] = 1; n_rescued += 1
    elif d_ecg[i] > CLEAN_NEG_US:
        label_new[i] = 0; n_clean_neg += 1
    else:
        n_dropped += 1

keep = label_new >= 0
t2, label2, snips2 = t[keep], label_new[keep], snips[keep]
print(f"original: {int(label_old.sum())} positive / {int((1-label_old).sum())} negative  (n={len(label_old)})")
print(f"kept as positive (SCG-matched): {n_kept_pos}")
print(f"ECG-rescued (was negative, now positive): {n_rescued}")
print(f"clean negative (unmatched, ECG also >250ms away): {n_clean_neg}")
print(f"dropped as ambiguous (unmatched, ECG 50-250ms away): {n_dropped}")
print(f"final relabeled dataset: n={len(t2)}  positive={int(label2.sum())}  negative={int((1-label2).sum())}")

with open("/tmp/cnn_dataset_relabel.pkl","wb") as f:
    pickle.dump(dict(t=t2, label=label2.astype(np.float32), snips=snips2, HALF_N=HALF_N, fsd_s=fsd_s,
                      n_rescued=n_rescued, n_clean_neg=n_clean_neg, n_dropped=n_dropped), f)
print("saved /tmp/cnn_dataset_relabel.pkl")

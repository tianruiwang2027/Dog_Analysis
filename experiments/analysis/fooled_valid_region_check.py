import pickle, sqlite3, datetime
import numpy as np

with open("/tmp/cnn_dataset.pkl","rb") as f:
    D = pickle.load(f)
t = D["t"]
with open("/tmp/cnn_fooling_negs.pkl","rb") as f:
    FN = pickle.load(f)
fool = FN["fool"]; prob_all = FN["prob_all_orig"]
fool_idx = np.where(fool)[0]
order_f = np.argsort(-prob_all[fool_idx])
picks_f = fool_idx[order_f[:: max(1, len(order_f)//10)][:10]]

with open("/tmp/valid_regions.pkl","rb") as f:
    VR = pickle.load(f)
valid_ivs = VR["valid_ivs"]

def load_peaks_and_good(path):
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = ''")
    pk = np.array([r[0] for r in cur.fetchall()], dtype="int64")
    cur.execute("SELECT DISTINCT timestamp, label FROM events WHERE label != ''")
    rows = cur.fetchall()
    ev_ts = np.array([r[0] for r in rows], dtype="int64"); ev_lab=[r[1].lower() for r in rows]
    return pk, ev_ts, ev_lab
def good_intervals(ev_ts, ev_lab, span_end):
    ivs=[]; state="bad"; cur_start=None
    for tt,l in zip(ev_ts, ev_lab):
        if "good" in l and "start" in l:
            if state!="good": cur_start=tt; state="good"
        elif "bad" in l and "start" in l:
            if state=="good": ivs.append((cur_start,tt)); state="bad"
        elif "bad" in l and "end" in l:
            if state!="good": cur_start=tt; state="good"
        elif "good" in l and "end" in l:
            if state=="good": ivs.append((cur_start,tt)); state="bad"
        elif "stop" in l:
            if state=="good": ivs.append((cur_start,tt)); state="bad"
    if state=="good": ivs.append((cur_start, span_end))
    return ivs
def in_ivs(tc, ivs):
    return any(a<=tc<b for a,b in ivs)

scg_pk, ev_ts, ev_lab = load_peaks_and_good("/tmp/annotation_chelten_scg2.db")
span_end = max(scg_pk.max(), t.max())
scg_ivs = good_intervals(ev_ts, ev_lab, span_end)
ecg_pk = load_peaks_and_good("/tmp/annotation_chelten_ecg2.db")[0]

with open("/tmp/final_4way_compare.pkl","rb") as f:
    Dc = pickle.load(f)
LAG_US = Dc.get("LAG_US", 7_000_000)
ecg_shift = ecg_pk + LAG_US

print(f"{'time':>16s} {'score':>6s} {'in valid_ivs?':>14s} {'in hand-good SCG?':>18s} {'SCG click dist':>15s} {'ECG click dist':>15s}  verdict")
for idx in picks_f:
    tc = t[idx]
    is_valid = in_ivs(tc, valid_ivs)
    is_scg_good = in_ivs(tc, scg_ivs)
    d_scg = np.min(np.abs(scg_pk-tc))/1e3
    d_ecg = np.min(np.abs(ecg_shift-tc))/1e3
    tstr = datetime.datetime.utcfromtimestamp(tc/1e6).strftime("%H:%M:%S.%f")[:-3]
    if is_valid:
        verdict = f"VALID region -> check BOTH: no SCG click within {d_scg:.0f}ms, ECG click {d_ecg:.0f}ms away"
    else:
        verdict = f"NOT valid (bad/excluded) -> ECG-only: nearest ECG click {d_ecg:.0f}ms away"
    print(f"{tstr:>16s} {prob_all[idx]:6.2f} {str(is_valid):>14s} {str(is_scg_good):>18s} {d_scg:13.0f}ms {d_ecg:13.0f}ms  {verdict}")

import pickle, datetime, sqlite3
import numpy as np

with open("/tmp/final_4way_compare.pkl","rb") as f:
    D = pickle.load(f)

LAG_US = D["LAG_US"]
UTC = datetime.timezone.utc
def fmt(us):
    return datetime.datetime.fromtimestamp(us/1e6, tz=UTC).strftime("%H:%M:%S.%f")[:-3]

# reload raw hand click lists (not just smoothed HR) for both channels
def load_peaks(path):
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = '' ORDER BY timestamp")
    return np.array([r[0] for r in cur.fetchall()], dtype="int64")

ecg_pk = load_peaks("/tmp/annotation_chelten_ecg2.db")
scg_pk = load_peaks("/tmp/annotation_chelten_scg2.db")

def window(pk, lo, hi):
    return pk[(pk>=lo)&(pk<=hi)]

# cluster around 17:43:00 - 17:43:15 (SCG time axis); ECG needs -LAG_US shift to compare on SCG axis... 
# actually let's just print both on their own native clocks, and also ECG shifted onto SCG's clock (t_ecg_on_scg = t_ecg - LAG_US)
lo = int(datetime.datetime(2026,6,26,17,42,58,tzinfo=UTC).timestamp()*1e6)
hi = int(datetime.datetime(2026,6,26,17,43,16,tzinfo=UTC).timestamp()*1e6)

scg_w = window(scg_pk, lo, hi)
ecg_w_native = ecg_pk[(ecg_pk-LAG_US>=lo)&(ecg_pk-LAG_US<=hi)]

print("SCG hand clicks (native clock) in window:")
prev=None
for t in scg_w:
    rr = f" RR={((t-prev)/1e3):.0f}ms" if prev is not None else ""
    print(f"  {fmt(t)}{rr}")
    prev=t

print("\nECG hand clicks (SHIFTED onto SCG's clock, t_ecg-7.0s) in same window:")
prev=None
for t in ecg_w_native:
    ts = t-LAG_US
    rr = f" RR={((ts-prev)/1e3):.0f}ms" if prev is not None else ""
    print(f"  {fmt(ts)}{rr}")
    prev=ts

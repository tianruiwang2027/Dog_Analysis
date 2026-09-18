import pickle, datetime, sqlite3
import numpy as np

with open("/tmp/final_4way_compare.pkl","rb") as f:
    D = pickle.load(f)

LAG_US = D["LAG_US"]
xa_scg, yb_scg, ta_scg = D["xa_scg"], D["yb_scg"], D["ta_scg"]
err = xa_scg - yb_scg

def load_peaks(path):
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("SELECT DISTINCT timestamp FROM events WHERE label = '' ORDER BY timestamp")
    return np.array([r[0] for r in cur.fetchall()], dtype="int64")

ecg_pk = load_peaks("/tmp/annotation_chelten_ecg2.db")
scg_pk = load_peaks("/tmp/annotation_chelten_scg2.db")
ecg_on_scg = ecg_pk - LAG_US   # ECG clicks expressed on SCG's clock

UTC = datetime.timezone.utc
def fmt(us):
    return datetime.datetime.fromtimestamp(us/1e6, tz=UTC).strftime("%H:%M:%S.%f")[:-3]

idx = np.argsort(-np.abs(err))[:25]

WIN = int(2.5e6)
print(f"{'time':>14} {'err':>7} {'n_SCG_clicks':>13} {'n_ECG_clicks':>13}  {'diff':>5}")
for i in idx:
    t = ta_scg[i]
    n_scg = ((scg_pk>=t-WIN)&(scg_pk<=t+WIN)).sum()
    n_ecg = ((ecg_on_scg>=t-WIN)&(ecg_on_scg<=t+WIN)).sum()
    print(f"{fmt(t):>14} {err[i]:+7.1f} {n_scg:13d} {n_ecg:13d}  {n_scg-n_ecg:+5d}")

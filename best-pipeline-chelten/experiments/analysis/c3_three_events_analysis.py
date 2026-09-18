import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/tmp/axes_zoom_data.pkl", "rb") as f:
    d = pickle.load(f)

ts_w = d["ts_w"]  # microseconds int64 timestamps
xf = d["xf"]      # dict of bandpassed arrays
raw = d["raw"]
fs = d["fs"]
print("fs=", fs)

t_s = (ts_w - ts_w[0]) / 1e6  # seconds relative to window start
t0_abs = ts_w[0]

# restrict window like before: 17:56:03 to 17:56:06.3
import datetime
def to_us(hhmmss_frac, base_date):
    pass

# We know ts_w spans window w0..w1 from compare_axes_zoom (17:56:00 or with pad). Let's just find abs offsets using raw ts.
# Find index range for 17:56:03.0 - 17:56:06.3 using ts_w absolute epoch microseconds.
# ts_w likely already in raw epoch us. Let's check ts_w[0], ts_w[-1] as datetime.
t_start_dt = datetime.datetime.utcfromtimestamp(ts_w[0]/1e6)
t_end_dt = datetime.datetime.utcfromtimestamp(ts_w[-1]/1e6)
print("window abs:", t_start_dt, t_end_dt)

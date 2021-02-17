import detchar
from detchar import IVCurveTaker, IVPointTaker
import pylab as plt
import numpy as np
import time


plt.ion()
plt.close("all")
curve_taker = IVCurveTaker(
    IVPointTaker("DB1", "BY", column_number=7),
    temp_settle_delay_s=0,
    shock_normal_dac_value=40000,
)
curve_taker.prep_fb_settings(I=16, fba_offset=8192, ARLoff=False)
curve_taker.set_temp_and_settle(0.075, sleep_time_s=40)
t = []
fbs = []
dac_lo = 0
dac_hi = 50
while True:
    try:
        print(f"n={len(t)}")
        t.append(curve_taker.adr_gui_control.get_temp_k())
        print(f"t={t[-1]*1e3:.2f} mK")
        fb0 = curve_taker.pt.get_iv_pt(dac_lo)
        print(f"fb0={fb0}")
        fb1 = curve_taker.pt.get_iv_pt(dac_hi)
        print(f"fb1={fb1}")
        fbs.append(fb1 - fb0)
        plt.clf()
        plt.plot(np.array(t) * 1e3, fbs)
        plt.ylabel(f"fb change when DB dac from {dac_lo} to {dac_hi}")
        plt.xlabel("temp (mK)")
        plt.title(
            f"tc tickle for bay {curve_taker.pt.bayname}, col {curve_taker.pt.col}"
        )
        plt.pause(0.1)
        curve_taker.set_temp_and_settle(0.074 - len(t) * 1e-3 / 50)

    except KeyboardInterrupt:
        break
fbs_out = np.vstack(fbs)

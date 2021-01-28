
import pylab as plt
import numpy as np
import detchar

plt.ion()
plt.close("all")
curve_taker = detchar.IVCurveTaker(detchar.IVPointTaker("DB", "AX"), temp_settle_delay_s=0, shock_normal_dac_value=100)
curve_taker.set_temp_and_settle(setpoint_k=0.190)
curve_taker.prep_fb_settings(I=10, fba_offset=8000)
dacs = np.linspace(7000,0,50)
data = curve_taker.get_curve(dacs, extra_info = {"magnetic field current (amps)": 1e-6})
data.plot()
data.to_file("ivtest.json", overwrite=True)
data2 = detchar.IVCurveColumnData.from_json(data.to_json())
assert data2.pre_time_epoch_s == data.pre_time_epoch_s
data = detchar.IVCurveColumnData.from_file("ivtest.json")
x, y = data.xy_arrays_zero_subtracted()
r = iv_to_frac_rn_array(x, y, superconducting_below_x=2000, normal_above_x=5000)
plt.figure()
plt.plot(x, r)
plt.xlabel("dac value")
plt.ylabel("fraction R_n")
plt.legend([f"row{i}" for i in range(r.shape[1])])
plt.vlines(2750, 0, 1)
plt.grid(True)


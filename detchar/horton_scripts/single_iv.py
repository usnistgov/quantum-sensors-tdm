import pylab as plt
import numpy as np
import detchar

plt.ion()
# plt.close("all")
curve_taker = detchar.IVCurveTaker(
    detchar.IVPointTaker("DB1", "BX", column_number=6),
    temp_settle_time_out_s=180,
    temp_settle_tolerance_k=0.15 * 1e-3,
    shock_normal_dac_value=40000,
)
curve_taker.set_temp_and_settle(setpoint_k=61 * 1e-3)
curve_taker.prep_fb_settings(I=10, fba_offset=8000, ARLoff=True)
# curve_taker.prep_fb_settings(I=60, fba_offset=3500, ARLoff=True)
dacs = detchar.sparse_then_fine_dacs(a=10000, b=2500, c=0, n_ab=30, n_bc=200)
# ZERO_V = 26214
# dacs = detchar.sparse_then_fine_dacs(a=0, b=4000, c=6000, n_ab=50, n_bc=300)[::-1]
# dacs2 = np.array(list(dacs) + list(-dacs[::-1])) + ZERO_V
# dacs3 = np.array(list(dacs2) + list(dacs2[::-1]))
data = curve_taker.get_curve(dacs, extra_info={"field coil (amps)": 0})
data.plot()
data.to_file("latest_single_iv.json", overwrite=True)

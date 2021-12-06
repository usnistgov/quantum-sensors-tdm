from detchar.iv_utils import AdrGuiControlDummy
import pylab as plt
import numpy as np
import detchar



plt.ion()
# plt.close("all")
curve_taker = detchar.IVCurveTaker(
    detchar.IVPointTaker("DB1", "CX", column_number=0),
    temp_settle_time_out_s=120,
    temp_settle_tolerance_k=0.15 * 1e-3,
    shock_normal_dac_value=40000,
    adr_gui_control=detchar.iv_utils.AdrGuiControlDummy()
)
#hacks to take IV with lakeshore off
# curve_taker.adr_gui_control.get_temp_k = lambda: 0
# curve_taker.adr_gui_control.get_hout = lambda: 0
# curve_taker.adr_gui_control.get_slope_hout_per_hour = lambda: 0
# curve_taker.adr_gui_control.get_temp_rms_uk = lambda: 0
# curve_taker.set_temp_and_settle(setpoint_k=75 * 1e-3)
curve_taker.prep_fb_settings(I=1, fba_offset=8000, ARLoff=True)
# curve_taker.prep_fb_settings(I=60, fba_offset=3500, ARLoff=True)
dacs = detchar.sparse_then_fine_dacs(a=14000, b=6000, c=0, n_ab=40, n_bc=250)
# ZERO_V = 26214
# dacs = detchar.sparse_then_fine_dacs(a=0, b=4000, c=6000, n_ab=50, n_bc=300)[::-1]
# dacs2 = np.array(list(dacs) + list(-dacs[::-1])) + ZERO_V
# dacs3 = np.array(list(dacs2) + list(dacs2[::-1]))
data = curve_taker.get_curve(dacs, extra_info={"field coil (amps)": 0})
data.plot()
data.to_file("latest_single_iv.json", overwrite=True)

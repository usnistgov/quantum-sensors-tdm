import pylab as plt
import numpy as np
import detchar

plt.ion()
# plt.close("all")
curve_taker = detchar.IVCurveTaker(
    detchar.IVPointTaker("DB1", "BX", column_number=0, voltage_source="bluebox"),
    temp_settle_time_out_s=180,
    temp_settle_tolerance_k=0.05 * 1e-3,
    shock_normal_dac_value=100,
)
curve_taker.set_temp_and_settle(setpoint_k=70.2 * 1e-3)
curve_taker.prep_fb_settings(I=10, fba_offset=8000)
curve_taker.prep_fb_settings(I=60, fba_offset=3500, ARLoff=True)
# dacs = (
#     detchar.sparse_then_fine_dacs(a=25000, b=1500, c=0, n_ab=30, n_bc=50) * 2.5 / 6.5535
# )
# dacs = detchar.sparse_then_fine_dacs(a=0, b=4000, c=6000, n_ab=50, n_bc=300)[::-1]
# dacs2 = np.array(list(dacs) + list(-dacs[::-1])) + ZERO_V
# dacs3 = np.array(list(dacs2) + list(dacs2[::-1]))
dacs = (
    detchar.sparse_then_fine_dacs(a=0, b=900, c=4000, n_ab=50, n_bc=150) * 2.5 / 6.5535
)
dacs = np.linspace(0, 6000, 600)
data = curve_taker.get_curve(dacs, extra_info={"field coil (amps)": 0})
data.plot()
data.to_file("latest_single_iv.json", overwrite=True)


fb_values = data.fb_values_array()
flip_fb = np.max(fb_values) - fb_values
diffs = np.diff(fb_values, axis=0)
sign = np.median(np.sign(diffs[0, :]))
inds, rows = np.nonzero(np.diff(np.sign(diffs), axis=0) == -2 * sign)
inds += 1

plt.plot(dacs[inds], data.fb_values_array()[inds, rows], "o")
ic_dacs = np.zeros(fb_values.shape[1])
for ind, row in zip(inds, rows):
    ic_dacs[row] = dacs[ind]

import pylab as plt
import numpy as np
import detchar

plt.ion()
plt.close("all")
curve_taker = detchar.IVCurveTaker(detchar.IVPointTaker("DB", "AX"), temp_settle_delay_s=0, shock_normal_dac_value=100)
curve_taker.set_temp_and_settle(setpoint_k=0.21)
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



    # plt.ion()
    # plt.close("all")
    # curve_taker = IVCurveTaker(IVPointTaker("DB1", "BX"), temp_settle_delay_s=0, shock_normal_dac_value=40000)
    # curve_taker.set_temp_and_settle(setpoint_k=0.075)
    # curve_taker.pt.prep_fb_settings(I=10, fba_offset=3000)
    # dacs = sparse_then_fine_dacs(a=40000, b = 4000, c=0, n_ab=40, n_bc=250)
    # data = curve_taker.get_curve(dacs, extra_info = {"magnetic field current (amps)": 1e-6})
    # data.plot()
    # data.to_file("ivtest.json", overwrite=True)
    # data2 = IVCurveColumnData.from_json(data.to_json())
    # assert data2.pre_time_epoch_s == data.pre_time_epoch_s
    # data = IVCurveColumnData.from_file("ivtest.json")
    # x, y = data.xy_arrays_zero_subtracted()
    # r = iv_to_frac_rn_array(x, y, superconducting_below_x=2000, normal_above_x=24000)
    # plt.figure()
    # plt.plot(x, r)
    # plt.xlabel("dac value")
    # plt.ylabel("fraction R_n")
    # plt.legend([f"row{i}" for i in range(r.shape[1])])
    # plt.vlines(2750, 0, 1)
    # plt.grid(True)



    # t, y = tc_tickle()
    # y = np.vstack(fbs)
    # plt.clf()
    # plt.plot(np.array(t)*1e3, y)
    # plt.xlabel("temp (mK)")
    # plt.ylabel("delta fb from 50 dac unit tickle")
    # plt.pause(0.1)

    plt.ion()
    plt.close("all")
    curve_taker = IVCurveTaker(IVPointTaker("DB1", "BX"), temp_settle_delay_s=180, shock_normal_dac_value=40000)
    curve_taker.prep_fb_settings()
    temp_sweeper = IVTempSweeper(curve_taker)
    dacs = sparse_then_fine_dacs(a=40000, b = 10000, c=0, n_ab=20, n_bc=100)
    temps_mk = np.linspace(60,100,16)
    print(f"{temps_mk} mK")
    sweep = temp_sweeper.get_sweep(dacs, 
        set_temps_k=temps_mk*1e-3, 
        extra_info={"field coil current (Amps)":0})
    sweep.to_file("iv_sweep_test2.json", overwrite=True)
    sweep2 = IVTempSweepData.from_file("iv_sweep_test2.json")
    sweep2.plot_row(row=0)
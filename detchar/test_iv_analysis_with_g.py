from . import test_data
from detchar import IVCircuit, IVCurveColumnData, IVTempSweepData
import pylab as plt
import os, time
import lmfit

import numpy as np

def pause_ci_safe(t):
    v = os.environ.get("CI", False)
    if not v:
        tstart = time.time()
        while time.time()-tstart < t:
            # small pauses until total duration has elapsed
            # use plt.pause to allow plots to appear
            plt.pause(5)
            any_figs_open = len(plt.get_fignums()) > 0
            if not any_figs_open:
                break


def test_g_from_temp_sweep():
    sdata = IVTempSweepData.from_file(test_data.horton_temp_sweep_with_zero_tracking_filename)


    circuit = IVCircuit(rfb_ohm = 4e3, 
    rbias_ohm = 1e3, # need to check notes for true value
    rsh_ohm = 200e-6,
    rpar_ohm = 0,
    m_ratio = 3.46, # from inspecting the mux19 test pdfs, Mr = Min/Mfb = 265.7 pH/76.6 pH = 3.46
    vfb_gain = 1.0/2**14,
    vbias_gain = 2.5/2**16)
    
    # at lowest temp point, loop over rows to learn r_parasitic
    rpar_ohm_by_row = sdata.fit_for_rpar(circuit, 
    sc_below_vbias_arb=500, #db value in dac units below which he device is superconducting
    temp_index=0) # do fits at the lowest temp

    print(f"{1e6*rpar_ohm_by_row=}")
    print(f"{sdata.data[-1]=}")

    for row in range(sdata.get_nrows()//16):
        sdata.plot_row(row=row)

    for temp_index in range(len(sdata.set_temps_k)//16):
        sdata.plot_temp(temp_index)
    # plt.figure()
    # plt.plot(vbias_arbs, fb_arbs)
    for row in range(sdata.get_nrows()//16):
        sdata.plot_row_iv(row=row, circuit=circuit, rpar_ohm_by_row=rpar_ohm_by_row)
        sdata.plot_row_iv(row=row, circuit=circuit, rpar_ohm_by_row=rpar_ohm_by_row, y_quantity="resistance")
        sdata.plot_row_iv(row=row, circuit=circuit, rpar_ohm_by_row=rpar_ohm_by_row, y_quantity="power")


    tb_k = np.array([45, 50, 55, 60, 65])*1e-3
    p_w = np.array([4.5, 3.9, 3.1, 2.2, 1.1])*1e-13
    def p_model(tb_k, tc_k, gigak, n):
        return 1e-9*gigak*(tc_k**n-tb_k**n)
    model = lmfit.model.Model(p_model)
    params = model.make_params()
    params["tc_k"].set(70e-3, vary=False)
    params["gigak"].set(1, min=1e-18)
    params["n"].set(3, min=0.5)
    result = model.fit(data=p_w, tb_k=tb_k, params=params)
    p_model_out = result.eval()
    print(result.fit_report())
    result.params.pretty_print()
    k = result.params["gigak"].value*1e-9
    n = result.params["n"].value
    tc_k = result.params["tc_k"].value
    G_W_per_K = n*k*tc_k**(n-1)
    plt.figure()
    plt.plot(tb_k, p_w, "o")
    plt.plot(tb_k, p_model_out, label=f"tc={tc_k:.2f} K, k={k:.2g} W/K^n, n={n:.2f}, G={G_W_per_K*1e12:.2f} pW/K")
    plt.xlabel("temp (K)")
    plt.ylabel("power (W)")
    plt.legend()

    pause_ci_safe(60)

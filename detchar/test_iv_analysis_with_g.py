from . import test_data
from detchar import IVCircuit, IVCurveColumnData, IVTempSweepData
import detchar
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

    powers = sdata.get_power_at_r(r_ohm=0.005, row=2, circuit=circuit, rpar_ohm_by_row=rpar_ohm_by_row, 
    sc_below_vbias_arb=None, plot=True)
    result, k, tc_k, n, G_W_per_k = detchar.g_fit(sdata.set_temps_k, powers, plot=True)
  

    pause_ci_safe(60)

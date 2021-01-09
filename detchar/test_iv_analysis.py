from . import test_data
from detchar import IVCircuit, IVCurveColumnData, IVTempSweepData
import pylab as plt
import os, time

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


def test_horton_column_data():
    data = IVCurveColumnData.from_file(test_data.horton_column_data_filename)
    sdata = IVTempSweepData.from_file(test_data.horton_temp_sweep_data_filename)


    circuit = IVCircuit(rfb_ohm = 4e3, 
    rbias_ohm = 1e3, # need to check notes for true value
    rsh_ohm = 200e-6,
    m_ratio = -8.3, # copied from Hannes,need accurate value from Malcolm
    vfb_gain = 1.0/2**14,
    vbias_gain = 2.5/2**16)

    itess = []
    vtess = []
    for d in sdata.data:
        vbias_arbs, fb_arbs = d.xy_arrays_zero_subtracted_at_origin()
        sc_below_vbias_arb = 500
        normal_above_fb = 20000
        vbias_arbs, fb_arbs = d.xy_arrays_zero_subtracted_at_normal_y_intersect(normal_above_fb)


        ites, vtes, rpar_ohm = circuit.iv_raw_to_physical_fit_rpar(vbias_arbs, 
            fb_arbs[:, 0], sc_below_vbias_arb)
        ites, vtes = circuit.iv_raw_to_physical(vbias_arbs, 
            fb_arbs[:, 0], rpar_ohm=0.0)
        print(f"{rpar_ohm=}")
        # itess.append(fb_arbs[:, 1])
        # vtess.append(vbias_arbs)
        itess.append(ites)
        vtess.append(vtes)


    plt.figure()
    for vt, it, d in zip(vtess, itess, sdata.data):
        plt.plot(vt, it, label=f"{d.nominal_temp_k*1e3:.2f} mK")
    plt.xlabel("V tes / V")
    plt.ylabel("I tes / A")
    plt.legend()
    plt.tight_layout()

    sdata.plot_row(0, zero="origin")
    sdata.plot_row(0, zero="fit normal")

    pause_ci_safe(60)
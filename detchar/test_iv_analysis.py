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

    itess_by_temp = []
    vtess_by_temp = []
    dlast = sdata.data[-1]
    vbias_last, fb_arbs_last = dlast.xy_arrays()
    
    nrows = sdata.data[0].xy_arrays()[1].shape[1]

    # at lowest temp point, loop over rows to learn r_parasitic
    d = sdata.data[0]
    rpar_by_row = []
    for row in range(nrows):
        sc_below_vbias_arb = 500
        vbias_arbs, fb_arbs = d.xy_arrays()
        try:
            ites, vtes, rpar_ohm = circuit.iv_raw_to_physical_fit_rpar(vbias_arbs, 
                    fb_arbs[:, row], sc_below_vbias_arb)
        except np.linalg.LinAlgError:
            rpar_ohm = np.nan
        rpar_by_row.append(rpar_ohm)

    for d in sdata.data:
        # vbias_arbs, fb_arbs = d.xy_arrays_zero_subtracted_at_origin()
        normal_above_fb = 20000
        vbias_arbs, fb_arbs = d.xy_arrays()
        
        # we assume the highest temperature IV (100 mK) has kept lock
        # and we assume that at the highest bias, all IVs shoudl have roughly
        # the same TES current
        # so we make all IVs agree with the highest temp IV at the highest bias
        # and make the highest temp IV go thru 0,0
        fb_arbs[:,:] -= fb_arbs[0,:]-fb_arbs_last[0,:]+fb_arbs_last[-1,:]

        itess = np.zeros_like(fb_arbs, dtype="float64")
        vtess = np.zeros_like(fb_arbs, dtype="float64")

        for row in range(nrows):
            ites, vtes = circuit.iv_raw_to_physical(vbias_arbs, 
                fb_arbs[:, row], rpar_ohm=rpar_by_row[row])
            itess[:, row] = ites
            vtess[:, row] = vtes
        ites[ites<0] = np.nan
        vtess_by_temp.append(vtess)
        itess_by_temp.append(itess)




    # plot a row vs temperature
    plt.figure()
    ntemps = len(sdata.data)
    row = 0
    for q in range(ntemps):
        x = vtess_by_temp[q][:, row]
        y = vtess_by_temp[q][:, row]/itess_by_temp[q][:, row]
        d = sdata.data[q]
        temp_mk = (d.pre_temp_k + d.post_temp_k)*1e3/2
        plt.plot(x, y, label=f"{temp_mk:.2f} mK")
    plt.xlabel("V tes / V")
    plt.ylabel("R tes / Ohm")
    plt.title(f"row = {row} r vs v_tes vs temp")
    plt.legend()
    plt.tight_layout()
    ymax = y[0]
    plt.ylim(0, ymax) # set the ylim to the maximum r value


    # plot all rows at one temp with physical units
    plt.figure()
    q = 0
    for row in range(5):
        x = vtess_by_temp[q][:, row]
        y = vtess_by_temp[q][:, row]/itess_by_temp[q][:, row]
        y[np.logical_or(y<0, y>ymax)] = np.nan
        plt.plot(x, y, label=f"row {row}")
    d = sdata.data[q]
    temp_mk = (d.pre_temp_k + d.post_temp_k)*1e3/2
    plt.xlabel("V tes / Volts")
    plt.ylabel("R tes / Ohm")
    plt.title(f"all rows at T={temp_mk:.2f} mK")
    # plt.legend()
    plt.tight_layout()
    plt.ylim(0, ymax) # set the ylim to the maximum r value    
    plt.grid(True, which="both")

    # plot all rows at one temp with experimental units
    plt.figure()
    q = 0
    for row in range(5):
        # x = vtess_by_temp[q][:, row]
        x = vbias_arbs
        y = vtess_by_temp[q][:, row]/itess_by_temp[q][:, row]
        y[np.logical_or(y<0, y>ymax)] = np.nan
        y=y/ymax
        plt.plot(x, y, label=f"row {row}")
    d = sdata.data[q]
    temp_mk = (d.pre_temp_k + d.post_temp_k)*1e3/2
    plt.xlabel("vbias dac values")
    plt.ylabel("R tes / Rmax")
    plt.title(f"all rows at T={temp_mk:.2f} mK")
    # plt.legend()
    plt.tight_layout()
    # plt.ylim(0, ymax) # set the ylim to the maximum r value    
    plt.grid(True, which="both")


    pause_ci_safe(60)
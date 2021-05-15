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


def test_g_from_temp_sweep():
    sdata = IVTempSweepData.from_file(test_data.horton_temp_sweep_with_zero_tracking_filename)


    circuit = IVCircuit(rfb_ohm = 4e3, 
    rbias_ohm = 1e3, # need to check notes for true value
    rsh_ohm = 200e-6,
    rpar_ohm = 0,
    m_ratio = 3.46, # from inspecting the mux19 test pdfs, Mr = Min/Mfb = 265.7 pH/76.6 pH = 3.46
    vfb_gain = 1.0/2**14,
    vbias_gain = 2.5/2**16)
    
    nrows = sdata.get_nrows()

    # at lowest temp point, loop over rows to learn r_parasitic
    rpar_ohm_by_row = sdata.fit_for_rpar(circuit, 
    sc_below_vbias_arb=500, #db value in dac units below which he device is superconducting
    temp_index=0) # do fits at the lowest temp

    print(f"{1e6*rpar_ohm_by_row=}")
    print(f"{sdata.data[-1]=}")
    # zero subtraction
    # we assume the highest temperature IV has kept lock
    # and use the zero db value taken after relocking after each IV to correct each other IV
    vbias_arbs, fb_arbs_last = sdata.data[-1].xyarrays_zero_subtracted_with_post_iv_zero_fb_value()
    vtess_by_temp = []
    itess_by_temp = []
    assert vbias_arbs[-1] == 0
    for d in sdata.data[:-3]:
        vbias_arbs, fb_arbs = d.xyarrays_zero_subtracted_with_post_iv_zero_fb_value()
        # we assume the highest temperature IV (100 mK) has kept lock
        # so we subtract its last point (db = 0)
        fb_arbs[:,:] -= fb_arbs_last[-1,:]
        itess = np.zeros_like(fb_arbs, dtype="float64")
        vtess = np.zeros_like(fb_arbs, dtype="float64")

        for row in range(nrows):
            ites, vtes = circuit.iv_raw_to_physical(vbias_arbs, 
                fb_arbs[:, row], rpar_ohm=rpar_ohm_by_row[row])
            itess[:, row] = ites
            vtess[:, row] = vtes
        ites[ites<0] = np.nan
        vtess_by_temp.append(vtess)
        itess_by_temp.append(itess)


    for row in range(sdata.get_nrows()//4):
        sdata.plot_row(row=row)

    for temp_index in range(len(sdata.set_temps_k)//4):
        sdata.plot_temp(temp_index)
    # plt.figure()
    # plt.plot(vbias_arbs, fb_arbs)
    for row in range(sdata.get_nrows()//4):
        sdata.plot_row_iv(row=row, circuit=circuit, rpar_ohm_by_row=rpar_ohm_by_row)
        sdata.plot_row_iv(row=row, circuit=circuit, rpar_ohm_by_row=rpar_ohm_by_row, y_quantity="resistance")
        sdata.plot_row_iv(row=row, circuit=circuit, rpar_ohm_by_row=rpar_ohm_by_row, y_quantity="power")

    pause_ci_safe(60)
    # # plot a row vs temp raw no circuit
    # plt.figure()
    # ntemps = len(sdata.data)
    # row = 0
    # for q in range(ntemps):
    #     d = sdata.data[q]

    #     x = vtess_by_temp[q][:, row]
    #     y = vtess_by_temp[q][:, row]/itess_by_temp[q][:, row]
    #     d = sdata.data[q]
    #     temp_mk = (d.pre_temp_k + d.post_temp_k)*1e3/2
    #     plt.plot(x, y, label=f"{temp_mk:.2f} mK")
    # plt.xlabel("V tes / V")
    # plt.ylabel("R tes / Ohm")
    # plt.title(f"row = {row} r vs v_tes vs temp")
    # plt.legend()
    # plt.tight_layout()
    # ymax = y[0]
    # plt.ylim(0, ymax) # set the ylim to the maximum r value    

    # # plot a row vs temperature
    # plt.figure()
    # ntemps = len(sdata.data)
    # row = 0
    # for q in range(ntemps):
    #     x = vtess_by_temp[q][:, row]
    #     y = vtess_by_temp[q][:, row]/itess_by_temp[q][:, row]
    #     d = sdata.data[q]
    #     temp_mk = (d.pre_temp_k + d.post_temp_k)*1e3/2
    #     plt.plot(x, y, label=f"{temp_mk:.2f} mK")
    # plt.xlabel("V tes / V")
    # plt.ylabel("R tes / Ohm")
    # plt.title(f"row = {row} r vs v_tes vs temp")
    # plt.legend()
    # plt.tight_layout()
    # ymax = y[0]
    # plt.ylim(0, ymax) # set the ylim to the maximum r value


    # # plot all rows at one temp with physical units
    # plt.figure()
    # q = 0
    # for row in range(nrows):
    #     x = vtess_by_temp[q][:, row]
    #     y = vtess_by_temp[q][:, row]/itess_by_temp[q][:, row]
    #     y[np.logical_or(y<0, y>ymax)] = np.nan
    #     plt.plot(x, y, label=f"row {row}")
    # d = sdata.data[q]
    # temp_mk = (d.pre_temp_k + d.post_temp_k)*1e3/2
    # plt.xlabel("V tes / Volts")
    # plt.ylabel("R tes / Ohm")
    # plt.title(f"all rows at T={temp_mk:.2f} mK")
    # # plt.legend()
    # plt.tight_layout()
    # plt.ylim(0, ymax) # set the ylim to the maximum r value    
    # plt.grid(True, which="both")

    # # plot all rows at one temp with experimental units
    # plt.figure()
    # q = 0
    # for row in range(nrows):
    #     # x = vtess_by_temp[q][:, row]
    #     x = vbias_arbs
    #     y = vtess_by_temp[q][:, row]/itess_by_temp[q][:, row]
    #     y[np.logical_or(y<0, y>ymax)] = np.nan
    #     y=y/ymax
    #     plt.plot(x, y, label=f"row {row}")
    # d = sdata.data[q]
    # temp_mk = (d.pre_temp_k + d.post_temp_k)*1e3/2
    # plt.xlabel("vbias dac values")
    # plt.ylabel("R tes / Rmax")
    # plt.title(f"all rows at T={temp_mk:.2f} mK")
    # # plt.legend()
    # plt.tight_layout()
    # # plt.ylim(0, ymax) # set the ylim to the maximum r value    
    # plt.grid(True, which="both")


    # x, y = sdata.data[0].xyarrays_zero_subtracted_with_post_iv_zero_fb_value()
    # plt.figure()
    # plt.plot(x,y)

    # pause_ci_safe(60)
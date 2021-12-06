import detchar
from detchar import IVCircuit, IVCurveColumnData, IVTempSweepData
import numpy as np
import pylab as plt
plt.ion()
plt.close("all")

filenames = ["iv55p4mK_-90mVfc_bayCX_col0.json",
             "iv60p3mK_-90mVfc_bayCX_col0.json",
             "iv65p4mK_-90mVfc_bayCX_col0.json",
             "iv70p4mK_-90mVfc_bayCX_col0.json",
             "iv72p4mK_-90mVfc_bayCX_col0.json",
             "iv73p4mK_-90mVfc_bayCX_col0.json",
             "iv75p4mK_-90mVfc_bayCX_col0.json"]

datas = [IVCurveColumnData.from_file(filename) for filename in filenames]
temps_k = np.array([55.4,60.3,65.4,70.4,72.4,73.4,75.4])*1e-3
sdata = IVTempSweepData(temps_k, datas)
sdata.to_file("20211121_-90mVfc_bayCX_col0_IVtempsweep_12row.json", overwrite=True)
sdata.plot_temp(0)
circuit =  IVCircuit(rfb_ohm = 4e3, 
    rbias_ohm = 1e3, # need to check notes for true value
    rsh_ohm = 200e-6,
    rpar_ohm = 0,
    m_ratio = 3.46, # from inspecting the mux19 test pdfs, Mr = Min/Mfb = 265.7 pH/76.6 pH = 3.46
    vfb_gain = 1.0/2**14,
    vbias_gain = 2.5/2**16)

rpar_ohm_by_row = sdata.fit_for_rpar(circuit, sc_below_vbias_arb=1500, temp_index=0)

for row in range(sdata.get_nrows()):
    sdata.plot_row_iv(row=row, circuit=circuit, rpar_ohm_by_row=rpar_ohm_by_row)
    sdata.plot_row_iv(row=row, circuit=circuit, rpar_ohm_by_row=rpar_ohm_by_row, y_quantity="resistance")
    # sdata.plot_row_iv(row=row, circuit=circuit, rpar_ohm_by_row=rpar_ohm_by_row, y_quantity="power")

# powers = sdata.get_power_at_r(r_ohm=0.005, row=0, circuit=circuit, rpar_ohm_by_row=rpar_ohm_by_row, 
# sc_below_vbias_arb=None, plot=True)
# result, k, tc_k, n, G_W_per_k = detchar.g_fit(sdata.set_temps_k, powers, plot=True)

for row in range(12):
    powers = sdata.get_power_at_r(r_ohm=0.004, row=row, circuit=circuit, rpar_ohm_by_row=rpar_ohm_by_row, 
    sc_below_vbias_arb=None, plot=False)
    result, k, tc_k, n, G_W_per_k = detchar.g_fit(sdata.set_temps_k, powers, plot=False)        
    plt.title(f"row {row}")
    print(f"row={row} k={k:.2g} n={n:.2f} G_W_per_k={G_W_per_k:.3g}")

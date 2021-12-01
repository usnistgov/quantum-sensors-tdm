import pylab as plt
import numpy as np
import detchar
from detchar import IVCircuit, IVCurveColumnData, IVTempSweepData



plt.ion()
plt.close("all")



sdata = IVTempSweepData.from_file("20210921_2_CX_NSLS_temp_sweep_IVs_with_zero_bias_track.json")


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

print(f"1e6*rpar_ohm_by_row{1e6*rpar_ohm_by_row}")
print(f"sdata.data[-1]={sdata.data[-1]}")

for row in range(sdata.get_nrows()//16):
    sdata.plot_row(row=row)

for temp_index in range(len(sdata.set_temps_k)//16):
    sdata.plot_temp(temp_index)
# plt.figure()
# plt.plot(vbias_arbs, fb_arbs)
for row in range(sdata.get_nrows()//16):
    sdata.plot_row_iv(row=row, circuit=circuit, rpar_ohm_by_row=rpar_ohm_by_row)
    sdata.plot_row_iv(row=row, circuit=circuit, 
    rpar_ohm_by_row=rpar_ohm_by_row, y_quantity="resistance")

    sdata.plot_row_iv(row=row, circuit=circuit, rpar_ohm_by_row=rpar_ohm_by_row, y_quantity="power")

powers = sdata.get_power_at_r(r_ohm=0.005, row=0, circuit=circuit, rpar_ohm_by_row=rpar_ohm_by_row, 
sc_below_vbias_arb=None, plot=True)
result, k, tc_k, n, G_W_per_k = detchar.g_fit(sdata.set_temps_k, powers, plot=True)

for row in range(16):
    powers = sdata.get_power_at_r(r_ohm=0.005, row=row, circuit=circuit, rpar_ohm_by_row=rpar_ohm_by_row, 
    sc_below_vbias_arb=None, plot=False)
    result, k, tc_k, n, G_W_per_k = detchar.g_fit(sdata.set_temps_k, powers, plot=False)        
    print(f"row={row} k={k:.2g} n={n:.2f} G_W_per_k={G_W_per_k:.3g}")

i,v =sdata.iv_temp_val_row(circuit, sc_below_vbias_arb=1000)
r = v/i
dacs = sdata.data[0].dac_values
plt.figure()
row=0
for temp_index in range(len(sdata.data)):
    curves = sdata.data[temp_index]
    t_mK = curves.nominal_temp_k * 1e3
    dt_mK = (curves.post_temp_k - curves.pre_temp_k) * 1e3
    plt.plot(dacs, r[temp_index,:,row], label=f"{t_mK:0.2f} mK, dt {dt_mK:0.2f} mK")
plt.xlabel("dac")
plt.ylabel("resistance (ohm)")
plt.legend()
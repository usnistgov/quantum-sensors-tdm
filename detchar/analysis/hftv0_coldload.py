''' hftv0_coldload.py

analysis of data collected during 12/2/2020 velma run.
Main data product is the detector efficiency, as determined from IV
as a function of cold load temperature
'''
import ivAnalysis_utils as iva

filename_json = 'lbird_hftv0_coldload_sweep_20210203.json'
dm = iva.DetectorMap('detector_map.csv')
cl_indicies = [0,1,2,3,4,5,6,7,8]
row_indicies = [1] # list(range(24))
bath_temp_index=0

# circuit parameters
iv_circuit = iva.IVCircuit(rfb_ohm=5282.0+50.0,
                       rbias_ohm=10068.0,
                       rsh_ohm=0.0662,
                       rx_ohm=0,
                       m_ratio=8.259,
                       vfb_gain=1.017/(2**14-1),
                       vbias_gain=6.5/(2**16-1))
df = iva.IVColdloadSweepAnalyzer(filename_json,dm.map_dict,iv_circuit) #df is the main "data format" of the coldload temperature sweep
#df.plot_measured_bath_temps() # first check if all measurements taken at intended temperatures
#plt.show()
for row in row_indicies:
    df.plot_sweep_analysis_for_row(row_index=row,bath_temp_index=1,cl_indicies=cl_indicies,showfigs=True,savefigs=False)

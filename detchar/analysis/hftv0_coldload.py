''' hftv0_coldload.py

analysis of data collected during 12/2/2020 velma run.
Main data product is the detector efficiency, as determined from IV
as a function of cold load temperature
'''
import ivAnalysis_utils as iva
import numpy as np

filename_json = 'lbird_hftv0_coldload_sweep_20210203.json'
dm = iva.DetectorMap('detector_map2.csv')
cl_indices = [0,1,2,3,4,5,6,7,8]
bath_temp_index=1

# circuit parameters
iv_circuit = iva.IVCircuit(rfb_ohm=5282.0+50.0,
                       rbias_ohm=10068.0,
                       rsh_ohm=0.0662,
                       rx_ohm=0,
                       m_ratio=8.259,
                       vfb_gain=1.017/(2**14-1),
                       vbias_gain=6.5/(2**16-1))
df = iva.IVColdloadSweepAnalyzer(filename_json,dm,iv_circuit) #df is the main "data format" of the coldload temperature sweep
df.full_analysis(bath_temp_index=bath_temp_index,cl_indices=cl_indices,showfigs=False,savefigs=True,dark_rnfrac=0.8)
# # get dP versus T coldload for dark bolometer
# dacs,fb_dark = df.get_cl_sweep_dataset_for_row(row_index=dark_row_index,bath_temp_index=bath_temp_index,cl_indices=cl_indices)
# iva_dark = iva.IVColdloadAnalyzeOneRow(dacs,fb_dark,
#                               cl_temps_k=list(np.array(df.set_cl_temps_k)[cl_indices]),
#                               bath_temp_k=df.set_bath_temps_k[bath_temp_index],
#                               device_dict=None,
#                               iv_circuit=df.iv_circuit,
#                               passband_dict=None)
# dark_dP = iva_dark.get_pt_delta_for_rnfrac(0.8)
#
# # do dark subtracted data analysis on optical bolo
# for row in row_indices:
#     print('Analyzing Row%02d'%row)
#     dacs,fb = df.get_cl_sweep_dataset_for_row(row_index=row,bath_temp_index=bath_temp_index,cl_indices=cl_indices)
#     xx = iva.IVColdloadAnalyzeOneRow(dacs,fb,
#                                   cl_temps_k=list(np.array(df.set_cl_temps_k)[cl_indices]),
#                                   bath_temp_k=df.set_bath_temps_k[bath_temp_index],
#                                   device_dict=dm.get_onerow_device_dict(row),
#                                   iv_circuit=df.iv_circuit,
#                                   passband_dict={'freq_edges_ghz':dm.map_dict['Row%02d'%row]['freq_edges_ghz']},
#                                   dark_dP_w=dark_dP)
#     xx.plot_full_analysis(showfigs=False,savefigs=True)

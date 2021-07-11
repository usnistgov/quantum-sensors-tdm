''' hftv0_eta_ltd19.py

Optical efficiency analysis for LTD19
analysis of data collected during 12/2/2020 velma run.
'''
import sys
sys.path.append('/Users/hubmayr/nistgit/nistqsptdm/')
import ivAnalysis_utils as iva
from fts_utils import Passband
import numpy as np
from detector_map import DetectorMap
import matplotlib.pyplot as plt
import pickle


def get_lb_pixel_type_from_devname(devname):
    ptype = devname.split(' ')[0]
    if ptype == 'HFT1':
        return 'hft1'
    elif ptype == 'HFT2':
        return 'hft2'

filename_json = 'lbird_hftv0_coldload_sweep_20210203.json'
dm = DetectorMap('detector_map_run20201202.csv')
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

# Measured passbands used for dP prediction
f=open('tmpfiles/hftv0_passbands.pkl','rb')
pb_dict = pickle.load(f)
f.close()
# pb_dict[hft1 or 2][band][f or S_complex]
# example: pb_dict['hft2']['337A']['f'], S_measure_complex=pb_dict['hft2']['337A']['S_complex']
passband_f_mask_ghz={'195':[140,240],
                     '280':[210,350],
                     '235':[170,300],
                     '337':[250,410]
                     }

# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------
df = iva.IVColdloadSweepAnalyzer(filename_json,dm,iv_circuit) #df is the main "data format" of the coldload temperature sweep
rn_frac_index = 3 #(0-5) for this dataset
row_dict = dm.get_rowdict_from_keyval_list(search_list=[['position',1],['band','337']])
row_dict.update(dm.get_rowdict_from_keyval_list(search_list=[['position',1],['type','dark']]))
row_dict = dm.get_rowdict_from_keyval_list(search_list=[['position',1],['band','337']])

for ii,position in enumerate([1,2,3,4]): # loop over positions (ie bays on mK board)
#for ii,position in enumerate([1]): # loop over positions (ie bays on mK board)
    # get indices corr to dark and optical pixels in the position/bay
    dark_dict = dm.get_rowdict_from_keyval_list(search_list=[['position',position],['type','dark']])
    dark_row_index = dm.get_row_nums_from_keys(row_keys=list(dark_dict.keys()))[0]
    opt_dict = dm.get_rowdict_from_keyval_list(search_list=[['position',position],['type','optical']])
    opt_row_indices = dm.get_row_nums_from_keys(row_keys=list(opt_dict.keys()))

    # get dP for darks first to feed into OneRow instance of optical devices
    dacs,fb = df.get_cl_sweep_dataset_for_row(row_index=dark_row_index,bath_temp_index=bath_temp_index,cl_indices=cl_indices)
    cl_row_dark = iva.IVColdloadAnalyzeOneRow(dacs,fb,
                                              cl_temps_k=list(np.array(df.set_cl_temps_k)[cl_indices]),
                                              bath_temp_k=df.set_bath_temps_k[bath_temp_index],
                                              device_dict=dm.get_onerow_device_dict(dark_row_index),
                                              iv_circuit=iv_circuit)
    #cl_row_dark.plot_full_analysis(showfigs=True,savefigs=False)
    dP_dark = cl_row_dark.dP_w[rn_frac_index,:]

    # now loop over optically coupled pixels
    for jj,opt_row_index in enumerate(opt_row_indices):
        rowmap = dm.get_mapval_for_row_index(opt_row_index)
        print(list(rowmap.keys())[0],rowmap['devname'])
        lb_pixel_type = get_lb_pixel_type_from_devname(rowmap['devname'])
        bandpol = rowmap['band']+rowmap['polarization']
        pb = Passband(f_measure_ghz=pb_dict[lb_pixel_type][bandpol]['f'], S_measure_complex=pb_dict[lb_pixel_type][bandpol]['S_complex'])
        # pb.plot()
        # plt.show()
        dacs,fb = df.get_cl_sweep_dataset_for_row(row_index=opt_row_index,bath_temp_index=bath_temp_index,cl_indices=cl_indices)
        cl_row = iva.IVColdloadAnalyzeOneRow(dacs,fb,
                                             cl_temps_k=list(np.array(df.set_cl_temps_k)[cl_indices]),
                                             bath_temp_k=df.set_bath_temps_k[bath_temp_index],
                                             device_dict=dm.get_onerow_device_dict(opt_row_index),
                                             iv_circuit=iv_circuit,
                                             passband_dict=None,
                                             dark_dP_w=dP_dark,
                                             passband_instance=pb,
                                             passband_f_mask_ghz=passband_f_mask_ghz[rowmap['band']])
        #cl_row.plot_full_analysis(showfigs=True,savefigs=False)
        cl_row.plot_efficiency(cl_row.cl_dT_k, cl_row.eta_passband_measured, cl_row.rn_fracs, fig_num=1, eta_dark_subtracted=cl_row.eta_passband_measured_ds)
        plt.show()

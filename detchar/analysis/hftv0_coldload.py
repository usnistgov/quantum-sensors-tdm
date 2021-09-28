''' hftv0_coldload.py

analysis of data collected during 12/2/2020 velma run.
Main data product is the detector efficiency, as determined from IV
as a function of cold load temperature
'''
import sys
sys.path.append('/Users/hubmayr/nistgit/nistqsptdm/')
import ivAnalysis_utils as iva
from fts_utils import Passband
import numpy as np
from detector_map import DetectorMap
import matplotlib.pyplot as plt
import pickle

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

# original code------------------
df = iva.IVColdloadSweepAnalyzer(filename_json,dm,iv_circuit) #df is the main "data format" of the coldload temperature sweep
# df.full_analysis(bath_temp_index=bath_temp_index,cl_indices=cl_indices,showfigs=True,savefigs=False,dark_rnfrac=0.8)

# First make dP vs dT plot with data from: HFT2 337A, 337B, dark, prediction --------------------------------------
rn_frac_index = 2 #(0-5) for this dataset

fig,ax = plt.subplots(num=1)

row_dict = dm.get_rowdict_from_keyval_list(search_list=[['position',1],['band','337']])
row_dict.update(dm.get_rowdict_from_keyval_list(search_list=[['position',1],['type','dark']]))

dPs = []
for row in ['Row15','Row12','Row16']:
    subdict = row_dict[row]
    idx = dm.get_row_nums_from_keys([row])[0]
    if subdict['band'] is not None:
        legend_label = subdict['band']
    else:
        legend_label=''
    if subdict['polarization'] is not None:
        legend_label = legend_label+subdict['polarization']
    print(row,': ',legend_label)
    dacs,fb = df.get_cl_sweep_dataset_for_row(row_index=idx,bath_temp_index=bath_temp_index,cl_indices=cl_indices)
    cl_row = iva.IVColdloadAnalyzeOneRow(dacs,fb,
                                  cl_temps_k=list(np.array(df.set_cl_temps_k)[cl_indices]),
                                  bath_temp_k=df.set_bath_temps_k[bath_temp_index],
                                  device_dict=dm.get_onerow_device_dict(idx),
                                  iv_circuit=iv_circuit,
                                  passband_dict=None,
                                  dark_dP_w=None)

    ax.plot(cl_row.cl_dT_k, cl_row.dP_w[rn_frac_index,:]*1e12,'o-',label=legend_label)
    dPs.append(cl_row.dP_w[rn_frac_index,:]*1e12)

# add prediction
# load phase corrected spectra from pickle file
f=open('tmpfiles/hftv0_passbands.pkl','rb')
pb_dict = pickle.load(f)
f.close()

#make instance of fts_utils.Passband
f_range_ghz = [250,400]
pb = Passband(f_measure_ghz=pb_dict['hft2']['337A']['f'], S_measure_complex=pb_dict['hft2']['337A']['S_complex'],f_model_ghz=None,S_model=None,f_range_ghz=f_range_ghz)
dT,dP = pb.get_dT_and_dP_from_measured_passband(temp_k_list=cl_row.cl_temps_k,f_mask=f_range_ghz,zero_index=0)
plt.plot(dT,dP*1e12,'k--',label='$\eta$=1')
ax.legend()
ax.set_xlabel('T$_{load}$ - %.1f K'%cl_row.cl_temps_k[cl_row.T_cl_index])
ax.set_ylabel('$\Delta{P}$ (pW)')
ax.set_ylim((-.2,3))
ax.grid()
fig.savefig('dPdT_example.pdf')

fi2,ax2 = plt.subplots(num=2)
for ii in range(2):
    eta = (dPs[ii]-dPs[2])/(dP*1e12)
    ax2.plot(dT,eta,'o-')
ax2.set_xlabel('T$_{load}$ - %.1f K'%cl_row.cl_temps_k[cl_row.T_cl_index])
ax2.set_ylabel('$\eta$')

plt.show()


#plt.show()



# path = '/Users/hubmayr/projects/lbird/HFTdesign/hft_v0/modeled_response/' # on Hannes' machine
# filename='hftv0_hft2_diplexer_model.txt'
# pbm = PassbandModel(path+filename)
# #pbm.plot_model()
# #plt.show()
#
# d=np.load('measured_spectrum_example.npz')
# f_ghz = d['f']
# S = d['B']
# f_range_ghz = [175,300]
# # plt.plot(f_ghz,np.real(S))
# # plt.show()
#
# pb = Passband(f_measure_ghz=f_ghz,S_measure_complex=S,f_model_ghz=pbm.f_ghz,S_model=pbm.model[:,2],f_range_ghz=f_range_ghz)
# pb.plot_PvTs(temp_k_list=list(range(4,12)),f_mask=f_range_ghz,freq_edges_ghz=[200,275],fig_num=1)
# pb.print_passband_metrics()
# plt.show()

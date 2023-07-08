 '''
uber_omt_PvT.py based on lb_bolov3_PvT.py

Take raw data from json, convert to physical units, fit power vs temperature and
extract Tc, G, n and save files

'''    
import os, sys
import numpy as np
from detchar.analysis.ivAnalysis_utils import IVCurveColumnDataExplore
from detchar.iv_data import IVTempSweepData
from detchar.analysis.ivAnalysis_utils import IVversusADRTempOneRow
from detchar.analysis.ivAnalysis_utils import IVSetAnalyzeColumn
from IPython import embed
from detchar.iv_data import IVCircuit
import pylab as pl
import pickle

def todict(obj, classkey=None):
    '''
    Turn an object into a dict so we can pickle it
    '''
    if isinstance(obj, dict):
        data = {}
        for (k, v) in obj.items():
            data[k] = todict(v, classkey)
        return data
    elif hasattr(obj, "_ast"):
        return todict(obj._ast())
    elif hasattr(obj, "__iter__") and not isinstance(obj, str):
        return [todict(v, classkey) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = dict([(key, todict(value, classkey)) 
            for key, value in obj.__dict__.items() 
            if not callable(value) and not key.startswith('_')])
        if classkey is not None and hasattr(obj, "__class__"):
            data[classkey] = obj.__class__.__name__
        return data
    else:
        return obj

def main():
    path = '/home/pcuser/data/uber_omt/20230421/'
    #fname = 'uber_omt_ColumnC_ivs_20230503_1683154594.json' # 90 to 170 mK
    # fname = 'uber_omt_ColumnC_ivs_20230504_1683223467.json' #90, 100, then 170 to 190
    fname = 'uber_omt_ColumnC_ivs_20230504_all_temps.json'  # 90 to 200 mK


    hwmappath = '/home/pcuser/qsp/src/nistqsptdm/detchar/analysis/uber_omt'
    hwmapfile = 'hwmap.pkl'    
    path = '/home/pcuser/data/uber_omt/20230421/'
    fnames = [#'uber_omt_ColumnC_ivs_20230503_1683154594.json', # 90 to 170 mK
              #'uber_omt_ColumnC_ivs_20230504_1683223467.json', #90, 100, then 170 to 190
              'uber_omt_ColumnC_ivs_20230504_all_temps.json',  # 90 to 200 mK
            ]
    outfile = 'tsweep_analyzed_170mK.pkl'        
    with open(os.path.join(hwmappath,hwmapfile),'rb') as opf:
        hwmap = pickle.load(opf)        
    outpath = os.path.join(path,'uber_omt_PvT_output_20230505')
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    results = {}
    for fname in fnames:
        df = IVTempSweepData.from_file(os.path.join(path,fname))
        # Cut all IVs with temps 170 mK and above
        t_cut_high = 0.17
        t_cut_index = [nn for nn,tt in enumerate(df.set_temps_k) if tt >= t_cut_high]
        for nn in t_cut_index[::-1]:
            df.data.pop(nn)
            df.set_temps_k.pop(nn)

        # Cut specific IVs from specific channels in the most inconvenient way possible
        # This doesn't work because the xy_arrays method df.data doesn't care if you want to change values
        # channel_cuts = {'A':{34,[0.16,0.165]}} # channel, row, temps to cut
        # for cc in channel_cuts:
        #     if cc in fnames[0].split('Column')[1][0]:
        #         for rr in channel_cuts[cc]:
        #             for tt in channel_cuts[cc][rr]:
        #                 t_cut_index = df.set_temps_k.index(tt)
        #                 n_len = len(df.data[t_cut_index].xy_arrays()[1][:,rr])
        #                 #df.data[t_cut_index].xy_arrays()[1][:,rr] = np.empty(n_len)
        #                 df.data[t_cut_index].xy_arrays()[1][:,rr][:] = np.nan

        bayname = df.data[0].bayname # 'A'
        results[bayname] = {}
        # get rows from hwmap
        i2r = 2 # index to row conversion, usually 2, use 4 for 20220214_1644858306.json data
        if bayname in hwmap:
            rows = list(hwmap[bayname].keys())
            row_indices = i2r*np.array(rows)
        else:
            rows = df.data[0].extra_info['config']['detectors']['Rows']
            row_indices = range(len(rows)) #i2r*np.array(rows) # we only used even rows to get around xtalk/shadow issues
        if i2r==4:
            row_indices = row_indices[row_indices<48] # jsut keep the lowest 48
        
        # Use IVCurveColumnDataExplore to get ivex.iv_circuit
        # ivex = IVCurveColumnDataExplore(df.data[0])
        # or 
        iv_circuit = IVCircuit(rfb_ohm=1698.0+50.0,
                        rbias_ohm=219.3,
                        rsh_ohm=0.000150,
                        rx_ohm=0,
                        m_ratio=15,
                        vfb_gain=1.017/(2**14-1),
                        vbias_gain=2.5/(2**16-1)) # 6.5 for bluebox

        # construct fb_arr versus Tbath for a single row
        dac, fb = df.data[0].xy_arrays()
        n_sweeps = len(df.set_temps_k)
        
        # loop over rows and analyze with IVversusADRTempOneRow
        for row_index in row_indices:
            #if not row_index: continue # skip first as a test
            print('\nWorking column {} row {}'.format(bayname,row_index))
            fb_arr = np.zeros((len(dac),n_sweeps))
            for ii in range(n_sweeps):
                dac, fb = df.data[ii].xy_arrays()
                #embed();asdf
                fb_arr[:,ii] = fb[:,row_index]

            iv_tsweep = IVversusADRTempOneRow(dac_values=dac,
                                            fb_values_arr=fb_arr, 
                                            temp_list_k=df.set_temps_k, 
                                            normal_resistance_fractions=[0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2],
                                            iv_circuit=iv_circuit, #ivex.iv_circuit,
                                            )
            
            savebasename = 'bay{}_row{:02d}'.format(df.data[0].bayname,row_index)
                    
            iv_tsweep.plot_raw(1)
            pl.title(savebasename)
            outname = 'raw_plot_{}.png'.format(savebasename)
            pl.savefig(os.path.join(outpath,outname),bbox_inches='tight')

            iv_tsweep.plot_vipr(2)
            pl.suptitle(savebasename)
            outname = 'vipr_plot_{}.png'.format(savebasename)
            pl.savefig(os.path.join(outpath,outname),bbox_inches='tight')

            iv_tsweep.plot_pr(fig_num=3)
            pl.title(savebasename)
            outname = 'pr_plot_{}.png'.format(savebasename)
            pl.savefig(os.path.join(outpath,outname),bbox_inches='tight')

            iv_tsweep.plot_pt(fig_num=4)
            pl.title(savebasename)
            outname = 'pt_plot_{}.png'.format(savebasename)
            pl.savefig(os.path.join(outpath,outname),bbox_inches='tight') 

            print(iv_tsweep.pfits)
            results[bayname][row_index] = todict(iv_tsweep)
            #embed();sys.exit()
            #pl.ion(); pl.show()
            pl.close('all')
    #embed();sys.exit()
    with open(os.path.join(outpath,outfile),'wb') as opr:
        pickle.dump(results,opr)
    return

if __name__=='__main__':
    main()

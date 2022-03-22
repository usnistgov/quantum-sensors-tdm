# want to plot N IV curves converted to physical units for same row under different conditions


from detchar.iv_data import IVTempSweepData
fnames = ['/home/pcuser/data/lbird/20210320/lbird_hftv0_ColumnA_ivs_20210413_1618344868.json',\
          '/home/pcuser/data/lbird/20210320/lbird_hftv0_ColumnA_ivs_20210413_1618345154.json',\
          '/home/pcuser/data/lbird/20210320/lbird_hftv0_ColumnA_ivs_20210413_1618346022.json']
ivs=[]
N = len(fnames)
for fname in fnames:
    ivs.append(IVTempSweepData.from_file(fname)) 

xys = []
for iv in ivs:
    xys.append(iv.data[0].xy_arrays_zero_subtracted_at_dac_high())

# plot darks
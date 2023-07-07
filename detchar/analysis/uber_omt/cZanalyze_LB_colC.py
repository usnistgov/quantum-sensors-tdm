from detchar.iv_data import CzData, SineSweepData, IVCurveColumnData
import matplotlib.pyplot as plt
import numpy as np
import os
from detchar.analysis.ivAnalysis_utils import IVCurveColumnDataExplore
from detchar.iv_data import IVTempSweepData
from IPython import embed

path='/data/uber_omt/20230421/'
ofile = 'colC_row13_15_19_cz_fine.json'
ivfile = 'uber_omt_ColumnC_ivs_20230504_all_temps.json'
cz = CzData.from_file(os.path.join(path,ofile))

# #data = IVCurveColumnData.from_file(os.path.join(path,ivfile))
# data = IVTempSweepData.from_file(os.path.join(path,ivfile))

# x, y = data.xy_arrays_zero_subtracted()
# r = iv_to_frac_rn_array(x, y, superconducting_below_x=2000, normal_above_x=5000)

#print(cz.temp_list_k)
#cz.plotZ(temp_k=0.11, Tc_k=0.19,semilogx=True,f_max_hz=1000)
cz.analyzeZ(temp_k=0.12, Tc_k=0.19,semilogx=True,f_max_hz=10000)

embed();sys.exit()

plt.ion()
plt.show()

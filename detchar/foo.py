from iv_data import CzData, SineSweepData 
import matplotlib.pyplot as plt
import numpy as np

path='/data/uber_omt/20230421/'
file = 'colC_row7_cz_fine.json'
#file = 'colC_row13_15_19_cz_fine.json'
cz = CzData.from_file(path+file)
#print(cz.temp_list_k)
cz.plotZ(0.13,f_max_hz=1000)
plt.show()

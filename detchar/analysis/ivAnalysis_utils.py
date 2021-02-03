''' ivAnalysis_utils.py '''

from detchar import IVColdloadSweepData
import numpy as np
import matplotlib.pyplot as plt

# produce IV plot for each coldload temp for single row at one temperature
class IVColdloadSweepAnalyzer(): 
    def __init__(self,filename_json):
        self.data = IVColdloadSweepData.from_file(filename_json) 
        #self.data.data is a list of IVColdloadSweepData, one per coldload temperature setpoint
        #self.data.data[0].data is a list of IVCurveColumnData, one per bath temperature setpoint
        #therefore self.data.data[ii].data[jj]
        
        # globals about coldload temperatures
        self.set_cl_temps_k = self.data.set_cl_temps_k 
        self.max_cl_temp_k = np.max(self.set_cl_temps_k)
        self.min_cl_temp_k = np.min(self.set_cl_temps_k)
        self.n_cl_temps = len(self.set_cl_temps_k)
        self.pre_cl_temps_k = self.data.extra_info['pre_cl_temps_k']
        self.post_cl_temps_k = self.data.extra_info['post_cl_temps_k']
        self.measured_cl_temps_k = np.array(self.data.extra_info['pre_cl_temps_k'])
        self.cl_therm_index = 0

        # globals about bath temperatures
        self.set_bath_temps_k = self.data.data[0].set_temps_k
    
    def plot_measured_cl_temps(self):
        plt.figure(1)
        plt.xlabel('Setpoint Temperature (K)')
        plt.ylabel('Measured Temperature (K)')
        cl_temp_list = self._package_cl_temp_to_list()
        plt.plot(self.set_cl_temps_k,self.pre_cl_temps_k,'*')
        plt.plot(self.set_cl_temps_k,self.post_cl_temps_k,'*')
        plt.plot(list(range(self.max_cl_temp_k+1)),'b--')
        plt.legend(('ChA pre','ChB pre','ChA post','ChB post'))
        plt.grid()

        plt.figure(2)
        plt.xlabel('Time (arb)')
        plt.ylabel('Temperature (K)')
        x = list(range(self.n_cl_temps))
        plt.plot(x,self.set_cl_temps_k,'ko-')
        plt.plot(x,self.pre_cl_temps_k,'*')
        plt.plot(x,self.post_cl_temps_k,'*')
        plt.legend(('Setpoint','ChA pre','ChB pre','ChA post','ChB post'))

        plt.show()

    def _package_cl_temp_to_list(self):
        cl_temp_list = []
        cl_temp_list.append(list(self.measured_cl_temps_k[:,0]))
        cl_temp_list.append(list(self.measured_cl_temps_k[:,1]))
        return cl_temp_list

    
         
if __name__ == "__main__":
    filename_json = '/home/pcuser/data/lbird/20201202/lbird_hftv0_coldload_sweep.json'
    df = IVColdloadSweepAnalyzer(filename_json)
    df.plot_measured_cl_temps()
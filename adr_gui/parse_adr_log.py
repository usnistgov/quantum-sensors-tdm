''' 
The start of some code to parse the adr log file. 

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

class AdrLogParser():
    def __init__(self,logfile,path='/data/ADRLogs/'):
        self.path = path 
        self.logfile = logfile 
        self.df = pd.read_csv(path+logfile) # cols: date/time, epoch time (s), adr temp, heater output 
        self.epoch_time_start = self.df.iloc[0,1]
        self.epoch_time_end = self.df.iloc[0,-1]
        self.datetime_start_str = self.df.iloc[0,0]
        self.datetime_end_str = self.df.iloc[-1,0]

    def print_metadata(self):
        print('ADR log file: %s spans %s -- %s'%(self.path+self.logfile,self.datetime_start_str,self.datetime_end_str))
        print('The min (max) ADR temperature is %.1f mK (%.1f)'%(self.df.iloc[:,2].min()*1000,self.df.iloc[:,2].max()*1000))
        print('The min (max) heater output is %.1f (%.1f)'%(self.df.iloc[:,3].min(),self.df.iloc[:,3].max()))

    def plot(self,index=2,fig=None,ax=None):
        if index==2:
            ylabel='ADR temp (K)'
        elif index==3:
            ylabel='Heater output'
        else:
            print('index must be 2 (adr temp) or 3 (heater output)')
        
        if np.logical_and(fig is None, ax is None):
            fig,ax = plt.subplots(1,1)
        ax.plot(self.df.iloc[:,1],self.df.iloc[:,index])
        ax.set_xlabel('Epoch time (s)')
        ax.set_ylabel(ylabel)
        ax.grid('on')

if __name__ == "__main__":
    al = AdrLogParser('ADRLog_20230708_t154911.txt')
    al.print_metadata()
    al.plot()
    plt.show()

    
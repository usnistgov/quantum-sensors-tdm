''' 
The start of some code to parse the adr log file. 

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import glob
import os

class AdrLogParser():
    def __init__(self,logfile=None,path='/data/ADRLogs/'):
        self.path = path 
        self.logfile = self.__handle_logfile__(logfile) # if logfile is none, find the most recent
        self.df = self.__get_data_frame__() 
        self.epoch_time_start = self.df.iloc[0,1]
        self.epoch_time_end = self.df.iloc[0,-1]
        self.datetime_start_str = self.df.iloc[0,0]
        self.datetime_end_str = self.df.iloc[-1,0]

    def __handle_logfile__(self,logfile):
        ''' find most recent logfile if not explicitly given '''
        if logfile is None: # use the most recent file if logfile is None
            list_of_files = glob.glob(self.path+'*') # * means all if need specific format then *.csv
            logfile = max(list_of_files, key=os.path.getctime)
            logfile = logfile.split('/')[-1]
        return logfile
        
    def __get_data_frame__(self):
        ''' determine file type and make header / columns 

            There are two distinct adr log types: 
                1) from adr_gui (tracks single thermometer and heater out), with filename structure "ADRLog_..." 
                2) from adrTempMonitorGui.py, which tracks many thermometers and has filename structure "adrFullLog_..."
        '''
        f=open(self.path+self.logfile,'r')
        line = f.readline()
        f.close()
        
        if line[0]=='#':
            names = line[1:].rstrip().split(',')
            # cols: date/time, epoch time (s), sensor #1, sensor #2, ... number of columns can vary
            self.logtype = 'full'
            df = pd.read_csv(self.path+self.logfile,names=names) 

        else:
            # cols: date/time, epoch time (s), adr temp, heater output
            self.logtype = 'adr_gui'
            names = ['date','epoch_time','adr','heater']
            df = pd.read_csv(self.path+self.logfile,header=None,names=names) 
        return df

    def print_metadata(self):
        print('ADR log file %s is of type %s and spans %s -- %s.'%(self.path+self.logfile,self.logtype,self.datetime_start_str,self.datetime_end_str))
        for column in self.df.columns[2:]:
            print(column, ' min | max = %.4f | %.4f'%(self.df[column].min(),self.df[column].max()))
        # print('ADR log file: %s spans %s -- %s'%(self.path+self.logfile,self.datetime_start_str,self.datetime_end_str))
        # print('The min (max) ADR temperature is %.1f mK (%.1f)'%(self.df.iloc[:,2].min()*1000,self.df.iloc[:,2].max()*1000))
        # print('The min (max) heater output is %.1f (%.1f)'%(self.df.iloc[:,3].min(),self.df.iloc[:,3].max()))

    def plot(self,fig=None,ax=None):
        # if index==2:
        #     ylabel='ADR temp (K)'
        # elif index==3:
        #     ylabel='Heater output'
        # else:
        #     print('index must be 2 (adr temp) or 3 (heater output)')

        if self.logtype == 'adr_gui':
            if np.logical_and(fig is None, ax is None):
                fig,ax1 = plt.subplots()
                fig.suptitle(self.logfile)
                ax1.set_xlabel('epoch time (s)')
                color = 'tab:blue'
                ax1.set_ylabel('adr temp (K)',color=color)
                ax1.plot(self.df.iloc[:,1],self.df.iloc[:,2],color=color)
                ax1.tick_params(axis='y', labelcolor=color)
                ax1.grid('on')

                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis  
                color = 'tab:red'
                ax2.set_ylabel('heater out (%)',color=color)  # we already handled the x-label with ax1
                ax2.plot(self.df.iloc[:,1],self.df.iloc[:,3],color=color)
                ax2.tick_params(axis='y', labelcolor=color)
                fig.tight_layout()  # otherwise the right y-label is slightly clipped

if __name__ == "__main__":
    al = AdrLogParser()
    al.print_metadata()
    al.plot()
    plt.show()

    
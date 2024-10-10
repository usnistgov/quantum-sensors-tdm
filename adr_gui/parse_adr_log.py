#!/usr/bin/env python

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
            self.header = line[1:].rstrip().split(',')
            # cols: date/time, epoch time (s), sensor #1, sensor #2, ... number of columns can vary
            self.logtype = 'full'
            df = pd.read_csv(self.path+self.logfile,header=0,names=self.header) 

        else:
            # cols: date/time, epoch time (s), adr temp, heater output
            self.logtype = 'adr_gui'
            self.header = ['date','epoch_time','adr','heater','magnet_current']
            df = pd.read_csv(self.path+self.logfile,names=self.header) 
            print(df)
        return df

    def print_metadata(self):
        print('ADR log file %s is of type %s and spans %s -- %s.'%(self.path+self.logfile,self.logtype,self.datetime_start_str,self.datetime_end_str))
        for column in self.df.columns[2:]:
            print(column, ' min | max = %.4f | %.4f'%(self.df[column].min(),self.df[column].max()))
            #print(column, self.df[column][0])

    def plot(self,time_hrs=True,semilog=False):
        ''' plot data as function of epoch time '''
        fig,ax1 = plt.subplots()
        fig.suptitle(self.logfile)
        
        if time_hrs:
            ax1.set_xlabel('Time (hrs)')
            t = (self.df.iloc[:,1]-self.df.iloc[0,1])/3600
        else:
            ax1.set_xlabel('epoch time (s)')
            t = self.df.iloc[:,1]

        if semilog:
            plot = ax1.semilogy
        else:
            plot = ax1.plot
        if self.logtype == 'adr_gui':
            color = 'tab:blue'
            ax1.set_ylabel('adr temp (K)/ magnet current [A]',color=color)
            plot(t,self.df.iloc[:,2],color=color,label="ADR temperature")
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid('on')

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis  
            color = 'tab:red'
            ax2.set_ylabel('heater out (%)',color=color)  # we already handled the x-label with ax1
            if semilog:
                ax2.semilogy(t,self.df.iloc[:,3],color=color,label="Heater")
            else:
                ax2.plot(t,self.df.iloc[:,3],color=color,label="heater")
            ax2.tick_params(axis='y', labelcolor=color)
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            ax1.plot(t,self.df.iloc[:,4],label="magnet current",color="C2")
            ax1.legend()

        elif self.logtype == 'full':
            ax1.set_ylabel('Temperature (K)')
            for ii in range(2,len(self.df.columns)):
                plot(t,self.df.iloc[:,ii],label=self.header[ii])
            ax1.grid('on')
            ax1.legend()

if __name__ == "__main__":
    import argparse
    def make_parser():
        parser = argparse.ArgumentParser(description='Plot ADR temperature log file')
        parser.add_argument('--filename', type=str, default=None,
                        help='logfile name, not full path.')
        parser.add_argument('--logplot', type=bool, default=False,
                        help='plot on logscale.  Boolean')
        args = parser.parse_args()
        return args

    args=make_parser()
    al = AdrLogParser(args.filename)
    al.print_metadata()
    al.plot(semilog=args.logplot)
    plt.show()

    
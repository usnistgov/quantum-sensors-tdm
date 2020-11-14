'''
iv_foo.py

stupid script to collect an IV assuming everything is already setup in cringe & dastard

data[col,row,frame,error=0/fb=1]
'''

import matplotlib.pyplot as plt 
import numpy as np 
from nasa_client import EasyClient
from instruments import BlueBox


def getDataAverageAndCheck(ec,v_min=0.1,v_max=0.9,npts=10000,verbose=False):
    flag=False
    data = ec.getNewData(minimumNumPoints=npts,exactNumPoints=True,toVolts=True)
    data_mean = np.mean(data,axis=2)
    data_std = np.std(data,axis=2)

    if verbose:
        for ii in range(ec.ncol):
            for jj in range(ec.nrow):
                print('Col ',ii, 'Row ',jj, ': %0.4f +/- %0.4f'%(data_mean[ii,jj,1],data_std[ii,jj,1]))

    a = data_mean[:,:,1][data_mean[:,:,1]>v_max]
    b = data_mean[:,:,1][data_mean[:,:,1]<v_min]
    if a.size: 
        print('Value above ',v_max,' detected')
        # relock here
        flag=True
    if b.size:
        print('Value below ',v_min,' detected')
        # relock here
        flag=True

    # have some error handling about if std/mean > threshold

    return data_mean, flag

def iv_sweep(ec, vs, v_start=0.1,v_stop=0,v_step=0.01,sweepUp=False,showPlot=False,verbose=False,):
    ''' ec: instance of easy client
        vs: instance of bluebox (voltage source)
        v_start: initial voltage
        v_stop: final voltage
        v_step: voltage step size
        sweepUp: if True sweeps in ascending voltage
    '''
    v_arr = np.arange(v_stop,v_start+v_step,v_step)
    if not sweepUp:
        v_arr=v_arr[::-1]
    
    N=len(v_arr)
    data_ret = np.zeros((ec.ncol,ec.nrow,N,2))
    for ii in range(N):
        vs.setvolt(v_arr[ii])
        data, flag = getDataAverageAndCheck(ec,verbose=verbose)
        data_ret[:,:,ii,:] = data
        #data_ret[:,:,ii,1] = data[:,:,1]
    
    if showPlot:
        for ii in range(ec.ncol):
            plt.figure(ii)
            for jj in range(ec.nrow):
                plt.subplot(211)
                plt.plot(v_arr,data_ret[ii,jj,:,1])
                plt.xlabel('V_bias')
                plt.ylabel('V_fb')
                plt.subplot(212)
                plt.plot(v_arr,data_ret[ii,jj,:,0])
                plt.xlabel('V_bias')
                plt.ylabel('V_err')
        plt.show()
    return v_arr, data_ret

c = EasyClient()
c.setupAndChooseChannels()
bb = BlueBox(port='vbox', version='mrk2', channel=0)
iv_sweep(c,bb,v_start=0.1,v_stop=0.0,v_step=0.001,sweepUp=True,showPlot=True,verbose=False)
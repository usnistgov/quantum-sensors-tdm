#! /usr/bin/env python
'''test_psd.py
Using easyclient, grab data and take psd

Usage: ./test_psd.py <acquistion time> <number of averages>

averages each PSD

JH 1/2021
'''

import numpy as np
import sys
from nasa_client import EasyClient
import matplotlib.pyplot as plt
from scipy.signal import welch

def highestPowerOfTwo(n):
    return int(np.log2(n))

R_fb = 5282.0
Mr = 8.259
Vfb_gain = 1.017

ec = EasyClient() # easy client for streaming data
nargs=len(sys.argv)
if nargs==1:
    npts = 10000
    nave = 1
elif nargs==2:
    npts = int(float(sys.argv[1])*ec.sample_rate)
    nave = 1
elif nargs==3:
    npts = int(float(sys.argv[1])*ec.sample_rate)
    nave = int(sys.argv[2])


#nfft = 2**highestPowerOfTwo(npts)
nfft = npts
if nave==1:
    data = ec.getNewData(minimumNumPoints=npts,exactNumPoints=True,toVolts=False)
    f,Pxx = welch(data[0,:,:,1], fs=ec.sample_rate, window=np.ones(nfft), nperseg=None, noverlap=None, nfft=nfft, detrend=False, scaling='density', axis=-1, average='mean')
else:
    Pxx=0
    for ii in range(nave):
        data = ec.getNewData(minimumNumPoints=npts,exactNumPoints=True,toVolts=False)
        f,Pxx_temp = welch(data[0,:,:,1], fs=ec.sample_rate, window=np.ones(nfft), nperseg=None, noverlap=None, nfft=nfft, detrend=False, scaling='density', axis=-1, average='mean')
        Pxx = Pxx+Pxx_temp
    Pxx = Pxx/nave
nplots = int(ec.nrow/8)

print(f[1])

# plotting
if nplots==0:
    for ii in range(ec.nrow):
    #Pxx, freqs = plt.psd(data[0,ii,:,1],NFFT=nfft,Fs=ec.sample_rate,detrend=None,window=None) # I am more familiar with this function, but it automatically plots
    # plt.close()
        plt.loglog(f,Pxx[ii,:]**0.5/2**16,label=ii)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FFT (V/$\sqrt{Hz}$)')
    plt.legend()
    plt.show()
else:
    for ii in range(nplots):
        plt.figure(ii)
        for jj in range(8):
            if jj+8*ii > ec.nrow-1:
                break
            plt.loglog(f,Pxx[jj+8*ii,:]**0.5/2**16,label=int(jj+8*ii))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('FFT (V/$\sqrt{Hz}$)')
        plt.legend()
    plt.show()


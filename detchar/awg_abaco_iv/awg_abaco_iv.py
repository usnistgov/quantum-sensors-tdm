#!/usr/bin/env python3

import sys, time
import abaco_util
import util
import numpy as np
import pylab as plt
import awg
import qsghw.instruments.lakeshore370_ser as lakeshore370_ser

plt.ion()
plt.close("all")


srate =  2000e1 # rate at which the awg steps thru the profile
profile0 = awg.make_ramp_dwell_ramp_profile(n_ramp=120, n_dwell=20, blip_delta=0.0)
profile_extra = [1, 1, 1, 1, 1, 1, 1, 1, 0.125, 0.125, 0.125, 0.125, 0.135, 0.125, 0.145, 0.125, 0.155, 0.125, 0.165, 0.125, 0.175, 0.125, 0.185, 0.125, 0.195, 0.125, 0.205, 0.125, 0.215, 0.125, 0.225, 0.125, 0.235, 0.125, 0, 0,0,0]
profile = profile0
ramp_extreme = 3 # voltage scale to multiply profile by
phi0_fb = 1024
col = 0 # easyClient treast dastard as having 1 columns and nchan rows
ls = lakeshore370_ser.Lakeshore370_ser(port="/dev/ttyUSB0")


awg.ch1_setup(profile=profile, srate=srate)
awg.ch2_setup()

def get_atan1():
        tstart_unix_nano = awg.ch1_trigger(ramp_extreme=ramp_extreme)
        alldata = abaco_util.getdata(toread = 1024*1024*100, tosleep_s=0.12)
        #get sync signal and unwrap + convert atan
        sync = alldata.data['syncbus'].astype(int) & 32
        sync_in = alldata.data['syncbus'].astype(int) & 16 == 16
        sync_ind = np.argmax(sync_in[:,0])
        atan1_0 = np.unwrap(2*np.pi*alldata.data['scal2']/2**32, axis=0)
        print(f"{atan1_0.shape=} {sync_ind=}")
                
        atan1 = atan1_0[sync_ind:,:]
        return atan1
def find_ic_by_max_before_crazy(a):
        d = np.diff(a)
        crazy_ind = np.argmax(np.abs(d)) # find where we're slipping phi0
        ic_ind = np.argmax(a[:crazy_ind]) # look for the peak before slipping phi0
        return ic_ind

def find_all_ics_by_max_before_crazy(atan1):
        n = atan1.shape[1]
        ic_inds = np.zeros(n, dtype="int64")
        for j in range(n):
                ic_inds[j] = find_ic_by_max_before_crazy(atan1[:,j])
        return ic_inds

atan1 = get_atan1()

ramp_start_ind = 258
ic_inds = np.argmax(atan1, axis=0)
ic_inds = find_all_ics_by_max_before_crazy(atan1[:,:])
ch=0
plt.figure()
# plt.plot(np.diff(atan1[sync_ind:,:],axis=1))
plt.plot(atan1)
plt.plot(ramp_start_ind*np.ones(atan1.shape[1]),atan1[ramp_start_ind,:],"o")
for j in range(len(ic_inds)):
        ic_ind = ic_inds[j]
        plt.plot(ic_ind, atan1[ic_ind, j], "o")

plt.ylabel(f"atan1")
plt.xlabel("sample number")


# a = atan1[:,5]
# ic_ind = find_ic_by_max_before_crazy(a)

# plt.figure()
# plt.plot(a)
# # plt.plot(crazy_ind, a[crazy_ind],"x")
# plt.plot(ic_ind, a[ic_ind],"o")

def ic_ind_to_ic_amp(ic_ind, a, amps_per_unit):
        # a[0] must be in flat part
        delta = a[ic_ind]-a[0]
        ic_amps = delta*amps_per_unit
        return ic_amps

def ic_inds_to_ic_amps(ic_inds, atan1, amps_per_unit):
        n = len(ic_inds)
        ic_amps = np.zeros(n)
        for j in range(n):
                ic_amps[j] = ic_ind_to_ic_amp(ic_inds[j], atan1[:,j], amps_per_unit)
        return ic_amps

ic_amps = ic_inds_to_ic_amps(ic_inds, atan1, amps_per_unit=2.207e-7)





# plt.figure()
# plt.plot(sync_in[sync_ind:, ch])
# plt.ylabel(f"sync_in ch {ch}")
# plt.xlabel("sample number")


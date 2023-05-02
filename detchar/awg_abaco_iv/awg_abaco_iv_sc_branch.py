#!/usr/bin/env python3

import sys, time
import abaco_util
# import util
import numpy as np
import pylab as plt
import awg
import qsghw.instruments.lakeshore370_ser as lakeshore370_ser
import h5py

plt.ion()
plt.close("all")


srate =  33e1 # rate at which the awg steps thru the profile
profile0 = awg.make_ramp_dwell_ramp_profile(n_ramp=100, n_dwell=25, blip_delta=0.0)
profile_extra = awg.make_pulse_profile(baseline=0.125, peaks=[1, 0.125, 0.125, 0.125, 0.127, 0.129, 0.129, 0.129, 0.129, 0.129, 0.129, 0.13, 0.135, 0.14, 0.15, 0.2, 0.25, 0.3], n_dwell=10)
# profile = np.array(list(profile0)+profile_extra)
ramp_extreme = 10 # voltage scale to multiply profile by
phi0_fb = 1024
col = 0 # easyClient treast dastard as having 1 columns and nchan rows
ls = lakeshore370_ser.Lakeshore370_ser(port="/dev/ttyUSB0")


awg.ch1_setup(profile=profile0, srate=srate)
awg.ch2_setup()
awg.ch2_setvolt(0)
# awg.ch1_set_ramp_extreme(10)
# time.sleep(5)
def getdata(profile, ramp_extreme):
        awg.ch1_setup(profile=profile, srate=srate)
        awg.ch1_set_ramp_extreme(ramp_extreme)
        time.sleep(2)
        for i in range(5):
                alldata = abaco_util.getdata(toread = 1024*1024*300, tosleep_s=0.12)
                if len(alldata.data) >= 150000:
                        break
                if i ==4:
                        raise Exception("too short data")
                else:
                        print(f"alldata.data.shape {alldata.data.shape}")
        sync_in = (alldata.data['syncbus'][:,6].astype(int) & 8 == 8).astype(int)
        phase = np.unwrap(alldata.data['scal2']/2**32, axis=0, period=1)
        # plt.figure()
        # plt.plot(phase[:,6], label="atan1_0 ch7")
        # plt.plot(sync_in*100, label="sync_in")
        # plt.grid()
        # plt.legend()
        # plt.pause(30)
        print(f"len(sync_in)={len(sync_in)}")
        ind_edge_up = np.where(np.diff(sync_in)==1)[0][0]
        ind_edge_down = np.where(np.diff(sync_in)==-1)[0][0]
        d = max(ind_edge_up, ind_edge_down) - min(ind_edge_up, ind_edge_down)
        print(f"2*d {2*d}")
        return sync_in, phase, ind_edge_up, ind_edge_down, d, profile, ramp_extreme

def makeplot(sync_in, phase, ind_edge_up, ind_edge_down, d, profile, ramp_extreme):
        plt.figure()
        plt.plot(phase[:,6], label="atan1_0 ch7")
        plt.plot(sync_in*100, label="sync_in")
        plt.plot(np.linspace(0,2*d, len(profile)), 100*profile,label="profile")
        plt.plot(ind_edge_up, 100,"o")
        plt.plot(ind_edge_down, 100, "o")
        plt.grid()
        plt.legend()

def get_temp_info():
        temp_k = ls.getTemperature(5)
        setpoint = ls.getTemperatureSetPoint()
        hout = ls.getHeaterOut()
        timestamp = time.time()
        return temp_k, setpoint, hout, timestamp

def set_and_settle(setpoint_mk):
        setpoint_k = setpoint_mk/1000
        ls.setTemperatureSetPoint(setpoint_k)
        tstart = time.time()
        timeout = 3600
        while time.time()-tstart < timeout:
                elapsed_s = time.time()-tstart 
                temp_k = ls.getTemperature(5)
                print(f"elapsed {elapsed_s:0.2f} s, setpoint {setpoint_mk:0.2f} mk, temp {temp_k*1000:0.2f} mK, diff {(temp_k-setpoint_k)*1000:0.2f} mK")
                if np.abs(temp_k-setpoint_k) < 0.03e-3:
                        print("break temp settle")
                        break
                plt.pause(10)
        n = 10
        for i in range(n):
                elapsed_s = time.time()-tstart 
                temp_k = ls.getTemperature(5)
                print(f"extra settle {i+1}/{n}")
                print(f"elapsed {elapsed_s:0.2f} s, setpoint {setpoint_mk:0.2f} mk, temp {temp_k*1000:0.2f} mK, diff {(temp_k-setpoint_k)*1000:0.2f} mK")
                plt.pause(5)
        
def take_data_and_to_group(g, profile, ramp_extreme, plot=True):
        temp_k_before, setpoint_before, hout_before, timestamp_before = get_temp_info()
        sync_in, phase, ind_edge_up, ind_edge_down, d, profile, ramp_extreme = getdata(profile, ramp_extreme)
        temp_k, setpoint, hout, timestamp = get_temp_info()
        g["temp_k_before"] = temp_k_before
        g["setpoint_before"] = setpoint_before
        g["hout_before"] = hout_before
        g["timestamp_before"] = timestamp_before
        g["sync_in"] = sync_in
        g["phase"] = phase
        g["temp_k_after"] = temp_k
        g["setpoint_after"] = setpoint
        g["hout_after"] = hout
        g["timestamp_after"] = timestamp
        g["profile"] = profile
        g["ramp_extreme"] = ramp_extreme
        g["srate"] = srate

        if plot:
                makeplot(sync_in, phase, ind_edge_up, ind_edge_down, d, profile, ramp_extreme)
                plt.title(f"setpoint {setpoint*1000} mK")

temp_k, setpoint, hout, timestamp = get_temp_info()
sync_in, phase, ind_edge_up, ind_edge_down, d, profile, ramp_extreme = getdata(profile0, ramp_extreme)
makeplot(sync_in, phase, ind_edge_up, ind_edge_down, d, profile, ramp_extreme)
# plt.pause(20)


datafilename = "galens_iv_data_sc_branch.h5"
h5 = h5py.File(datafilename,"w")

g = h5.create_group("test_iv")
take_data_and_to_group(g, profile0, 10)
g = h5.create_group("test_pulsey")
take_data_and_to_group(g, profile_extra, 10)

i=0
temps_mk = [25]
for j in range(3):
        for temp_mk in temps_mk:
                set_and_settle(temp_mk)
                plt.close("all")
                for q in range(4):
                        plt.pause(q*30) #pause longer and longer each time, want to see how long
                        # we need to really settle
                        print(f"j {j}, temp_mk {temp_mk}, q {q}")
                        i+=1
                        g=h5.create_group(f"{i}")
                        take_data_and_to_group(g, profile0, ramp_extreme)
                        print(f"j {j}, temp_mk {temp_mk}, q {q}")
                        g=h5.create_group(f"{i}p")
                        take_data_and_to_group(g, profile_extra, ramp_extreme)
                        print(f"j {j}, temp_mk {temp_mk}, q {q}")



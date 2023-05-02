import awg
import pylab as plt
import numpy as np
from dataclasses import dataclass
from typing import Any
import time
import pickle
import qsghw.instruments.lakeshore370_ser as lakeshore370_ser
import os



@dataclass
class TempInfo:
    temp_before_k: float
    temp_after_k: float
    temp_setpoint_k: float
    heater_out_before: float
    heater_out_after: float

def get_temp_k():
    for i in range(3): # it was returning old data somehow
        t = ls.getTemperature(5)
    return t

def get_setpoint_hout():
    for i in range(3):
        setpoint = ls.getTemperatureSetPoint()
    for i in range(3):
        hout = ls.getHeaterOut()
    return setpoint, hout



# setup function generator with both ouputs at zero, and ch1
# ready to ramp upon trigger
# awg.ch2_setup()
# awg.ch1_setup(profile=profile, srate=srate)
# awg.ch1_set_ramp_extreme(ramp_extreme)
# _ = awg.ch1_trigger()
# time.sleep(profile_duration_s+0.1)

# setup readout and get fba offsets

@dataclass
class RawAWGRowIVData:
    row: int
    ramp_extreme: float
    profile: Any
    profile_period_s: float
    phi0_fb: float
    fb_sample_period_s: float
    # fb_ic_fixed: Any
    fb: Any
    tempinfo: TempInfo

    def get_ic_ind(self):
        fb_unwrapped = self.get_fb_unwrapped()
        ic_ind = np.argmax(np.sign(self.ramp_extreme)*fb_unwrapped[:len(fb_unwrapped)//2])
        return ic_ind
    
    def get_fb_unwrapped(self):
        fb_unwrapped = awg.add_flux_jumps(self.fb, phi0_fb = self.phi0_fb, 
        fb_step_threshold=int(self.phi0_fb*0.95))
        return fb_unwrapped

    def get_fb_ic_fixed(self):
        ic_ind = self.get_ic_ind()
        fb_unwrapped = self.get_fb_unwrapped()
        fb_ic_fixed = fb_unwrapped[:]-fb_unwrapped[0]
        fb_ic_fixed[ic_ind:]-=fb_ic_fixed[-1]
        # ic_fb_units = fb_ic_fixed[ic_ind]
        return fb_ic_fixed

    def get_t_fb(self):
        t_fb=np.arange(len(self.fb))*self.fb_sample_period_s
        return t_fb

    def get_t_profile(self):
        t_profile = np.arange(len(self.profile))*self.profile_period_s
        return t_profile

    def plot(self, t_profile_offset_s=0.25):
        t_fb = self.get_t_fb()
        t_profile = self.get_t_profile()
        fb_ic_fixed = self.get_fb_ic_fixed()
        ic_ind = self.get_ic_ind()
        plt.figure()
        plt.plot(t_fb, self.fb, label="fb raw")
        plt.plot(t_fb[:-1], self.get_fb_unwrapped(), label="fb_unwrapped")
        plt.plot(t_fb[:-1], fb_ic_fixed, label="fb_ic_fixed")
        plt.plot(t_fb[ic_ind], fb_ic_fixed[ic_ind],"o", label="ic_ind")
        plt.plot(t_profile+t_profile_offset_s, self.profile*1e4*self.ramp_extreme, label="profile/0.1 mV")
        plt.grid(True)
        plt.legend()

def getIVs(profile, ramp_extreme, srate):
    awg.ch1_setup(profile=profile, srate=srate)
    awg.ch1_set_ramp_extreme(ramp_extreme)
    profile_duration_s = len(profile)/srate
    num_sample_needed = 1.1*(profile_duration_s)//awg.c.samplePeriod
    temp_before_k = get_temp_k()
    setpoint_before, heater_out_before = get_setpoint_hout()
    time_nano_after_trigger = awg.ch1_trigger()
    data = awg.c.getNewData(-0.001,minimumNumPoints=num_sample_needed)
    time_nano_first_sample = awg.c._lastGetNewDataFirstTriggerUnixNano-awg.c.nPresamples*awg.c.samplePeriod*1e9
    temp_after_k = get_temp_k()
    setpoint_after, heater_out_after = get_setpoint_hout()
    assert setpoint_after == setpoint_before
    tempinfo = TempInfo(temp_before_k, temp_after_k, setpoint_after,
    heater_out_before, heater_out_after)

    iv_datas = []
    for row in range(awg.c.numRows):
        fb = data[col, row, :, 1]
        # fb_offset = fba_offsets[col,row]
        fb_fixed = awg.add_flux_jumps(fb, phi0_fb = phi0_fb, 
        fb_step_threshold=int(phi0_fb*0.95))

        t_ms = awg.c.samplePeriod*np.arange(len(fb))*1000
        t_ms_first_trigger_after_first_sample = (time_nano_after_trigger-time_nano_first_sample)*1e-6
            
        vbias = profile*ramp_extreme

        # calculate iv vs vbias
        iv = RawAWGRowIVData(row, ramp_extreme, profile, 1/srate,
                                        phi0_fb, awg.c.samplePeriod,
                                        fb, tempinfo)
        iv_datas.append(iv)
    return iv_datas

def getIVs_retry(profile, ramp_extreme, srate):
    nmax = 3
    for i in range(nmax):
        try:
            if i != nmax-1:
                print(f"retry after {i+1} failure")
            iv_datas = getIVs(profile, ramp_extreme, srate)
            return iv_datas
        except:
            continue
    raise Exception("failed 3 times in a row! :(")


d = "./data"
temp_setpoint_mK = int(ls.getTemperatureSetPoint()*1e3)


# getting IVs
ivs = getIVs_retry(profile, ramp_extreme, srate)
with open(os.path.join(d, f"ivs_pos_{temp_setpoint_mK}mK.pkl"), "wb") as f:
    pickle.dump(ivs, f)
iv = ivs[5]
iv.plot()
del ivs

ivs_neg = getIVs_retry(profile, -ramp_extreme, srate)
with open(os.path.join(d, f"ivs_neg_{temp_setpoint_mK}mK.pkl"), "wb") as f:
    pickle.dump(ivs_neg, f)
iv2 = ivs_neg[5]
iv2.plot()
del ivs_neg

ivs_pulsey = getIVs_retry(pulsey_profile, ramp_extreme, srate)
with open(os.path.join(d, f"ivs_pulsey_{temp_setpoint_mK}mK.pkl"), "wb") as f:
    pickle.dump(ivs_pulsey, f)
iv3 = ivs_pulsey[5]
iv3.plot()
del ivs_pulsey


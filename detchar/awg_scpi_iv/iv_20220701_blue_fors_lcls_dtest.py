import awg
import pylab as plt
import numpy as np
from dataclasses import dataclass
from typing import Any
import time
import pickle

plt.ion()
plt.close("all")

srate =  5e1 # rate at which the awg steps thru the profile
profile0 = awg.make_ramp_dwell_ramp_profile(n_ramp=120, n_dwell=20, blip_delta=0.1)
profile_extra = [1, 1, 1, 1, 1, 1, 1, 1, 0.125, 0.125, 0.125, 0.125, 0.135, 0.125, 0.145, 0.125, 0.155, 0.125, 0.165, 0.125, 0.175, 0.125, 0.185, 0.125, 0.195, 0.125, 0.205, 0.125, 0.215, 0.125, 0.225, 0.125, 0.235, 0.125, 0, 0,0,0]
profile = profile0
ramp_extreme = -1.2 # voltage scale to multiply profile by
phi0_fb = 1024
col = 0 # easyClient treast dastard as having 1 columns and nchan rows


profile_duration_s = len(profile)/srate
num_sample_needed = 1.1*(profile_duration_s)//awg.c.samplePeriod

# setup function generator with both ouputs at zero, and ch1
# ready to ramp upon trigger
awg.ch2_setup()
awg.ch1_setup(profile=profile, srate=srate)
awg.ch1_set_ramp_extreme(ramp_extreme)
_ = awg.ch1_trigger()
time.sleep(profile_duration_s+0.1)

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

    def get_ic_ind(self):
        fb_unwrapped = self.get_fb_unwrapped()
        ic_ind = np.argmax(np.sign(ramp_extreme)*fb_unwrapped[:len(fb_unwrapped)//2])
        return ic_ind
    
    def get_fb_unwrapped(self):
        fb_unwrapped = awg.add_flux_jumps(self.fb, phi0_fb = phi0_fb, 
        fb_step_threshold=int(phi0_fb*0.95))
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
        plt.plot(t_profile+t_profile_offset_s, profile*1e4*ramp_extreme, label="profile/0.1 mV")
        plt.grid(True)
        plt.legend()

def getIVs(profile, ramp_extreme, srate):
    awg.ch1_setup(profile=profile, srate=srate)
    awg.ch1_set_ramp_extreme(ramp_extreme)
    time_nano_after_trigger = awg.ch1_trigger()
    data = awg.c.getNewData(-0.001,minimumNumPoints=num_sample_needed)
    time_nano_first_sample = awg.c._lastGetNewDataFirstTriggerUnixNano-awg.c.nPresamples*awg.c.samplePeriod*1e9

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
                                        fb)
        iv_datas.append(iv)
    return iv_datas


@dataclass
class IVvsField:
    ivs_pos: Any
    ivs_neg: Any
    v_fcs: Any

# getting IVs
ivs = getIVs(profile, ramp_extreme, srate)
iv = ivs[5]
iv.plot()
del ivs
ivs_neg = getIVs(profile, ramp_extreme, srate)
iv2 = ivs_neg[5]
iv2.plot()
del ivs_neg
raise Exception()


fb_fixed = awg.add_flux_jumps(iv.fb, phi0_fb = phi0_fb, 
fb_step_threshold=int(phi0_fb*0.95))
# lets find IC by saying is the extreme (with correct sign) in the first half
ic_ind = np.argmax(np.sign(ramp_extreme)*fb_fixed[:len(fb_fixed)//2])
fb_ic_fixed = fb_fixed[:]-fb_fixed[0]
fb_ic_fixed[ic_ind:]-=fb_ic_fixed[-1]
ic_fb_units = fb_ic_fixed[ic_ind]
t_fb=np.arange(len(fb_fixed))*awg.c.samplePeriod
t_profile = np.arange(len(profile))*1/srate
plt.figure()
plt.plot(t_fb, iv.fb[:-1], label="fb")
plt.plot(t_fb, fb_fixed, label="fb_fixed")
plt.plot(t_fb, fb_ic_fixed, label="fb_ic_fixed")
plt.plot(t_fb[ic_ind], fb_ic_fixed[ic_ind],"o", label="ic_ind")
plt.plot(t_profile+0.25, profile*1e4*ramp_extreme, label="profile/0.1 mV")
plt.grid(True)
plt.legend()

ivs = []
ivs_neg = []
v_fcs = np.linspace(-4,4,15)
# v_fcs = np.array([0.5, 0, -0.5])
for v_fc in v_fcs:
    print(v_fc)
    awg.ch2_setvolt(v_fc)
    time.sleep(.5)
    awg.ch1_set_ramp_extreme(ramp_extreme)
    for i in range(3):
        try:
            ivs.append(getIVs())
            break
        except AssertionError as ex:
            print("IV retry")
            pass
    awg.ch1_set_ramp_extreme(-ramp_extreme)
    for i in range(3):
        try:
            ivs_neg.append(getIVs())
            break
        except AssertionError as ex:
            print("IV retry")
            pass

ivs_vs_field = IVvsField(ivs, ivs_neg, v_fcs)

ic_row_vfc = np.zeros((awg.c.numRows, len(v_fcs)))
ic_row_vfc_neg = np.zeros((awg.c.numRows, len(v_fcs)))
for row in range(awg.c.numRows):
    for i in range(len(v_fcs)):
        ic_row_vfc[row, i] = ivs[i][row].ic_fb_units
        ic_row_vfc_neg[row, i] = ivs_neg[i][row].ic_fb_units

with open("last_ivs_vs_field.pkl", "wb") as f:
    pickle.dump(ivs_vs_field, f)

plt.figure()
cm = plt.get_cmap("cool",len(ivs[0]))
for row in range(awg.c.numRows):
    lines=plt.plot(v_fcs, ic_row_vfc[row,:], color=cm(row))
    lines_neg=plt.plot(v_fcs, ic_row_vfc_neg[row, :], "--", color=cm(row))
plt.xlabel("V field coil (V)")
plt.ylabel("ic fb units")
plt.legend()
plt.grid()
plt.tight_layout()
plt.legend([lines[0], lines_neg[0]], ["positive detector bias", "negative detector bias"])

# ch2_volt = 0
# awg.ch2_setvolt(ch2_volt)
# ivs = getIVs()
# plt.figure()
# for row in range(awg.c.numRows):
#     plt.plot(ivs[row].vbias, ivs[row].fb_sampled, label=f"row{row:d}")
# plt.legend()
# plt.title(f"ch2_volt={ch2_volt}")
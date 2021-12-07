import awg
import pylab as plt
import numpy as np
from dataclasses import dataclass
from typing import Any
import time

plt.ion()
plt.close("all")

srate =  1e2 # rate at which the awg steps thru the profile
profile = awg.make_ramp_dwell_ramp_profile(n_ramp=150, n_dwell=20, blip_delta=0.00)
ramp_extreme = -2 # voltage scale to multiply profile by
col=0
row=1
phi0_fb = 1754
flux_jump_threshold_dac_units = phi0_fb


profile_duration_s = len(profile)/srate
num_sample_needed = (profile_duration_s)//awg.c.samplePeriod

# setup function generator with both ouputs at zero, and ch1
# ready to ramp upon trigger
awg.ch2_setup()
awg.ch1_setup(profile=profile, srate=srate)
awg.ch1_set_ramp_extreme(ramp_extreme)
_ = awg.ch1_trigger()
time.sleep(profile_duration_s+0.1)

# setup readout and get fba offsets
awg.set_arl_params(flux_jump_threshold_dac_units)
fba_offsets = awg.get_fba_offsets()

@dataclass
class RawAWGRowIVData:
    row: int
    ic_fb_units: float
    vbias: Any
    # fb_ic_fixed: Any
    # fb: Any
    # err: Any
    

def getIVs():
    time_nano_after_trigger = awg.ch1_trigger()
    data = awg.c.getNewData(-0.001,minimumNumPoints=num_sample_needed)
    time_nano_first_sample = awg.c._lastGetNewDataFirstTriggerUnixNano-awg.c.nPresamples*awg.c.samplePeriod*1e9

    iv_datas = []
    for row in range(awg.c.numRows):
        fb = data[col, row, :, 1]
        err = data[col, row, :, 0]
        fb_offset = fba_offsets[col,row]
        fb_fixed = awg.add_flux_jumps(fb, phi0_fb = phi0_fb, 
        fb_step_threshold=int(phi0_fb*0.95))

        t_ms = awg.c.samplePeriod*np.arange(len(fb))*1000
        t_ms_first_trigger_after_first_sample = (time_nano_after_trigger-time_nano_first_sample)*1e-6

        #fix the jump due to IC
        ic_fix_inds = np.where(np.abs(err)>500)[0]
        fb_ic_fixed = fb_fixed[:]-fb_fixed[0]
        if len(ic_fix_inds)>0:
            ic_fix_ind = ic_fix_inds[0]
            fb_ic_fixed[ic_fix_ind:]-=fb_ic_fixed[-1]
            ic_ind = np.argmax(np.abs(fb_ic_fixed[:ic_fix_ind])) # limit the range we look over a bi
            ic_fb_units = fb_ic_fixed[ic_ind]
            
        vbias = profile*ramp_extreme

        # calculate iv vs vbias
        iv_datas.append(RawAWGRowIVData(row, ic_fb_units, vbias))#, fb_ic_fixed))
    return iv_datas


ivs = []
ivs_neg = []
v_fcs = np.linspace(-4,4,61)
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

ic_row_vfc = np.zeros((awg.c.numRows, len(v_fcs)))
ic_row_vfc_neg = np.zeros((awg.c.numRows, len(v_fcs)))
for row in range(awg.c.numRows):
    for i in range(len(v_fcs)):
        ic_row_vfc[row, i] = ivs[i][row].ic_fb_units
        ic_row_vfc_neg[row, i] = ivs_neg[i][row].ic_fb_units

plt.figure()
cm = plt.get_cmap("pink",len(ivs[0]))
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
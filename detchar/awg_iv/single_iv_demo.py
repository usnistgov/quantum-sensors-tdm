import pymeasure
from pymeasure import instruments
from pymeasure.instruments.agilent import Agilent33500
import numpy as np
import nasa_client
import pylab as plt
from cringe.cringe_control import CringeControl
import time
import mass
from mass.mathstat.fitting import fit_kink_model
plt.ion()
plt.close("all")


a = Agilent33500("TCPIP::A-33522B-01982.local::5025::SOCKET")
# a = Agilent33500("USB::2391::11271::MY57801982::0::INSTR")



def ch2_output_enable(enable_bool):
    a.write(f"OUTP2 {enable_bool:d}")

def ch2_setup():
    a.write(f"SOUR2:FUNC DC")
    a.write(f"SOUR2:VOLT:OFFS 0")
    a.write(f"OUTP2:LOAD INF")
    ch2_output_enable(True)

def ch2_setvolt(v):
    a.write(f"SOUR2:VOLT:OFFS {v:f}")

def ch1_set_ramp_extreme(extreme):
    if extreme>=0:
        a.arb_file = "iv_ramp"
    else:
        a.arb_file = "iv_ramp_neg"
    a.amplitude = np.abs(extreme)


def ch1_ramp_setup(profile):
    a.data_volatile_clear()         # Clear volatile internal memory
    a.data_arb(                     # Send data points of arbitrary waveform
        'iv_ramp',
        profile,          # the wave form
        data_format='float'              # float format takes values from -1 to 1
    )
    a.data_arb(                     # Send data points of arbitrary waveform
        'iv_ramp_neg',
        -profile,          # the wave form
        data_format='float'              # float format takes values from -1 to 1
    )
    a.arb_file = 'iv_ramp'             # Select the transmitted waveform 'test'


def ch1_setup(profile, srate):
    a.write("OUTP2:POL NORM")
    a.write("FUNC:ARB:FILT OFF")
    a.shape = "ARB"
    a.arb_srate = srate
    a.output = True
    a.output_load = "INF"
    a.burst_state = True
    a.busrt_ncycles = 1
    a.trigger_source ="BUS"
    ch1_ramp_setup(profile)
    ch1_set_ramp_extreme(.5)


def ch1_trigger():
    time_nano = time.time()*1e9
    a.trigger()
    return time_nano

def make_ramp_dwell_ramp_profile(n_ramp, n_dwell, blip_delta):
    _ramp = np.linspace(0,1,n_ramp)
    dwell = np.ones(n_dwell)
    dwell[len(dwell)//2]=1-blip_delta # blipfor alignment
    start_dwell = np.zeros(n_dwell)
    profile = np.hstack([start_dwell, _ramp, dwell, _ramp[::-1], start_dwell])
    return profile

def add_flux_jumps(fb, phi0_fb, fb_step_threshold):
    """return an array like fb, but with flux jumps resolved and one value removed from the end
    method: find indicies with np.diff(fb) greater ro less than fb_step_threshold
    and add phi0 units with the correct sign"""
    out = fb[1:]+np.cumsum(np.diff(fb)<-fb_step_threshold)*phi0_fb
    out -= np.cumsum(np.diff(fb)>fb_step_threshold)*phi0_fb
    return out

ch2_setup()
ch2_setvolt(-0.5)
n_dwell=10
profile = make_ramp_dwell_ramp_profile(n_ramp=300, n_dwell=n_dwell, blip_delta=0.00)
srate =  1e2
ch1_setup(profile=profile, srate=srate)
ramp_extreme = -2
ch1_set_ramp_extreme(-2)
# print(f"any errors? {a.check_errors()}")
c = nasa_client.EasyClient()


cc = CringeControl()
reply, fba_offsets = cc.get_fba_offsets()
flux_jump_threshold_dac_units = 1754
cc.set_arl_params(flux_jump_threshold_dac_units=flux_jump_threshold_dac_units,
    plus_event_reset_delay_frm_units=0, minus_event_reset_delay_frm_units=0)
print("wait 1 s for cringe timer")
time.sleep(1) # wait out cringe timer
print("cringe timer done")
time_nano_after_trigger = ch1_trigger()

data = c.getNewData(minimumNumPoints=2200000)
time_nano_first_sample = c._lastGetNewDataFirstTriggerUnixNano-c.nPresamples*c.samplePeriod*1e9
col=0
row=1
fb = data[col, row, :, 1]
err = data[col, row, :, 0]
fb_offset = fba_offsets[col,row]
fb_fixed = add_flux_jumps(fb, phi0_fb = 1754, fb_step_threshold=1300)

t_ms = c.samplePeriod*np.arange(len(fb))*1000
t_ms_first_trigger_after_first_sample = (time_nano_after_trigger-time_nano_first_sample)*1e-6


plt.figure(figsize=(12,8))
ax1=plt.subplot(211)
plt.plot(t_ms,fb, label="raw")
plt.plot(t_ms[:-1],fb_fixed, label="fixed")
plt.axhline(fb_offset, color="cyan")
plt.axhline(fb_offset+flux_jump_threshold_dac_units, color="cyan")
plt.axhline(fb_offset-flux_jump_threshold_dac_units, color="cyan")
plt.axvline(t_ms_first_trigger_after_first_sample, color="red", label="profile start")
plt.xlabel("t_ms")
plt.ylabel("fb")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.subplot(212, sharex=ax1)
plt.plot(t_ms,err)
plt.grid(True)
plt.xlabel("t_ms")
plt.ylabel("err")
plt.tight_layout()

#fix the jump due to IC
ic_fix_ind = np.where(np.abs(err)>500)[0][0]
fb_ic_fixed = fb_fixed[:]-fb_fixed[0]
fb_ic_fixed[ic_fix_ind:]-=fb_ic_fixed[-1]
ic_ind = np.argmax(np.abs(fb_ic_fixed[:ic_fix_ind])) # limit the range we look over a bi
ic_fb_units = fb_ic_fixed[ic_ind]



# find where triangle starts
offset_ind_from_timing = (t_ms_first_trigger_after_first_sample*1e-3/c.samplePeriod)
offset_ind = int(offset_ind_from_timing+0e-3/c.samplePeriod)
# the timing method isnt reliable, lets look for the first kink
def find_kink_approx(fb, delta_threshold = 1000, pointstep = 10):
    for i in range(10, len(fb)):
        a = fb[i]-fb[i-pointstep]
        b = fb[pointstep] - fb[i]
        if np.abs(b)-np.abs(a) > delta_threshold:
            break
    return i
kink_ind_approx = find_kink_approx(fb_ic_fixed)
model, (kbest,a,b,kink_c), X2 = fit_kink_model(np.arange(kink_ind_approx),
                            fb_ic_fixed[:kink_ind_approx])
kink_ind = int(kbest)


# calculate iv vs vbias
vbias = profile[n_dwell:-n_dwell]*ramp_extreme
step_period_ind = int(c.sample_rate//srate)
sample_inds = kink_ind-3+np.arange(len(vbias))*step_period_ind
fb_sampled = fb_ic_fixed[sample_inds]

plt.figure(figsize=(6,4))
plt.plot(t_ms[:-1],fb_ic_fixed)
plt.plot(t_ms[ic_ind], ic_fb_units, "o", label="ic")
plt.axvline(t_ms[ic_fix_ind], color="red", label="ic fix ind")
plt.plot(t_ms[kink_ind], fb_ic_fixed[kink_ind], "s", label="kink")
plt.plot(t_ms[sample_inds], fb_ic_fixed[sample_inds], ".", label="sample")
plt.xlabel("t_ms")
plt.ylabel("fb ic_fixed")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(6,4))
plt.plot(vbias,fb_sampled)
plt.xlabel("vbias/V")
plt.ylabel("fb_ic_fixed")
plt.grid(True)
plt.legend()
plt.tight_layout()

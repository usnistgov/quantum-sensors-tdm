import pymeasure
from pymeasure import instruments
from pymeasure.instruments.agilent import Agilent33500
import numpy as np
import nasa_client
import pylab as plt
from cringe.cringe_control import CringeControl
import time
plt.ion()


a = Agilent33500("TCPIP::A-33522B-01982.local::inst0::INSTR")

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


def ch1_setup(profile):
    a.write("OUTP2:POL NORM")
    a.shape = "ARB"
    a.arb_srate = 1e4
    a.output = True
    a.output_load = "INF"
    a.burst_state = True
    a.busrt_ncycles = 1
    a.trigger_source ="BUS"
    ch1_ramp_setup(profile)
    ch1_set_ramp_extreme(.5)


def ch1_trigger():
    a.trigger()

def make_ramp_dwell_ramp_profile(n_ramp, n_dwell):
    _ramp = np.linspace(0,1,n_ramp)
    dwell = np.ones(n_dwell)
    profile = np.hstack([_ramp, dwell, _ramp[::-1]])
    return profile

def add_flux_jumps(fb, phi0_fb, fb_step_threshold):
    """return an array like fb, but with flux jumps resolved and one value removed from the end
    method: find indicies with np.diff(fb) greater ro less than fb_step_threshold
    and add phi0 units with the correct sign"""
    out = fb[:-1]+np.cumsum(np.diff(fb)<-fb_step_threshold)*phi0_fb
    out -= np.cumsum(np.diff(fb)>fb_step_threshold)*phi0_fb
    return out

ch2_setup()
ch2_setvolt(.1)
profile = make_ramp_dwell_ramp_profile(n_ramp=15000, n_dwell=500)
ch1_setup(profile)
ch1_set_ramp_extreme(-2)
print(f"any errors? {a.check_errors()}")
c = nasa_client.EasyClient()


cc = CringeControl()
reply, fba_offsets = cc.get_fba_offsets()
flux_jump_threshold_dac_units = 1350
cc.set_arl_params(flux_jump_threshold_dac_units=flux_jump_threshold_dac_units,
    plus_event_reset_delay_frm_units=0, minus_event_reset_delay_frm_units=0)
print("wait 1 s for cringe timer")
time.sleep(1) # wait out cringe timer
print("cringe timer done")
ch1_trigger()

data = c.getNewData(-0.001, minimumNumPoints=2200000)
col=0
row=0
fb = data[col, row, :, 1]
err = data[col, row, :, 0]
fb_offset = fba_offsets[col,row]
fb_fixed = add_flux_jumps(fb, phi0_fb = 1350, fb_step_threshold=1300)

t_ms = c.samplePeriod*np.arange(len(fb))*1000

plt.figure(figsize=(12,8))
ax1=plt.subplot(211)
plt.plot(t_ms,fb, label="raw")
plt.plot(t_ms[:-1],fb_fixed, label="fixed")
plt.axhline(fb_offset, color="cyan")
plt.axhline(fb_offset+flux_jump_threshold_dac_units, color="cyan")
plt.axhline(fb_offset-flux_jump_threshold_dac_units, color="cyan")
plt.xlabel("t_ms")
plt.ylabel("fb")
plt.legend()
plt.grid(True)
plt.subplot(212, sharex=ax1)
plt.plot(t_ms,err)
plt.grid(True)
plt.xlabel("t_ms")
plt.ylabel("err")
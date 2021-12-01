import pymeasure
from pymeasure import instruments
from pymeasure.instruments.agilent import Agilent33500
import numpy as np
import nasa_client
import pylab as plt
from cringe.cringe_control import CringeControl
import time
import sys, os


a = Agilent33500("TCPIP::A-33522B-01982.local::inst0::INSTR")
c = nasa_client.EasyClient()
c.setupAndChooseChannels()
cc = CringeControl()

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def ch2_output_enable(enable_bool):
    a.write(f"OUTP2 {enable_bool:d}")

def ch2_setvolt(v):
    a.write(f"SOUR2:VOLT:OFFS {v:f}")

def ch2_setup():
    a.write(f"SOUR2:FUNC DC")
    a.write(f"SOUR2:VOLT:OFFS 0")
    a.write(f"OUTP2:LOAD INF")
    ch2_output_enable(True)
    ch2_setvolt(0)

def ch1_set_ramp_extreme(extreme):
    if extreme>=0:
        a.arb_file = "iv_ramp"
    else:
        a.arb_file = "iv_ramp_neg"
    a.amplitude = np.abs(extreme)


def ch1_ramp_setup(profile):
    a.data_volatile_clear()         # Clear volatile internal memory
    blockPrint()
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
    enablePrint()
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

last_flux_jump_threshold_dac_units = None
def set_arl_params(flux_jump_threshold_dac_units):
    # this function is slow due to the cringe timer, so
    # we skip it if its already been done
    global last_flux_jump_threshold_dac_units
    if flux_jump_threshold_dac_units == last_flux_jump_threshold_dac_units:
        print("skpping arl set since flux_jump_threshold_dac_units == last_flux_jump_threshold_dac_units")
        return
    cc.set_arl_params(flux_jump_threshold_dac_units=flux_jump_threshold_dac_units,
        plus_event_reset_delay_frm_units=0, minus_event_reset_delay_frm_units=0)
    last_flux_jump_threshold_dac_units = flux_jump_threshold_dac_units
    print("wait 1 s for cringe timer")
    time.sleep(1) # wait out cringe timer
    print("cringe timer done")

def check_errors():
    errs = a.check_errors()
    if len(errs)>1:
        print("AWG Errors!")
        print(errs)

def get_fba_offsets():
    reply, fba_offsets = cc.get_fba_offsets()
    assert reply.startswith("ok")
    return fba_offsets

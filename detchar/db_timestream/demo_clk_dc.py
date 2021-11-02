from cringe.cringe_control import CringeControl
from adr_gui.adr_gui_control import AdrGuiControl
from nasa_client import EasyClient
import time
import numpy as np
import pylab as plt
import progress.bar
import os
from detchar.iv_data import (
    IVCurveColumnData,
    IVTempSweepData,
    IVColdloadSweepData,
    IVCircuit,
)
from instruments import BlueBox
import detchar
plt.ion()

def add_flux_jumps(fb, phi0_fb=1670, fb_step_threshold=1000):
    """return an array like fb, but with flux jumps resolved and one value removed from the end
    method: find indicies with np.diff(fb) greater ro less than fb_step_threshold
    and add phi0 units with the correct sign"""
    out = fb[:-1]+np.cumsum(np.diff(fb)<-fb_step_threshold)*phi0_fb
    out -= np.cumsum(np.diff(fb)>fb_step_threshold)*phi0_fb
    return out

# hardware ineraction
ec = EasyClient()
cc = CringeControl()
adr_gui_control = AdrGuiControl()

flux_jump_threshold_dac_units=1800
i = 40
p = 0 # just require this
nsamp = 4 # just require this, not actually sure how it enters
lookback_samples = 20
lookback_threshold = 500
reply, fba_offsets = cc.get_fba_offsets()
fb_offset = fba_offsets[0,0]

# set ARL params
cc.set_arl_params(flux_jump_threshold_dac_units=flux_jump_threshold_dac_units,
    plus_event_reset_delay_frm_units=0, minus_event_reset_delay_frm_units=0)
cc.set_fb_i(col=0, fb_i=i)
# cc.set_fb_p(col=0, fb_p=0)
time.sleep(1) # wait out cringe timer



ec.setMixToZero()


db_cardname = "DB1"
bayname = "CX"
column_number=0

time_nano_after_last_set_volt = None
# setting the bias point
def set_volt(dacvalue):
    cc.set_clk_fba_dc(dacvalue)
    global time_nano_after_last_set_volt
    time_nano_after_last_set_volt = time.time()*1e9

set_volt(20000)
time.sleep(1)
set_volt(19000)
data = ec.getNewData(delaySeconds=-.01, minimumNumPoints=50000) # collecting the data over this changing bias point
# data[col, row, frame (time), 0=err/1=fb]

def intergral_term_effect(err_scalar, i):
    return 3*i*err_scalar//400 #how should nsamp enter, this probably only works for nsamp=4

def find_arl_resets_no_lookboth(fb, err, flux_jump_threshold_dac_units, db_offset):
    """assumes the reset delay =0 so only one sample is needed
    """
    # find samples outside the bounds
    lo = db_offset-flux_jump_threshold_dac_units
    hi =  db_offset+flux_jump_threshold_dac_units
    ups = np.where(fb < lo)[0] # not sure if this should be <=
    downs = np.where(fb > hi)[0] # not sure if this should be >=
    return ups, downs

def find_arl_resets_lookboth(fb, err, flux_jump_threshold_dac_units, db_offset, lookback_samples, lookback_threshold):
    """when we change the db we often see an electrical interference blip
    that blip can cause an ARL event, but one that is not associated with a vphi slip
    so we filter them out by noting that they have very steep slopes leading up to them
    AND THEN check to see if the current settles to the same or a different value
    assuming that we let the detectors settle out, then change bias, the blip ARL event should
    only ever be able to be the first event
    """
    ups_all, downs_all = find_arl_resets_no_lookboth(fb, err, flux_jump_threshold_dac_units, db_offset)
    if len(ups_all) > 0:
        i = ups_all[0]
        if np.abs(fb[i] - fb[i-lookback_samples]) < lookback_threshold:
            ups = ups_all[:]
        else:
            # now we know we are on a noise blip... but need to check if it actually slips a phi0
            if np.abs(fb[i]-fb[i+lookback_samples]) < lookback_threshold:
                # settled back to the same vphi
                ups = ups_all
            else:
                # settled to a different vphi
                ups = ups_all[1:]
    else: 
        ups = ups_all[:]
    if len(downs_all) > 0:
        i = downs_all[0]
        if np.abs(fb[i] - fb[i-lookback_samples]) < lookback_threshold:
            downs = downs_all[:]
        else:
            # now we know we are on a noise blip... but need to check if it actually slips a phi0
            if np.abs(fb[i]-fb[i+lookback_samples]) < lookback_threshold:
                # settled back to the same vphi
                downs = downs_all
            else:
                # settled to a different vphi
                downs = downs_all[1:]
    else:
        downs = downs_all[:]

    return ups, downs
    
    ups = []
    for i in ups_all:
        if np.abs(fb[i]-fb[i-lookback_samples]) < lookback_threshold:
            ups.append(i)        
    downs = []
    for i in downs_all:
        if np.abs(fb[i]-fb[i-lookback_samples]) < lookback_threshold:
            downs.append(i) 

    return np.array(ups,dtype="int64"), np.array(downs, dtype="int64")


fb = data[0,0, :, 1]
err = data[0,0, :, 0]

ups, downs = find_arl_resets_lookboth(fb, err, flux_jump_threshold_dac_units, fb_offset,
lookback_samples=20, lookback_threshold=500)


def debug_plot(fb, err, ups, downs):
    plt.figure()
    plt.plot(fb, label="fb")
    # plt.plot(fb_ideal, label="fb ideal")
    plt.plot(err, label="err")
    plt.plot(ups, fb[ups], "o")
    plt.plot(downs, fb[downs], "s")
    plt.axhline(fb_offset, color="cyan")
    plt.axhline(fb_offset+flux_jump_threshold_dac_units, color="cyan")
    plt.axhline(fb_offset-flux_jump_threshold_dac_units, color="cyan")
    plt.show()
    plt.legend()
    plt.title("debug plot")

# debug_plot(fb, err, ups, downs)

def getpoint_ch0(dac):
    set_volt(dac)
    for i in range(3):
        try:
            data = ec.getNewData(delaySeconds=-.01, minimumNumPoints=50000) # collecting the data over this changing bias point
            break
        except:
            continue
    try:
        fb = data[0,0, :, 1]
        err = data[0,0, :, 0]
        ups, downs = find_arl_resets_lookboth(fb, err, flux_jump_threshold_dac_units, fb_offset,
        lookback_samples=20, lookback_threshold=500)
        if len(ups)>0: assert (len(fb)-ups[-1])>200
        if len(downs)>0: assert (len(fb)-downs[-1])>200
    except Exception as ex:
        plt.figure()
        plt.title("EROROROROROR")
        plt.plot(fb, label="fb")
        # plt.plot(fb_ideal, label="fb ideal")
        plt.plot(err, label="err")
        plt.axhline(fb_offset)
        plt.axhline(fb_offset+flux_jump_threshold_dac_units)
        plt.axhline(fb_offset-flux_jump_threshold_dac_units)
        plt.show()
        plt.legend()
        try:
            plt.plot(ups, fb[ups], "o")
            plt.plot(downs, fb[downs], "s")
            plt.legend()
        except:
            pass
        raise ex
    return np.mean(fb[-200:]), len(ups)-len(downs), (fb,err, ups, downs, data, ec._lastGetNewDataFirstTriggerUnixNano, time_nano_after_last_set_volt)

dacs = np.arange(20000,-1,-500)
fbs, phi0s = np.zeros_like(dacs), np.zeros_like(dacs)
debug_vars = []
for i, dac in enumerate(dacs):
    print(dac)
    v, n, debug_vars_local = getpoint_ch0(dac)
    fbs[i] = v
    phi0s[i] = n
    debug_vars.append(debug_vars_local)

def debug_point(i):
    (fb,err, ups, downs, data, nano_first_sample, nano_after_setvolt) = debug_vars[i]
    blip_loc_est_s = (nano_after_setvolt-nano_first_sample)*1e-9
    blip_loc_est_ind = blip_loc_est_s/ec.samplePeriod+5092
    debug_plot(fb, err, ups, downs)
    plt.axvline(blip_loc_est_ind, color="red")
    plt.title(f"debug plot point {i}")

plt.figure()
plt.plot(fbs, label="fbs")
plt.plot(phi0s*flux_jump_threshold_dac_units, label="phi0s")
plt.xlabel("dacs")
plt.ylabel("fb out")
plt.legend()

a=[]
b=[]
# check about sum of abs error near blip
for i in range(len(debug_vars)):
    (fb,err, ups, downs, data, nano_first_sample, nano_after_setvolt) = debug_vars[i]
    blip_loc_est_s = (nano_after_setvolt-nano_first_sample)*1e-9
    blip_loc_est_ind = int(blip_loc_est_s/ec.samplePeriod+5092)
    print(np.sum(np.abs(err[blip_loc_est_ind-5:blip_loc_est_ind+5])))
    a.append(np.sum(err[blip_loc_est_ind-5:blip_loc_est_ind+5]))

plt.figure()
plt.plot(a, label="a")
plt.plot(fbs, label="fbs")
plt.legend()
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
fb_offset = 8511
i = 40
p = 0 # just require this
nsamp = 4 # just require this, not actually sure how it enters

# set ARL params
cc.set_arl_params(flux_jump_threshold_dac_units=flux_jump_threshold_dac_units,
    plus_event_reset_delay_frm_units=2, minus_event_reset_delay_frm_units=2)
cc.set_fb_i(col=0, fb_i=i)
# cc.set_fb_p(col=0, fb_p=0)
time.sleep(1) # wait out cringe timer



ec.setMixToZero()


db_cardname = "DB1"
bayname = "CX"
column_number=0

# setting the bias point
def set_volt(dacvalue):
    cc.set_tower_channel(db_cardname, bayname, int(dacvalue))


set_volt(20000)
time.sleep(1)
set_volt(0)
data = ec.getNewData(delaySeconds=-.05, minimumNumPoints=60000) # collecting the data over this changing bias point
# data[col, row, frame (time), 0=err/1=fb]

def intergral_term_effect(err_scalar, i):
    return 3*i*err_scalar//400 #how should nsamp enter, this probably only works for nsamp=4

def find_arl_resets(fb, err, flux_jump_threshold_dac_units, db_offset):
    """assumes the reset delay =0 so only one sample is needed
    """
    # find samples outside the bounds
    lo = db_offset-flux_jump_threshold_dac_units
    hi =  db_offset+flux_jump_threshold_dac_units
    ups_all = np.where(fb < lo)[0] # not sure if this should be <=
    downs_all = np.where(fb > hi)[0] # not sure if this should be >=
    
    # then find the last point of runs of exactly 3 samples
    ups = []
    for i in range(2, len(ups_all)):
        if ups_all[i-2]+2 == ups_all[i]:
            ups.append(ups_all[i])
    downs = []
    for i in range(2, len(downs_all)):
        if downs_all[i-2]+2 == downs_all[i]:
            downs.append(downs_all[i])

    return np.array(ups,dtype="int64"), np.array(downs, dtype="int64")



fb = data[0,0, :, 1]
err = data[0,0, :, 0]

ups, downs = find_arl_resets(fb, err, flux_jump_threshold_dac_units, fb_offset)

plt.figure()
plt.plot(fb, label="fb")
# plt.plot(fb_ideal, label="fb ideal")
plt.plot(err, label="err")
plt.plot(ups, fb[ups], "o")
plt.plot(downs, fb[downs], "s")
plt.axhline(fb_offset)
plt.axhline(fb_offset+flux_jump_threshold_dac_units)
plt.axhline(fb_offset-flux_jump_threshold_dac_units)
plt.show()
plt.legend()

# plt.figure()
# plt.plot(np.diff(data[0,0,20000:20050,1]), label="fb diff")
# plt.plot(-3*data[0,0,20000:20050,0]//10, label="err")
# plt.show()
# plt.legend()

# datas=[]
# dacs = np.arange(6500,0,-500)

# for dacVal in dacs:
#     print('setting dac to {}'.format(dacVal))
#     set_volt(dacVal)
#     dataloop = ec.getNewData(delaySeconds=0, minimumNumPoints=40000)
#     datas.append(dataloop)

#     # save data (should eventually make to match json format of old iv's)
#     #np.save('iv_timestream_20210708_'+str(T)+'mK_'+bayname+'_v4.npy', data_all)
#     #np.save('iv_timestream_20210708_'+str(T)+'mK_'+bayname+'_dacVals_v4.npy', dacs)



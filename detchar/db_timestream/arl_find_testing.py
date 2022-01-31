from cringe.cringe_control import CringeControl
from adr_gui.adr_gui_control import AdrGuiControl
import time
import numpy as np
import pylab as plt


data = np.load("tempdata.npy")

# set ARL params
flux_jump_threshold_dac_units=1800
fb_offset = 8490
i = 40
p = 0 # just require this
nsamp = 4 # just require this, not actually sure how it enters

db_cardname = "DB1"
bayname = "CX"
column_number=0

def intergral_term_effect(err_scalar, i):
    return 3*i*err_scalar//400 #how should nsamp enter, this probably only works for nsamp=4

def find_arl_resets(fb, err, flux_jump_threshold_dac_units, db_offset):
    """assumes the reset delay =0 so only one sample is needed
    """
    # find samples outside the bounds
    lo = db_offset-flux_jump_threshold_dac_units
    hi =  db_offset+flux_jump_threshold_dac_units
    ups = np.where(fb < lo)[0] # not sure if this should be <=
    downs = np.where(fb > hi)[0] # not sure if this should be >=
    
    # validate those by seeing if they reset to the right location
    for up in ups:
        # first we remove the effect of the I term on the next sample
        fb_ideal_next = fb[up+1] + intergral_term_effect(err[up], i)
        print(f"up {up} is followed by fb={fb[up+1]} and fb_ideal_next={fb_ideal_next}")
        if np.abs(fb_ideal_next - db_offset)>1:
            print(f'{up} invalid!!')


    return ups, downs



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



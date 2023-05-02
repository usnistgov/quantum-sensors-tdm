import h5py
import numpy as np
import pylab as plt
plt.ion()
plt.close("all")

# h5 = h5py.File("/data/20230501galens_iv_data.h5","r")
h5 = h5py.File("20230501_2_galens_iv_data.h5","r")



g = h5["1"]
i0 = np.where(np.diff(g["sync_in"][()])==1)[0][0]
i0_2 = np.where(np.diff(g["sync_in"][()])==1)[0][1]
d = i0_2-i0
i_ic_bound = np.argmin(np.diff(g["phase"][i0:,6]))+i0
i_ic = np.argmin(g["phase"][i0:i_ic_bound,6])+i0

plt.figure()
plt.plot(g["phase"][:,6])
plt.plot(g["sync_in"][:])
plt.plot(np.diff(g["phase"][:,6]))
plt.plot(i0, 1, "o")
plt.plot(i0_2, 1, "o")
plt.plot(i_ic, g["phase"][i_ic,6],"o")

phase = g["phase"][i0:i0_2,6]
phase0 = np.mean(phase[:10])
phase0_after_ic = np.mean(phase[-10:])
phase_shift = np.round(phase0_after_ic-phase0) # round to whole phi0
phase[i_ic-i0+1:i0_2] -= phase_shift
phase -= phase0
phase *= -1
profile_V = g["profile"][()]*g["ramp_extreme"][()]
Rfb = 3960 #ohm, measured with 2 2kohm resisotrs stacked May2 1037am 2023 by gco
profile_A = profile_V/Rfb
try: 
    srate = g["srate"][()] # forgot to record in first runs
except:
    srate = 1000
timebase_abaco_s = len(profile_V)/srate/d
setpoint_mK = g["setpoint_before"][()]*1000

plt.figure()
plt.plot(np.arange(len(phase))*timebase_abaco_s,phase)
plt.plot((i_ic-i0)*timebase_abaco_s, phase[i_ic-i0],"o")
plt.plot(np.arange(len(profile_V))/srate, profile_V,".-", drawstyle="steps-post")
plt.title(f"setpoint {setpoint_mK:0.2f} mK")
plt.grid(True)

for k in g.keys():
    if g[k].shape==():
        v = g[k][()]
    else:
        v = g[k]
    print(f"{k} {v}")
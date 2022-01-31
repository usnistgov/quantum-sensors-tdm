import numpy as np
import pylab as plt

dacVals = np.load('iv_timestream_20210708_65mK_AY_dacVals_v0.npy')
fb = np.load('iv_timestream_20210708_65mK_AY_v0.npy')


#stack a bunch of the fb sections together
for i in range(100,200):
    if i == 100:
        fb_all = fb[i,:]
    else:
        fb_temp = fb[i,:]
        fb_all = np.hstack([fb_all, fb[i,:]])

plt.plot(fb_all)
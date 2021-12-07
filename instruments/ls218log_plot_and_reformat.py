import numpy as np
import pylab as plt
import os
import datetime

filename = "ls218_log_20210717_t090500.txt"
filename = os.path.join(os.path.expanduser( "~/ADRLogs"), filename)

epoch_time, t3k, t50k =np.genfromtxt(filename, delimiter=",", 
usecols=[0,1,2], invalid_raise=False, unpack=True, skip_header=1)
datetimes = [datetime.datetime.fromtimestamp(et) for et in epoch_time]

plt.ion()
fig,(ax1,ax2)=plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, 
squeeze=True, subplot_kw=None, gridspec_kw=None)
ax1.plot(datetimes[1::10], t3k[1::10], label="3K")
ax2.plot(datetimes[1::10], t50k[1::10], label="50K")
ax2.set_xlabel("time")
ax1.set_ylabel("temperature 2nd stage (K)")
ax2.set_ylabel("temperature 1st stage (K)")
ax1.grid(True)
ax2.grid(True)
# plt.suptitle(os.path.split(filename)[-1])
fig.autofmt_xdate()
plt.tight_layout()

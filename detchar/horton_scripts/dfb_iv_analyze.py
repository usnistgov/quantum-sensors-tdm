import pylab as plt
import numpy as np
from cringe.tune.analysis import conditionvphi
plt.ion()
plt.close("all")
data = np.load("20210507_SSRL_AX_56p6_DFB_IV.npy")

tri = data[0,2,:,1]
fb = data[1,1,:,1]

dwell = 2**10
o = 950
last = dwell*161



plt.plot(tri[o:last:dwell], fb[o:last:dwell], label=f"{o}")
plt.plot(tri[last+o:last:dwell], fb[last+o:last:dwell], label=f"{o}")
plt.legend()

# out = conditionvphi(tri, fb, tridwell=10, tristeps=9, tristepsize=31)
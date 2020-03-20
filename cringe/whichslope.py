import numpy as np
import pylab as plt
from tune import analysis
from tune import vphistats

fbb2_firstMinimumX = np.load("last_fbb2_firstMinimumX.npy")
lfba_modDepth = np.load("last_lfba_modDepth.npy")

fbb2_vphi = np.load("last_fbb2_vphi.npy")
tridwell,tristeps,tristepsize=2,9,20
fbb2triangle, fbb2sigsup, fbb2sigsdown = analysis.conditionvphis(fbb2_vphi[:,:,:,1], fbb2_vphi[:,:,:,0], tridwell, tristeps, tristepsize)
fbb2stats = vphistats.vPhiStats(fbb2triangle, fbb2sigsup)

sf = 0.36 # fraction of period taken up by the slope we want the vphi on
fbbshift = (fbb2stats["periodXUnits"]*sf-lfba_modDepth)/2.0

startofsweepX = fbb2stats["firstMinimumX"]+fbbshift
endofsweepX = startofsweepX+lfba_modDepth




plt.ion()
plt.close("all")
plt.figure()
plt.plot(fbb2stats["firstMinimumX"][0,:], fbb2stats["firstMinimumY"][0,:],"o")
col = 0
for row in range(2,6):
    a=np.min([startofsweepX[col,row], endofsweepX[col,row]])
    b=np.max([startofsweepX[col,row], endofsweepX[col,row]])
    plt.plot(fbb2triangle, fbb2sigsup[col,row,:])
    inds = np.logical_and(fbb2triangle<b, fbb2triangle>a)
    plt.plot(fbb2triangle[inds], fbb2sigsup[col,row,inds],lw=2)

plt.draw()
plt.show()
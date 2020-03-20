from . import analysis
from . import vphistats
import numpy as np


data = np.load("../last_last_fba_vphi.npy")
fba = data[0,0,:,1] #triangle
err = data[0,0,:,0] #signal
tridwell,tristeps,tristepsize=2,9,20
lastfbatriangle, lastfbasigsup, lastfbasigsdown = analysis.conditionvphis(data[:,:,:,1], data[:,:,:,0], tridwell, tristeps, tristepsize)
lastfbastats = vphistats.vPhiStats(fbatriangle, fbasigsup, fracFromBottom=fracFromBottom)

from cringe.tune import analysis
from cringe.tune import vphistats
import numpy as np
import pylab as plt

import os

def gen_triangle_output(nsamples, tridwell, tristeps, tristepsize):
    dwell, period, steps, stepsize = analysis.lineartriangleparams(tridwell, tristeps, tristepsize)
    assert dwell >= 1
    assert stepsize >= 0

    out = np.zeros(nsamples, dtype="int64")
    v = 0
    i = 0
    ndwell, nstep = 0, 0
    while True:
        nstep += 1
        while ndwell < dwell:
            ndwell += 1
            out[i] = v
            # print(f"{i=} {ndwell=} {nstep=} {stepsize=}")
            i += 1
            if i == len(out):
                return out
        ndwell = 0
        v += stepsize
        if nstep == steps//2:
            nstep = 0
            stepsize *= -1






def test_vphi_analysis_steps():
    tridwell, tristeps, tristepsize = 2, 9, 30
    lindwell, period, linsteps, linstepsize = analysis.lineartriangleparams(tridwell, tristeps, tristepsize)
    print(f"{lindwell=}, {period=}, {linsteps=}, {linstepsize=}")


    fba = np.reshape(gen_triangle_output(2**14, tridwell, tristeps, tristepsize), (1,1,-1))
    err = np.reshape(np.sin((fba-1000)*2*np.pi/1022.2), fba.shape)

    # plt.close("all")
    # plt.figure()
    # plt.subplot(311)
    # plt.xlabel("sample num")
    # plt.ylabel("fb")
    # plt.plot(fba[0,0,:],".")
    # plt.subplot(312)
    # plt.plot(fba[0,0,:], err[0,0,:],".")
    # plt.xlabel("fb")
    # plt.ylabel("err")
    # plt.subplot(313)
    # plt.plot(err[0,0,:],".")
    # plt.xlabel("sample num")
    # plt.ylabel("err")
    # plt.tight_layout()
    # plt.show()

    triout, sigup, sigdown = analysis.conditionvphi(fba[0, 0, :], err[0, 0, :], tridwell, tristeps, tristepsize)

    fbatriangle, fbasigsup, fbasigsdown = analysis.conditionvphis(fba, err, tridwell, tristeps, tristepsize)
    assert len(fbatriangle) == fbasigsup.shape[2]
    assert len(fbatriangle) == linsteps//2-1
    fbastats = vphistats.vPhiStatsSingle(fbatriangle, fbasigsup[0,0,:], fracFromBottom=.15)
    (periodInds, periodXUnits, positiveCrossingSlope, negativeCrossingSlope, positiveCrossingFirstX, negativeCrossingFirstX,
    firstMinimumInd, firstMinimumX, firstMinimumY, modDepth, midPoint, crossingPoint, firstMaximumInd, firstMaximumX, firstMaximumY) = fbastats
    assert np.abs(periodXUnits-1022.2)<0.01
    assert np.abs(modDepth-2)<0.01
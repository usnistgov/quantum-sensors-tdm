import numpy as np
import matplotlib.pylab as plt
from cringe.shared import log
from cringe.shared import get_savepath


def lineartriangleparams(tridwell, tristeps, tristepsize):
    lindwell = 2**tridwell
    linsteps = 2**(tristeps + 1)
    linperiod = lindwell * linsteps
    linstepsize = tristepsize
    return lindwell, linperiod, linsteps, linstepsize


def conditionvphi(triangle, signal, tridwell, tristeps, tristepsize):
    llog = log.child("analysis.conditionvphi")
    """Returns a ramping up part of triangle, averaged ramping up and down parts of signal.
    """
    lindwell, linperiod, linsteps, linstepsize = lineartriangleparams(
        tridwell, tristeps, tristepsize)

    # truncate to a full number of periods
    minval = np.amin(triangle)
    # throw out min inds where the next value is also a minind
    mininds, = np.nonzero(triangle == minval)
    # mininds[plt.find(np.diff(mininds) > 1)]  # throw out min inds where the next value is also a minind
    assert len(mininds) >= 2
    triangle = triangle[mininds[0]:mininds[-1] - lindwell + 1]
    signal = signal[mininds[0]:mininds[-1] - lindwell + 1]

    # sample only the last point from each dwell (to let PID settle)
    sampledtriangle = triangle[lindwell - 1::lindwell]
    sampledsignal = signal[lindwell - 1::lindwell]

    nperiods = sampledtriangle.shape[0] // linsteps

    # make sure the triangle is periodic
    onetriangle = sampledtriangle[:linsteps]
    for i in range(nperiods):
        # this should be all, but lets have some slop for bit errors
        # plt.plot((sampledtriangle[i*linsteps:(i+1)*linsteps]-onetriangle)*(2**14-1))
        y = sampledtriangle[i*linsteps:(i + 1)*linsteps]
        a = np.abs(y - onetriangle)
        b = a >= 2  # sometimes we're seeing 1 unit errors... lets not worry
        if any(a > 0):
            llog.error(
                "biterrors detected, ignoring them for now. but you should recalibrate")
        s = np.sum(b)
        # if s < 1024:
        #     ind = np.where(~a)[0][0]
        #     print(f"i={i} s={s} {ind}")
        if s > 1:  # also allow one tolerance here
            np.save(os.path.join(
                savedir, "last_failed_sampledtriangle"), sampledtriangle)
            np.save(get_savepath("last_failed_triangle"), triangle)
            np.save(get_savepath("last_failed_signal"), signal)
            ind = np.where(~b)[0][0]
            print(
                f"i={i} s={s} ind={ind}. onetriangle[ind]={onetriangle[ind]}, y[ind]={y[ind]}")
            print(f"{b}")
            # plt.figure()
            # plt.plot(y,".-")
            # plt.plot(ind,y[ind],".")
            # plt.show()
            # plt.pause(10)
            raise Exception("triangle appears imperfect")
    oneramp = sampledtriangle[1:linsteps//2]

    # average all the values at the same place in the triangle together
    sampledsignal = sampledsignal.reshape(nperiods, -1)
    outSignalUp = np.median(sampledsignal[:, 1:linsteps//2], axis=0)
    outSignalDown = np.median(sampledsignal[:, :linsteps//2:-1], axis=0)

    # make sure oneramp is monotonic increasing with even spacing
    assert np.std(np.diff(oneramp)) < 1e-17
    assert np.mean(np.diff(oneramp)) > 0

    return oneramp, outSignalUp, outSignalDown


def conditionvphis(triangle, signal, tridwell, tristeps, tristepsize):
    ncol, nrow, _ = signal.shape
    lindwell, linperiod, linsteps, linstepsize = lineartriangleparams(
        tridwell, tristeps, tristepsize)
    outsigsup = np.zeros((ncol, nrow, linsteps//2 - 1))
    outsigsdown = np.zeros_like(outsigsup)
    outtriangles = np.zeros_like(outsigsup)
    for col in range(ncol):
        for row in range(nrow):
            try:
                triout, sigup, sigdown = conditionvphi(
                    triangle[col, row, :], signal[col, row, :], tridwell, tristeps, tristepsize)
            except Exception as ex:
                raise type(ex)("{}, col {}, row {}".format(ex, col, row))
            outsigsup[col, row, :] = sigup
            outsigsdown[col, row, :] = sigdown
            outtriangles[col, row, :] = triout
    # make sure all triangles are the same
    for col in range(ncol):
        for row in range(nrow):
            assert all(outtriangles[col, row, :] == outtriangles[0, 0, :])

    return outtriangles[0, 0, :], outsigsup, outsigsdown

import numpy as np
import matplotlib.pylab as plt


def lineartriangleparams(tridwell, tristeps, tristepsize):
    lindwell = 2**tridwell
    linsteps = 2**(tristeps + 1)
    linperiod = lindwell * linsteps
    linstepsize = tristepsize
    return lindwell, linperiod, linsteps, linstepsize


def conditionvphi(triangle, signal, tridwell, tristeps, tristepsize):
    """Returns a ramping up part of triangle, averaged ramping up and down parts of signal.
    """
    lindwell, linperiod, linsteps, linstepsize = lineartriangleparams(tridwell, tristeps, tristepsize)

    # truncate to a full number of periods
    minval = np.amin(triangle)
    mininds, = np.nonzero(triangle == minval)  # throw out min inds where the next value is also a minind
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
        if not np.sum(sampledtriangle[i*linsteps:(i + 1)*linsteps] == onetriangle) == linsteps * 1:
            np.save("last_failed_sampledtriangle",sampledtriangle)
            np.save("last_failed_triangle",triangle)
            np.save("last_failed_signal",signal)
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
    lindwell, linperiod, linsteps, linstepsize = lineartriangleparams(tridwell, tristeps, tristepsize)
    outsigsup = np.zeros((ncol, nrow, linsteps//2 - 1))
    outsigsdown = np.zeros_like(outsigsup)
    outtriangles = np.zeros_like(outsigsup)
    for col in range(ncol):
        for row in range(nrow):
            try:
                triout, sigup, sigdown = conditionvphi(triangle[col, row, :], signal[col, row, :], tridwell, tristeps, tristepsize)
            except Exception as ex:
                raise type(ex)("{}, col {}, row {}".format(ex,col,row))
            outsigsup[col, row, :] = sigup
            outsigsdown[col, row, :] = sigdown
            outtriangles[col, row, :] = triout
    # make sure all triangles are the same
    for col in range(ncol):
        for row in range(nrow):
            assert all(outtriangles[col, row, :] == outtriangles[0, 0, :])

    return outtriangles[0, 0, :], outsigsup, outsigsdown

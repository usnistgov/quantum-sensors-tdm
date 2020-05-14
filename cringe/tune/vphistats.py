# Embedded file name: /home/pcuser/nist_lab_internals/viper/cringe/tune/analysis.py
import numpy as np
import pylab as plt
import scipy.signal
from cringe.tune.analysis import conditionvphis, conditionvphi


def vPhiStatsSingle(triangle, signal, fracFromBottom=0.5):
    sMax = np.amax(signal)
    sMin = np.amin(signal)
    modDepth = sMax - sMin
    midPoint = (sMax + sMin)/2.0

    crossingPoint = sMin + fracFromBottom*modDepth
    # find crossings
    if crossingPoint is None:
        crossingPoint = midPoint
    crossingArray = np.diff((signal - crossingPoint) > 0)
    triangleStepSize = triangle[1] - triangle[0]
    slopes = []
    interpolatedCrossingInds = []
    for i in np.where(crossingArray)[0]:
        if signal[i] - crossingPoint > 0:
            pastCrossingIndex = i + 1
        else:
            pastCrossingIndex = i

        if not (pastCrossingIndex <= 1 or pastCrossingIndex >= len(signal) - 1):
            dy = signal[pastCrossingIndex]-signal[pastCrossingIndex-1]
            interpolatedCrossingIndex = pastCrossingIndex-(signal[pastCrossingIndex]-crossingPoint)/float(dy)
            slopes.append(dy/float(triangleStepSize))
            interpolatedCrossingInds.append(interpolatedCrossingIndex)
    slopes = np.array(slopes)
    interpolatedCrossingInds = np.array(interpolatedCrossingInds)
    assert(np.sum(slopes > 0) >= 2)
    assert(np.sum(slopes < 0) >= 2)
    periodInds = np.mean(np.diff(interpolatedCrossingInds[slopes > 0]))
    periodXUnits =periodInds*triangleStepSize
    positiveCrossingSlope = np.mean(slopes[slopes > 0])
    negativeCrossingSlope = np.mean(slopes[slopes < 0])
    positiveCrossingFirstX = triangle[0] + triangleStepSize*interpolatedCrossingInds[slopes > 0][0]
    negativeCrossingFirstX = triangle[0] + triangleStepSize*interpolatedCrossingInds[slopes < 0][0]

    # find bottom
    # look only in the first period
    signaloneperiod = signal[:int(np.ceil(periodInds))]
    firstMinimumInd = np.argmin(signaloneperiod)
    firstMinimumX = triangle[firstMinimumInd]
    firstMinimumY = signal[firstMinimumInd]
    firstMaximumInd = np.argmax(signaloneperiod)
    firstMaximumX = triangle[firstMaximumInd]
    firstMaximumY = signal[firstMaximumInd]

    return (periodInds, periodXUnits, positiveCrossingSlope, negativeCrossingSlope, positiveCrossingFirstX, negativeCrossingFirstX,
    firstMinimumInd, firstMinimumX, firstMinimumY, modDepth, midPoint, crossingPoint, firstMaximumInd, firstMaximumX, firstMaximumY)


def vPhiStats(triangle, signals, fracFromBottom=0.5):
    stats = {}
    statnames = ["periodInds", "periodXUnits", "positiveCrossingSlope", "negativeCrossingSlope",
                 "positiveCrossingFirstX", "negativeCrossingFirstX", "firstMinimumInd",
                 "firstMinimumX", "firstMinimumY", "modDepth", "midPoint", "crossingPoint",
                 "firstMaximumInd", "firstMaximumX", "firstMaximumY"]
    log.debug("vphistats:signals.shape",signals.shape)
    for statname in statnames:
        stats[statname] = np.zeros((signals.shape[0], signals.shape[1]))
    for col in range(signals.shape[0]):
        for row in range(signals.shape[1]):
            try:
                stattuple = vPhiStatsSingle(triangle, signals[col,row], fracFromBottom)
                (periodInds, periodXUnits, positiveCrossingSlope, negativeCrossingSlope,
    positiveCrossingFirstX, negativeCrossingFirstX, firstMinimumInd,
    firstMinimumX, firstMinimumY, modDepth, midPoint, crossingPoint,
    firstMaximumInd, firstMaximumX, firstMaximumY) = stattuple
                for i, statname in enumerate(statnames):
                    stats[statname][col, row] = stattuple[i]
            except AssertionError as ex:
                log.debug(("AssertionErrors col %d, row %d"%(col,row)))
                # we really shouldn't be using try for flow control here

    return stats


def testVPhiStats():
    triangle = np.arange(0, 2500, 10)
    signal = np.zeros((2, 2, len(triangle)), dtype="int64")
    signal[0, 0, :] = 1000*np.sin(triangle*2*np.pi/1000.) + 2000
    stats = vPhiStats(triangle, signal)

    plt.figure()
    plt.plot(triangle, signal[0, 0, :])
    plt.plot(interpolatedCrossingInds, np.ones(len(slopes))*stats['midPoint'][0, 0], ".")
    plt.plot(interpolatedCrossingInds+1, np.ones(len(slopes))*stats['midPoint'][0, 0] + slopes, ".")


if __name__ == '__main__':
    plt.ion()

    def sqarray(fb):
        return 1000*np.sin(fb*2*np.pi/508.6)+2000

    def sq1(fb):
        return 140*np.sin(fb*2*np.pi/707.2)+1000

    triangle = np.arange(0,2500,1)
    signals = np.zeros((2,2,len(triangle)), dtype="int64")
    for col in range(signals.shape[0]):
        for row in range(signals.shape[1]):
            signals[col, row, :] = sqarray(triangle)
    signal = signals[0, 0, :]

    (periodInds, periodXUnits, positiveCrossingSlope, negativeCrossingSlope,
    positiveCrossingFirstX, negativeCrossingFirstX, firstMinimumInd,
    firstMinimumX, firstMinimumY, modDepth, midPoint, crossingPoint) = vPhiStatsSingle(triangle, signal)
    stats = vPhiStats(triangle, signals)

    plt.figure()
    plt.plot(triangle, signal, "-")
    plt.plot(negativeCrossingFirstX, crossingPoint, "bo")
    dx = 0.02*periodXUnits
    dy = dx*negativeCrossingSlope
    plt.plot(negativeCrossingFirstX+dx, crossingPoint+dy, "bs")
    plt.plot(positiveCrossingFirstX, crossingPoint, "ro")
    dx = 0.02*periodXUnits
    dy = dx*positiveCrossingSlope
    plt.plot(positiveCrossingFirstX+dx, crossingPoint+dy, "rs")
    plt.plot(firstMinimumX, firstMinimumY, "gs")
    plt.xlabel("fbb")
    plt.ylabel("err")

    triangle2 = np.arange(0, 3000, 8)
    signals2 = np.zeros((2, 2, len(triangle2)), dtype="int64")
    for col in range(signals.shape[0]):
        for row in range(signals.shape[1]):
            signals2[col, row, :] = sq1(triangle2)
    lockedsignal2=signals2[0, 0, :]
    stats2 = vPhiStats(triangle2, signals2)

    newoffset = stats["firstMinimumX"][0,0]-stats2["firstMinimumY"][0,0]
    #offset must be positive, so add enough periodXUnits to make it posistive
    while newoffset <= 0:
        newoffset += stats["periodXUnits"][0,0]
    newoffset = np.round(newoffset)

    xmin,xmax = stats["firstMinimumX"][0, 0], stats["firstMinimumX"][0, 0]+stats2["modDepth"][0, 0]
    inds = np.logical_and(triangle>xmin, triangle < xmax)
    plt.plot(triangle[inds], signal[inds], lw=2)

    plt.figure()
    plt.plot(triangle2, lockedsignal2, "-")
    plt.xlabel("fba")
    plt.ylabel("fbb")

    plt.figure()
    for offset2 in [newoffset]:
        modulatedsignal2 = sqarray(sq1(triangle2)+offset2)
        plt.plot(triangle2, modulatedsignal2, label=offset2)
    plt.xlabel("fba")
    plt.ylabel("err")
    plt.legend()

    plt.close("all")
    data = np.load('last_fbb_vphi.npy')
    fb = data[0, 0, :, 1]
    err = data[0, 0, :, 0]
    triangle = fb
    signal = err
    tridwell = 2
    tristeps = 8
    tristepsize = 10
#     outtriangle, outsignalup, outsignaldown = conditionvphi(triangle, signal, tridwell, tristeps, tristepsize)
#     outtriangle, outsigsup, outsigsdown = conditionvphis(data[:, :, :, 1], data[:, :, :, 0], tridwell, tristeps, tristepsize)
    #plt.plot(outtriangle, outsigsup[0, :, :].T)
#     data2 = np.load('last_locked_fba_vphi.npy')
#     fba2 = data2[0, 0, :, 0]
#     fbb2 = data2[0, 0, :, 1]
#     outtriangle, outsigsup, outsigsdown = conditionvphis(data2[:, :, :, 0], data2[:, :, :, 1], tridwell, tristeps, tristepsize)
#     plt.figure()
#     plt.plot(outtriangle, outsigsup[0,0,:])
#     plt.plot(outtriangle, outsigsdown[0,0,:])
#
#     os = np.arange(-50,50)
#     s=np.zeros_like(os)
#     for i,o in enumerate(os):
#         a=outsigsup[0,0,50:-50]
#         b=outsigsdown[0,0,50+o:-50+o]
#         s[i]=np.sum(np.abs(a-b))
#     plt.figure()
#     plt.plot(os,s,".-")
#
#     plt.figure()
#     a=outsigsup[0,0,:]-np.mean(outsigsup[0,0,:])
#     b=outsigsdown[0,0,:]-np.mean(outsigsdown[0,0,:])
#     plt.plot(scipy.signal.correlate(a,b))

import matplotlib.pyplot as plt
import numpy as np

def SQ1bStep(start, end, stepsize):
    return np.arange(start, end, stepsize)

def listMaker(columns, rows):
    a = [[[] for i in range(0, rows)] for j in range(0, columns)]
    return a

def Averager(startIndx=0, pointsPerSlice=4096):
    import numpy as np
    import easyClient

    c = easyClient.EasyClient()
    c.setupAndChooseChannels()
    data = c.getNewData()

    slices = np.int_(data.shape[2] / pointsPerSlice)
    start = startIndx
    end = start + pointsPerSlice

    i = 0
    fb = np.zeros(pointsPerSlice)

    while i != slices:
        fb = fb + data[0, :, start:end, 1]
        start += pointsPerSlice
        end += pointsPerSlice
        i += 1
    return fb / slices, data[0, :, startIndx:startIndx + pointsPerSlice, 0]

def MinMax(fb, a):
    for i, element in enumerate(fb):
        min = np.amin(element)
        max = np.amax(element)
        a[0][i].append([min, max])
    return a

fb, err = Averager()
print(fb.shape)
plt.plot(fb[0])
plt.show()
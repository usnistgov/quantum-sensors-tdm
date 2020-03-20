import numpy as np
import easyClient
import matplotlib.pyplot as plt

c = easyClient.EasyClient()
c.setupAndChooseChannels()

class CryoQA(object):
    def __init__(self, rows, columns, SQ1bStart=0, SQ1bEnd=40000, SQ1bSteps=500, startIndx=0, pointsPerSlice=4096):
        self.rows = rows
        self.columns = columns
        self.start = SQ1bStart
        self.end = SQ1bEnd
        self.steps = SQ1bSteps
        self.startIndx = startIndx
        self.pointsPerSlice = pointsPerSlice

    def SQ1bStep(self):
        return np.arange(self.start, self.end, self.steps)

    def listMaker(self):
        a = [[[] for i in range(0, self.rows)] for j in range(0, self.columns)]
        return a

    #def biasStepper(self, val):
        #do some stuff I don't quite know how to do yet
        #follows Galen's code:
        #towercard = self.mm.cringe.tower_widget.towercards["SQ1b"]
        #for towerchannel in towercard.towerchannels:
            #towerchannel.dacspin.setValue(val)


    def test(self):
        a = self.listMaker()
        for element in self.SQ1bStep():
            #self.biasStepper(element)
            fb, err = self.averager()
            print("Toot!")
            for i, row in enumerate(fb):
                min = np.amin(row)
                max = np.amax(row)
                a[0][i].append([min, max])
        return a


    def averager(self):
        data = c.getNewData()

        slices = np.int_(data.shape[2] / self.pointsPerSlice)
        start = self.startIndx
        end = start + self.pointsPerSlice

        i = 0
        fb = np.zeros(self.pointsPerSlice)

        while i != slices:
            fb = fb + data[0, :, start:end, 1]
            start += self.pointsPerSlice
            end += self.pointsPerSlice
            i += 1

        return fb / slices, data[0, :, self.startIndx:self.startIndx + self.pointsPerSlice, 0]


a = CryoQA(12, 1)
print(a.test())
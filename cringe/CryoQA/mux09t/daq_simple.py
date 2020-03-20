import numpy as np
import easyClient


class DaqSimple:

    def __init__(self, tri_period=512, averages=10, row_slice=0):
        self.c = easyClient.EasyClient(clockmhz=125, )
        self.c.setupAndChooseChannels()
        self.pointsPerSlice = tri_period
        self.averages = averages
        self.row_slice = row_slice

    def take_average_data(self):
        #
        # Gathers data from easyClient and averages one period of it over as many periods as possible in the data.
        # Returns two arrays: fb, err
        #
        data = self.c.getNewData(minimumNumPoints=self.pointsPerSlice * self.averages, exactNumPoints=True)

        slices = np.int_(data.shape[2] / self.pointsPerSlice)
        start = 0
        end = start + self.pointsPerSlice

        i = 0
        fb = np.zeros(self.pointsPerSlice)
        err = np.zeros(self.pointsPerSlice)

        while i != slices:
            fb += data[0, self.row_slice, start:end, 1]
            err += data[0, self.row_slice, start:end, 0]
            start += self.pointsPerSlice
            end += self.pointsPerSlice
            i += 1

        return fb / slices, err / slices

    def take_data(self):
        #
        # Gathers data from easyClient and averages one period of it over as many periods as possible in the data.
        # Returns two arrays: fb, err
        #
        data = self.c.getNewData(minimumNumPoints=self.pointsPerSlice, exactNumPoints=True)

        fb = np.zeros(self.pointsPerSlice)
        err = np.zeros(self.pointsPerSlice)

        fb += data[0, self.row_slice, :, 1]
        err += data[0, self.row_slice, :, 0]

        return fb, err

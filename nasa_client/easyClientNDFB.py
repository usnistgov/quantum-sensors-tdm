#!/usr/bin/env python

'''
Sweeper client based on text client of a NASA server.

The sweeper connects a client to the server and then retrieves FB data when asked.

Created on Nov 10, 2011

@author: bennettd
'''

import time
import nasa_client
import numpy
from nasa_client import xcaldaq_commands
import zmq

class EasyClientNDFB(nasa_client.client.ZMQClient):
    """This client will connect to a server's summary channels."""
    def __init__(self, host='localhost', port=2011, clockmhz=50):
        nasa_client.client.ZMQClient.__init__(self, host=host, port=port,clockmhz=clockmhz,noblock=True)
        self.debug = False



    def setupAndChooseChannels(self, streamFbChannels = True, streamErrorChannels = True):
        """ sets up the server to stream all Fb Channels or all error channels or both
        """
        self.connect_server()
        if streamErrorChannels is True and streamFbChannels is True:
            self.stream_channels = list(range(self.nchan))
        elif streamErrorChannels is True and streamFbChannels is False:
            self.stream_channels = list(range(0, self.nchan, 2))
        elif streamErrorChannels is False and streamFbChannels is True:
            self.stream_channels = list(range(1, self.nchan, 2))
        self.start_streaming()
        print(('streaming channels: '+str(self.stream_channels)))

    def getSummaryPackets(self):
        '''getSummaryPackets(self)
        leaves stream_channels in same state as when called
        starts streaming, takes some data, then stops streaming

        return fb_mean, fb_std, error_mean, error_std
        '''
        raise Exception("not fixed for ZMQ")
        self.clearSocket()
        old_stream_channels = self.stream_channels[:] # forces a copy
        self.stream_channels = (self.CHANNEL_CODE_FB_SUMMARY, self.CHANNEL_CODE_FB_STDDEV, self.CHANNEL_CODE_ERR_SUMMARY, self.CHANNEL_CODE_ERR_STDDEV)
        for j in range(100):
            newpayloads, newheaders = self.get_data_packets() # this is done out here so we can find out the number of samples, so we can preallocate dataOut
            if len(newpayloads)>0:
                break
            time.sleep(0.005)
            time.sleep(0.005)
            if j >=99:
                raise ValueError('server wont give me any data')
        self.stream_channels = old_stream_channels
        # might need a loop here to grab more data if we don't have enough yet, but it seems to work without it
        for payload, header in zip(newpayloads,newheaders):
            if header['chan'] == self.CHANNEL_CODE_FB_SUMMARY:
                fb_mean = payload.reshape(self.ncol, self.nrow)
            elif header['chan'] == self.CHANNEL_CODE_FB_STDDEV:
                fb_std = payload.reshape(self.ncol, self.nrow)
            elif header['chan'] == self.CHANNEL_CODE_ERR_SUMMARY:
                error_mean = payload.reshape(self.ncol, self.nrow)
            elif header['chan'] == self.CHANNEL_CODE_ERR_STDDEV:
                error_std = payload.reshape(self.ncol, self.nrow)
        return fb_mean, fb_std, error_mean, error_std


    def clearSocket(self):
        # i think this will throw away old data
        i = 0
        while True:
            try:
                self.dataPort.recv(zmq.NOBLOCK)
            except zmq.ZMQError:
                break
            i+=1
            if i > 10000:
                break
        return

    def clearWithLatencyCheck(self, delaySeconds):
        delay_usec=int(delaySeconds*1e6)
        mytime_usec = time.time()*1e6
        self.clearSocket()
        while True:
            payloads, headers = self.get_data_packets(max_bytes=1)
            if len(payloads) == 0:
                continue
            servertime_usec = headers[-1]["packet_timestamp"]
            if servertime_usec - mytime_usec > delay_usec:
                return headers[-1]["count_of_last_sample"]


    def reshapeDataToColRowFrame(self, dataIn):
        if len(self.stream_channels)<self.ncol*self.nrow*2:
            raise ValueError('will not work unless streaming all possible channels')
        # print(f"ncol {self.ncol} nrow {self.nrow} shape {dataIn.shape}")
        dataOut = numpy.zeros((self.ncol, self.nrow, dataIn.shape[1], 2),dtype="int32")
        for col in range(self.ncol):
            for row in range(self.nrow):
                errorIndex = self.stream_channels.index(self.errorChannel(col, row))
                fbIndex = self.stream_channels.index(self.fbChannel(col, row))
                # [col, row, frame, 0=error/1=feedback]
                dataOut[col,row,:,0] = dataIn[errorIndex,:]
                dataOut[col,row,:,1] = dataIn[fbIndex,:]
        return dataOut

    def fbChannel(self, col, row):
        return 2*(col*self.nrow+row)+1

    def errorChannel(self, col, row):
        return 2*(col*self.nrow+row)

    def setMixToZero(self):
        for channel in self.stream_channels:
            self.setMixChannel(channel,0)

    def setMix(self, mixVal):
        if len(numpy.shape(mixVal))==0:  # voltage is a single number, make a array out of it, and set all channels to the same value
            mixVal = numpy.ones((self.ncol, self.nrow))*mixVal
        if not numpy.all(numpy.shape(mixVal) == (self.ncol, self.nrow)):
            raise ValueError('mixVal should either a number or a list/array with (ncol, nrow) elements')
        for col in range(self.ncol):
            for row in range(self.nrow):
#                print('col %d, row %d, fbchannel %d, mixVal %f'%(col, row, self.fbChannel(col, row), mixVal[col,row]))
                self.setMixChannel(self.fbChannel(col, row), mixVal[col,row])

    def setMixChannel(self, channel, mixVal = 0):
        self.set_float(xcaldaq_commands.secondary_comm['MIXLEVEL'], channel, mixVal)
        if mixVal == 0:
            self._command(xcaldaq_commands.primary_comm['SET'], xcaldaq_commands.secondary_comm['MIXFLAG'], channel, 0)
        else:
            self._command(xcaldaq_commands.primary_comm['SET'], xcaldaq_commands.secondary_comm['MIXFLAG'], channel, 1)

    def getNewData(self, delaySeconds = 0.001, minimumNumPoints = 4000, exactNumPoints = False, sendMode = 0, toVolts=False, divideNsamp=True):
        '''
        getNewData(self, delaySeconds = 0.001, minimumNumPoints = 4000, exactNumPoints = False, sendMode = 0)
        returns dataOut[col,row,frame,error=1/fb=2]
        rejects data taken within delaySeconds of calling getNewData, used to ensure new data
        returns at least minimumNumPoints frames
        if exactNumPoints is True, returns exactly minimumNumPoints frames, but does throw away data to achieve this
        sendMode corresponds to dfb07_card setting (or "raw" for diagnostic mode from server")
        returned data is probably time continuous, there will be printed warning statements if it is not
        '''
        count_of_last_sample_to_avoid = self.clearWithLatencyCheck(delaySeconds) # works best on same computer, use bigger latency on different computers
        headers, payloads = [],[]
        firstSampleCount = [-1]*len(self.stream_channels)
        lastSampleCount = [0]*len(self.stream_channels)
        numPoints = [0]*len(self.stream_channels)
        while True:
            newpayloads, newheaders = self.get_data_packets(max_bytes=1) # this is done out here so we can find out the number of samples, so we can preallocate dataOut
            for payload, header in zip(newpayloads, newheaders):
                if not header['chan'] in self.stream_channels:
                    continue
                payloadChannelIndex = self.stream_channels.index(header['chan'])
                countOfFirstSample = header['count_of_last_sample']-header['record_samples']
                if header['count_of_last_sample'] <= count_of_last_sample_to_avoid:
                    continue
                    # print('rejected channel %d count_of_last_sample %d <= max_count_of_last_sample %d'%(header['chan'], header['count_of_last_sample'], throw_away_last_count))
                elif numPoints[payloadChannelIndex] > minimumNumPoints:
                    continue
                    # print('rejected chann %d numPoints %d > minimumNumPoints %d'%(header['chan'],numPoints[payloadChannelIndex],minimumNumPoints ))
                else:
                    if lastSampleCount[payloadChannelIndex] < header['count_of_last_sample']:
                        if header['count_of_last_sample']-header['record_samples'] != lastSampleCount[payloadChannelIndex] and lastSampleCount[payloadChannelIndex]>0:
                            print('WARNING: getNewData is not getting continuous data')
                        lastSampleCount[payloadChannelIndex] = header['count_of_last_sample']
                        # print('channel %d lastSampleCount = %d'%(header['chan'],header['count_of_last_sample']))
                    if firstSampleCount[payloadChannelIndex] < 0:
                        firstSampleCount[payloadChannelIndex] = countOfFirstSample
                        # print('channel %d firstSampleCount = %d'%(header['chan'],countOfFirstSample))
                    payloads.extend([payload])
                    headers.extend([header])

            numPoints = numpy.array(lastSampleCount)-numpy.array(firstSampleCount)
            #print 'numPoints',numPoints
            if all(numPoints>minimumNumPoints):
                break
        if numpy.diff(firstSampleCount).sum()>0:
            print('WARNING: getNewData does not have all time aligned data')

        dataOut = self.sortPackets(payloads, headers, numPoints, firstSampleCount)
        dataOut = self.reshapeDataToColRowFrame(dataOut)
        if sendMode != "raw":
            dataOut[:,:,:,1]=dataOut[:,:,:,1]>>2 # ignore 2 lsbs (frame bit and trigger)
        if sendMode == 2:
            dataOut[:,:,:,0]=dataOut[:,:,:,0]>>2 # ignore 2 lsbs if this is also a fb (just scaling)
        if toVolts:
            dataOut = self.toVolts(dataOut, sendMode)
        if divideNsamp and sendMode==0:
            dataOut = numpy.array(dataOut,dtype="float32")
            dataOut[:,:,:,0]/=self.num_of_samples

        if exactNumPoints:
            dataOut = dataOut[:,:,:minimumNumPoints,:]
        return dataOut

    def convertPacketsToVolts(self, payloads, headers, numPoints, firstSampleCount, sendMode):
        print((numpy.min(numPoints), len(self.stream_channels)))
        try:
            dataOut = numpy.zeros((len(self.stream_channels),numpy.min(numPoints)))
        except ValueError as e:
            print(('Tried and failed to make an array of size (%d, %d)'%
                  (len(self.stream_channels),numpy.min(numPoints))))
            print("The numpoints array is: ")
            print(numPoints)
            raise e

    def sortPackets(self, payloads, headers, numPoints, firstSampleCount):
        dataOut = numpy.zeros((len(self.stream_channels),numpy.min(numPoints)),dtype="int32")
        for payload, header in zip(payloads, headers):
            payloadChannelIndex = self.stream_channels.index(header['chan'])
            indexOfLastSample = header['count_of_last_sample']-firstSampleCount[payloadChannelIndex]
            indexOfFirstSample = indexOfLastSample-header['record_samples']
            if indexOfLastSample >= dataOut.shape[1]:
                indexOfLastSample = dataOut.shape[1]
            if indexOfFirstSample<=indexOfLastSample:
                dataOut[payloadChannelIndex, indexOfFirstSample:indexOfLastSample] = payload[:indexOfLastSample-indexOfFirstSample]

        return dataOut

    def toVolts(self,dataOut, sendmode):
        print("doing toVolts")
        dataOut = numpy.array(dataOut,dtype="float64")
        if sendmode == 0:
            dataOut[:,:,:,0]/=float((2**12-1)*self.num_of_samples) # error
            dataOut[:,:,:,1]/=float(2**14-1) # FBA
        if sendmode == 2:
            dataOut[:,:,:,0]/=float(2**14-1) # FBA
            dataOut[:,:,:,1]/=float(2**14-1) # FBB
        return dataOut

def main():
    c = EasyClient()
    c.setupAndChooseChannels()
    print((c.getNewData()))

if __name__ == '__main__':
    main()

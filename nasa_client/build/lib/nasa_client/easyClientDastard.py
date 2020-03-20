import rpc_client
import numpy
import zmq
import time
import collections
import json
import numpy as np

DEBUG = True
rpc_client.DEBUG = True

SUMMARY_HEADER_DTYPE=np.dtype([("chan",np.uint16),("headerVersion",np.uint8),
     ("npresamples",np.uint32),("nsamples",np.uint32),("pretrig_mean","f4"),("peak_value","f4"),
     ("pulse_rms","f4"),("pulse_average","f4"),("residualStdDev","f4"),
     ("unixnano",np.uint64), ("trig frame",np.uint64)])

RECORD_HEADER_DTYPE = np.dtype([("chan",np.uint16),("headerVersion",np.uint8), ("dataTypeCode", np.uint8),
     ("npresamples",np.uint32),("nsamples",np.uint32),("samplePeriod","f4"),("voltsPerArb","f4"),
     ("unixnano",np.uint64), ("triggerFramecount",np.uint64)])


# // uint16: channel number
# // uint8: headerVersion number
# // uint8: code for data type (0-1 = 8 bits; 2-3 = 16; 4-5 = 32; 6-7 = 64; odd=uint; even=int)
# // uint32: # of pre-trigger samples
# // uint32: # of samples, total
# // float32: sample period, in seconds (float)
# // float32: volts per arb conversion (float)
# // uint64: trigger time, in ns since epoch 1970
# // uint64: trigger frame #

class EasyClientDastard():
    """This client will connect to a server's summary channels."""
    def __init__(self, host='localhost', baseport=5500, setupOnInit = True):
        self.host = host
        self.baseport = baseport
        self.context = zmq.Context()
        self.samplePeriod = None # learn this from first observed data packet
        if setupOnInit:
            self.setupAndChooseChannels()


    def _connectStatusSub(self):
        """ connect to the status update port of dastard """
        self.statusSub = self.context.socket(zmq.SUB)
        address = "tcp://%s:%d" % (self.host, self.baseport+1)
        self.statusSub.connect(address)
        print("Collecting updates from dastard at %s" % address)
        self.statusSub.setsockopt(zmq.SUBSCRIBE, "")
        self.messages_seen = collections.Counter()

    def _connectRecordSub(self):
        """ connect to the record update port of dastard """
        self.recordSub = self.context.socket(zmq.SUB)
        address = "tcp://%s:%d" % (self.host, self.baseport+2)
        self.recordSub.setsockopt(zmq.RCVTIMEO, 1000)
        self.recordSub.connect(address)
        print("Collecting records from dastard at %s" % address)
        self.recordSub.setsockopt(zmq.SUBSCRIBE, "")

    def _connectRPC(self):
        """ connect to the rpc port of dastard """
        self.rpc = rpc_client.JSONClient((self.host, self.baseport))
        print("Dastard is at %s:%d" % (self.host, self.baseport))

    def _getStatus(self):
        self._sourceRecieved = False
        self._statusRecieved = False
        self.rpc.call("SourceControl.SendAllStatus", "dummy")
        while True:
            try:
                [topic, contents] = self.statusSub.recv_multipart(zmq.NOBLOCK)
                self.messages_seen[topic] += 1
                self._handleStatusMessage(topic, contents)
            except zmq.ZMQError:
                break
        if (not self._sourceRecieved) or (not self._statusRecieved):
            raise Exception("didn't get source and status messages")
        return

    def _handleStatusMessage(self,topic, contents):
        if DEBUG:
            print topic
            print contents
        if topic in ["CURRENTTIME"]:
            if DEBUG:
                print("skipping topic %s"%topic)
            return
        d = json.loads(contents)
        if DEBUG:
            print d
        if topic == "STATUS":
            self._statusRecieved = True
            self.numChannels = d["Nchannels"]
            if d["Ncol"] == [] and DEBUG:
                self.numColumns = 1
            else:
                self.numColumns = d["Ncol"][0]
            if d["Nrow"] == [] and DEBUG:
                self.numRows = self.numChannels//2
            else:
                self.numRows = d["Nrow"][0]
            self.sourceName = d["SourceName"]
        if topic == "SIMPULSE" and self.sourceName=="SimPulses":
            if not DEBUG:
                raise Exception("dont use a SIMPULSE in non-debug mode")
                print("using SIMPULSE")
            self.nSamp = 2
            self.clockMhz = 125
            self._sourceRecieved = True
        if topic == "LANCERO" and self.sourceName=="Lancero":
            self.nSamp = d["NSamp"]
            self.clockMhz = d["ClockMhz"]
            self._sourceRecieved = True



    def _configDastardAutoTrigs(self):
        """ configure dastard to have auto trigger with about 1/20th of a second record lengths """
        recordLength = 5000
        if recordLength < 1:
            recordLength = 1
        configLengths  = {"Nsamp": recordLength, "Npre": 0}
        self.rpc.call("SourceControl.ConfigurePulseLengths", configLengths)
        autoDelayNanoseconds = int(0)
        configTriggers = {"AutoTrigger":True,
                  "AutoDelay":autoDelayNanoseconds,
                  "ChannelIndicies":self.channelIndicies}
        self.rpc.call("SourceControl.ConfigureTriggers", configTriggers)
        # possibly should be using CoupleErrToFB here


    def setupAndChooseChannels(self, streamFbChannels = True, streamErrorChannels = True):
        """ sets up the server to stream all Fb Channels or all error channels or both
        """
        self._connectStatusSub()
        self._connectRPC()
        self._getStatus()
        self._configDastardAutoTrigs()
        self._connectRecordSub()
        header, data = self.getMessage()
        self.samplePeriod = header["samplePeriod"]
        self.linePeriodSeconds = self.samplePeriod/self.numRows
        self.linePeriod = int(round(self.linePeriodSeconds*self.clockMhz))
        print self

    @property
    def channelIndicies(self):
        return range(self.numChannels)

    def getSummaryPackets(self):
        raise Exception("not implmented (there are no summary packets in dastard)")


    def clearSocket(self):
        """ recieve all messages queued on the sub socket and throw them away """
        while True:
            try:
                self.recordSub.recv(zmq.NOBLOCK)
            except zmq.ZMQError:
                break
        return

    def getMessage(self):
        m = self.recordSub.recv_multipart()
        header = np.frombuffer(m[0],RECORD_HEADER_DTYPE)[0]
        data = np.frombuffer(m[1], np.uint16)
        assert header["nsamples"] == len(data)
        assert header["headerVersion"] == 0
        return header, data

    def clearWithLatencyCheck(self, delaySeconds):
        """ return the first message (header,data) timestamped at least delaySeconds after the call
        to this function. if delayseconds is zero negative, returns the first message read """
        delayNano=int(delaySeconds*1e9)
        mytimeNano = time.time()*1e9
        self.clearSocket()
        header, data = self.getMessage()
        firstDastardNano = header["unixnano"]
        if mytimeNano<firstDastardNano:
            raise Exception("doesnt seem right")
        while True:
            dastardNano = header["unixnano"]
            if dastardNano - mytimeNano > delayNano or delayNano < 0:
                return header, data
            header, data = self.getMessage()



    def reshapeDataToColRowFrame(self, dataIn):
        dataOut = numpy.zeros((self.numColumns, self.numRows, dataIn.shape[1], 2),dtype="int32")
        for col in range(self.numColumns):
            for row in range(self.numRows):
                errorIndex = self.errorChannel(col, row)
                fbIndex = self.fbChannel(col, row)
                # [col, row, frame, 0=error/1=feedback]
                dataOut[col,row,:,0] = dataIn[errorIndex,:]
                dataOut[col,row,:,1] = dataIn[fbIndex,:]
        return dataOut

    def fbChannel(self, col, row):
        return 2*(col*self.numRows+row)+1

    def errorChannel(self, col, row):
        return 2*(col*self.numRows+row)

    def setMixToZero(self):
        for channel in self.channelIndicies:
            self.setMixChannel(channel,0)

    def setMix(self, mixVal):
        if len(numpy.shape(mixVal))==0:  # voltage is a single number, make a array out of it, and set all channels to the same value
            mixVal = numpy.ones((self.numColumns, self.numRows))*mixVal
        if not numpy.all(numpy.shape(mixVal) == (self.numColumns, self.numRows)):
            raise ValueError('mixVal should either a number or a list/array with (numColumns, numRows) elements')
        for col in xrange(self.numColumns):
            for row in xrange(self.numRows):
#                print('col %d, row %d, fbchannel %d, mixVal %f'%(col, row, self.fbChannel(col, row), mixVal[col,row]))
                self.setMixChannel(self.fbChannel(col, row), mixVal[col,row])

    def setMixChannel(self, channelIndex, mixFraction):
        config = {"ChannelIndex": channelIndex,
                  "MixFraction": mixFraction}
        self.rpc.call("SourceControl.ConfigureMixFraction", config)

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

        header, data = self.clearWithLatencyCheck(delaySeconds) # works best on same computer, use bigger latency on different computers
        acceptFramecount = header["triggerFramecount"]
        numPoints = [0]*self.numChannels
        firstFramecount = -np.ones(self.numChannels,dtype=np.int64) # triggerFramecount of first saple of first data in datas
        lastFramecount = -np.ones(self.numChannels,dtype=np.int64) # triggerFramecount of last sample of last data in datas
        headers, datas = [],[]
        while True:
            if header["chan"] >= self.numChannels:
                raise Exception("")
            channelIndex = header["chan"]
            triggerFramecount = header['triggerFramecount']
            # trigger framecount is the first sample framecount because we set npresamples = 0
            if triggerFramecount < acceptFramecount:
                raise Exception("got a triggerFramecount from backwards in time")
            elif numPoints[channelIndex] > minimumNumPoints:
                continue
            else:
                if lastFramecount[channelIndex] < triggerFramecount:
                    if lastFramecount[channelIndex] != triggerFramecount-1 and lastFramecount[channelIndex]>0:
                        raise Exception('getNewData is not getting continuous data')
                    lastFramecount[channelIndex] = triggerFramecount+len(data)-1
                if firstFramecount[channelIndex] < 0:
                    firstFramecount[channelIndex] = triggerFramecount
                datas.append(data)
                headers.append(header)
            numPoints = lastFramecount-firstFramecount+1
            #print 'numPoints',numPoints
            if all(numPoints>minimumNumPoints):
                break
            if numPoints.max()-numPoints.min() > len(data):
                raise Exception("missing some channels?")
            header, data = self.getMessage()


        if numpy.diff(firstFramecount).sum()>0:
            Exception('getNewData does not have all time aligned data')

        dataOut = self.sortPackets(datas, headers, numPoints, firstFramecount)
        dataOut = self.reshapeDataToColRowFrame(dataOut)

        if sendMode != "raw":
            dataOut[:,:,:,1]=dataOut[:,:,:,1]>>2 # ignore 2 lsbs (frame bit and trigger)
        if sendMode == 2:
            dataOut[:,:,:,0]=dataOut[:,:,:,0]>>2 # ignore 2 lsbs if this is also a fb (just scaling)
        if toVolts:
            dataOut = self.toVolts(dataOut, sendMode)
        if divideNsamp and sendMode==0:
            dataOut = numpy.array(dataOut,dtype="float32")
            dataOut[:,:,:,0]/=self.nSamp

        if exactNumPoints:
            dataOut = dataOut[:,:,:minimumNumPoints,:]
        return dataOut

    def sortPackets(self, datas, headers, numPoints, firstFramecount):
        dataOut = numpy.zeros((self.numChannels,numpy.min(numPoints)),dtype="int32")
        for data, header in zip(datas, headers):
            channelIndex = header["chan"]
            indexOfFirstSample = int(header["triggerFramecount"]-firstFramecount[channelIndex])
            indexOfLastSample = int(indexOfFirstSample+len(data))
            if indexOfLastSample >= dataOut.shape[1]:
                indexOfLastSample = dataOut.shape[1]
            if indexOfFirstSample<=indexOfLastSample:
                dataOut[channelIndex, indexOfFirstSample:indexOfLastSample] = data[:indexOfLastSample-indexOfFirstSample]

        return dataOut

    def toVolts(self,dataOut, sendmode):
        print "doing toVolts"
        dataOut = numpy.array(dataOut,dtype="float64")
        if sendmode == 0:
            dataOut[:,:,:,0]/=float((2**12-1)*self.nSamp) # error
            dataOut[:,:,:,1]/=float(2**14-1) # FBA
        if sendmode == 2:
            dataOut[:,:,:,0]/=float(2**14-1) # FBA
            dataOut[:,:,:,1]/=float(2**14-1) # FBB
        return dataOut

    @property
    def ncol(self):
        return self.numColumns
    @property
    def nrow(self):
        return self.numRows
    @property
    def nsamp(self):
        return self.nSamp
    @property
    def lsync(self):
        return self.linePeriod

    def __repr__(self):
        return "EasyClientDastard {} columns X {} rows, linePeriod {}, clockMhz {}, nsamp {}".format(self.ncol,self.nrow,self.lsync,self.clockMhz, self.nsamp)

if __name__ == '__main__':
    c = EasyClientDastard()
    for i in range(10):
        print i
        print c.getMessage()
    # c.setMixChannel(0,0)
    data = c.getNewData(delaySeconds=-1,minimumNumPoints=6000)
    print data.shape
    print data
    import pylab as plt
    print c
    for sendMode in ["raw",1,2]:
        data = c.getNewData(delaySeconds=-1,minimumNumPoints=6000,sendMode=sendMode)
        plt.figure()
        plt.plot(data[0,0,:,0],label="err (lastind 0)")
        plt.plot(data[0,0,:,1],label="fb (lastind 1)")
        plt.title("send mode = {}".format(sendMode))
        plt.xlabel("framecount")
        plt.ylabel("value")
        plt.legend()
    plt.show()

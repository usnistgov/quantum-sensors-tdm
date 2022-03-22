from . import rpc_client_for_easy_client
import numpy
import zmq
import time
import collections
import json
import numpy as np

DEBUG = False
rpc_client_for_easy_client.DEBUG = False

SUMMARY_HEADER_DTYPE=np.dtype([("chan",np.uint16),("headerVersion",np.uint8),
     ("npresamples",np.uint32),("nsamples",np.uint32),("pretrig_mean","f4"),("peak_value","f4"),
     ("pulse_rms","f4"),("pulse_average","f4"),("residualStdDev","f4"),
     ("unixnano",np.uint64), ("trig frame",np.uint64)])

RECORD_HEADER_DTYPE = np.dtype([("chan",np.uint16),("headerVersion",np.uint8), ("dataTypeCode", np.uint8),
     ("npresamples",np.uint32),("nsamples",np.uint32),("samplePeriod","f4"),("voltsPerArb","f4"),
     ("unixnano",np.uint64), ("triggerFramecount",np.uint64)])

class EasyClientDastard():
    """This client will connect to a server's summary channels."""
    def __init__(self, host='localhost', baseport=5500, setupOnInit = True):
        self.host = host
        self.baseport = baseport
        self.context = zmq.Context()
        self.samplePeriod = None # learn this from first observed data packet
        self._restoredOldTriggerSettings = False
        if setupOnInit:
            self.setupAndChooseChannels()


    def _connectStatusSub(self):
        """ connect to the status update port of dastard """
        self.statusSub = self.context.socket(zmq.SUB)
        address = "tcp://%s:%d" % (self.host, self.baseport+1)
        self.statusSub.setsockopt(zmq.RCVTIMEO, 1000) # this doesn't seem to do anything
        self.statusSub.setsockopt( zmq.LINGER,      0 )
        self.statusSub.connect(address)
        print(("Collecting updates from dastard at %s" % address))
        self.statusSub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.messagesSeen = collections.Counter()

    def _connectRecordSub(self, verbose):
        """ connect to the record update port of dastard """
        self.recordSub = self.context.socket(zmq.SUB)
        self.recordSubAddress = "tcp://%s:%d" % (self.host, self.baseport+2)
        self.recordSub.RCVTIMEO = 1000
        self.recordSub.RCVHWM = 5000
        self.recordSub.LINGER = 0
        self.recordSub.connect(self.recordSubAddress)
        if verbose: print(("Collecting records from dastard at %s" % self.recordSubAddress))
        self.recordSub.setsockopt_string(zmq.SUBSCRIBE, "")

    def _connectRPC(self):
        """ connect to the rpc port of dastard """
        self.rpc = rpc_client_for_easy_client.JSONClient((self.host, self.baseport))
        print(("Dastard is at %s:%d" % (self.host, self.baseport)))

    def _getStatus(self):
        self._sourceRecieved = False
        self._statusRecieved = False
        time.sleep(0.05) # these seem to help cringe be reliable
        self.rpc.call("SourceControl.SendAllStatus", "dummy")
        time.sleep(0.05) # these seem to help cringe be reliable
        tstart = time.time()
        while True:
            if time.time()-tstart > 2:
                raise Exception("took too long to get status")
            topic, contents = self.statusSub.recv_multipart()
            topic = topic.decode()
            contents = contents.decode()
            self.messagesSeen[topic] += 1
            self._handleStatusMessage(topic, contents)
            if DEBUG:
                print(f"messages seen so far = {self.messagesSeen}")
            if all([self.messagesSeen[t]>0 for t in ["STATUS", "LANCERO", "SIMPULSE"]]):
                if self.sourceName == "Lancero":
                    self.numRows = self.sequenceLength
                    self.numColumns = self.numChannels//(2*self.numRows)
                    assert self.numChannels%(2*self.numRows) == 0
                else:
                    raise Exception(f"source {self.sourceName} not yet supported")
                print("returned from _getStatus")
                return
        raise Exception(f"didn't get source and status messages, messagesSeen: {self.messagesSeen}")

    def _handleStatusMessage(self,topic, contents):
        if DEBUG:
            print(("topic=%s"%topic))
            print(contents)
        if topic in ["CURRENTTIME"]:
            if DEBUG:
                print(("skipping topic %s"%topic))
            return
        d = json.loads(contents)
        if DEBUG:
            print(d)
        if topic == "STATUS":
            self._statusRecieved = True
            self.numChannels = d["Nchannels"]
            self.sourceName = d["SourceName"]
            self._oldNSamples = d["Nsamples"]
            self._oldNPresamples = d["Npresamp"]
        if topic == "SIMPULSE" and self.sourceName=="SimPulses":
            if not DEBUG:
                raise Exception("dont use a SIMPULSE in non-debug mode")
            print("using SIMPULSE")
            self.nSamp = 2
            self.clockMhz = 125
            self.sequenceLength=self.numChannels
        if topic == "LANCERO" and self.sourceName=="Lancero":
            self.nSamp = d["DastardOutput"]["Nsamp"]
            self.clockMhz = d["DastardOutput"]["ClockMHz"]
            self.sequenceLength = d["DastardOutput"]["SequenceLength"]
        if topic == "TRIGGER":
            self._oldTriggerDict = d[0]

    def restoreOldTriggerSettings(self):
        configLengths  = {"Nsamp": self._oldNSamples, "Npre": self._oldNPresamples}
        self.rpc.call("SourceControl.ConfigurePulseLengths", configLengths)
        self.rpc.call("SourceControl.ConfigureTriggers", self._oldTriggerDict)
        self._restoredOldTriggerSettings = True


    def _configDastardAutoTrigs(self):
        """ configure dastard to have auto trigger with about 1/20th of a second record lengths """
        configLengths  = {"Nsamp": 10000, "Npre": 5000}
        self.rpc.call("SourceControl.ConfigurePulseLengths", configLengths)
        configTriggers = {"AutoTrigger":True,
                  "AutoDelay":int(0),
                  "ChannelIndices":self.channelIndicies}
        # sending new trigger settings should reset the last trigger value in Dastard
        # so all channels trigge at the same starting point
        print(f"configTriggers {configTriggers}")
        self.rpc.call("SourceControl.ConfigureTriggers", configTriggers)
        # possibly should be using CoupleErrToFB here


    def setupAndChooseChannels(self, streamFbChannels = True, streamErrorChannels = True):
        """ sets up the server to stream all Fb Channels or all error channels or both
        """
        self._connectStatusSub()
        self._connectRPC()
        self._getStatus()
        self._configDastardAutoTrigs()
        self._connectRecordSub(verbose=True)
        header, data = self.getMessage()
        self.samplePeriod = header["samplePeriod"]
        self.linePeriodSeconds = self.samplePeriod/self.numRows
        self.linePeriod = int(round(self.linePeriodSeconds*self.clockMhz*1e6))
        print(self)

    @property
    def channelIndicies(self):
        return list(range(self.numChannels))

    def getSummaryPackets(self):
        raise Exception("not implmented (there are no summary packets in dastard)")


    def clearSocket(self):
        """ recieve all messages queued on the sub socket and throw them away """
        # close and reconnect to flush the recieve side buffer
        # recieve data until there is no more data to recieve to
        self.recordSub.close() # close socket and reconnect, rely on ZMQ_LINGER=0
        self._connectRecordSub(verbose=False)
        # https://stackoverflow.com/questions/43771830/zeromq-packets-being-lost-even-after-setting-hwm-and-bufsize
        # says never set SENDHWM = 1
        # dastard has SENDHWM = 10 and a channel with buffer size 500
        # while True:
        #     try:
        #         self.recordSub.recv(zmq.NOBLOCK)
        #     except zmq.ZMQError:
        #         break
        # return

    def getMessage(self, sendMode=0):
        h, m = self.recordSub.recv_multipart()
        header = np.frombuffer(h,RECORD_HEADER_DTYPE)[0]
        if sendMode == 0:
            if header["chan"]%2==0:
                data = np.frombuffer(m, np.int16)
            else:
                data = np.frombuffer(m, np.uint16)
        else:
            raise Exception("sendMode {} not implemented".format(sendMode))
        assert header["nsamples"] == len(data)
        assert header["headerVersion"] == 0
        return header, data

    def fbChannelIndex(self, col, row):
        return 2*(col*self.numRows+row)+1

    def errorChannelIndex(self, col, row):
        return 2*(col*self.numRows+row)

    def setMixToZero(self):
        self.setMix(0)

    def setMix(self, mixFractions):
        if len(numpy.shape(mixFractions))==0:  # voltage is a single number, make a array out of it, and set all channels to the same value
            mixFractions = numpy.ones((self.numColumns, self.numRows))*mixFractions
        if not numpy.all(numpy.shape(mixFractions) == (self.numColumns, self.numRows)):
            raise ValueError('mixFractions should either a number or a list/array with (numColumns, numRows) elements')
        config = {"ChannelIndices":  np.arange(1,self.numColumns*self.numRows*2,2).tolist(),
                  "MixFractions": mixFractions.flatten().tolist()}
        self.rpc.call("SourceControl.ConfigureMixFraction", config)

    def getNewData(self, delaySeconds = 0.001, minimumNumPoints = 4000, exactNumPoints = False, sendMode = 0, toVolts=False, divideNsamp=True, retries = 3):
        '''
        getNewData(self, delaySeconds = 0.001, minimumNumPoints = 4000, exactNumPoints = False, sendMode = 0)
        returns dataOut[col,row,frame,error=0/fb=1]
        rejects data taken within delaySeconds of calling getNewData, used to ensure new data
        returns at least minimumNumPoints frames
        if exactNumPoints is True, returns exactly minimumNumPoints frames, but does throw away data to achieve this
        sendMode corresponds to dfb07_card setting (or "raw" for diagnostic mode from server")
        retries - how many times to retry upon an error
        '''

        if self._restoredOldTriggerSettings:
            self.setupAndChooseChannels()

        delayNano=int(delaySeconds*1e9)
        mytimeNano = time.time()*1e9
        self.clearSocket()
        counter = collections.Counter()
        firstTriggerFramecount = None
        target_triggerFramecount = -1 # impossible framecount
        # map triggerFramecount to a list containing data for each channel
        datas_dict = collections.defaultdict(lambda: [None]*self.numChannels)
        i = 0
        n_thrown_away_for_delay_seconds = 0
        # tstart = time.time()
        while True:
            i+=1
            if i>40*self.numChannels:
                raise Exception("couldn't get the data you wantd")
            header, data = self.getMessage()
            dastardNano = header["unixnano"]
            if delayNano >= 0 and (mytimeNano + delayNano > dastardNano):
                n_thrown_away_for_delay_seconds+=1
                continue
            # if delayNano < 0 accept any data
            if firstTriggerFramecount is None:
                firstTriggerFramecount=header["triggerFramecount"]
            counter[header["triggerFramecount"]-firstTriggerFramecount] += 1
            n_observed = counter[header["triggerFramecount"]-firstTriggerFramecount]
            datas_dict[header["triggerFramecount"]][header["chan"]] = data
            if n_observed == self.numChannels:
                # check if we have enough samples
                keys = sorted(datas_dict.keys())
                k_complete = [k for k in keys if counter[k-firstTriggerFramecount] == self.numChannels]
                n_have = len(data)*len(k_complete)
                if n_have >= minimumNumPoints:
                    break
            # print(f"i={i} n_observed={n_observed} counter={counter}")
            # x = header["triggerFramecount"]-firstTriggerFramecount
            # y = x//10000
            # ch = header["chan"]
            # nsamp = len(data)
            # dt = (time.time()-tstart)*1000
            # unixnano = header["unixnano"]
            # dtb = (time.time()*1000-unixnano*1e-6)
            # print(f"timestamp={y} chan={ch} nsamp={nsamp} dt={dt:.2f}ms dtb={dtb:.2f}ms")
            # if dt>200:
            #     raise Exception()
            # v = list(counter.values())
            # j = np.argmax(v)
            # fc = list(counter.keys())[j]
            # print(i, fc, v[j])
        
        # print( "keys",keys)
        # print( "lengths", [len(datas_dict[k]) for k in keys])
        # print( "k_complete", k_complete)
        # print( "n_have", n_have)
        # print( "n_thrown_away_for_delay_seconds", n_thrown_away_for_delay_seconds)
        # print( self.numChannels, self.numRows, self.numColumns)
        # print( "triggerFramecount Counter", counter)
        assert(all(np.diff(k_complete)==len(data)))


        dataOut = np.zeros((self.numColumns, self.numRows, n_have, 2),dtype="int32")
        for col in range(self.numColumns):
            for row in range(self.numRows):
                errorIndex = self.errorChannelIndex(col, row)
                fbIndex = self.fbChannelIndex(col, row)
                # [col, row, frame, 0=error/1=feedback]
                j=0    
                for k in k_complete:
                    data_error = datas_dict[k][errorIndex]
                    data_fb = datas_dict[k][fbIndex]
                    n = len(data_fb)
                    assert(len(data_error)==n)
                    dataOut[col,row,j:j+n,0] = data_error
                    dataOut[col,row,j:j+n,1] = data_fb
                    j += len(data_fb)


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


    def toVolts(self,dataOut, sendmode):
        #print("doing toVolts")
        dataOut = numpy.array(dataOut,dtype="float64")
        if sendmode == 0:
            dataOut[:,:,:,0]/=float((2**12-1)*self.nSamp) # error
            dataOut[:,:,:,1]/=float(2**14-1) # FBA
        if sendmode == 2:
            dataOut[:,:,:,0]/=float(2**14-1) # FBA
            dataOut[:,:,:,1]/=float(2**14-1) # FBB
        return dataOut


    # emulate easyClientNDFB
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
    def num_of_samples(self):
        return self.nSamp
    @property
    def lsync(self):
        return self.linePeriod
    @property
    def sample_rate(self):
        return 1/self.samplePeriod


    def __repr__(self):
        return "EasyClientDastard {} columns X {} rows, linePeriod {}, clockMhz {}, nsamp {}".format(self.ncol,self.nrow,self.lsync,self.clockMhz, self.nsamp)

if __name__ == '__main__':
    if False:
        import pylab as plt
        plt.ion()
        plt.close("all")
        c = EasyClientDastard()
        c.clearSocket()
        sendModes = [0]
        sendModes = ["raw",0,1,2]
        for sendMode in sendModes:
            data = c.getNewData(delaySeconds=0.001,minimumNumPoints=6000,sendMode=sendMode)
            plt.figure()
            plt.plot(data[0,0,:,0],label="err (lastind 0)")
            plt.plot(data[0,0,:,1],label="fb (lastind 1)")
            plt.title("send mode = {}".format(sendMode))
            plt.xlabel("framecount")
            plt.ylabel("value")
            plt.legend()
        plt.show()

        if c.sourceName == "Lancero":
            mixFractions = [0,0.01,0.1,0.2,0.4,1]
            plt.figure()
            for mixFraction in mixFractions:
                c.setMixChannel(1,mixFraction)
                data = c.getNewData(.1)
                plt.plot(data[0,0,:,1],label="mixFrac {}".format(mixFraction))
            plt.legend()
            plt.ylabel("fb (lastind = 1)")
            plt.figure()
            for mixFraction in mixFractions:
                c.setMixChannel(1,.0001)
                data = c.getNewData(0.1)
                plt.plot(data[0,0,:,0],label="mixFrac {}".format(mixFraction))
            plt.ylabel("err (lastind = 0)")
            plt.legend()

        plt.show()
        eval(input())

    if True:
        # search for drops in one channel
        firsts = []
        c = EasyClientDastard()
        i=0
        ndrops=0
        while True:
            header,data = c.getMessage()
            if header["chan"]!=1:
                continue
            i+=1
            if i%100==0:
                print(i)
            firsts.append(header["triggerFramecount"])
            if np.sum(np.diff(firsts)<0)>0:
                print((np.sum(np.diff(firsts)<0), firsts))
            if np.sum(np.diff(firsts)>len(data))>ndrops:
                ndrops+=1
                print(("drop", ndrops, np.diff(firsts)/len(data)))

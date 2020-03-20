from time import sleep
import serial_instrument
import serial
import numpy as np
import ethernet_instrument
import socket

class uxMAN_Serial(serial_instrument.SerialInstrument):
    def __init__(self,port="",baud=115200):
       super(uxMAN_Serial, self).__init__(port,baud=baud,stopbits=serial.STOPBITS_ONE, parity=serial.PARITY_NONE,
        bytesize=serial.EIGHTBITS, min_time_between_writes=0.1, readtimeout=0.1)

    def send(CMD,ARG):
        cmd = uxMAN_Command(CMD,ARG)
        self.write(cmd.fullstring())

    def requestkvsetpoint(self):
        cmd = uxMAN_Command(14)
        print cmd.fullstring()
        return self.ask(cmd.fullstring())

    def requestmasetpoint(self):
        cmd = uxMAN_Command(15)
        print cmd.fullstring()
        return self.ask(cmd.fullstring())

    def requestanalogmonitorreadbacks(self):
        cmd = uxMAN_Command(20)
        print cmd.fullstring()
        return self.ask(cmd.fullstring())

    def requeststatus(self):
        cmd = uxMAN_Command(22)
        print cmd.fullstring()
        return self.ask(cmd.fullstring())

    def programkvsetpoint(self):
        cmd=uxMAN_Command(10,4095)
        print cmd.fullstring()
        return self.ask(cmd.fullstring())

class uxMAN(ethernet_instrument.EthernetInstrument):
        ERRORCODES = {"E#1":"uxMAN error code 1: out of range",
                      "E#2":"uxMAN error code 2: interlock enabled"}
        ANALOGMONITORMEANINGS =             ["control board temperature",
                    "low voltage supply monitor",
                    "kV feedback",
                    "mA feedback",
                    "filament current",
                    "filament voltage",
                    "high voltage board temperature"]
        STATUSMEANINGS = ["HV On", "Interlock Open", "Fault Condition"]
        def __init__(self,hostname="192.168.1.4",tcpPort=50001, verbose=False, enable_highvoltagestatus=True):
        #    super(uxMAN, self).__init__(hostname, tcpPort, useUDP=False)
           self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
           self.sock.connect( (hostname,tcpPort))
           self.hostname = hostname
           self.port = tcpPort
           self.verbose = verbose
           if enable_highvoltagestatus:
               self.programhighvoltagestatus(True) # remote commands wont work without this

        def send(CMD,ARG):
            cmd = uxMAN_Command(CMD,ARG,add_checksum=False)
            self.write(cmd.fullstring())

        def requestkvsetpoint(self):
            cmd = uxMAN_Command(14,add_checksum=False)
            reply = self.ask(cmd.fullstring(),"requestkvsetpoint")
            return int(reply[0])

        def requestmasetpoint(self):
            cmd = uxMAN_Command(15,add_checksum=False)
            reply = self.ask(cmd.fullstring(),"request ma setpoint")
            return int(reply[0])

        def requestfilamentcurrentlimit(self):
            cmd = uxMAN_Command(17,add_checksum=False)
            reply = self.ask(cmd.fullstring(),"request filament current limit")
            return int(reply[0])


        def requestanalogmonitorreadbacks(self):
            cmd = uxMAN_Command(20,add_checksum=False)
            reply =  self.ask(cmd.fullstring(),"request analog monitor readbacks")
            reply_int = map(int,reply)
            return {k:v for (k,v) in zip(self.ANALOGMONITORMEANINGS,reply_int)}


        def requeststatus(self):
            cmd = uxMAN_Command(22,add_checksum=False)
            reply =  self.ask(cmd.fullstring(),"request status")
            reply_bool = map(bool,map(int,reply))
            return {k:v for (k,v) in zip(self.STATUSMEANINGS,reply_bool)}


        def programkvsetpoint(self,val):
            cmd=uxMAN_Command(10,val,add_checksum=False)
            reply =  self.ask(cmd.fullstring(),"program kv setpoint %g"%val)
            assert reply == ['$']

        def programmasetpoint(self,val):
            cmd=uxMAN_Command(11,val,add_checksum=False)
            reply =  self.ask(cmd.fullstring(),"program ma setpoint %g"%val)
            assert reply == ['$']

        def programhighvoltagestatus(self,val):
            cmd=uxMAN_Command(99,int(val),add_checksum=False)
            reply =  self.ask(cmd.fullstring(),"program high voltage status %g"%val)
            assert reply == ['$']

        def askraw(self,msg):
            assert self.sock.send(msg)==len(msg)
            data = self.sock.recv(4096)
            return data

        def ask(self,msg,description):
            data = self.askraw(msg)
            if self.verbose:
                print(description)
                print("sent: ",msg)
                print("reply: ",data)
            if data[0] == uxMAN_Command.STX:
                assert data[-1]==uxMAN_Command.ETX
                return data.split(",")[1:-1]
            else:
                return self.ERRORCODES.get(data,data) # return the error code description if there is one, otherwise return the data


class uxMAN_Command():
    STX = chr(0x02)
    ETX = chr(0x03)
    ARGRANGES = {7:(0,5),
                 10:(0,4095),
                 11:(0,4095),
                 12:(0,4095),
                 13:(0,4095),
                 14:None,
                 15:None,
                 16:None,
                 17:None,
                 19:None,
                 20:None,
                 21:None,
                 22:None,
                 23:None,
                 24:None,
                 26:None,
                 30:None,
                 32:None,
                 52:None,
                 65:None,
                 66:None,
                 99:(0,1)}
    def __init__(self,CMD,ARG="",add_checksum=True):
        self.CMD = "%i"%CMD # force CMD to be a string of an integer, eg 1 not 1.0
        if not ARG=="":
            ARG = "%i"%ARG # force ARG to be a string of an integer, eg 1 not 1.0
        self.ARG = ARG
        self.add_checksum = add_checksum
        self.validateARG()

    def validateARG(self):
        intcmd = int(self.CMD)
        argrange = self.ARGRANGES[intcmd]
        if argrange is None:
            if self.ARG=="":
                return
        lo,hi = argrange
        intarg = int(self.ARG)
        if lo<=intarg<=hi:
            return
        raise Exception("ARG {} not in range {} for CMD {}".format(self.ARG,argrange,self.CMD))

    def corestring(self):
        if len(self.ARG)>0:
            return self.CMD+","+self.ARG+","
        else:
            return self.CMD+","

    def checksum(self):
        if not self.add_checksum: return ""
        b = map(ord,self.corestring())
        s = np.sum(b) # sum bytes of the corestring
        tc = 256-s # twos complement
        trunc = tc&0x7F # take only 7 least significant bits
        result = trunc|0x40 # set 6th bit high
        return chr(result)

    def fullstring(self):
        return self.STX+self.corestring()+self.checksum()+self.ETX


c = uxMAN_Command(10,4095)
assert c.checksum()=="u"
assert c.fullstring()=='\x0210,4095,u\x03'
c2 = uxMAN_Command(22)
assert c2.checksum() == "p"
failed = False
try:
    c3 = uxMAN_Command(10,100000)
except:
    failed = True
assert failed

if __name__=="__main__":
    uxman = uxMAN(verbose=True)
    print uxman.requeststatus()
    print uxman.requestmasetpoint()
    print uxman.requestkvsetpoint()
    print uxman.requestanalogmonitorreadbacks()
    print uxman.programmasetpoint(0)
    print uxman.programhighvoltagestatus(True)

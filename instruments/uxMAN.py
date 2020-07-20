import time
from . import serial_instrument
import serial
import numpy as np
from . import ethernet_instrument
import socket

settle_std_thresholds = {'control board temperature': 2,
    'filament current': 2,
    'filament voltage': 2,
    'high voltage board temperature': 2,
    'kV feedback': 1,
    'low voltage supply monitor': 10,
    'mA feedback': 1}

class uxMAN(ethernet_instrument.EthernetInstrument):
    """Simplest usgage:
    ux = instruments.uxMAN()
    ux.on(kvsetpoint=2000, masetpoint=1000)
    # use xrays
    ux.off()

    Both setpoints have a range 0,4095
    The actual voltage and current seems to depend on both the setpoint and the
    analog knob position

    To get working on Horton:
    1. Connect a cable from the 2nd ethernet port to uxMAN
    2. In top right Network Icon->eno1->WiredSettings
    3. Disable eno1
    4. Edit eno1 settings icon-ipv4 tab
      a. IPv4 method = Manual
      b. address = 192.168.1.1
      c. netmask = 255.255.255.0
      d. gateway = 192.168.1.1
      e. apply
    5. enable eno1
    6. eno1 settings icon
    7. check that it shows the ip you just entered
    9. unplug uxMAN, set DIP switches according to chart, plug back ing
    8. now it should connect when you insantiate uxMan(), also you can go to 192.168.1.4 in your address bar
    
    See manual at: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiOvuyh1tzqAhWTB50JHQHKCY8QFjAAegQIBBAB&url=https%3A%2F%2Fwww.spellmanhv.com%2F-%2Fmedia%2Fen%2FTechnical-Resources%2FManuals%2FuXMAN.pdf%3Fla%3Den%26hash%3D6A65DA8146DBE20B0DC12BD6AFB263A9&usg=AOvVaw3FiR5VipG4PX_UXNwbYjqr
    """
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
        self.write(cmd.bytes())

    def requestkvsetpoint(self):
        cmd = uxMAN_Command(14,add_checksum=False)
        reply = self.ask(cmd.bytes(),b"requestkvsetpoint")
        return int(reply[0])

    def requestmasetpoint(self):
        cmd = uxMAN_Command(15,add_checksum=False)
        reply = self.ask(cmd.bytes(),b"request ma setpoint")
        return int(reply[0])

    def requestfilamentcurrentlimit(self):
        cmd = uxMAN_Command(17,add_checksum=False)
        reply = self.ask(cmd.bytes(),b"request filament current limit")
        return int(reply[0])


    def requestanalogmonitorreadbacks(self):
        cmd = uxMAN_Command(20,add_checksum=False)
        reply =  self.ask(cmd.bytes(),b"request analog monitor readbacks")
        reply_int = list(map(int,reply))
        return {k:v for (k,v) in zip(self.ANALOGMONITORMEANINGS,reply_int)}


    def requeststatus(self):
        cmd = uxMAN_Command(22,add_checksum=False)
        reply =  self.ask(cmd.bytes(),b"request status")
        reply_bool = list(map(bool,list(map(int,reply))))
        return {k:v for (k,v) in zip(self.STATUSMEANINGS,reply_bool)}


    def programkvsetpoint(self,val):
        cmd=uxMAN_Command(10,val,add_checksum=False)
        reply =  self.ask(cmd.bytes(),b"program kv setpoint %g"%val)
        assert reply == ['$']

    def programmasetpoint(self,val):
        cmd=uxMAN_Command(11,val,add_checksum=False)
        reply =  self.ask(cmd.bytes(),b"program ma setpoint %g"%val)
        assert reply == ['$']

    def programhighvoltagestatus(self,val):
        cmd=uxMAN_Command(99,int(val),add_checksum=False)
        reply =  self.ask(cmd.bytes(),b"program high voltage status %g"%val)
        assert reply == ['$']

    def askraw(self,msg):
        assert self.sock.send(msg)==len(msg)
        data = self.sock.recv(4096)
        return data

    def ask(self,msg,description):
        data = self.askraw(msg).decode()
        if self.verbose:
            print(description)
            print(("sent: ",msg))
            print(("reply: ",data))
        if data[0] == uxMAN_Command.STX:
            assert data[-1]==uxMAN_Command.ETX
            return data.split(",")[1:-1]
        else:
            return self.ERRORCODES.get(data,data) # return the error code description if there is one, otherwise return the data

    def is_settled(self, r_history, n, settle_std_thresholds):
        " look at the last n sampleof each entry in r_history and compare the std dev to the std_threshold"
        N = len(r_history['filament current'])
        if N<n:
            return False
        for k,v in r_history.items():
            if np.std(v[-n:])>settle_std_thresholds[k]:
                return False
        return True

    def wait_for_settle(self, n=5, pause_seconds = 0.1, settle_std_thresholds=settle_std_thresholds):
        r_history = {'control board temperature': [],
        'filament current': [],
        'filament voltage': [],
        'high voltage board temperature': [],
        'kV feedback': [],
        'low voltage supply monitor': [],
        'mA feedback': []}
        tstart = time.time()
        ERROR_AFTER = 10 # seconds
        while True:
            r = self.requestanalogmonitorreadbacks()
            for (k,v) in r.items():
                r_history[k].append(float(v))
            if self.is_settled(r_history, n, settle_std_thresholds):
                break
                if time.time()-tstart > ERROR_AFTER:
                    raise Exception("settling took long, ERROR_AFTER = {}".format(ERROR_AFTER))
            time.sleep(pause_seconds)
        return time.time()-tstart

    def on(self, kvsetpoint, masetpoint):
        self.programkvsetpoint(kvsetpoint)
        self.programmasetpoint(masetpoint)
        self.programhighvoltagestatus(True)
        self.wait_for_settle()
        print(self.requestanalogmonitorreadbacks())
        masetpoint = self.requestmasetpoint()
        kvsetpoint = self.requestkvsetpoint()
        print(f"read back kvsetpoint={kvsetpoint}, masetpoint={masetpoint}")

    def off(self):
        self.programkvsetpoint(0)
        self.programmasetpoint(0)
        self.programhighvoltagestatus(False)
        self.wait_for_settle()


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
        b = list(map(ord,self.corestring()))
        s = np.sum(b) # sum bytes of the corestring
        tc = 256-s # twos complement
        trunc = tc&0x7F # take only 7 least significant bits
        result = trunc|0x40 # set 6th bit high
        return chr(result)

    def bytes(self):
        return (self.STX+self.corestring()+self.checksum()+self.ETX).encode()


c = uxMAN_Command(10,4095)
assert c.checksum()=="u"
assert c.bytes()==b'\x0210,4095,u\x03'
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
    print((uxman.requeststatus()))
    print((uxman.requestmasetpoint()))
    print((uxman.requestkvsetpoint()))
    print((uxman.requestanalogmonitorreadbacks()))
    print((uxman.programmasetpoint(0)))
    print((uxman.programhighvoltagestatus(True)))

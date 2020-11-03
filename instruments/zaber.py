import math
import time
import struct

from . import serial_instrument

#########################################################################
#
# Zaber Stepper Motor Class
#
# Class to control the Zaber motor via serial port. This is usually used
# as a heat switch.
#
# protocol reference:
# https://www.zaber.com/protocol-manual?protocol=Binary#topic_action_055_echo_data
#
# by Doug Bennett
# Based of similar class for Matlab by Dan Schmidt
# modified by Frank Schima
##########################################################################

ZABER_ERRORS = {
    1  : "cannot home",
    2  : "device number invalid",
    14 : "voltage low",
    15 : "voltage high",
    17 : "stored position invalid",
    18 : "stored postion invalid",
    20 : "absolute position invalid",
    21 : "relative position invalid",
    22 : "velocity invalid",
    36 : "peripheral id invalid",
    37 : "resolution invalid",
    38 : "run current invalid",
    39 : "hold current invalid",
    64 : "requested command number is invalid in this firmware version"
} # this dict is incomplete, but basically it just means the command failed and the keyis the command number

class Zaber(serial_instrument.SerialInstrument):

######################################################################################
    # Zaber class

    def __init__(self, port='zaber', baud=9600, shared=True, unit=2):
        '''Zaber motor - parameters:
        port - a logical portname defined in namedserialrc
        baud -  9600 for zaber
        shared - allow other processes to open this device
                 only works on posix
        unit - Unit number. 0 for all.
        '''

        super(Zaber, self).__init__(port, baud, shared, readtimeout=1, min_time_between_writes=0.5)

        self.manufacturer = 'Zaber'
        self.model_number = 'T-NM17A200'

        self.port = port
        self.baud = baud
        self.shared = shared
        self.unit = unit
        #self.serial = None
#        self.value = None  #unknown value
        self.MicroStepsPerRev = 12800  #microsteps per revolution
        self.RevsPerSecAtSpeed1 = 7.0000e-004  #rev/sec at speed = 1

        # Values from Dan - may be good for all heat switches
        self.OpeningSpeed = 50
        self.OpeningCurrent = 10  #full current
        self.OpeningRevs = 10  #how much to open switch
        self.SlowRevs = 2  #how many revs to open slowly at high torque
        self.ClosingSpeed = 1000
        self.ClosingCurrent = 10 #less than full or not
        self.debug = False # set to true to enable debug prints
        _ = self.serial.readall() # just make sure there isn't a backlog of comms
    
########################################### Private Methods #################################################

    def _debug(self, s):
        if self.debug: print(s)

    def __getBytes(self, value):
        assert isinstance(value, int)
        # convert Data to 4 bytes for serial writing, following algorithm
        # in Appendix B of zaber binary protocol manual
        if value < 0: # handle negatie values by calculating the twos complement
            value += 256**4
        b = [0, 0, 0, 0]
        b[3] = value // 256**3
        value -= b[3]*256**3
        b[2] = value // 256**2
        value -= b[2]*256**2
        b[1] = value // 256
        value -= b[1]*256
        b[0] = value
        return b

    def __sendCommand(self, value, command):
        self._debug(f"value={value} command={command}")
        value_bytes = self.__getBytes(value)
        b = [self.unit, command] + value_bytes        
        nwritten = self.serial.write(bytes(b))
        assert nwritten == len(b), f"tried to write {len(b)} bytes, actually wrote {nwritten}"
        self._debug(f"bytes being sent{b}")        

    def __sendCommandParseReply(self, value, command):
        self.__sendCommand(value, command)
        reply = self.serial.read(6) # will block for timeout
        return self.__parseReplyToInt(command, reply, value)

    def __parseReplyToInt(self, command, reply, command_value):
        self._debug(f"reply={reply}")
        if len(reply) == 0:
            raise Exception(f"command {command} got empty reply")
        elif len(reply) > 6:
            raise Exception(f"reply too long, length={len(reply)} reply={reply}")
        elif len(reply) != 6: 
            raise Exception(f"all reply should be length 6, got length {len(reply)}, reply={reply}")
        b = struct.unpack("BBBBBB", reply)
        self._debug(f"reply_bytes={b}")
        # assert b[0] == self.unit
        # in principle we should check the reply is from the right unit number
        # but in practice we always use just one, so who cares?
        v = b[2] + b[3]*256 + b[4]*256**2 + b[5]*256**3
        if b[5] > 127: # handle twos complement
            v -= 256**4
        if b[1] == 255:
            raise Exception(self.__errorString(v, command, command_value))
        else:
            assert b[1] == command, f"2nd reply byte was {b[1]}, should equal the command {command}"
        return v
    
    def __errorString(self, v, command, command_value):
        error_string = f"zaber command {command} with data {command_value} failed with code {v}: " + ZABER_ERRORS.get(v, f"unknown code")
        return error_string

########################################### Public Methods #################################################

    def getFirmwareVersion(self):
        reply_int = self.__sendCommandParseReply(value=0, command=51)
        version_float = reply_int / 100.0
        return version_float

    def close(self):
        self.serial.close()

    def Stop(self):
        #stop motor
        self.__sendCommandParseReply(value=0, command=23)

    def MoveConstantVelocity(self, velocity=500):
        #constant velocity move , sign indicates direction
        #keeps moving until another command is issued
        self.__sendCommandParseReply(value=velocity, command=22)


    def SetCurrentPosition(self, position=200000):
        #set current position
        self.__sendCommandParseReply(value=position, command=45)


    def SetTargetVelocity(self, velocity=50):
        #set target velocity
        self.__sendCommandParseReply(value=velocity, command=42)


    def SetHoldCurrent(self, current=0):
        #set running current
        #set running current Range 0, 10-127   0: no current, 10 max , 127 min
        self.__sendCommandParseReply(value=current, command=39)

    def SetRunningCurrent(self, current=10):
        #set running current
        #set running current Range 0, 10-127   0: no current, 10 max , 127 min
        self.__sendCommandParseReply(value=current, command=38)

    # dont use this, because it doesnt return a reply until it finishes, so you need logic to
    # to wait for the reply
    # def MoveRelative(self, steps=0):
    #     #move steps # of microsteps
    #     self.__sendCommandParseReply(value=steps, command=21)

    def OpenHeatSwitch(self, OpeningRevs = None, SlowRevs = None):

        if OpeningRevs is None:
            OpeningRevs = self.OpeningRevs

        if SlowRevs is None:
            SlowRevs = self.SlowRevs

        # high torque low speed
        self.SetRunningCurrent(self.OpeningCurrent)
        self.SetTargetVelocity(self.OpeningSpeed)
        self.SetCurrentPosition(1000000) # makes negative relative moves work
        self.MoveRelativeThenWait(int(SlowRevs*self.MicroStepsPerRev), self.OpeningSpeed) # open 2 revolutions
        # time.sleep(SlowRevs/(self.OpeningSpeed*self.RevsPerSecAtSpeed1)*1.2) # wait for motion to complete


        # drop torque , up speed and finish
        self.SetRunningCurrent(self.ClosingCurrent)
        self.SetTargetVelocity(self.ClosingSpeed)
        self.SetCurrentPosition(1000000) # makes negative relative moves work
        self.MoveRelativeThenWait(int((OpeningRevs-SlowRevs)*self.MicroStepsPerRev), self.ClosingSpeed) # open rest of revs
        # time.sleep((OpeningRevs-SlowRevs)/(self.ClosingSpeed*self.RevsPerSecAtSpeed1)*1.2) # wait for motion to complete

    def CloseHeatSwitch(self, ClosingRevs = None):

        if ClosingRevs is None:
            ClosingRevs = self.OpeningRevs + 2 # +2 to ensure it closes

        self.SetRunningCurrent(self.ClosingCurrent)
        self.SetTargetVelocity(self.ClosingSpeed)
        self.SetCurrentPosition(1000000) # makes negative relative moves work
        self.MoveRelativeThenWait(int(-ClosingRevs*self.MicroStepsPerRev), self.ClosingSpeed)
        # time.sleep(ClosingRevs/(self.ClosingSpeed*self.RevsPerSecAtSpeed1)*1.2)

    def getPosition(self):
        value = self.__sendCommandParseReply(value=0, command=60)

        return value

    def echo(self, v):
        """
        send value v to the zaber, the zaber should return the same value
        useful for testing
        """
        assert isinstance(v, int)
        return self.__sendCommandParseReply(value=v, command=55)

    def getCurrentPosition(self):
        return self.__sendCommandParseReply(value=0, command=60)

    def getSerialNumber(self):
        return self.__sendCommandParseReply(value=0, command=63)

    def getStatus(self):
        code = self.__sendCommandParseReply(value=0, command=54)
        interp_dict = {
            0  : "idle",
            21 : "motion", # seems to be returned during motion, even though manual says it should be 99
            65 : "parked",
            90 : "diabled",
            93 : "inactive",
            99 : "motion"
        }
        if code not in interp_dict:
            raise Exception(f"status code = {code} invalid, not in interp_dict = {interp_dict}")
        return interp_dict[code]

    def getPowerSupplyVoltage(self):
        return self.__sendCommandParseReply(value=0, command=52)

    def getEncoderPosition(self):
        return self.__sendCommandParseReply(value=0, command=89)

    def waitForRelativeMoveToComplete(self, timeout_s):
        """ this function should work, but it always causes an error of some sort, 
        don't expect it to work"""
        tstart = time.time()
        while True:
            reply = self.serial.read(6)
            if len(reply) == 0:
                continue
                # move is not complete, because it hasn't responded
            elif len(reply) == 6:
                b = struct.unpack("BBBBBB", reply)
                assert b[1] == 21, f"expected respond to move relative (21), got {b[1]}"    
                break
            else:
                raise Exception(f"reply should be length 0 or 6, got {length(reply)}, reply={reply}")
            elapsed_s = time.time()-tstart
            if elapsed_s > timeout_s:
                raise Exception(f"should have finished relative move by now. elapsed_s = {elapsed_s}, timeout_s = {timeout_s}")

    def MoveRelativeThenWait(self, microsteps, speed):
        revolutions = microsteps/self.MicroStepsPerRev
        estimated_time_s = revolutions/(self.RevsPerSecAtSpeed1*speed)
        self.__sendCommand(value=microsteps, command=21)
        self.waitForRelativeMoveToComplete(1.3*estimated_time_s) 


    # in practice this just causes the heat switch to turn a lot, so commenting it out
    # def setKnobDisable(self, val: bool):
    #     assert isinstance(val, bool)
    #     self.__sendCommand(107, int(val))

    # setting it to "displacement" causes the motor to just keep turning
    # def setKnobMovementMode(self, mode):
    #     if mode == "displacement":
    #         val = 1
    #     elif mode == "velocity": 
    #         val = 0
    #     else:
    #         raise Exception("only displacement and velocity are supported movement modes")

    #     self.__sendCommand(109, val)

    # setting it to 1 causes the motor to just keep turning
    # def setKnobJogSize(self, size):
    #     self.__sendCommand(110, size)
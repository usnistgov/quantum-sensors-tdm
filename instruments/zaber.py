import math
from time import sleep
import struct

from . import serial_instrument

#########################################################################
#
# Zaber Stepper Motor Class
#
# Class to control the Zaber motor via serial port. This is usually used
# as a heat switch.
#
# by Doug Bennett
# Based of similar class for Matlab by Dan Schmidt
# modified by Frank Schima
##########################################################################


class Zaber(serial_instrument.SerialInstrument):

######################################################################################
    # Zaber class

    def __init__(self, port='zaber', baud=9600, shared=True, unit=0):
        '''Zaber motor - parameters:
        port - a logical portname defined in namedserialrc
        baud -  9600 for zaber
        shared - allow other processes to open this device
                 only works on posix
        unit - Unit number. 0 for all.
        '''

        super(Zaber, self).__init__(port, baud, shared,readtimeout=1,min_time_between_writes=0.0)

        self.manufacturer = 'Zaber'
        self.model_number = 'T-NM17A200'

        self.port = port
        self.baud = baud
        self.shared = shared
        self.unit = unit
        #self.serial = None
#        self.value = None  #unknown value
        self.MicroStepsPerRev = 12800  #microsteps per revolution
        self.SpeedConvert = 7.0000e-004  #rev/sec at speed = 1

        # Values from Dan - may be good for all heat switches
        self.OpeningSpeed = 50
        self.OpeningCurrent  = 10  #full current
        self.OpeningRevs = 10  #how much to open switch
        self.SlowRevs = 2  #how many revs to open slowly at high torque
        self.ClosingSpeed = 1000
        self.ClosingCurrent = 10 #less than full or not

########################################### Private Methods #################################################

    def __getBytes(self, value):

        #convert Data to 4 bytes for serial writing

        if value >= 0:
            #if positive convert normally
            bytevals = [0, 0, 0, 0]
            bytevals[3]= int(math.floor(value/(256*256*256)))
            value = value - (256*256*256)* bytevals[3]
            bytevals[2] = int(math.floor(value/(256*256)))
            value = value - (256*256)* bytevals[2]
            bytevals[1] = int(math.floor(value/(256)))
            bytevals[0] = int(value - 256* bytevals[1])
        else:
            #two's complement for negative
            value = -1*value
            value = int(value)
            bytevals = [0, 0, 0, 0]
            bytevals[3]= math.floor(value/(256*256*256))
            value = value - (256*256*256)* bytevals[3]
            bytevals[2] = math.floor(value/(256*256))
            value = value - (256*256)* bytevals[2]
            bytevals[1] = math.floor(value/(256))
            bytevals[0] = value - 256* bytevals[1]

            bytevals[3] = int(255 - bytevals[3])
            bytevals[2] = int(255 - bytevals[2])
            bytevals[1] = int(255 - bytevals[1])
            #bytevals[0] = int(255 - bytevals[0] + 1)
            bytevals[0] = int(255 - bytevals[0])

        return bytevals


    def __writeBytes(self, bytestowrite):

        result = self.serial.writelist(bytestowrite)
        if result != len(bytestowrite):
            print(('Error in serial write to Zaber, result, len(bytestowrite)', result, len(bytestowrite)))
        return result


    def __sendCommand(self, value, command):

        bytevals = self.__getBytes(value)
        bytevals.insert(0, command)
        bytevals.insert(0, self.unit)

        self.__writeBytes(bytevals)
        return self.serial.read(6)

########################################### Public Methods #################################################

    def getFirmwareVersion(self):

        Command = 51

        value = self.__sendCommand(value=0, command=Command)

        byteval_1 = value[0]
        byteval_2 = value[1]
        byteval_3 = value[2]
        byteval_4 = value[3]
        byteval_5 = value[4]
        byteval_6 = value[5]

        # Device #
        version1, = struct.unpack("B", byteval_1)
        # Command #
        version2, = struct.unpack("B", byteval_2)
        # Return data
        version3, = struct.unpack("B", byteval_3)
        version4, = struct.unpack("B", byteval_4)
        version5, = struct.unpack("B", byteval_5)
        version6, = struct.unpack("B", byteval_6)

        # Convert data to integer value
        version = 256**3 * version6 + 256**2 * version5 + 256 * version4 + version3
        if version6 > 127:
            version = version - 256^4

        version_float = version / 100.0

        return version_float

    def close(self):
        self.serial.close()

    def Stop(self):
        #stop motor

        Command = 23

        self.__sendCommand(value=0, command=Command)


    def MoveConstantVelocity(self, velocity=500):
        #constant velocity move , sign indicates direction
        #keeps moving until another command is issued

        Command = 22

        self.__sendCommand(value=velocity, command=Command)


    def SetCurrentPosition(self, position=200000):
        #set current position

        Command = 45

        self.__sendCommand(value=position, command=Command)


    def SetTargetVelocity(self, velocity=50):
        #set target velocity

        Command = 42

        self.__sendCommand(value=velocity, command=Command)


    def SetHoldCurrent(self, current=0):
        #set running current
        #set running current Range 0, 10-127   0: no current, 10 max , 127 min

        Command = 39

        self.__sendCommand(value=current, command=Command)

    def SetRunningCurrent(self, current=10):
        #set running current
        #set running current Range 0, 10-127   0: no current, 10 max , 127 min

        Command = 38

        self.__sendCommand(value=current, command=Command)

    def MoveRelative(self, steps=0):
        #move steps # of microsteps

        Command = 21

        self.__sendCommand(value=steps, command=Command)

    def OpenHeatSwitch(self, OpeningRevs = None, SlowRevs = None):

        if OpeningRevs is None:
            OpeningRevs = self.OpeningRevs

        if SlowRevs is None:
            SlowRevs = self.SlowRevs

        #print 'high torque low speed'
        self.SetRunningCurrent(self.OpeningCurrent)
        sleep(1)
        self.SetTargetVelocity(self.OpeningSpeed)
        sleep(1)
        self.SetCurrentPosition(200000)   #set to middle of range
        sleep(1)
        self.MoveRelative(int(SlowRevs*self.MicroStepsPerRev)) # open 2 revolutions
        sleep(SlowRevs/(self.OpeningSpeed*self.SpeedConvert)*1.2) #wait for motor to finish 1.2 to be certain

        #print 'drop torque , up speed and finish'
        self.SetRunningCurrent(self.ClosingCurrent)
        sleep(1)
        self.SetTargetVelocity(self.ClosingSpeed)
        self.MoveRelative(int((OpeningRevs-SlowRevs+.5)*self.MicroStepsPerRev)) # open 8.5 more revs
        sleep((OpeningRevs-SlowRevs+.5)/(self.ClosingSpeed*self.SpeedConvert)*1.3) #wait for motor to finish 1.3 to be certain

    def CloseHeatSwitch(self, ClosingRevs = None):

        if ClosingRevs is None:
            ClosingRevs = self.OpeningRevs + 2 # +2 to ensure it closes

        #print 'closing heat switch'
        self.SetRunningCurrent(self.ClosingCurrent)
        sleep(1)
        self.SetTargetVelocity(self.ClosingSpeed)
        self.SetCurrentPosition(200000)   #set to middle of range
        #move relative - OpeningRevs + 2revolutions (negative to give clockwise rotation)
        self.MoveRelative(int(-(ClosingRevs)*self.MicroStepsPerRev))
        sleep(ClosingRevs/(self.ClosingSpeed*self.SpeedConvert)*1.3)

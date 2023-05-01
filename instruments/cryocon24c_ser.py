'''
Lakesore370 
Created on Mar 11, 2009
@author: bennett
'''
import sys
from . import serial_instrument
from .lookup import Lookup
from time import sleep
import math
import numpy
from threading import Lock
#from scipy.io import read_array #obsolete, replace with numpy.genfromtxt
import pylab
#import scipy
#from scipy.interpolate import interp1d
#from tkSimpleDialog import askfloat

class Cryocon24c_ser(serial_instrument.SerialInstrument):
    '''
    The Cryocon 24C serial communication class
    '''


    def __init__(self, port='cryocon1', baud=9600, shared=True):
        '''Constructor  The PAD (Primary GPIB Address) is the only required parameter '''

        super(Cryocon24c_ser, self).__init__(port, baud, shared, readtimeout=0.25)
        
        self.id_string = ""
        self.manufacturer = 'Cryocon'
        self.model_number = '24c'
        self.description  = 'Temperature Controller'

    #
    # Now we have functions that deal with the details of the Cryocon24c command language
    #

    # xxx beckerd - this one has stuff for serial comm failure, but not others ... why?
    def getTemperature(self, channel, verbose=False):
        ''' Get temperature from a given channel as a float '''

        commandstring = 'INPut? ' + str(channel)
        try:
            response = self.askFloat(commandstring)
        except:
            #print('Serial Communication Failed')
            response = None
        if verbose is True:
            print(response)
        if response is None:
            temperature = numpy.nan
        else:
            temperature = response

        return temperature


    def getHeaterPower(self, channel=1):
        ''' Get temperature from a given channel as a float '''

        commandstring = f"LOOP {channel}:OUTPwr?"
        response = self.askFloat(commandstring)
        power = response

        return power

    def setLoopThermometer(self, loop, thermometer):
        '''Set the thermometer for a loop '''

        commandstring = 'LOOP ' + str(loop) + ':SOURce ' + str(thermometer)
        result = self.ask(commandstring)
        print("setLoopThermometer")
        print(commandstring)
        print(result)

    def setTemperature(self, channel, temp):
        '''Set temperature to a given channel '''

        if temp is not None:
            commandstring = 'LOOP ' + str(channel) + ':SETPt ' + str(temp)
            result = self.ask(commandstring)
            print("setTemperature")
            print(commandstring)
            print(result)
        else:
            print('Not a valid temperature')

    def getTemperatureSetpoint(self, channel):
        '''Get temperature setpoint for a given channel '''

        commandstring = 'LOOP? ' + str(channel) + ':SETPt?'
        ans = self.ask(commandstring)
        # print('Crycon24c getTemperatureSetpoint response:', ans)
        if ans[-3:] != b'K\r\n':
            raise Exception("response should end with K")
        return float(ans[:-3])

    def STOP(self):
        '''Stop regulating the temperature'''
        commandstring = 'STOP'
        self.ask(commandstring)

    def CONTrol(self):
  
        commandstring = 'CONT'
        self.ask(commandstring)
        # sleep(5)

    def getControl(self):

        commandstring = 'CONT?'
        return self.ask(commandstring).strip()

    def setHeaterRange(self, loop, Vrange):
	
        commandstring = 'LOOP ' + str(loop) + ':RANGe ' + str(Vrange)
        self.ask(commandstring)

    def setHeaterOut(self, loop, Vpercent):

        commandstring = 'LOOP ' + str(loop) + ':PMANual ' + str(Vpercent)
        self.ask(commandstring)

    def setToManual(self, loop = 1):
	
        commandstring = 'LOOP ' + str(loop) + ':TYPe ' + 'MAN'
        self.ask(commandstring)



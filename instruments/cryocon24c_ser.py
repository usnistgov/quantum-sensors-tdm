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


    def __init__(self, port='cryocon', baud=9600, shared=True):
        '''Constructor  The PAD (Primary GPIB Address) is the only required parameter '''

        super(Cryocon24c_ser, self).__init__(port, baud, shared, readtimeout=5)
        
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

        commandstring = 'LOOP ' + str(channel) + ':OUTPwr?'
        response = self.askFloat(commandstring)
        power = response

        return power

    def setLoopThermometer(self, loop, thermometer):
        '''Set the thermometer for a loop '''

        commandstring = 'LOOP ' + str(loop) + ':SOURce ' + str(thermometer)
        self.write(commandstring)

    def setTemperature(self, channel, temp):
        '''Set temperature to a given channel '''

        if temp is not None:
            commandstring = 'LOOP ' + str(channel) + ':SETPt ' + str(temp)
            self.write(commandstring)
        else:
            print('Not a valid temperature')

    def getTemperatureSetpoint(self, channel):
        '''Get temperature setpoint for a given channel '''

        commandstring = 'LOOP? ' + str(channel) + ':SETPt?'
        ans = self.ask(commandstring)
        if ans[-3:] != 'K\r\n':
            pass # xxx need to report an error
        print('XXXXXXXX', ans)
        return float(ans[:-3])

    def STOP(self):
        '''Stop regulating the temperature'''
        commandstring = 'STOP'
        self.write(commandstring)

    def CONTrol(self):
  
        commandstring = 'CONT'
        self.write(commandstring)
        sleep(5)

    # xxx beckerd - just prints result to stdout, seems pointless
    def getControl(self):

        commandstring = 'CONT?'
        return self.ask(commandstring).strip()

    def setHeaterRange(self, loop, Vrange):
	
        commandstring = 'LOOP ' + str(loop) + ':RANGe ' + str(Vrange)
        self.write(commandstring)

    def setHeaterOut(self, loop, Vpercent):

        commandstring = 'LOOP ' + str(loop) + ':PMANual ' + str(Vpercent)
        self.write(commandstring)

    def setToManual(self, loop = 1):
	
        commandstring = 'LOOP ' + str(loop) + ':TYPe ' + 'MAN'
        self.write(commandstring)


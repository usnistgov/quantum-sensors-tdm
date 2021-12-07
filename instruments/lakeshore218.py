'''
Lakesore370
Created on Mar 11, 2009
@author: bennett
'''

from .lookup import Lookup
from time import sleep
import math
import numpy
#from scipy.io import read_array #obsolete, replace with numpy.genfromtxt
import pylab
import scipy
from scipy.interpolate import interp1d
from . import serial_instrument
import serial
from instruments import retry



class Lakeshore218(serial_instrument.SerialInstrument):
    '''
    The Lakeshore 370 AC Bridge GPIB communication class
    '''


    def __init__(self, port="ls218"):

        super(Lakeshore218, self).__init__(port, bytesize=serial.SEVENBITS, parity=serial.PARITY_ODD,
        stopbits=serial.STOPBITS_ONE, min_time_between_writes=0.05, readtimeout=0.05)

        # GPIB identity string of the instrument
        self.id_string = "??"
        self.manufacturer = 'Lakeshore'
        self.model_number = '218'
        self.description  = 'Temperature Montior'


        #self.compare_identity()

    @retry(tries=3, delay_s=0.1)
    def getTemperature(self, channel=1):
        ''' Get temperature from a given channel as a float '''

        commandstring = 'KRDG? ' + str(channel)
        self.voltage = self.askFloat(commandstring)

        return self.voltage

    @retry(tries=3, delay_s=0.1)
    def getResistance(self, channel=1):
        '''Get resistance from a given channel as a float.'''

        commandstring = 'SRDG? ' + str(channel)
        resistance = self.askFloat(commandstring)

        return resistance

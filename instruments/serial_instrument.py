'''
Created on Feb 17, 2010

@author: schimaf
@version: 1.0.1

1.0.1  Show warning if serial port does not actually exist.
'''

#import time

import instrument
from named_serial import Serial
import serial
import numpy as np
import time

class SerialInstrument(instrument.Instrument):
    '''
    Base class for serial controlled instrument
    '''

    def __init__(self, port, baud=9600, shared=True, readtimeout=1.0, min_time_between_writes=0.6, lineend = "\r\n", **kwargs):
        super(SerialInstrument, self).__init__()

        self.port = port
        self.baud = baud
        self.shared = shared
        self.serial = None
        self.lineend = lineend
        self.min_time_between_writes = min_time_between_writes # Lakeshore370 manual section 6.2.6 requires 50 ms
        self.time_of_last_write = 0 # intial value chosen to avoid wait on first write

        try:
            self.serial = Serial(port=self.port, baud=self.baud, shared=self.shared, timeout=readtimeout, **kwargs)
            #self.serial.setTimeout(readtimeout)

        except serial.serialutil.SerialException:
            print "WARNING: serial port could not be found!"

    def identify(self):
        identify_string = "%s %s" % (self.manufacturer, self.model_number)
        return identify_string

    def write(self,x):
        x+=self.lineend
        tosleep = self.time_of_last_write+self.min_time_between_writes-time.time()
        if tosleep>0:
            time.sleep(tosleep)
        self.serial.write(x)
        self.time_of_last_write=time.time()

    def read(self):
        result = self.serial.readline()
        return result

    def ask(self, x):
        self.write(x)
        result = self.read()
        return result

    def askFloat(self, string):
        '''
        send a command and get a float in return. Has error handling for empty returns.
        '''
        self.write(string)
        result = self.read()
        float_result = np.nan
        if len(result) > 0:
            float_result = float(result)

        return float_result

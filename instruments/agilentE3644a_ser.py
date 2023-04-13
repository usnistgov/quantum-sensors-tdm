'''
Created on Jan 12, 2010

@author: schimaf
'''

from . import serial_instrument

class AgilentE3644A(serial_instrument.SerialInstrument):
    '''
    Agililent E3644A DC Power Supply control class
    '''

    def __init__(self, port, baud=9600, shared=True):
        super(AgilentE3644A, self).__init__(port, baud, shared, timeout=5)
        
        self.id_string = ""
        self.manufacturer = 'Agilent'
        self.model_number = 'E3441A'
        self.description  = 'DC Power Supply'

    def outputOff(self):
        self.write("OUTP OFF")

    def outputOn(self):
        self.write("OUTP ON")

    def setVoltage(self, voltage):
        self.write('VOLT %8.6f' % voltage)

    def setCurrentLimit(self, output, voltage, amps_limit):
        '''
        Set Current Limit
        '''
        voltage_string = "%8.6f" %( voltage ) # V setting
        amps_limit_string = "%8.6f" %( amps_limit )
        self.write("APPL " + output + "," + voltage_string + "," + amps_limit_string)

    def measureCurrent(self):
        return self.askFloat("MEAS:CURR:DC?")

    def measureVoltage(self):
        return self.askFloat("MEAS:VOLT:DC?")

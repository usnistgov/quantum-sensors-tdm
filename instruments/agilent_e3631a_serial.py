'''
Created on Jan 12, 2010

@author: schimaf
'''

from time import sleep
import serial_instrument
import serial
import numpy as np

class AgilentE3631A(serial_instrument.SerialInstrument):
    '''
    Agililent E3631A control class
    '''

    def __init__(self, port):
        '''Constructor - The PAD (Primary GPIB Address) is the only required parameter '''

        super(AgilentE3631A, self).__init__(port,stopbits=serial.STOPBITS_TWO, parity=serial.PARITY_NONE,
        bytesize=serial.EIGHTBITS, min_time_between_writes=0.7)

        # GPIB identity string of the instrument
        self.id_string = "HEWLETT-PACKARD,E3631A,0,2.1-5.0-1.0"
        self.manufacturer = 'Agilent'
        self.model_number = 'E3631A'
        self.description  = 'DC Power Supply'

        self.allowed_outputs = ["P6V", "P25V", "N25V"]

        self.voltage = None
        self.powered = False

        self.initRS232()

    def initRS232(self):
        self.reset()
        self.write("SYST:REM") # put into remote operation mode
        # self.write("*RST;*CLS") # dont sent *RST, it turns off the power supplies
        self.write("*CLS") # clear status, eg clear the error queue

    def reset(self):
        self.write("\x03") # this is sending Ctrl-C, which the Agilent #3631A wants for reset
        # see https://stackoverflow.com/questions/7018139/pyserial-how-to-send-ctrl-c-command-on-the-serial-line
        # and page 87 of https://edisciplinas.usp.br/pluginfile.php/2405092/mod_resource/content/1/Agilent_E3631%20Power%20Supply_Users_Guide%20%284%29.pdf

    def powerOff(self):
        '''
        shutOff() = Shut off the power
        '''
        self.write("OUTP OFF")
        # self.displayText("--OFF--")
        self.powered = False

    def powerOn(self):
        '''
        turnOn() = Turn on the power
        '''
        self.reset()
        self.write("OUTP ON")
        # Allow time to stabilize
        self.powered = True
        sleep(1)

    def displayText(self, text_string):

        self.write('DISP:TEXT "%s"' % text_string )

    def setCurrentLimit(self, output, voltage, amps_limit):
        '''
        Set Current Limit
        '''
        voltage_string = "%8.6f" %( voltage ) # V setting
        amps_limit_string = "%8.6f" %( amps_limit )
        self.write("APPL " + output + "," + voltage_string + "," + amps_limit_string)

    def measureCurrent(self, output):
        '''
        result = measureCurrent(output)
        output = "P6V" or "P25V" or "N25V"
        P6V  =  +6V output
        P25V = +25V output
        N25V = -25V output
        Measure one of the currents from the specified output
        '''
        the_result = self.askFloat("MEAS:CURR:DC? " + output)
        return the_result

    def measureVoltage(self, output):
        '''
        result = measureVoltage(output)
        output = "P6V" or "P25V" or "N25V"
        P6V  =  +6V output
        P25V = +25V output
        N25V = -25V output
        Measure one of the voltages from the specified output
        '''
        the_result = self.askFloat("MEAS:VOLT:DC? " + output)
        return the_result

    @property
    def pad(self):
        return self.port

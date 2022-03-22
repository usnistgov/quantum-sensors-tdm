'''
cryocon22_serial
Created on Nov, 2020 adapted from cryocon_gpib.py
@author: hubmayr

From the manual:

USB configuration
The USB connection on the Model 22C is a simple serial port emulator. Therefore,
installation of drivers is not generally required. Once connected, configuration and
use is identical to a standard RS-232 serial port.
The USB serial port emulator interface supports Baud Rates of 9600, 19,200, 38,400,
57,600 and 115200. The factory default is 9600.

Other USB communications parameters are fixed in the instrument. They are:
Parity: None, Bits: 8, Stop Bits: 1, Mode: Half Duplex

Note: Ensure that the baud rate expected by your computer
matches the baud rate set in the instrument. The rate is
changeable from the instrument's front panel by using the System
Functions Menu. Default is 9600.

The USB interface uses a "New Line", or Line Feed character as a line termination. In
LabView or the C programming language, this character is \n or hexadecimal 0xA.
The controller will always return the \n character at the end of each line.

My Notes:

Cryocon22 can readout two thermometers labeled "CHA" or "CHB".  Thermometers are referred to as "channel" in this software.
Unit has capability of four control loops labeled 1,2,3,4.  These are referred to as "loop_channel" in this software.
Unit allows an number of calibration curves.  These are called by the curve_number in the software, and each curve has a name associated with it. 

On software itself:
1) line termination is actually b'\r\n' and not \n 

On application to cryogenic cold load:
1) low, mid, high control range outputs .1, .33 and 1.0A
2) Cold load setup (50Ohm termination resistor at room temperature, 1000 ohm heater 
resistor on the cold load itself is in parallel with this) gives max powers:

low: 17.9 mW
mid: 195 mW
high 1.8 W, 

Therefore never want to use the high range, and this software forbids it.
'''

from . import serial_instrument
import serial
from time import sleep
import numpy as np
import cmath
import pickle
import pylab
import math

class Cryocon22(serial_instrument.SerialInstrument):
    '''
    cryocon 22C temperature controller serial communication class
    '''

    def __init__(self, port='cryocon'):
        '''Constructor  port is the only required parameter '''

        super(Cryocon22, self).__init__(port, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE, min_time_between_writes=0.3, readtimeout=0.5, lineend = b"\r\n")
        # Empirically found that readoutout must = None (port just waits as long as it needs to get a response), then things work.  

        # identity string of the instrument
        self.id_string = "Cryocon22C SNo:205346"
        self.manufacturer = 'Cryocon'
        self.model_number = '22'
        self.description  = 'Temperature Controller' 
    
    def parseAsk(self,result):
        return result.decode().split(self.lineend.decode())[0]

    def writeAndFlush(self,thestring,verbose=False):
        # This method is a cludge used for all write commands that do not receive a response 
        # (all "sets" that do not return floats.
        # The problem: after any write command, a read command gives \r\n, the lineend.
        # This causes problems when subsequent "get" commands, which need a response. 
        # The "get" response only returns \r\n.  Thus the solution is to readline() after every write.  
        # Note: read_all() doesn't work!
        self.write(thestring)
        result = self.serial.readline()
        if verbose: print(result)

    def myAsk(self,command):
        # currently not used.
        self.write(command.encode())
        result = self.serial.read_until(self.lineend)
        result = self.parseAsk(result)
        return result
    
    # The gets ------------------------------------------------
    def getTemperature(self, channel='both'):
        if channel in ['A','B','a','b']:
            data = self.askFloat('input '+channel+':temp?')
            if data == '.......':
                print('WARNING: Cryocon returned a voltage out of range signal, Temperature = -7777')
                return -7777
            elif data == '-------':
                print('WARNING: Cryocon returned a temperature out of range signal, Temperature = -9999')
                data = -9999
            else:
                return data
        elif channel in ['both','BOTH']:
            dataA = self.askFloat('input a:temp?')
            dataB = self.askFloat('input b:temp?')
            return dataA, dataB
        else:
            raise ValueError
    def getVoltage(self, channel='both'):
        if channel in ['A','B','a','b']:
            data = self.askFloat('input '+channel+':senpr?')
            if data == '.......':
                print('WARNING: Cryocon returned a voltage out of range signal, Temperature = -7777')
                return -7777
            elif data == '-------':
                print('WARNING: Cryocon returned a temperature out of range signal, Temperature = -9999')
                data = -9999
            else:
                return data
        elif channel in ['both','BOTH']:
            dataA = self.askFloat('input a:senpr?')
            dataB = self.askFloat('input b:senpr?')
            return dataA, dataB
        else:
            raise ValueError
    def getControlLoopState(self):
        self.serial.flushInput()
        result = self.ask('control?')
        result = self.parseAsk(result)
        return result
    
    def getControlLoopMode(self,loop_channel=1):
        # note if PID is returned there are two extra spaces.
        result = self.ask('loop '+str(loop_channel)+':type?')
        result = self.parseAsk(result)
        return result

    def getControlTemperature(self,loop_channel=1):
        ''' return float in K '''
        # note askFloat() doesn't work because 'K' follows the value.
        # when a return attaches units, it cannot convert to float
        result = self.ask('loop '+str(loop_channel)+':setpt?')
        result = self.parseAsk(result)
        return float(result.split('K')[0])
    
    def getControlSource(self,loop_channel):
        result = self.ask('loop '+str(loop_channel)+':source?')
        result = self.parseAsk(result)
        return result
        
    def getHeaterRange(self,loop_channel):
        result = self.ask('loop '+str(loop_channel)+':range?')
        result = self.parseAsk(result)
        return result
    
    def getPID(self,loop_channel):
        P = self.askFloat('loop '+str(loop_channel)+':pgain?') 
        I = self.askFloat('loop '+str(loop_channel)+':igain?')
        D = self.askFloat('loop '+str(loop_channel)+':dgain?')
        return P,I,D
    
    def getCalibrationCurveName(self,curve_number):
        result = self.ask('sensorix '+str(curve_number)+':name?')
        result = self.parseAsk(result)
        return result
    
    def getCalibrationCurveForSensor(self,t_channel):
        curve_num=int(self.ask('input '+str(t_channel)+':sensor?'))
        curve_name = self.getCalibrationCurveName(curve_num)
        return curve_num,curve_name
    
    # The sets ------------------------------------------------
    
    def setUnits(self,channel='a',unit='k'): #
        if channel in ['a','b','A','B']:
            if unit in ['k','K','s','S']:
                self.writeAndFlush('inp '+channel+':unit '+ unit)
            else:
                print('ValueError: unit can only be \'k\' or \'s\'')
                raise ValueError
        else:
            print('ValueError: channel can only be \'a\' or \'b\'')
    
    def setControlTemperature(self,temp,loop_channel=1): #
        self.writeAndFlush('loop '+str(loop_channel)+ ':setpt '+str(temp))
    
    def setControlState(self,state='off'): #
        if state == 'off' or state == 'OFF':
            self.writeAndFlush('stop')
        elif state == 'on' or state == 'ON':
            self.writeAndFlush('control')
    
    def setTemperatureUnitsToKelvin(self,channel='A'): #
        if channel in ['A','B','a','b']:
            self.writeAndFlush('Input '+channel+':units k')
    
    def setControlLoopMode(self,loop_channel,mode='off'): #
        if loop_channel not in [1,2,3,4]:
            print('Invalid loop_channel: '+str(loop_channel)+'. Must be 1,2,3, or 4')
            raise ValueError
        if mode in ['off','man','pid','table','rampt','rampp','OFF','MAN','PID','TABLE','RAMPT','RAMPP']:
            self.writeAndFlush('loop '+str(loop_channel)+':type '+mode)
        else:
            print('Invalid mode: '+mode+'. Only off, man, pid, table, rampt, rampp allowed.')
            raise ValueError

    def setControlSource(self,loop_channel=1, t_channel='a'): #
        self.writeAndFlush('loop '+str(loop_channel)+':source '+t_channel)
    
    def setPID(self,loop_channel,P,I,D):
        loop_channel_string=str(loop_channel)
        self.writeAndFlush('loop '+loop_channel_string+':pgain '+str(P)) 
        self.writeAndFlush('loop '+loop_channel_string+':igain '+str(I)) 
        self.writeAndFlush('loop '+loop_channel_string+':dgain '+str(D))
    
    def setHeaterRange(self,loop_channel,heater_range='low'):
        if loop_channel not in [1,2,3,4]:
            print('Invalid loop_channel: '+str(loop_channel)+'. Only 1->4 allowed')
            raise ValueError
        if heater_range not in ['low', 'mid', 'LOW','MID']:
            print('Invalid heater range: '+range+'. Only low and mid ranges allowed')
            raise ValueError
        self.writeAndFlush('loop '+str(loop_channel)+':range '+heater_range)
        #return self.getHeaterRange(loop_channel) # commented out because it is different than all the other "sets"

    def setSensorToCurve(self,t_channel='a', curve_number=2):
        ''' Assign a calibration curve to a temperature sensor.  
            curve_number2 is the factory installed LS DT-670 curve
        '''
        thestring = 'input '+str(t_channel)+':sensorix '+str(curve_number)
        self.writeAndFlush(thestring)
        
    def disableControlLoops(self):
        self.writeAndFlush('stop')
    
    def controlLoopSetup(self,loop_channel=1,control_temp=3.0,t_channel='a',PID=[1,5,0],heater_range='low'):
        ''' initialize the control loop setup '''
        loop_channel_string=str(loop_channel)
        
        self.setControlSource(loop_channel,t_channel) # links control loop to thermometer t_channel 
        self.setTemperatureUnitsToKelvin(t_channel) 
        self.setHeaterRange(loop_channel, heater_range) 
        self.setPID(loop_channel,P=PID[0],I=PID[1],D=PID[2]) 
        self.setControlLoopMode(loop_channel,'PID') 
        self.setControlTemperature(control_temp, loop_channel)
        for ii in [1,2,3,4]: # turn all other loops off
             if ii != loop_channel:
                 self.setControlLoopMode(ii,'off') 

        print('loop channel '+loop_channel_string+' control loop config:')
        print('source: '+ self.getControlSource(loop_channel))
        print('Heater range: ' + self.getHeaterRange(loop_channel))
        print('PID = ',self.getPID(loop_channel))
        print('Control mode: ' + self.getControlLoopMode(loop_channel))
        print('Control temperature: ', self.getControlTemperature(loop_channel))
        print('Control status: ' + self.getControlLoopState())
        
    def isTemperatureStable(self,loop_channel,tolerance=0.1,control_channel=None,control_temp=None):
        if control_channel==None: t_channel=self.getControlSource(loop_channel).split('CH')[1]
        else: t_channel = control_channel
        if control_temp==None: t_c = self.getControlTemperature(loop_channel)
        else: t_c = control_temp
        t_m = self.getTemperature(t_channel)
        t_error = t_m - t_c
        print(t_m, t_c, t_error)
        if abs(t_error)<tolerance:
            stability=True
        else:
            stability=False
        return stability
    
        
                
        
    
    
    
        
            
        
        
        

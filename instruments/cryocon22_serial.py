'''
cryocon22_serial
Created on Nov, 2020 adapted from cryocon_gpib.py
@author: hubmayr

Notes:
1) self.ask() gives a string of ###\x00\x00..., current getting rid of \x00 with split and globally in the class with get() function
2) low, mid, high control range outputs .1, .33 and 1.0A
3) Cold load setup (50Ohm termination resistor at room temperature, 1000 ohm heater 
resistor on the cold load itself is in parallel with this) gives max powers:

low: 17.9 mW
mid: 195 mW
high 1.8 W, 

Therefore never want to use the high range.  

4) front panel doesn't update when commands are issued.  Need to flip the screen. 
'''

from . import serial_instrument
import serial
from time import sleep
import numpy as np
import cmath
from tkinter.simpledialog import askfloat
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
        stopbits=serial.STOPBITS_ONE, min_time_between_writes=0.05, readtimeout=0.05)
        
        # identity string of the instrument
        self.id_string = "Cryocon22C SNo:205346"
        self.manufacturer = 'Cryocon'
        self.model_number = '22'
        self.description  = 'Temperature Controller'
    
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
        data = self.ask('control?')
        return data
    
    def getControlLoopMode(self,loop_channel=1):
        return self.ask('loop '+str(loop_channel)+':type?')
    
    def getControlTemperature(self,loop_channel=1):
        return float(self.ask('loop '+str(loop_channel)+':setpt?').split('K')[0])
    
    def getControlSource(self,loop_channel):
        return self.ask('loop '+str(loop_channel)+':source?')
    
    def getHeaterRange(self,loop_channel):
        return self.ask('loop '+str(loop_channel)+':range?')
    
    def getPID(self,loop_channel):
        P = self.ask('loop '+str(loop_channel)+':pgain?')
        I = self.ask('loop '+str(loop_channel)+':igain?')
        D = self.ask('loop '+str(loop_channel)+':dgain?')
        return P,I,D
    
    def getCalibrationCurveName(self,curve_number):
        return self.ask('sensorix '+str(curve_number)+':name?')
    
    def getCalibrationCurveForSensor(self,t_channel):
        curve_num=int(self.ask('input '+str(t_channel)+':sensor?'))
        curve_name = self.getCalibrationCurveName(curve_num)
        return curve_num,curve_name
    
    # The sets ------------------------------------------------
    
    def setUnits(self,channel='a',unit='k'):
        if channel in ['a','b','A','B']:
            if unit in ['k','K','s','S']:
                self.write('inp '+channel+':unit '+ unit)
            else:
                print('ValueError: unit can only be \'k\' or \'s\'')
                raise ValueError
        else:
            print('ValueError: channel can only be \'a\' or \'b\'')
    
    def setControlTemperature(self,temp,loop_channel=1):
        self.write('loop '+str(loop_channel)+ ':setpt '+str(temp))
    
    def setControlState(self,state='off'):
        if state == 'off' or state == 'OFF':
            self.write('stop')
        elif state == 'on' or state == 'ON':
            self.write('control')
    
    def setTemperatureUnitsToKelvin(self,channel='A'):
        if channel in ['A','B','a','b']:
            self.write('Input '+channel+':units k')
    
    def setControlLoopMode(self,loop_channel,mode='off'):
        if loop_channel not in [1,2,3,4]:
            print('Invalid loop_channel: '+str(loop_channel)+'. Must be 1,2,3, or 4')
            raise ValueError
        if mode in ['off','man','pid','table','rampt','rampp','OFF','MAN','PID','TABLE','RAMPT','RAMPP']:
            self.write('loop '+str(loop_channel)+':type '+mode)
        else:
            print('Invalid mode: '+mode+'. Only off, man, pid, table, rampt, rampp allowed.')
            raise ValueError

    def setControlSource(self,loop_channel=1, t_channel='a'):
        self.write('loop '+str(loop_channel)+':source '+t_channel)
    
    def setPID(self,loop_channel,P,I,D):
        loop_channel_string=str(loop_channel)
        self.write('loop '+loop_channel_string+':pgain '+str(P))
        self.write('loop '+loop_channel_string+':igain '+str(I))
        self.write('loop '+loop_channel_string+':dgain '+str(D))
    
    def setHeaterRange(self,loop_channel,heater_range='low'):
        if loop_channel not in [1,2,3,4]:
            print('Invalid loop_channel: '+str(loop_channel)+'. Only 1->4 allowed')
            raise ValueError
        if heater_range not in ['low', 'mid', 'LOW','MID']:
            print('Invalid heater range: '+range+'. Only low and mid ranges allowed')
            raise ValueError
        self.write('loop '+str(loop_channel)+':range '+heater_range)
        return self.get('loop '+str(loop_channel)+':range?')
    
    def setSensorToCurve(self,t_channel='a', curve_number=2):
        ''' Assign a calibration curve to a temperature sensor.  
            curve_number2 is the factory installed LS DT-670 curve
        '''
        self.write('input '+str(t_channel)+' sensorix '+str(curvenumber))
        
    def disableControlLoops(self):
        self.write('stop')
    
    def controlLoopSetup(self,loop_channel=1,control_temp=3.0,t_channel='a',PID=[1,1,1],heater_range='low'):
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
        print('source:' + self.getControlSource(loop_channel))
        print('Heater range: ' + self.getHeaterRange(loop_channel))
        print('PID = ',self.getPID(loop_channel))
        print('Control mode: ' + self.getControlLoopMode(loop_channel))
        print('Control temperature: ', self.getControlTemperature(loop_channel))
        print('Control status: ' + self.getControlLoopState())
        
    def isTemperatureStable(self,loop_channel,tolerance=0.1):
        t_channel=self.getControlSource(loop_channel).split('CH')[1]
        t_m = self.getTemperature(t_channel)
        t_c = self.getControlTemperature(loop_channel)
        t_error = t_m - t_c
        print(t_m, t_c, t_error)
        if abs(t_error)<tolerance:
            stability=True
        else:
            stability=False
        return stability
    
        
                
        
    
    
    
        
            
        
        
        

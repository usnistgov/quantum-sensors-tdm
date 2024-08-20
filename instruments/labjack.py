'''
Labjack
NIST class for Labjack U3,U6 and potentialy U9
Created on January 30, 2011
@author: bennettd
'''

import u3
from time import sleep

class Labjack(object):
    '''
    The Labjack High Level Interface Class
    '''

    def __init__(self, isVerbose = False):
        '''
        Constructor  for Labjack class
        '''

        #super(Agilent33220A, self).__init__(board_number, name, pad, sad, timeout, send_eoi, eos_mode)
        
        self.manufacturer = 'Labjack'
        self.model_number = 'U6'
        self.description  = 'DAC Card'
        self.isVerbose = isVerbose
        self.lj = u3.U3() # Opens first found u3 over USB
        self.lj.close()
        #self.lj = u6.U6() # Opens first found u6 over USB
        print("Opened labjack device...")

    def getAnalogInput(self, analog_input,verbose=True):
        '''Get the voltage from one of the analog inputs'''
        
        # Reading Analog Input using Modbus
        #register = 0 + analog_input*2 # Channel 0 is register 0 and then they go up by 2s
        #voltage = self.lj.readRegister(register)
        # Reading Analog Input using lower level commands
        self.safeLJOpen()
        voltage = self.lj.getAIN(analog_input)
        self.lj.close()
        if verbose:
            print(("AI %s: %s" % (analog_input, voltage)))
        
        return voltage
 
    def getAnalogInputs(self):
        '''Get the voltage from all of the analog inputs'''
        
        # Reading Analog Inputs.
        print("Analog Inputs:")
        for i in range(4):
            register = 0 + i*2 # Starting register 0, 2 registers at a time
            print(("AIN%s (register %s): %s" % (i, register, self.lj.readRegister(register))))
        
    def setDACVoltage(self, dac_channel, voltage):
        '''Set the voltage on the DAC0'''
        
        # Set DAC using Modbus
        #if dac_channel == 0:
        #    DAC0_REGISTER = 5000
        #    self.lj.writeRegister(DAC0_REGISTER, voltage)
        #elif dac_channel == 1:
        #    DAC1_REGISTER = 5002
        #    self.lj.writeRegister(DAC1_REGISTER, voltage)
        #else:
        #    print 'Error: Not a channel'
        
        # Set DAC using 8 bit low level command
        dac_value = int(voltage * 255/4.95)
        if dac_channel == 0:
            self.lj.getFeedback(u3.DAC0_8(dac_value))
        elif dac_channel == 1:
            self.lj.getFeedback(u3.DAC1_8(dac_value))
        else:
            print('Error: Not a channel')
            
    def setDigIODirection(self, io_channel, direction):
        '''Set the direction of a Digital IO Channel '''
        
        self.safeLJOpen()
        
        if direction == 'output':
            self.lj.getFeedback(u3.BitDirWrite(io_channel, 1))   # Set IO channel to digital output
        elif direction == 'input':
            self.lj.getFeedback(u3.BitDirWrite(io_channel, 0))   # Set IO channel to digital input
        else:
            print('Error: Direction not valid')
        self.lj.close()

    def getDigIODirection(self, io_channel):
        '''Get the direction of a Digital IO Channel '''
        self.safeLJOpen()
        direction_bool = self.lj.getFeedback(u3.BitDirRead(io_channel))[0]   #GSet direction of digital output
        
        if direction_bool == 0:
            direction = 'input'
        elif direction_bool == 1:
            direction = 'output'
        else:
            print('Returned unknown state')
            direction = None
            
        self.lj.close()
        return direction
        
    def setDigIOState(self, io_channel, state):
        '''Set the state of a Digital IO Channel '''
        
        if state == 'high':
            self.safeLJOpen()
            self.lj.getFeedback(u3.BitStateWrite(io_channel, 1))   # Set IO channel to high
            self.lj.close()
        elif state == 'low':
            self.safeLJOpen()
            self.lj.getFeedback(u3.BitStateWrite(io_channel, 0))   # Set IO channel to low
            self.lj.close()
        else:
            print('Error: Direction not valid')

    def getDigIOState(self, io_channel):
        '''Get the state of a Digital IO Channel '''
        self.safeLJOpen()
        state_bool = self.lj.getFeedback(u3.BitStateRead(io_channel))[0]   #GSet direction of digital output
        
        if state_bool == 0:
            state = 'low'
        elif state_bool == 1:
            state = 'high'
        else:
            print('Returned unknown state')
            state = None
            
        self.lj.close()
        return state
    
    def configStarupFIOs(self, ai_inputs=[], outputs=[4,5,6,7], output_states=[0,0,0,0]):
        '''Setup the FIOs at startup. Currently for U3 only.'''
        
        lj_config = self.lj.configU3() # Get the current configuration for a U3
        device_name = lj_config['DeviceName']
        
        # If the model is a u3-HV then the first 4 FIOs have to be analog inputs
        # Remove the  0 through 3 from ai_inputs if they are there and set the starting mask to 15 for 1111
        if device_name == 'U3-HV':
            for index in range(4):
                try:
                    ai_inputs.remove(index)
                except:
                    pass
            ai_mask = 15
        else:
            ai_mask = 0
            
        for ai_input in ai_inputs:
            ai_mask = ai_mask+2**ai_input  #FIOs that are analog inputs depending if the bit is 1

        fio_dir_mask = 0
        fio_state_mask = 0
        for index,output in enumerate(outputs):
            fio_dir_mask = fio_dir_mask+2**output  #FIO is a output or input depending on if the bit in the mask is a 1 or 0
            output_state = output_states[index]
            if output_state == 1:
                fio_state_mask = fio_state_mask+2**output
            
        self.safeLJOpen()
        self.lj.configU3(FIOAnalog=ai_mask,FIODirection=fio_dir_mask,FIOState=fio_state_mask) 
        self.lj.close()
        
    def configStarupForADRControl(self, ai_inputs=[0,1,2,3], outputs=[4,5,6,7], output_states=[0,0,0,0]):
        '''Setup FIO 0-3 for analog input and FIO 4-7 at startup to be output and low. Currently for U3 only'''
        
        print('Setting up Labjack for controlling the ADR Control Box')
        self.configStarupFIOs(ai_inputs=ai_inputs, outputs=outputs, output_states=output_states)

    def setRelayControl(self, io_channel):
        '''Turn on digital io channel for 2 seconds then turn back off to switch latching relay '''
        
        self.setDigIOState(io_channel=io_channel, state='high')
        sleep(0.2)
        self.setDigIOState(io_channel=io_channel, state='low')
        sleep(0.5)
        
    def setRelayToRamp(self, io_channel=4):
        '''Switch relay to ramp mode. Default assumes ramp setting is on io=4.'''
        
        self.setRelayControl(io_channel)
        
    def setRelayToControl(self, io_channel=5):
        '''Switch relay to control mode. Default assumes control setting is on io=5.'''
        
        self.setRelayControl(io_channel)
        
    
    def safeLJOpen(self):
        ''' open the labjack if it isn't already open'''
        if self.lj.handle is None: # when it is close, handle is None
            self.lj.open()
            if self.isVerbose:
                print('repoened lj\n')
        
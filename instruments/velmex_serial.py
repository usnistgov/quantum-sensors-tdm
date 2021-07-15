'''
velmex_serial.py

Class to control Velmex Rotary stage 

9/15/16 - Created (JA; "rotary_stage.py") Based partially off CU 6-axis beam map control code (CU_velmex_control.py)
6/2021 Updated to python3 and integrated with the "Previous Generation TDM Electronics" / 2 stage ADR system architecture.
       Jay opened and closed the serial port a zillion times.  Hannes removed this.  Was there a reason for this? 

'''

import time, serial
from . import serial_instrument
#import serial_instrument


class Velmex(serial_instrument.SerialInstrument):
    '''
    Velmex VXM Stepper Motor Communication Class.  
    Hardware allows for control of two motors, but this class only controls one. 
    '''
    def __init__(self, port="velmex",doInit=True):
        ''' Constructor.  port is the only required parameter. 
            
            Note that you have to set the correct kind of motor (and limit swich) in the Velmex controller
            separately.  This can be done via comm port directly or use Windows softare from vender if confused

        '''

        super(Velmex, self).__init__(port, baud=9600, bytesize=8, parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE, min_time_between_writes=0.05, readtimeout=0.05)
        
        # identity string of the instrument
        self.id_string = "Velmex VXM"
        self.manufacturer = 'Velmex'
        self.model_number = 'VXM'
        self.description  = 'Stepping Motor Controller' 
        
        self.timeout_s = 60
        if doInit:
            self.initialize(verbose=True)
    
    def initialize(self, verbose=False):
        self.reset()
        self.home(home_offset=200, verbose=verbose)  
        return

    def reset(self):
        '''
        Clear and kill all motor controllers then take offline
        '''
        command_str = 'F C K Q'
        self.write(command_str)

    def home(self, home_offset=200, wait=False, verbose=False):
        '''
        Just go home.  Same as rotary_absolute, but no additional move after finding home
        
        Parameters
        ----------
        home_offset: int
            Offset needed for this particular motor/limit switch to hit ~0 as home
            steps (e.g. (1/20 degree)
        wait : boolean (optional)
            wait for motor to finish moving before allowing another command
            for parallel motion all but last move should wait
        verbose: print some stuff
        
        Return
        ------
        None 
        '''

        if verbose:
            print('Velmex moving home ')
        # note in next command, last move before commanded rotation is unique home to a particular motor/limit switch being used, 
        cmd_str = 'F N C setL1M-2,SIM400,A1M1,I1M-400,I1M0,I1M-400,I1M0,I1M'+str(home_offset)+',I1M'+',R'
        self.write(cmd_str)
        if wait:
            self.wait_for_ready(verbose=verbose)
        return

    def move_absolute(self,phi_100th_deg,home_offset=200,wait=False,verbose=False):
        '''
        Move azimuth to absolute angle from any starting angle
        Beware, will do several movements to home first, then find absolute
        
        Parameters
        ----------
        phi_100th_deg : int
            1/100 degrees of rotation
        wait : boolean (optional)
            wait for motor to finish moving before allowing another command
            for parallel motion all but last move should wait
        home_offset: int
            Offset needed for this particular motor/limit switch to hit ~0 as home
            steps (e.g. (1/20 degree)
        Return
        ------
        None 
        '''
        if verbose:
            print('Absolute azimuth move to '+str(phi_100th_deg/100)+' deg')
        # note in next command, last move before commanded phi_100th_deg is unique home to a particular motor/limit switch being used, 
        cmd_str = 'F N C setL1M-2,SIM400,A1M1,I1M-400,I1M0,I1M-400,I1M0,I1M'+str(home_offset)+',I1M'+str(-phi_100th_deg/5)+',R'
        self.write(cmd_str)
        if wait:
            self.wait_for_ready(verbose=verbose)
        return 
    
    def move_relative(self,phi_100th_deg,wait=False,verbose=False):
        '''
        Move azimuth from current position
        Parameters
        ----------
        phi_100th_deg : int
            1/100 degrees of rotation
        wait : boolean (optional)
            wait for motor to finish moving before allowing another command
            for parallel motion all but last move should wait
        Return
        ------
        None 
        '''
        if verbose:
            print('Incremental azimuth move of '+str(phi_100th_deg/100)+' deg')
        cmd_str = 'F N C S1M400,I1M'+str(-phi_100th_deg/5)+',R'
        self.write(cmd_str)
        if wait:
            self.wait_for_ready(verbose=verbose)
        return
    
    def wait_for_ready(self,verbose=False):
        ready = False
        if verbose:
            print('Number of characters in buffer',self.inWaiting())
        start = time.time()
        now = time.time()
        while not ready == '^' and now < start+self.timeout_s:
            time.sleep(0.1)
            ready = self.read()
            now = time.time()
            if verbose:
                print(now-start,ready,ser.inWaiting())
        return

if __name__ == "__main__":
    vm = VelmexSerial(port='/dev/ttyUSB8',doInit=True)
    vm.move_relative(1000,verbose=True)
    time.sleep(5)
    vm.move_absolute(2000,verbose=True)





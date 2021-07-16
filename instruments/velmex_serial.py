'''
velmex_serial.py

Class to control Velmex Rotary stage

9/15/16 - Created (JA; "rotary_stage.py") Based partially off CU 6-axis beam map control code (CU_velmex_control.py)
6/2021 Updated to python3 and integrated with the "Previous Generation TDM Electronics" / 2 stage ADR system architecture.

Manual at https://www.velmex.com/Downloads/User_Manuals/vxm_user_manl.pdf
Some notes on syntax
"E" ("F"): echo on (off)

Things to check:
lineend currently b'\r\n'.  Should it be b'\r'?


'''

import time, serial
from . import serial_instrument
#import serial_instrument


class Velmex(serial_instrument.SerialInstrument):
    '''
    Velmex VXM stepper motor controller communication class appled to
    B5990 rotary table, which has 0.01 deg / step and to be used in interactive mode.
    Hardware allows for control of two motors, but this class only controls one.
    Manual at: https://www.velmex.com/Downloads/User_Manuals/vxm_user_manl.pdf
    Full command summary: https://www.velmex.com/Downloads/Spec_Sheets/VXM%20-%20%20Command%20Summary%20Rev%20B%20814.pdf
    '''
    def __init__(self, port="velmex",doInit=True):
        ''' Constructor.  port is the only required parameter.

            Note that you have to set the correct kind of motor (and limit switch) in the Velmex controller
            separately.  This can be done via comm port directly or use Windows softare from vender if confused

        '''

        super(Velmex, self).__init__(port, baud=9600, bytesize=8, parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE, min_time_between_writes=0.05, readtimeout=0.05)

        # need to keep track of angle in software due to IR source wires attached to stepper motor
        # if going more than 2 rotations, may break the wires
        self.current_angle_100th_deg = 0
        self.max_angle_100th_deg = 72000 # maximum angle allowed in 100th degree steps

        print('Making instance of VELMEX stepper motor')
        print('WARNING: IF USING WITH NIST POLCAL SETUP. SPINNING INDEFINITELY WILL BREAK LEADS TO THE HAWKEYE IR SOURCE.')
        print('An attempt to prevent this is made through software keeping track of the current angle, and limiting to a safe range of +/- %d deg'%(self.max_angle_100th_deg/100))
        print('HOWEVER A KNOWN BUG EXISTS!  Calling home will take the shortest path to the phi = 0.  And the software currently gets confused on modulo 2pi')

        # globals
        self.motor_id = 1 # must match velmex controller output
        self.motor_type_int = 4 # hardware defined.  INCORRECT VALUE CAN DAMAGE MOTOR OR CONTROLLER!
        self.steps_per_revolution = 36000 # hardware defined.
        # In full the controller steps per revolution = 40 and the B5990 rotary table has a 90:1 gear ratio
        self.steps_second = 400 # speed of rotation
        self.acceleration = 1 # can be 1-127

        self.id_string = "Velmex VXM-2"
        self.manufacturer = 'Velmex'
        self.model_number = 'VXM-2'
        self.description  = 'Velmex VXM Stepper Motor Control System for 2 motors, one at a time'
        self.timeout_s = 20
        self.wait_cycle_time_s = 0.5

        #self.setup()
        if doInit:
            self.initialize(verbose=True)

    def setup(self):
        print('to be written')

    def run_command(self,cmd_str):
        self.write(cmd_str+",R")

    def ensure_safe_angle(self, angle_100th_deg):
        assert self.is_safe_angle(angle_100th_deg),print('Angle %d is outside safe zone (phi +/- %d deg)!'%(angle_100th_deg/100,self.max_angle_100th_deg/100))

    def is_safe_angle(self, angle_100th_deg):
        result = True
        if angle_100th_deg > self.max_angle_100th_deg or angle_100th_deg < -1*self.max_angle_100th_deg:
            result = False
        return result

    def initialize(self, verbose=False):
        self.reset()
        self.home(home_offset=200, verbose=verbose)
        self.current_angle_100th_deg = 0

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
        # F N C setL1M-2: on-line mode with echo off; null motors 1,2,3,4; clear all commands; set axis 1 limit switch mode to -2;
        # SIM400: should this be S1M400 ?
        #
        self.write(cmd_str)
        self.current_angle_100th_deg = 0
        if wait: self.wait_for_ready()

    def move_absolute(self,angle_100th_deg,home_offset=200,wait=False,verbose=False):
        '''
        Move azimuth to absolute angle from any starting angle
        Beware, will do several movements to home first, then find absolute

        Parameters
        ----------
        angle_100th_deg : int
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

        self.ensure_safe_angle(angle_100th_deg)

        if verbose:
            print('Absolute azimuth move to '+str(angle_100th_deg/100)+' deg')
        # note in next command, last move before commanded angle_100th_deg is unique home to a particular motor/limit switch being used,
        cmd_str = 'F N C setL1M-2,SIM400,A1M1,I1M-400,I1M0,I1M-400,I1M0,I1M'+str(home_offset)+',I1M'+str(-angle_100th_deg/5)+',R'
        self.write(cmd_str)
        if wait: self.wait_for_ready()
        self.current_angle_100th_deg = angle_100th_deg

    def move_relative(self,angle_100th_deg,wait=False,verbose=False):
        '''
        Move azimuth from current position
        Parameters
        ----------
        angle_100th_deg : int
            1/100 degrees of rotation
        wait : boolean (optional)
            wait for motor to finish moving before allowing another command
            for parallel motion all but last move should wait
        Return
        ------
        None
        '''
        self.ensure_safe_angle(self.current_angle_100th_deg + angle_100th_deg)

        if verbose:
            print('Incremental azimuth move of '+str(angle_100th_deg/100)+' deg')
        cmd_str = 'F N C S1M400,I1M'+str(-angle_100th_deg/5)+',R'
        self.write(cmd_str)
        self.current_angle_100th_deg = self.current_angle_100th_deg + angle_100th_deg
        if wait:
            self.wait_for_ready(verbose=verbose)

    def wait_for_ready(self):
        ready = self.is_ready()
        t = time.time()
        time_out = t+self.timeout_s
        if not ready and t<time_out:
            time.sleep(self.wait_cycle_time_s)
            ready = self.is_ready()
            t+=time.time()

    # def wait_for_ready(self,verbose=False):
    #     if verbose:
    #         print('Number of characters in buffer',self.inWaiting())
    #     ready = False
    #     now = start = time.time()
    #     t_end = start+self.timeout_s
    #     while not ready == '^' and now < t_end:
    #         time.sleep(0.1)
    #         ready = self.read()
    #         now = time.time()
    #         if verbose:
    #             print(now-start,ready,ser.inWaiting())

    def get_current_position(self,motor_id=1):
        if motor_id==1: cmd_str = "X"
        elif motor_id==2: cmd_str = "Y"
        elif motor_id==3: cmd_str = "Z"
        elif motor_id==4: cmd_str = "T"
        else:
            print('unknown motor_id')
            return None
        return int(self.askFloat(cmd_str))

    def get_current_motor_number(self):
        return int(self.askFloat("#"))

    def get_jog_speed_for_motor(self,motor_id):
        return int(self.askFloat("getj%dM"%motor_id))

    def set_jog_speed(self,speed):
        self.write("setj1M%d"%speed)

    def kill_motion(self):
        self.write("K")

    def clear(self):
        self.write("C")

    def is_ready(self):
        result = self.ask("V").decode()
        if result == 'B':
            return False
        elif result == 'R':
            return True

    def get_motor_type(self,motor_id):
        return int(self.askFloat("getM%dM"%motor_id))

    def set_motor_type_func(self,motor_id=1,motor_type_int=1):
        self.write("setM%dM%d"%(motor_id,motor_type_int))

    def set_motor_type(self):
        self.set_motor_type_func(self.motor_id,self.motor_type_int)


if __name__ == "__main__":
    vm = VelmexSerial(port='/dev/ttyUSB8',doInit=True)
    vm.move_relative(1000,verbose=True)
    time.sleep(5)
    vm.move_absolute(2000,verbose=True)

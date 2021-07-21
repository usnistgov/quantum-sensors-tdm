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

    a positive angle rotates the grid clockwise as viewed from the top.  This is + on the physical angular ruler on the rotary table; 
    however the velmex manual claims that a negative value is CCW.  A negative angle rotates CW.

    '''
    def __init__(self, port="velmex",doInit=True):
        ''' Constructor.  port is the only required parameter.

            Note that you have to set the correct kind of motor (and limit switch) in the Velmex controller
            separately.  This can be done via comm port directly or use Windows softare from vender if confused

        '''

        super(Velmex, self).__init__(port, baud=9600, bytesize=8, parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE, min_time_between_writes=0.05, readtimeout=0.05)

        # fixed hardware details
        self.id_string = "Velmex VXM-2"
        self.manufacturer = 'Velmex'
        self.model_number = 'VXM-2'
        self.description  = 'Velmex VXM Stepper Motor Control System for 2 motors, one at a time'
        self.motor_id = 1 # must match velmex controller output
        self.motor_type_int = 4 # hardware defined.  INCORRECT VALUE CAN DAMAGE MOTOR OR CONTROLLER!
        self.limitswitch_mode = -2
        self.index_per_revolution = 7200  # 1 commanded step (index) is 1/20th of a degree

        # definitions of motion speed
        self.index_per_second = 400 # speed of rotation
        self.acceleration_setting = 1 # 1-127, with 1 unit = 4000 steps/s^2.  With 90:1 gear ratio, this is .11 rev/s^2

        # keep track of current index so that infinite rotations don't break wires beneath the rotary stage
        self.current_angle_deg = 0
        self.max_index_allowed = 14400 # 2 revolutions allowed as baseline 

        # timing stuff 
        self.timeout_s = 20
        self.wait_cycle_time_s = 0.5

        # do some initialization
        self.configure()
        if doInit:
            self.initialize(verbose=True)

    def configure(self):
        ''' set to online mode, define motor type, jog speed, acceleration, and limit switch mode '''
        cmd_list = [self._cmdstr_online_mode(echo_on=False),\
                    self._cmdstr_kill(),\
                    self._cmdstr_set_motor_type(self.motor_type_int,self.motor_id),\
                    self._cmdstr_set_limitswitch_mode(-2,self.motor_id),\
                    self._cmdstr_set_speed(self.index_per_second, self.motor_id),\
                    self._cmdstr_set_acceleration(self.acceleration_setting, self.motor_id)]
        cmd_str = ''
        for cmd in cmd_list:
            cmd_str=cmd_str+cmd+','
        cmd_str=cmd_str[:-1]
        #print(cmd_str)
        self.write(cmd_str)
        response = self.get_configuration(print_back=True)

    def get_configuration(self,print_back=True):
        #mode = self.get_communication_mode()
        print(self.askFloat("getM1M"))
        motor_type = self.get_motor_type(self.motor_id) 
        limitswitch_mode = self.get_limitswitch_mode(self.motor_id)
        speed = self.get_speed(self.motor_id) 
        accel = self.acceleration_setting #self.get_acceleration() 

        if print_back:
           #labels = ['mode','motor_type','limit switch mode','speed','acceleration']
           labels = ['motor_type','limit switch mode','speed','acceleration']
           vals = [mode,motor_type,limitswitch_mode,speed,accel]
           for ii in range(len(labels)):
               print(label,':: ',vals[ii])

        return mode, motor_type, limitswitch_mode, speed, accel 

    # Command strings --------------------------------------------
    # written out individually so that they may be combined and passed to 
    # the instrument in one go, rather than many writes.
    def _cmdstr_set_motor_type(self,motor_type_int,motor_id=1):
        cmdstr = "setM%dM%d"%(motor_id, motor_type_int)
        return cmdstr 

    def _cmdstr_set_acceleration(self,acceleration_setting,motor_id=1):
        cmdstr = 'A%dM%d'%(motor_id, acceleration_setting)
        return cmdstr

    def _cmdstr_set_speed(self,speed, motor_id=1):
        return "S%dM%d"%(motor_id,speed)

    def _cmdstr_set_limitswitch_mode(self,mode_int=-2,motor_id=1):
        return "setL%d%d"%(motor_id,mode_int)

    def _cmdstr_home(self,motor_id=1):
        cmd_str = 'C setL%dM-2,I%dM-400,I%dM0,I%dM-400,I%dM0,I%dM200,I%dM'%([motor_id]*7)
        return cmd_str

    def _cmdstr_set_zero_index(self):
        return "N"

    def _cmdstr_kill(self):
        return "K"
    
    def _cmdstr_online_mode(self,echo_on=False):
        if echo_on:
            return "E"
        else:
            return "F"

    def _cmdstr_clear(self):
        return "C"

    def _cmdstr_run(self):
        return "R"

    def _cmdstr_get_motor_type(self,motor_id):
        return "getM%dM"%motor_id

    def _cmdstr_get_limitswitch_mode(self,motor_id):
        return "getM%dM"%motor_id

    def _cmdstr_get_speed(self,motor_id):
        return "getj%dM"%motor_id

    def _cmdstr_get_operating_mode(self):
        return "getDM"

    # def cmdstr_get_acceleration(self,motor_id):
    #     return "getj%dM"%motor_id

    # the gets ---------------------------------------------
    # some gets that don't exist: communication mode, acceleration_setting
    def _ask(self,cmd):
        self.serial.flushInput()
        return cmd 
    
    def get_operating_mode(self):
        return self._ask(int(self.askFloat(self._cmdstr_get_operating_mode())))
    
    #def get_communication_mode(self):

    def get_motor_type(self,motor_id=1):
        return self._ask(int(self.askFloat(self._cmdstr_get_motor_type(motor_id))))

    def get_limitswitch_mode(self,motor_id=1):
        return self._ask(int(self.askFloat(self._cmdstr_get_limitswitch_mode(motor_id))))
    
    def get_speed(self,motor_id):
        return self._ask(int(self.askFloat(cmdstr_get_speed(motor_id))))

    # def get_acceleration(self,motor_id):
    #     return int(self.askFloat(cmdstr_get_acceleration(motor_id)))
    
    def get_current_position(self,motor_id=1):
        self.serial.flushInput()
        if motor_id==1: cmd_str = "X"
        elif motor_id==2: cmd_str = "Y"
        elif motor_id==3: cmd_str = "Z"
        elif motor_id==4: cmd_str = "T"
        else:
            print('unknown motor_id')
            return None
        return int(self.askFloat(cmd_str))

    def get_current_motor_number(self):
        self.serial.flushInput()
        return int(self.askFloat("#"))

    # the sets ---------------------------------------------
    # dangerous ones do not allow arguments, apply only to motor 1
    def set_motor_type(self):
        self.write(self._cmdstr_set_motor_type(self.motor_type_int,self.motor_id))

    def set_limitswitch_mode(self,limitswitch_mode):
        self.limitswitch_mode = limitswitch_mode
        self.write(self._cmdstr_set_limitswitch_mode(limitswitch_mode, self.motor_id))

    def set_speed(self,speed):
        self.index_per_second = speed 
        self.write(self._cmdstr_set_speed(speed, self.motor_id))

    def set_acceleration(self,acceleration_setting):
        self.acceleration_setting = acceleration_setting
        self.write(self._cmdstr_set_speed(acceleration_setting, self.motor_id))

    def set_zero_angle(self):
        self.write(self._cmdstr_set_zero_index())

    def _run(self):
        ''' Used to send a series of commands '''
        self.write(self._cmdstr_run())

    def kill(self):
        self.write(self.cmdstr_kill())

    def clear_commands(self):
        ''' clear all commands from currently selected program '''
        self.write(self._cmdstr_clear())

    # higher level send methods -----------------------------------

    def initialize(self, verbose=False):
        self.home()
        self.run()
        
        print(self.get_current_position())

    def home(self):
        ''' performs a series of motions and lands at angle = 0 '''
        self.write(self._cmdstr_home(self.motor_id))

    def print_configuration(self):
        print('To be written')

    

    

    def is_ready(self):
        result = self.ask("V").decode()
        if result == 'B':
            return False
        elif result == 'R':
            return True

    def run_command(self,cmd_str):
        self.write(cmd_str+",R")

    def ensure_safe_angle(self, angle_100th_deg):
        assert self.is_safe_angle(angle_100th_deg),print('Angle %d is outside safe zone (phi +/- %d deg)!'%(angle_100th_deg/100,self.max_angle_100th_deg/100))

    def is_safe_angle(self, angle_100th_deg):
        result = True
        if angle_100th_deg > self.max_angle_100th_deg or angle_100th_deg < -1*self.max_angle_100th_deg:
            result = False
        return result

    

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

    # deprecated methods --------------------

    # def move_absolute(self,angle_100th_deg,home_offset=200,wait=False,verbose=False):
    #     '''
    #     Move azimuth to absolute angle from any starting angle
    #     Beware, will do several movements to home first, then find absolute

    #     Parameters
    #     ----------
    #     angle_100th_deg : int
    #         1/100 degrees of rotation
    #     wait : boolean (optional)
    #         wait for motor to finish moving before allowing another command
    #         for parallel motion all but last move should wait
    #     home_offset: int
    #         Offset needed for this particular motor/limit switch to hit ~0 as home
    #         steps (e.g. (1/20 degree)
    #     Return
    #     ------
    #     None
    #     '''

    #     self.ensure_safe_angle(angle_100th_deg)

    #     if verbose:
    #         print('Absolute azimuth move to '+str(angle_100th_deg/100)+' deg')
    #     # note in next command, last move before commanded angle_100th_deg is unique home to a particular motor/limit switch being used,
    #     cmd_str = 'F N C setL1M-2,SIM400,A1M1,I1M-400,I1M0,I1M-400,I1M0,I1M'+str(home_offset)+',I1M'+str(-angle_100th_deg/5)+',R'
    #     self.write(cmd_str)
    #     if wait: self.wait_for_ready()
    #     self.current_angle_100th_deg = angle_100th_deg

    # def reset(self):
    #     '''
    #     Clear and kill all motor controllers then take offline
    #     '''
    #     command_str = 'F C K Q'
    #     self.write(command_str)

    # def home(self, home_offset=200, wait=False, verbose=False):
    #     '''
    #     Run the motor to defined 0 angle.  

    #     Parameters
    #     ----------
    #     home_offset: int
    #         Offset needed for this particular motor/limit switch to hit ~0 as home
    #         steps (e.g. (1/20 degree)
    #     wait : boolean (optional)
    #         wait for motor to finish moving before allowing another command
    #         for parallel motion all but last move should wait
    #     verbose: print some stuff

    #     Return
    #     ------
    #     None
    #     '''

    #     if verbose:
    #         print('Velmex moving home ')
    #     # note in next command, last move before commanded rotation is unique home to a particular motor/limit switch being used,
    #     cmd_str = 'F N C setL1M-2,SIM400,A1M1,I1M-400,I1M0,I1M-400,I1M0,I1M'+str(home_offset)+',I1M'+',R'
    #     # F N C:  on-line mode with echo off; null motors 1,2,3,4; clear all commands; 
    #     # setL1M-2: set axis 1 limit switch mode to -2= Disabled N/O (normally open) for Home Switch use
    #     # SIM400: should this be S1M400, set speed of motor 1 to 400 steps/second ?
    #     # A1M1: set acceleration of motor1 to 1 (1 to 127)
    #     # I1M-400: index motor 1 CCW by -400 steps (incremental).  rotates clockwise 20 deg
    #     # I1M0: index motor until positive limit is encountered.   goes to 10 deg
    #     # I1M-400: index motor 1 CCW by -400 steps (incremental).  goes to 30 deg
    #     # I1M0: index motor until positive limit is encounted.     goes to 10 deg
    #     # I1M(home_offset): move home offset steps                 goes to 0 deg (for home offset = 200)
    #     # I1M: I guess this runs home again                        goes to 0 deg after a series of moves
    #     # R: run
    #     self.write(cmd_str)
    #     self.current_angle_100th_deg = 0
    #     if wait: self.wait_for_ready()

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

    


if __name__ == "__main__":
    vm = Velmex(doInit=False)

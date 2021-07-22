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
    B4818 rotary table, which has 0.05 deg / step and to be used in interactive mode.
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
        stopbits=serial.STOPBITS_ONE, min_time_between_writes=0.05, readtimeout=0.05, lineend = b"\r")

        # fixed hardware details
        self.id_string = "Velmex VXM-2"
        self.manufacturer = 'Velmex'
        self.model_number = 'VXM-2'
        self.description  = 'Velmex VXM Stepper Motor Control System for 2 motors, one at a time'
        self.motor_id = 1 # must match velmex controller output
        self.motor_type_int = 4 # hardware defined.  INCORRECT VALUE CAN DAMAGE MOTOR OR CONTROLLER!
        self.limitswitch_mode = -2   
        self.num_index_per_deg = 20 # 1 commanded step (index) is 1/20th of a degree 
                                    # 18:1 rotary stage and 400 steps/rev from VMX
        self.controller_mode_int = 3 # default=3 and hard coded.  Makes it such that positive  
                                     # requested angles rotate CW, which is consistent with physical 
                                     # label on rotary table

        # definitions of motion speed
        self.index_per_second = 2000 # speed of rotation in indices/ sec.  
        self.acceleration_setting = 1 # 1-127, with 1 unit = 4000 steps/s^2.  

        # keep track of current index so that infinite rotations don't break wires beneath the rotary stage
        self.current_angle_deg = 0
        self.max_index_allowed = 14400 # 2 revolutions allowed as baseline 

        # timing stuff 
        self.timeout_s = 20
        self.wait_cycle_time_s = 0.5

        # do some initialization
        #print('Initial read from unit: ', self.read())
        self.system_homed = False
        self.configure()
        if doInit:
            self.initialize(verbose=True)

    def configure(self):
        ''' set to online mode, define motor type, jog speed, acceleration, and limit switch mode '''
        cmd_list = [self._cmdstr_kill(),\
                    self._cmdstr_clear(),\
                    self._cmdstr_online_mode(echo_on=False),\
                    self._cmdstr_set_controller_mode(self.controller_mode_int),\
                    self._cmdstr_set_motor_type(self.motor_type_int,self.motor_id),\
                    self._cmdstr_set_limitswitch_mode(-2,self.motor_id),\
                    self._cmdstr_set_speed(self.index_per_second, self.motor_id),\
                    self._cmdstr_set_acceleration(self.acceleration_setting, self.motor_id)]
        cmd_str = ''
        for cmd in cmd_list:
            cmd_str=cmd_str+cmd+','
        cmd_str=cmd_str[:-1]
        #print('sending command string: ',cmd_str)
        self.write(cmd_str)
        self.read()
        response = self.get_configuration(print_back=True)

    # Command strings --------------------------------------------
    # written out individually so that they may be combined and passed to 
    # the instrument in one go, rather than many writes.
    def _cmdstr_set_controller_mode(self,mode_int):
        return "setDM%d"%(mode_int)
    
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

    def _cmdstr_home(self,speed,motor_id=1):
        cmd_str = 'C setL%dM-2,S%dM%d'%(motor_id,motor_id,speed)+',I%dM-400,I%dM0,I%dM-400,I%dM0,I%dM200,I%dM,R'%tuple([motor_id]*6)
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

    def _cmdstr_get_operating_mode(self):
        return "getDM"

    def _cmdstr_move_rel(self,angle_deg,speed,motor_id):
        return "C, S%dM%d,I%dM%d,R"%(motor_id,speed,motor_id,self._deg_to_index(angle_deg))

    # def cmdstr_get_acceleration(self,motor_id):
    #     return "getj%dM"%motor_id

    # some helper methods --------------------------------------------
    def _ask(self,cmd):
        self.serial.flushInput()
        return cmd 

    def _deg_to_index(self, angle_deg):
        return int(round(angle_deg*self.num_index_per_deg))

    def _index_to_deg(self, index):
        return index/self.num_index_per_deg

    def is_ready(self):
        result = self.read().decode()
        if "^" in result:
            print(result)
            return True
        else:
            return False
        
    def wait_for_ready(self):
        ready = self.is_ready()
        t = time.time()
        time_out = t+self.timeout_s
        if not ready and t<time_out:
            time.sleep(self.wait_cycle_time_s)
            ready = self.is_ready()
            t+=time.time()

    def ensure_safe_angle(self, angle_100th_deg):
        assert self.is_safe_angle(angle_100th_deg),print('Angle %d is outside safe zone (phi +/- %d deg)!'%(angle_100th_deg/100,self.max_angle_100th_deg/100))

    def is_safe_angle(self, angle_100th_deg):
        result = True
        if angle_100th_deg > self.max_angle_100th_deg or angle_100th_deg < -1*self.max_angle_100th_deg:
            result = False
        return result

    # the gets ---------------------------------------------
    # some gets that don't exist: communication mode, acceleration_setting
    
    def get_operating_mode(self):
        return self._ask(int(self.askFloat(self._cmdstr_get_operating_mode())))
    
    #def get_communication_mode(self):

    def get_motor_type(self,motor_id=1):
        return self._ask(int(self.askFloat(self._cmdstr_get_motor_type(motor_id))))

    def get_limitswitch_mode(self,motor_id=1):
        return self._ask(int(self.askFloat(self._cmdstr_get_limitswitch_mode(motor_id))))

    # def get_acceleration(self,motor_id):
    #     return int(self.askFloat(cmdstr_get_acceleration(motor_id)))
    
    def get_current_position(self,motor_id=1,convert_to_deg=False):
        self.serial.flushInput()
        if motor_id==1: cmd_str = "X"
        elif motor_id==2: cmd_str = "Y"
        elif motor_id==3: cmd_str = "Z"
        elif motor_id==4: cmd_str = "T"
        else:
            print('unknown motor_id')
            return None
        index = int(self.askFloat(cmd_str))
        if convert_to_deg:
            return self._index_to_deg(index)
        else:
            return index

    def get_current_motor_number(self):
        self.serial.flushInput()
        return int(self.askFloat("#"))

    def get_configuration(self,print_back=True):
        mode = self.get_operating_mode()
        motor_type = self.get_motor_type(self.motor_id) 
        limitswitch_mode = self.get_limitswitch_mode(self.motor_id)
        speed = self.get_speed(self.motor_id) 
        accel = self.acceleration_setting #self.get_acceleration() 

        if print_back:
           labels = ['mode','motor_type','limit switch mode','speed','acceleration']
           labels = ['motor_type','limit switch mode','speed','acceleration']
           vals = [mode,motor_type,limitswitch_mode,speed,accel]
           for ii in range(len(labels)):
               print(labels[ii],':: ',vals[ii])

        return motor_type, limitswitch_mode, speed, accel 

    # the sets ---------------------------------------------
    # dangerous ones do not allow arguments, apply only to motor 1
    def set_controller_mode():
        self.write(self._cmdstr_set_controller_mode(3))
    
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
        print('Recording zero angle = ',self.get_current_position(self.motor_id))

    def _run(self):
        ''' Used to send a series of commands '''
        self.write(self._cmdstr_run())

    def kill(self):
        self.write(self.cmdstr_kill())

    def clear_commands(self):
        ''' clear all commands from currently selected program '''
        self.write(self._cmdstr_clear())

    def run_command(self,cmd_str):
        self.write(cmd_str+",R")

    # motion methods -----------------------------------

    def initialize(self, verbose=False):
        self.home()
        self.wait_for_ready()
        time.sleep(10)
        self.set_zero_angle()
        
    def home(self):
        ''' performs a series of motions and lands at angle = 0 '''
        self.write(self._cmdstr_home(self.index_per_second, self.motor_id))
        self.system_homed = True

    def move_relative(self,angle_deg,wait_for_complete=False):
        '''
        Index the motor by angle_deg relative to current position. 
        '''
        #self.ensure_safe_angle(self.current_angle_100th_deg + angle_100th_deg)
        self.write(self._cmdstr_move_rel(angle_deg,self.index_per_second,self.motor_id))
        if wait_for_complete:
            self.wait_for_ready()

    def move_absolute(self,angle_deg,wait_for_complete=False):
        '''
        Index the motor by angle_deg relative to current position. 
        '''
        assert self.system_homed, 'The system must be homed before move_absolute can be used'
        cur_index = self.get_current_position(self.motor_id)
        self.move_relative(angle_deg - self._index_to_deg(cur_index),wait_for_complete = wait_for_complete)

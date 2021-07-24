'''
velmex_serial.py

Class to control Velmex Rotary stage

@author: JH, 7/2021 based on rotary_stage.py by JEA, which was used in the Legacy TDM system

Manual at https://www.velmex.com/Downloads/User_Manuals/vxm_user_manl.pdf
'''

import time, serial
from . import serial_instrument
import numpy as np 

class Velmex(serial_instrument.SerialInstrument):
    '''
    Velmex VXM stepper motor controller communication class appled to
    B4818 rotary table, which has 0.05 deg / step and to be used in interactive mode.
    Hardware allows for control of two motors, but this class only controls one.
    Manual at: https://www.velmex.com/Downloads/User_Manuals/vxm_user_manl.pdf
    Full command summary: https://www.velmex.com/Downloads/Spec_Sheets/VXM%20-%20%20Command%20Summary%20Rev%20B%20814.pdf

    The expected use case is with an IR source mounted below the rotary stage, but co-rotates with the stage.
    As such infinite rotation will break the IR source lead wires.  Therefore this class includes smarts to limit the
    absolute rotation.

    '''
    def __init__(self, port="velmex",doInit=True):
        ''' Constructor.  port is the only required parameter.
            if doInit=True, the device will be homed and the zero index defined at the home position
        '''
        super(Velmex, self).__init__(port, baud=9600, bytesize=8, parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE, min_time_between_writes=0.05, readtimeout=0.05, lineend = b"\r")

        # fixed hardware details
        self.id_string = "Velmex VXM-2"
        self.manufacturer = 'Velmex'
        self.model_number = 'VXM-2'
        self.description  = 'Velmex VXM Stepper Motor Control System for 2 motors.  This class only supports communication to one motor.'
        self.motor_id = 1 # must match velmex controller output
        self.motor_type_int = 4 # hardware defined.  INCORRECT VALUE CAN DAMAGE MOTOR OR CONTROLLER!
        self.limitswitch_mode = -2  # -2 is default and required for homing
        self.num_index_per_deg = 20 # 1 commanded step (index) is 1/20th of a degree
                                    # 18:1 rotary stage and 400 steps/rev from VMX
        self.controller_mode_int = 3 # default=3 and hard coded.  Makes it such that positive
                                     # requested angles rotate CW, which is consistent with physical
                                     # labels on rotary table

        # definitions of motion speed
        self.index_per_second = 600 # speed of rotation in indices/sec.
        self.acceleration_setting = 1 # 1-127, with 1 unit = 4000 steps/s^2.

        # timing
        self.additional_wait_s = 0.2 # added time to wait for motion to complete

        # limits
        self.max_deg_allowed = 720 # + or - 2 revolutions is the max allowed

        # initialization/configuration
        self.system_homed = False
        self.send_configure()
        if doInit:
            self.initialize(verbose=True)

    def send_configure(self):
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
        self.write(cmd_str)
        self.read()
        response = self.get_configuration(print_back=True)

    # Command strings --------------------------------------------
    # written out individually so that they may be combined and passed to
    # the instrument in one go, rather than many single writes.
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

    def _cmdstr_home(self,motor_id=1):
        ''' purposefully moves 45 deg away from home to ensure initial state is not 
            in active area of limit switch (which is a couple degrees) 
        '''
        #cmd_str = 'C S%dM600, I%dM-0, I%dM900,I%dM-0,IA%dM-0,R'%tuple([motor_id]*5)
        cmd_str = 'C S%dM600, I%dM-0, I%dM900,I%dM-0,IA%dM-0,I%dM-200,R'%tuple([motor_id]*6)
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

    def _cmdstr_get_motor1_position(self):
        return "X"

    # def cmdstr_get_acceleration(self,motor_id):
    #     return "getj%dM"%motor_id

    # some helper methods ------------------------------------------------------------------------------------------
    def _deg_to_index(self, angle_deg):
        return int(round(angle_deg*self.num_index_per_deg))

    def _index_to_deg(self, index):
        return index/self.num_index_per_deg

    def angle_list_to_stepper_resolution(self,angle_list):
        index_list = np.round(np.array(angle_list)*self.num_index_per_deg,decimals=0).astype(int)
        return list(index_list/self.num_index_per_deg)

    def _motion_time(self,rel_angle_deg):
        ''' return estimated motion time for a move in seconds '''
        t = abs(rel_angle_deg)*self.num_index_per_deg/self.index_per_second
        #print("motion time is %.3f s"%t)
        return t

    def _wait_for_move_to_complete(self,rel_angle_deg):
        time.sleep(self._motion_time(rel_angle_deg)+self.additional_wait_s)

    def check_angle_safe(self,angle_deg):
        #assert abs(angle_deg) <= self.max_deg_allowed, print('Requested angle %.2f is outside the allowable range'%angle_deg)
        assert abs(angle_deg) <= self.max_deg_allowed, 'Requested angle %.2f deg is outside the allowable range'%angle_deg

    def _is_ready(self):
        ''' Determine if VMX is ready to receive a new command.
            It does not wait for motion to complete.
            Use case assumed to be directly after a run command sent.
        '''
        result = False
        if self.read().decode() == "^":
            result = True
        return result

    def _wait_for_ready(self,timeout_s=20,wait_cycle_time_s=0.5):
        ''' wait for VMX to be ready to receive a command.
            This does not wait for the actual motion to complete.
        '''
        ready = self._is_ready()
        t = time.time()
        time_out = t+timeout_s
        if not ready and t<time_out:
            time.sleep(wait_cycle_time_s)
            ready = self._is_ready()
            t+=time.time()

    def _ask_return_int(self,cmd):
        self.serial.flushInput() # required since I don't know what is sitting in the instrument queue
        response = int(self.askFloat(cmd))
        return response 

    def kill(self):
        self.write(self.cmdstr_kill())

    def clear_commands(self):
        ''' clear all commands from currently selected program '''
        self.write(self._cmdstr_clear())

    def run_command(self,cmd_str):
        self.write(cmd_str+",R")

    # the gets --------------------------------------------------------------------------------------
    def get_current_position(self,convert_to_deg=True):
        index = self._ask_return_int(self._cmdstr_get_motor1_position())
        if convert_to_deg:
            return self._index_to_deg(index)
        else:
            return index

    def get_current_position_for_motor(self,motor_id=1,convert_to_deg=False):
        if motor_id==1: cmd_str = "X"
        elif motor_id==2: cmd_str = "Y"
        elif motor_id==3: cmd_str = "Z"
        elif motor_id==4: cmd_str = "T"
        else:
            print('unknown motor_id')
            return None
        index = self._ask_return_int(cmd_str)
        if convert_to_deg:
            return self._index_to_deg(index)
        else:
            return index

    def get_operating_mode(self):
        return self._ask_return_int(self._cmdstr_get_operating_mode())

    def get_motor_type(self,motor_id=1):
        return self._ask_return_int(self._cmdstr_get_motor_type(motor_id))

    def get_limitswitch_mode(self,motor_id=1):
        return self._ask_return_int(self._cmdstr_get_limitswitch_mode(motor_id))

    def get_current_motor_number(self):
        return self._ask_return_int("#")

    def get_configuration(self,print_back=True):
        mode = self.get_operating_mode()
        motor_type = self.get_motor_type(self.motor_id)
        limitswitch_mode = self.get_limitswitch_mode(self.motor_id)
        speed = self.index_per_second
        accel = self.acceleration_setting #self.get_acceleration()

        if print_back:
            vals = [mode,motor_type,limitswitch_mode,speed,accel]
            labels = ['mode','motor_type','limit switch mode','index per sec','acceleration setting']
            print("Velmex Motor Controller Config:")
            for ii in range(len(labels)):
                print(labels[ii],':: ',vals[ii])

        return mode, motor_type, limitswitch_mode, speed, accel

    # Those commented out below do not seem possible given the VMX comm,
    # but maybe so with a deeper look at manual

    # def get_acceleration(self,motor_id):
    #     return int(self.askFloat(cmdstr_get_acceleration(motor_id)))

    #def get_communication_mode(self):

    # the sets --------------------------------------------------------------------------------------------
    # dangerous ones do not allow arguments.
    # apply these to self.motor_id only
    def set_controller_mode():
        self.write(self._cmdstr_set_controller_mode(3))

    def set_motor_type(self):
        self.write(self._cmdstr_set_motor_type(self.motor_type_int,self.motor_id))

    def set_limitswitch_mode(self,limitswitch_mode):
        self.limitswitch_mode = limitswitch_mode
        self.write(self._cmdstr_set_limitswitch_mode(limitswitch_mode, self.motor_id))

    # def set_speed(self,speed):
    #     self.index_per_second = speed
    #     self.write(self._cmdstr_set_speed(speed, self.motor_id))

    def set_acceleration(self,acceleration_setting):
        self.acceleration_setting = acceleration_setting
        self.write(self._cmdstr_set_acceleration(acceleration_setting, self.motor_id))

    def set_zero_angle(self):
        self.write(self._cmdstr_set_zero_index())
        print('Recording zero angle = ',self.get_current_position(convert_to_deg=True))

    # motion methods --------------------------------------------------------------------------
    def initialize(self, verbose=False):
        self.home(verbose)
        self.set_zero_angle()

    def home(self,verbose=True):
        ''' performs a series of motions and lands at angle = 0.
            Since home() command always approaches the limitswitch from the
            same direction, wires could wrap up.  To handle case where the
            system has already been homed and the current angle is negative,
            run to zero index before executing this home command.

        '''
        if self.system_homed:
            if verbose: print('Home: Moving to zero index.')
            self.move_to_zero_index(wait=True)
        self.write(self._cmdstr_home(self.motor_id))
        wait_time = self._motion_time(180+self._index_to_deg(900*2))+self.additional_wait_s
        if verbose: print('Home: performing home sequence for %.1f seconds.'%wait_time)
        time.sleep(wait_time)
        self.system_homed = True

    def move_relative(self,angle_deg,wait=True):
        '''
        Index the motor by angle_deg relative to current position.
        This is the main use method of the class.
        If wait=True, wait for the motion to complete.
        '''
        self.check_angle_safe(self.get_current_position(convert_to_deg=True) + angle_deg)
        self.write(self._cmdstr_move_rel(angle_deg,self.index_per_second,self.motor_id))
        if wait: self._wait_for_move_to_complete(angle_deg)

    def move_absolute(self,angle_deg,wait=False):
        '''
        Index the motor to absolute angle angle_deg.
        '''
        assert self.system_homed, 'The system must be homed before move_absolute can be used'
        rel_angle = angle_deg - self.get_current_position(convert_to_deg=True)
        self.move_relative(rel_angle,wait=wait)

    def move_to_zero_index(self,wait=False,verbose=False):
        if verbose: print('Moving to zero index.')
        cur_angle = self.get_current_position(convert_to_deg=True)
        self.move_relative(-cur_angle,wait)

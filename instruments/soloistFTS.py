'''
soloistFTS.py

Class to control NIST FTS mirror.  
Based on aerotechXY.py 
@author: JH 5/2025
'''

import socket
import time
import numpy as np

class SoloistFTS(object):
    '''
    Class for controlling the soloist FTS mirror


    '''

#'132.163.82.11'
    def __init__(self, motor_controller_IP='192.168.30.10', motor_controller_port=8000,
                 AsciiCmdEOSChar='\n', AsciiCmdAckChar='%',
                 AsciiCmdNakChar='!',AsciiCmdFaultChar='#',
                 AsciiCmdTimeoutChar='$'):
        '''
        motor_controller_IP: the IP address of the ensemble motor controller
        motor_controller_port: the port of the ensemble motor controller
        AsciiCmdEOSChar: thing needed to terminate every command
        '''

        self.motor_controller_IP=motor_controller_IP
        self.motor_controller_port=motor_controller_port
        self.AsciiCmdEOSChar=AsciiCmdEOSChar
        self.AsciiCmdAckChar = AsciiCmdAckChar
        self.AsciiCmdNakChar = AsciiCmdNakChar
        self.AsciiCmdFaultChar = AsciiCmdFaultChar
        self.AsciiCmdTimeoutChar = AsciiCmdTimeoutChar
        self.pause_after_motion_stop = 0.1
        self.post_command_sleep = 0.1
        self.home_speed_mmps=20

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__connect__()
        

    def __connect__(self):
        self.client_socket.connect((self.motor_controller_IP, self.motor_controller_port))

    def __SendStr__(self,string):
        foo = string+self.AsciiCmdEOSChar
        answer = self.client_socket.send(foo.encode())
        time.sleep(self.post_command_sleep)

    def close_connection(self):
        self.client_socket.close()

    def parse_return(self,ret_string):
        ''' parse the return string from motor controller '''
        return ret_string.split('\n')

    # def send_series_commands(self,CMD1,CMD2):
    #     self.__SendStr__(CMD1)
    #     self.__SendStr__('WAIT MOVEDONE')
    #     self.__SendStr__(CMD2)

    # def wait_move_done(self):
    #     self.__SendStr__('WAIT MOVEDONE X Y')

    #-----------------------------------------------------------------------------------------
    # commanding without returns
    def enable(self):
        self.__SendStr__('ENABLE')

    def disable(self):
        self.__SendStr__('DISABLE')

    def home(self):
        ''' return to the home position '''
        self.__SendStr__('HOME')

    def move_absolute(self,x_mm,v_mmps=1,verbose=False,wait=True):
        ''' Absolute movement 
            x_mm: position in millimeters
            v_mmps: velocity in millimeters per second
        '''
        
        assert np.logical_and(x_mm<=200,x_mm>=0),'Postion out of range.  X limits 0--200 mm'
        
        string='MOVEABS D'+str(x_mm)+' F'+str(v_mmps)
        if verbose:
            print('sending following string to ensemble: ',string)
        self.__SendStr__(string)
        if wait:
            x0,y0=self.get_position()
            t=np.max([abs(x-x0)/vx_mmps,abs(y-y0)/vy_mmps])
            time.sleep(t+self.pause_after_motion_stop)

    #-----------------------------------------------------------------------------------------
    # the gets 

    # def get_position(self):
    #     #self.client_socket.recv(1000) # clear the current buffer of returns from the controller
    #     x=self.__SendStr__('PFBK()')
    #     ret_string = self.client_socket.recv(1000)
    #     print(ret_string.decode())
    #     xy_raw = self.parse_return(ret_string.decode())[-3:-1]
    #     return float(xy_raw[0].split('%')[-1]),float(xy_raw[1].split('%')[-1])

    # def set_wait_mode(self,mode='MOVEDONE'):
    #     assert mode in ['MOVEDONE','NOWAIT','INPOS'], 'mode must be MOVEDONE,NOWAIT,or INPOS'
    #     self.__SendStr__('WAIT MODE '+mode)

    # def initialize(self,home=True):
    #     print('initializing the XY stage.')
    #     self.set_wait_mode(mode='MOVEDONE')
    #     self.enable_axis('X')
    #     self.enable_axis('Y')
    #     if home:
    #         print('Homing X and Y')
    #         self.home('both')

    # def shutdown(self):
    #     print('Shutting down the XY stage.  Homing X and Y...')
    #     self.home('both')
    #     print('Disabling the axes and closing communications')
    #     self.disable_axis('X')
    #     self.disable_axis('Y')
    #     self.close_connection()
    #     print('shutdown complete')

    

#     def move_incremental(self,dx,dy,vx_mmps=25,vy_mmps=25,verbose=False,wait=True):
#         ''' Incremental movement '''
#         # if dx is None:
#         #     string='MOVEINC Y'+str(dy)+' F'+str(vy_mmps)
#         # elif dy is None:
#         #     string='MOVEINC X'+str(dx)+' F'+str(vx_mmps)
#         # else:
#         #     string='MOVEINC X'+str(dx)+' F'+str(vx_mmps)+' Y'+str(dy)+' F'+str(vy_mmps)
#         string='MOVEINC X'+str(dx)+' F'+str(vx_mmps)+' Y'+str(dy)+' F'+str(vy_mmps)
#         if verbose:
#             print('sending following string to ensemble: ',string)
#         self.__SendStr__(string)
#         if wait:
#             t=np.max([abs(dx)/vx_mmps,abs(dy)/vy_mmps])
#             time.sleep(t+self.pause_after_motion_stop)        

    
        
    

# if __name__ == "__main__":
#     xy = AerotechXY()
#     print(xy.get_position())
#     xy.initialize(home=False)
#     print('moving to 100,0')
#     xy.move_absolute(x=100,y=0,vx_mmps=10)
#     print('I ought to be finished moving now')
#     # time.sleep(3)
#     # xy.shutdown()
    

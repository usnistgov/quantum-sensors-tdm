'''
aerotechXY.py

ported to python3 7/2021 from XY.py written JH 10/20/2011
'''

import socket
import time
import numpy as np

class AerotechXY(object):
    '''
    Class for controlling the aerotech XY stage.  Some setup is required on the Ensemble motor controller side.

    Standard setup on ensemble for ascii command
    AsciiCmdEOSChar = 10; return feed
    AsciiCmdAckChar = 37 (%) the command was acknowledged
    AsciiCmdNakChar = 33 (!) command failure
    AsciiCmdFaultChar = 35; (#) command recognized but not executed
    AsciiCmdTimeout = 10; return feed
    AsciiCmdTimeoutChar = 36; $

    For this to work, the following setup is needed:

    --- Controller --- (note may need to "connect" then hit "retrieve parameters" icon on top toolbare)
    1) In Parameters->Controller->Communication change the following parameters:
        AsciiCmdEnable, enable Ethernet Socket 1
    2) In Parameters->Controller->Communication->Ethernet->Sockets change the following:
        InetSock1Port to 8000
        InetSock1Flags to TCP server
    3) AsciiCmdEnable must be set to true

    To view current Aerotech connection settings:
    On windows box from within Ensemble IDE (XY stage motio control icon on desktop):
    Tools -> Configuration Manager -> click Entire Network "Ensemble" to view  

    '''

#'132.163.82.11'
    def __init__(self, motor_controller_IP='192.168.30.100', motor_controller_port=8000,
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

    def send_series_commands(self,CMD1,CMD2):
        self.__SendStr__(CMD1)
        self.__SendStr__('WAIT MOVEDONE')
        self.__SendStr__(CMD2)

    def azimuth_scan(self,xi,xf,yi,yf,v):
        self.move_absolute(xi,yi, v, v,True)
        self.__SendStr__('WAIT INPOS X Y')
        self.move_absolute(xf,yf,v,v,True)
        self.__SendStr__('WAIT INPOS X Y')
        self.move_absolute(xi,yi, v, v, True)
        #time.sleep(sleeptime)
        #self.__SendStr__('WAIT INPOS X Y')
        #time.sleep(sleeptime)

    def wait_move_done(self):
        self.__SendStr__('WAIT MOVEDONE X Y')
    #-----------------------------------------------------------------------------------------
    def enable_axis(self,axis='both'):
        if axis not in ['x','y','both','X','Y']:
            print('unknown axis')
            return False

        if axis=='x' or axis=='X':
            self.__SendStr__('ENABLE X')
        elif axis =='y' or axis=='Y':
            self.__SendStr__('ENABLE Y')
        elif axis=='both':
            self.__SendStr__('ENABLE X')
            self.__SendStr__('ENABLE Y')

    def disable_axis(self,axis='both'):
        if axis not in ['x','y','both','X','Y']:
            print('unknown axis')
            return False

        if axis=='x' or axis=='X':
            self.__SendStr__('DISABLE X')
        elif axis =='y' or axis=='Y':
            self.__SendStr__('DISABLE Y')
        elif axis=='both':
            self.__SendStr__('DISABLE X')
            self.__SendStr__('DISABLE Y')

    def set_wait_mode(self,mode='MOVEDONE'):
        assert mode in ['MOVEDONE','NOWAIT','INPOS'], 'mode must be MOVEDONE,NOWAIT,or INPOS'
        self.__SendStr__('WAIT MODE '+mode)

    def initialize(self,home=True):
        print('initializing the XY stage.')
        self.set_wait_mode(mode='MOVEDONE')
        self.enable_axis('X')
        self.enable_axis('Y')
        if home:
            print('Homing X and Y')
            self.home('both')

    def shutdown(self):
        print('Shutting down the XY stage.  Homing X and Y...')
        self.home('both')
        print('Disabling the axes and closing communications')
        self.disable_axis('X')
        self.disable_axis('Y')
        self.close_connection()
        print('shutdown complete')

    def home(self,axis='both'):
        ''' return to the home position '''
        if axis not in ['x','y','both','X','Y']:
            print('unknown axis')
            return False
        if axis=='x' or axis=='X':
            self.__SendStr__('HOME X')
        elif axis =='y' or axis=='Y':
            self.__SendStr__('HOME Y')
        elif axis=='both':
            self.__SendStr__('HOME X')
            self.__SendStr__('HOME Y')
        x,y=self.get_position()
        t=np.max([x,y])/self.home_speed_mmps
        time.sleep(t+self.pause_after_motion_stop)

    def move_incremental(self,dx,dy,vx_mmps=25,vy_mmps=25,verbose=False,wait=True):
        ''' Incremental movement '''
        # if dx is None:
        #     string='MOVEINC Y'+str(dy)+' F'+str(vy_mmps)
        # elif dy is None:
        #     string='MOVEINC X'+str(dx)+' F'+str(vx_mmps)
        # else:
        #     string='MOVEINC X'+str(dx)+' F'+str(vx_mmps)+' Y'+str(dy)+' F'+str(vy_mmps)
        string='MOVEINC X'+str(dx)+' F'+str(vx_mmps)+' Y'+str(dy)+' F'+str(vy_mmps)
        if verbose:
            print('sending following string to ensemble: ',string)
        self.__SendStr__(string)
        if wait:
            t=np.max([abs(dx)/vx_mmps,abs(dy)/vy_mmps])
            time.sleep(t+self.pause_after_motion_stop)        

    def move_absolute(self,x,y,vx_mmps=25,vy_mmps=25,verbose=False,wait=True):
        ''' Absolute movement '''
        # if x==None:
        #     string='MOVEABS Y'+str(y)+' F'+str(vy_mmps)
        # elif y==None:
        #     string='MOVEABS X'+str(x)+' F'+str(vx_mmps)
        # else:
        #     string='MOVEABS X'+str(x)+' F'+str(x)+' Y'+str(vy_mmps)+' F'+str(vx_mmps)
        assert np.logical_and(x<=400,x>=0),'Postion out of range.  X limits 0--400'
        assert np.logical_and(y<=400,y>=0),'Postion out of range.  Y limits 0--400'
        
        string='MOVEABS X'+str(x)+' F'+str(vx_mmps)+' Y'+str(y)+' F'+str(vy_mmps)
        if verbose:
            print('sending following string to ensemble: ',string)
        self.__SendStr__(string)
        if wait:
            x0,y0=self.get_position()
            t=np.max([abs(x-x0)/vx_mmps,abs(y-y0)/vy_mmps])
            time.sleep(t+self.pause_after_motion_stop)
        
    def get_position(self):
        #self.client_socket.recv(1000) # clear the current buffer of returns from the controller
        x=self.__SendStr__('PFBK X')
        y=self.__SendStr__('PFBK Y')
        ret_string = self.client_socket.recv(1000)
        #print(ret_string.decode())
        xy_raw = self.parse_return(ret_string.decode())[-3:-1]
        return float(xy_raw[0].split('%')[-1]),float(xy_raw[1].split('%')[-1])

if __name__ == "__main__":
    xy = AerotechXY()
    print(xy.get_position())
    xy.initialize(home=False)
    print('moving to 100,0')
    xy.move_absolute(x=100,y=0,vx_mmps=10)
    print('I ought to be finished moving now')
    # time.sleep(3)
    # xy.shutdown()
    

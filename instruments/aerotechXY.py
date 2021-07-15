'''
XY.py

JH 10/20/2011
'''

import socket
import time

class XY(object):
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

    --- Controller ---
    1) In Parameters->Controller->Communication change the following parameters:
        AsciiCmdEnable, enable Ethernet Socket 1
    2) In Parameters->Controller->Communication->Ethernet->Sockets change the following:
        InetSock1Port to 8000
        InetSock1Flags to TCP server
    3) AsciiCmdEnable must be set to true
    '''

#'132.163.82.11'
    def __init__(self, motor_controller_IP='192.168.0.11', motor_controller_port=8000,
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
        
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__connect__()
    
    def __connect__(self):
        self.client_socket.connect((self.motor_controller_IP, self.motor_controller_port))
        
    def __SendStr__(self,string):
        answer = self.client_socket.send(string+' '+self.AsciiCmdEOSChar)
        return answer
    
    def CloseConnection(self):
        self.client_socket.close()
        
    def ParseReturn(self,ret_string):
        ''' parse the return string from motor controller '''
        return ret_string.split('\n')
    
    def SendSeriesCommands(self,CMD1,CMD2):
        self.__SendStr__(CMD1)
        self.__SendStr__('WAIT MOVEDONE')
        self.__SendStr__(CMD2)
        
    def AzScan(self,xi,xf,yi,yf,v,sleeptime=.1):
        self.MoveAbs(xi,yi, v, v,True)
        time.sleep(sleeptime)
        self.__SendStr__('WAIT INPOS X Y')
        time.sleep(sleeptime)
        self.MoveAbs(xf,yf,v,v,True)
        time.sleep(sleeptime)
        self.__SendStr__('WAIT INPOS X Y')
        time.sleep(sleeptime)
        self.MoveAbs(xi,yi, v, v, True)
        #time.sleep(sleeptime)
        #self.__SendStr__('WAIT INPOS X Y')
        #time.sleep(sleeptime)
            
    def WaitMoveDone(self):
        self.__SendStr__('WAIT MOVEDONE X Y')        
    #-----------------------------------------------------------------------------------------
    def EnableAxis(self,axis='both'):
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
            
    def DisableAxis(self,axis='both'):
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
    
    def MoveInc(self,x_displacement,y_displacement,x_velocity=25,y_velocity=25,verbose=False):
        ''' Incremental movement '''
        if x_displacement==None:
            string='MOVEINC Y'+str(y_displacement)+' F'+str(y_velocity)
        elif y_displacement==None:
            string='MOVEINC X'+str(x_displacement)+' F'+str(x_velocity)
        else:
            string='MOVEINC X'+str(x_displacement)+' F'+str(x_velocity)+' Y'+str(y_displacement)+' F'+str(y_velocity)
        
        if verbose:
            print('sending following string to ensemble: ',string)
        self.__SendStr__(string)
        
    def MoveAbs(self,x_displacement,y_displacement,x_velocity=25,y_velocity=25,verbose=False):
        ''' Absolute movement '''
        if x_displacement==None:
            string='MOVEABS Y'+str(y_displacement)+' F'+str(y_velocity)
        elif y_displacement==None:
            string='MOVEABS X'+str(x_displacement)+' F'+str(x_velocity)
        else:
            string='MOVEABS X'+str(x_displacement)+' F'+str(x_velocity)+' Y'+str(y_displacement)+' F'+str(y_velocity)
        
        if verbose:
            print('sending following string to ensemble: ',string)
        self.__SendStr__(string)
        
    def Home(self,axis='both'):
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
        
    def GetPosition(self,sleeptime=.01):
        #self.client_socket.recv(1000) # clear the current buffer of returns from the controller
        x=self.__SendStr__('PFBK X')
        time.sleep(sleeptime)
        y=self.__SendStr__('PFBK Y')
        time.sleep(sleeptime)
        ret_string = self.client_socket.recv(1000)
        xy_raw = self.ParseReturn(ret_string)[-3:-1]
        return float(xy_raw[0].split('%')[-1]),float(xy_raw[1].split('%')[-1])
    
    def Initialize(self):
        print('initializing the XY stage.')
        self.EnableAxis('X')
        time.sleep(1)
        self.EnableAxis('Y')
        time.sleep(1)
        print('Homing X and Y')
        self.Home('X')
        time.sleep(10)
        self.Home('Y')
        time.sleep(10)
        print('done')
        
    def Shutdown(self):
        print('Shutting down the XY stage.  Homing X and Y...')
        self.Home('X')
        time.sleep(10)
        self.Home('Y')
        time.sleep(10)
        print('Disabling the axes and closing communications')
        self.DisableAxis('X')
        time.sleep(1)
        self.DisableAxis('Y')
        time.sleep(1)
        self.CloseConnection()
        print('Shutdown complete')
        
        
    

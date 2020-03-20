'''
 Crate_Card

 This class is the parent class of all other crate card classes, such as
 clock_card, ra8_card, and dfb_card.  It initializes with an address and
 knows the named_serial object to use for RS232/FiberOptic communication
 with the crate.  This means that any inheriting class, such as dfb_card,
 can simple use a self.send(cmd) to communicate with the crate.
 
 Alternately, I have added the ability to connect to a network socket, 
 initially for connecting to the NASA clock card, but eventually for
 connecting to the network aware NDFB card. 

 Note: Crate_Card is a modern class whose parent is "object"
'''

import serial
import socket
import struct

import named_serial

class Crate_Card(object):

    '''
    Crate_Card initializes with an address, set by a dip switch on
    the card itself.
    '''

    def __init__(self, address, port='rack', baud=115200, shared=True, use_sockets=False, slot=None, debug_mode=False):
        '''
        Crate_Card(address, port='rack', baud=115200, shared=True, use_sockets=False)
        address = card address (0-63)
        port = serial port based on entries in namedserialrc
        baud = serial baud rate
        shared = If True, the allow the port to be opened by other applications at the same time
        use_sockets = If True, then use network sockets to communicate rather than the serial port. 
        slot = Crate slot (0-20)
        '''

        self.address     = address
        self.port        = port
        self.baud        = baud
        self.shared      = shared
        self.use_sockets = use_sockets
        self.slot        = slot
        self.socket_host = 'localhost'
        self.socket_port = 7001
        self.the_socket  = 0
        self.serialport  = None
        self.card_type   = "crate_card"
        self.debug_mode  = debug_mode

        if self.use_sockets == True:
            # Instantiate a socket
            self.setSocket(self.socket_port)
        else:
            # Instantiates a serial port object
            # Note: It would be cleaner to have cards share a single name_serial object
            #       but duplication is easier and wastes very little memory
            #self.serialport = named_serial.Serial(self.port, baud = self.baud, shared = self.shared)
            self.setSerialPort(port)

    def setCardAddress(self, address):

        ''' setCardAddress(address)
        address = new card address (0-63)
        Sets the card address. 
        '''

        self.address = address

    def setUseSockets(self, use_sockets):
        
        '''Usage: setUseSockets(use_sockets)
        True - Use network sockets, False - use serial port
        '''
        
        self.use_sockets = use_sockets

    def setSerialPort(self, port):
        
        '''Usage: setSerialPort(port)
        Set the serial port from named_serial
        '''
        
        #print "crate_card: set serial [%s] %i" % (port, self.baud)
        
        self.port = port
        try:
            self.serialport = named_serial.Serial(self.port, baud = self.baud, shared = self.shared)
        except serial.serialutil.SerialException:
            print("WARNING: serial port could not be found!")

    def setSocket(self, socket_port, socket_host = 'localhost'):
        
        '''
        Usage: setSocket(socket_port, socket_host)
        socket_port = network socket port
        socket_host = hostname or ip address of the server
        Connect the socket
        '''
        
        self.socket_port = socket_port
        self.socket_host = socket_host

        print(("crate_card %i: connecting to socket [%s] %i" % (self.address, self.socket_host, self.socket_port)))
        self.the_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ## apparently the double parens is to make a single list
        ### or some other argument
        self.the_socket.connect((self.socket_host, self.socket_port))

    def setShared(self, shared):
        
        '''
        Usage: setShared(shared)   - Set the shared flag to True or False
        '''
        
        self.shared = shared
        #self.serialport = named_serial.Serial(self.port, baud = self.baud, shared = self.shared)
        self.setSerialPort(self.port)

    ######################################################################################
    # Send command uses the serial port to send a 28-bit binary command to the crate.

    def send(self, cmd, register=-1, value=-1):

        '''
        Usage: send(cmd, register, value)
        cmd      = Only for serial sends. serial command to send. 
        register = Default is -1. Only for network sockets. Register to send to. 
        value    = Default is -1. Only for network sockets. Value to set for that register. 
        Sends a command to the crate. Converts to low level binary command over the serial port, 
        or a higher level command over a network socket. 
        '''

        # Send the message
        if self.use_sockets == True:
            socket_string = "set %i %i %i\n" % (self.address, register, value)
            #self.the_socket.send(msg)
            self.the_socket.send(socket_string)
            print(("crate_card %i Sending [%s]" % (self.address, socket_string)))
        else:
            # Break the 28 bits of data into chunks
            b0 = (cmd & 0x7f ) << 1            # 1st 7 bits shifted up 1
            b1 = ((cmd >> 7) & 0x7f) <<  1     # 2nd 7 bits shifted up 1
            b2 = ((cmd >> 14) & 0x7f) << 1     # 3rd 7 bits shifted up 1
            b3 = ((cmd >> 21) & 0x7f) << 1     # 4th 7 bits shifted up 1
            b4 = (self.address << 1) + 1       # Address shifted up 1 bit with address bit set
    
            # Construct the serial message
            msg = struct.pack('BBBBB', b0, b1, b2, b3, b4)
            if self.debug_mode is True:
                print(("send address,cmd,register,value = %i,%i,%i,%i" % (self.address, cmd, register, value)))

            self.serialport.write(msg)


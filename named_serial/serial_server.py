#!/usr/bin/python

import socket
import struct
import named_serial

HOST = ''                 # Symbolic name meaning all available interfaces
PORT = 7001               # port number
serial_port = 'rack'
baud = 115200
address = 0
shared = True

def setSerialPort(port, baud=115200, shared=True):
    
    '''Usage: setSerialPort(port)   - Set the serial port from named_serial'''
    
    print "serial socket server: set serial [%s] %i" % (port, baud)
    
    serialport = named_serial.Serial(port, baud=baud, shared=shared)
    
    return serialport

def send(serialport, msg):

    print "Received [%s]" % msg
    #print serialport

    msg_array = msg.split(" ")

    command = msg_array[0]
    address = int(msg_array[1])
    wreg = int(msg_array[2])
    value = int(msg_array[3])

    # Convert to serial code
    cmd = (wreg << 25) + value
    b0 = (cmd & 0x7f ) << 1            # 1st 7 bits shifted up 1
    b1 = ((cmd >> 7) & 0x7f) <<  1     # 2nd 7 bits shifted up 1
    b2 = ((cmd >> 14) & 0x7f) << 1     # 3rd 7 bits shifted up 1
    b3 = ((cmd >> 21) & 0x7f) << 1     # 4th 7 bits shifted up 1
    b4 = (address << 1) + 1       # Address shifted up 1 bit with address bit set

    # Construct the serial message
    value = struct.pack('BBBBB',b0, b1, b2, b3, b4)


    # Send the message
    serialport.open()
    serialport.write(value)
    serialport.close()

def breakout_packets(serialport, packet):

    packet_array = packet.split('\n')
    print "Received %i data packets" % len(packet_array)
    #print packet
    
    for p in packet_array:
        if len(p) > 0:
            send(serialport, p)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))

s.listen(1)
conn, addr = s.accept()
print 'Connected by', addr

serialport = setSerialPort(serial_port)
#print serialport

while 1:
    ###receive up to 1024 bytes
    data = conn.recv(1024)
    if not data: break
    print "Received ", len(data), " bytes"
    #print "Data = ", data
    breakout_packets(serialport, data)

conn.close()
serialport.close()

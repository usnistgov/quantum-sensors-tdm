'''Assign names to serial ports in a configuration file.
Idea is to provicd a logical name for ports based on
desired function (eg "voltmeter")
Based on (and requires) http://pyserial.sourceforge.net/
'''
# The current version of pyserial has an important difference in
# implementation on different platforms.  On win32, the serial ports
# are not shared among different processes - the open call set the
# "exclusive access bit".  On all the posix platforms, any one with
# necessary permissions can open up and do whatever.  It seems the
# serial port can be opened about 1000 times (and each process can
# read and write)
#
# To unify the behaviour (at least a bit) I have added calls to fcntl.lockf
# as default and added an optional parameter "shared" to the init which
# allows the previous posix behaviour to continue.

import os
import platform
import serial
if os.name == 'posix':
    import fcntl #for locking
_namedports = {}


def __setup():
    '''On module import read rc file and setup namedports dictionary'''
    # stuck in a function to hide all the variables
    if platform.system() == 'Linux':
        rcfilename = '/etc/namedserialrc'
    elif platform.system() == 'Windows':
        # I don't know the best place for a win32 configuration file ...
        rcfilename = 'namedserialrc'
    elif platform.system() == 'Darwin':
        rcfilename = '/etc/namedserialrc'
    rcfile = open(rcfilename, 'rt')
    lines = rcfile.readlines()
    rcfile.close()#On module import, setup the list
    _namedports = {}
    for line in lines:
        if line[0] == '#' or len(line.strip()) <= 1:
            continue
        port, name = line.split()
        _namedports[name] = port
    return _namedports


def namedports():
    global _namedports
    if len(_namedports) == 0:
        _namedports = __setup()
    return _namedports


def getnames():
    '''Convenience routine for building ui - returns all defined names'''
    return list(namedports().keys())


def _setup_for_testing(d):
    global _namedports
    _namedports = d


class Serial(serial.Serial):
    ''' Wrapper class around the serial.Serial that uses logical device
    names instead of physical device names.
    '''
    def __init__(self, port=None, baud=115200, shared=False, **kwargs):
        '''Return a serial port object.  Parameter are:
        port - the logical name of the port defined in the config file
        baud - baudrate
        shared - allow others to used this serial port.  Only works
                 on posix system
        '''
        #print "port", port, baud, shared
        self.the_port   = port
        self.the_baud   = baud
        self.the_shared = shared

        # pass other serial init args?
        if port not in namedports():
            # Hmmm - raise an exception or just pass it on to serial
            # allowing use of "normal" device names?
            # This is probably better.
            raise ValueError("Named port '%s' not in configuration file" % port)
        myport = namedports()[port]
        if "dummy" in myport:
            self._set_as_dummy(port, myport)
            return
        try:
            # serial.Serial.__init__(self, myport, baud)
            super(Serial, self).__init__(port=myport, baudrate=baud, **kwargs) #better?
        except serial.serialutil.SerialException as e:
            e.args = ("Error auto-loading serial port %s" % port, e.args[0])
            raise e
        if not shared and os.name == 'posix':
            try:
                fcntl.lockf(self.fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
            except IOError as e:
                e.args = ('Error locking on serial port %s' % port, e.args[0])
                raise e
        if shared and os.name != 'posix':
            raise ValueError('No shared ports on non-posix systems')

    def _set_as_dummy(self, port, myport):
        # print("namedserial: _set_as_dummy: port={}, myport={}".format(port, myport))
        #self.write = lambda v: None
        self.write = self._dummy_write
        
    def _dummy_write(self, data):
        return len(data)

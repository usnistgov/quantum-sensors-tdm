'''Module for controlling  NIST  "bluebox" voltage sources.
They currently exist in three varieties, "mrk1", "mrk2" and "tower" which
use different input format'''
from named_serial import Serial

class BlueBox(object):
    '''Module for controlling NIST "bluebox" voltage sources.
    They currently exist in two varieties, "mrk1", "mrk2" and "tower" which
    use different input format'''
    def __init__(self, port='vbox', version='mrk1', 
        baud=9600, address=0, channel=0, shared=True):
        '''Return a bluebox object - parameters:
        port - a logical portname defined in namedserialrc
        version -  mrk1, mrk2 or tower so far
        baud -  only used for mrk1. Actually all of the
                mrk1 boxes are 9600 baud.
        address - only used for the mrk1 and tower.  I think all
                of the mrk1 boxes have an address of 0. tower
                range is 0-127. 
        channel - only used for the tower. Values are 0-7. 
        shared - allow other processes to open this device
                 only works on posix
        '''
        self.port = port
        self.serial = None
        self.value = None  #unknown value
        self.version = None
        self.address = 0
        self.configure(version, baud, address, channel, shared)

    def configure(self, version='mrk1', baud=9600, address=0, channel=0, shared=True):
        '''Configure' a bluebox object - parameters:
        version -  mrk1, mrk2 or tower so far
        baud -  only used for mrk1. Actually all of the
                mrk1 boxes are 9600 baud.
        address - only used for the mrk1 and tower.  I think all
                of the mrk1 boxes have an address of 0. tower
                range is 0-127. 
        channel - only used for the tower. Values are 0-7. 
        shared - allow other processes to open this device
                 only works on posix
        '''
        if version == 'mrk1' or version == 1:
            self.version = 'mrk1'
            self.baud = baud
            self.address = 0
            self.getbytes = self._getbytes_mrk1
            self.min_voltage = 0.0
            self.max_voltage = 6.5535
            #self.port = 'vbox' 
        elif version == 'mrk2' or version == 2:
            self.version = 'mrk2'
            self.baud = 115200 # mrk2 has fixed baud
            self.getbytes = self._getbytes_mrk2
            self.min_voltage = 0.0
            self.max_voltage = 6.5535
            #self.port = 'vbox'
        elif version == 'tower':
            self.version = 'tower'
            self.baud = 115200
            self.address = address
            self.channel = channel
            self.getbytes = self._getbytes_tower
            self.min_voltage = 0.0
            self.max_voltage = 2.5
            self.port = 'tower'
        else:
            raise ValueError('Unknown bluebox version')
        if self.serial:
            self.serial.close()
        #print "port = ", self.port
        self.serial = Serial(port=self.port, baud=self.baud, shared=shared)
        self.value = None
               
    def close(self):
        self.serial.close()

    def getvolt(self):
        '''Get the last set voltage of the bluebox'''
        return self.value

    def setvolt(self, volt): 
        '''Set the voltage of the bluebox. 
        Parameters:
        volt - number between the voltage source min and max
        '''
        if (volt < self.min_voltage or volt > self.max_voltage):
            raise ValueError('Voltage set value out of range')
        dac_units = int(round(volt * 65535.0 / self.max_voltage))
        return self.setVoltDACUnits(dac_units)

    def setVoltDACUnits(self, val): 
        '''Set the voltage of the bluebox. 
        Parameters:
        dacvalue
        '''
        b = self.getbytes(val)
        n = self.serial.write(b)
        if n != len(b):
            raise Exception(f"tried to write {len(b)} bytes, actually wrote {n}")
        self.value = self.max_voltage*val/65535.0
        return self.value

    def _getbytes_mrk1(self, value):
        #   word0  high6bit 00
        #   word1  mid6bit 01
        #   word2  low4bit 10
        #   word3  address 11   -  address is 4 bits
        bytevals = []
        bytevals.append((value / 1024) * 4)
        bytevals.append((((value / 16) & 0x3f) * 4) + 1)
        bytevals.append(((value & 0xf) * 4) + 2)
        bytevals.append((self.address * 4) + 3)
        return bytes(bytevals)

    def _getbytes_mrk2(self, value):
        #  word 0 - 7 least significant bits  + 0
        #  word 1 - 7 middle bits + 0
        #  word 2 - 5 error checking bits + 2 most significant bits + 1
        bytevals = []
        bytevals.append((value & 0x7f) << 1)
        bytevals.append((value & 0x3f80) >> 6)
        bytevals.append(( 10 << 3 ) + ( ( value & 0xc000 ) >> 13 ) + 1)
        return bytes(bytevals)

    def _getbytes_tower(self, value):
        ''' get 4 bytes for the tower '''
        address = int(self.address)
        channel = int(self.channel)
        #print 'address = [%3i] channel = [%3i]' % (address, channel)
        addr = (address<<3) + channel
        #print 'addr = ', addr
        bytevals = []
        bytevals.append(value & 0x7F)
        bytevals.append((value>>7) & 0x7F)
        bytevals.append((value>>14) + ((addr & 0x1F) << 2) )
        bytevals.append(0x80 + (addr >> 5))
        #print "bb bytevals", bytevals
        return bytes(bytevals)

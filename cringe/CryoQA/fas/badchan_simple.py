# import named_serial
import struct


class BadChanSimple:

    def __init__(self, chan=0, cardaddr=3, serialport=None):

        self.serialport=serialport

        if self.serialport is None:
            import named_serial
            self.serialport = named_serial.Serial(port='rack', shared=True)

        self.chan = chan
        self.cardaddr = cardaddr

        self.dc = False
        self.lohi = False
        self.tri = False

        self.d2a_hi_value = 0x0000
        self.d2a_lo_value = 0x0000

    def change_channel(self, channel):
        self.chan = channel
        self.send_wreg2()

    def set_d2a_hi_value(self, value_dac):
        self.d2a_hi_value = value_dac
        self.send_wreg4()

    def set_d2a_lo_value(self, value_dac):
        self.d2a_lo_value = value_dac
        self.send_wreg5()

    def set_lohi(self, hi_nlo):
        self.lohi = hi_nlo
        self.send_wreg4()

    def set_dc(self, dc):
        self.dc = dc
        self.send_wreg4()

    def set_tri(self, on_noff):
        self.tri = on_noff
        self.send_wreg4()

    def send_channel(self):
        self.send_wreg2()
        self.send_wreg4()
        self.send_wreg5()

    def send_wreg2(self):
        wreg = 2 << 25
        wregval = wreg | self.chan
        self.sendreg(wregval)

    def send_wreg4(self):
        wreg = 4 << 25
        wreg = wreg | (int(self.dc) << 21)
        wreg = wreg | (int(self.lohi) << 20)
        wreg = wreg | (int(self.tri) << 19)
        wregval = wreg | self.d2a_hi_value
        self.sendreg(wregval)

    def send_wreg5(self):
        wreg = 5 << 25
        wregval = wreg | (self.d2a_lo_value << 8)
        self.sendreg(wregval)

    def sendreg(self, wregval):
        b0 = (wregval & 0x7f) << 1  # 1st 7 bits shifted up 1
        b1 = ((wregval >> 7) & 0x7f) << 1  # 2nd 7 bits shifted up 1
        b2 = ((wregval >> 14) & 0x7f) << 1  # 3rd 7 bits shifted up 1
        b3 = ((wregval >> 21) & 0x7f) << 1  # 4th 7 bits shifted up 1
        b4 = (self.cardaddr << 1) + 1  # Address shifted up 1 bit with address bit set

        msg = struct.pack('BBBBB', b0, b1, b2, b3, b4)
        self.serialport.write(msg)

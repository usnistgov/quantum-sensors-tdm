import struct
import named_serial
from cringe.shared import terminal_colors as tc
import logging


#########################################################################
#
# Energy Management Unit (EMU)
#
# EMU is a child of Crate_Card, inheriting the ability to send serial
# port commands to the card in the crate at address 0x7F, the default for the
# EMU card.
#
#
# If you import this class to use to control a EMU card, use the following methods:
#
#    powerOn()
#    powerOff()
#    clockPass()
#
# Commands on the PCCC:
#
# Command	Data	Function	Description
# 0x9		N/A		powerOn		Turns the EMU on
# 0xA		N/A		powerOff	Turns the EMU off
# 0xB		1 bit	clackPass	Controls the clock pass through buffers so that the Power Card can pass through LSync and MLCK to the crate from
#                                    EMU Daughter Card. ! bit is located in the LSB of the data.
#
##########################################################################

# Not sure why Franks serial class requires irrelevant horse shit in the send command. Will look into and possible make another command

class EMU_Card(object):

    def __init__(self, address = 0x7F):

        super(EMU_Card, self).__init__()

        self.serialport = named_serial.Serial(port='rack', shared = True)

        self.revision      = 0.1
        self.card_type     = "emu"
        self.dummy_val = 0  #Work around Frank's socket weirdness
        
        self.address = address
        
        logging.debug(tc.INIT + "building EMU card: slot 19-21 / address"+str(self.address)+ tc.ENDC)
        

    ######################################################################################

    def powerOn(self):
        ''' Turn crate EMU/PCCC power card on. '''
        logging.debug(tc.FCTCALL + "switch power to crate through EMU:"+ tc.BOLD+ "ON"+ tc.ENDC)
        

        wregval = 0b001 << 25

        self.sendReg(wregval)

    ######################################################################################

    def powerOff(self):
        ''' Turn crate EMU/PCCC power card off. '''
        logging.debug(tc.FCTCALL + "switch power to crate through EMU:"+ tc.BOLD+ "OFF"+ tc.ENDC)
        

        wregval = 0b010 << 25

        self.sendReg(wregval)

    ######################################################################################

    def clockPass(self, clock_pass = False):
        ''' Backplane clocking pass through control '''

        wregval = 0b011 << 25
        
        wregval = wregval + (clock_pass << 24)

        self.sendReg(wregval)
                
    def sendReg(self, wregval):
        logging.debug(tc.COMMAND + "send to address"+str(self.address)+ ":"+ tc.BOLD+str(wregval)+ tc.ENDC)
        b0 = (wregval & 0x7f ) << 1            # 1st 7 bits shifted up 1
        b1 = ((wregval >> 7) & 0x7f) <<  1     # 2nd 7 bits shifted up 1
        b2 = ((wregval >> 14) & 0x7f) << 1     # 3rd 7 bits shifted up 1
        b3 = ((wregval >> 21) & 0x7f) << 1     # 4th 7 bits shifted up 1
        b4 = (self.address << 1) + 1           # Address shifted up 1 bit with address bit set

        msg = struct.pack('BBBBB', b0, b1, b2, b3, b4)
        self.serialport.write(msg)

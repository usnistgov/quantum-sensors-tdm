from . import crate_card

#########################################################################
#
# PCCC (Power Conditioning and Control Card) aka PC^3 (PC cubed)
#
# PCCC_Card is a child of Crate_Card, inheriting the ability to send serial
# port commands to the card in the crate at address 0x7F, the default for the
# PCCC card.
#
#
# If you import this class to use to control a PCCC card, use the following methods:
#
#    powerOn()
#    powerOff()
#    clockPass()
#
# Commands on the PCCC:
#
# Command	Data	Function	Description
# 0x9		N/A		powerOn		Turns the PCCC on
# 0xA		N/A		powerOff	Turns the PCCC off
# 0xB		1 bit	clackPass	Controls the clock pass through buffers so that the Power Card can pass through LSync and MLCK to the crate from
#                                    PCCC Daughter Card
#
##########################################################################

# Not sure why Franks serial class requires irrelevant horse shit in the send command. Will look into and possible make another command

class PCCC_Card(crate_card.Crate_Card):

    def __init__(self, address=0x7F):

        super(PCCC_Card, self).__init__(address=address)

        self.revision      = 0.1
        self.card_type     = "pccc_clock"
        self.dummy_val = 0  #Work around Frank's socket weirdness

    ######################################################################################

    def powerOn(self):
        ''' Turn crate EMU/PCCC power card on. '''

        wregval = 0b1001 << 25

        self.send(wregval, 1, self.dummy_val)

    ######################################################################################

    def powerOff(self):
        ''' Turn crate EMU/PCCC power card off. '''

        wregval = 0b1010 << 25

        self.send(wregval, 1, self.dummy_val)

    ######################################################################################

    def clockPass(self, clock_pass = False):
        ''' Backplane clocking pass through control '''

        wregval = 0b1011 << 25
        
        wregval = wregval + (clock_pass << 24)

        self.send(wregval, 1, self.dummy_val)

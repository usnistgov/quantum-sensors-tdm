import sys
import bluebox


class TowerChannel:
    
    def __init__(self, column=0, cardaddr=3, serialport="tower", shockvalue=65535):

        self.COMMAND = '\033[95m'
        self.FCTCALL = '\033[94m'
        self.INIT = '\033[92m'
        self.WARNING = '\033[93m'
        self.FAIL = '\033[91m'
        self.ENDC = '\033[0m'
        self.BOLD = "\033[1m"

        self.green = "90EE90"
        self.red = "F08080"
        self.yellow = "FFFFCC"
        self.grey = "808080"
        self.white = "FFFFFF"

        self.address = cardaddr
        self.column = column
        self.serialport = serialport
        self.shockvalue = shockvalue
        self.current_value = 0

        self.bluebox = bluebox.BlueBox(port=serialport,
                                       version='tower',
                                       address=self.address,
                                       channel=self.column,
                                       shared=True)

    def set_value(self, dac_value):
        self.current_value = dac_value
        #print("towerchannel sending %g to addr %g, chn %g"%(dac_value, self.address, self.column))
        self.bluebox.setVoltDACUnits(self.current_value)

from named_serial import Serial
from instruments.serial_instrument import SerialInstrument

class SRS980(SerialInstrument):
    def __init__(
        self,
        slot,
        port = "srs",
        baud = 9600
    ):
        self.slot=slot
        super().__init__(port, baud)

    def sndt(self, cmd):
        self.write(f'SNDT {self.slot},"{cmd}"')

    def setvolt(self, volt):
        self.sndt(f"VOLT {volt:.4e}")

    def output_on(self):
        self.sndt('OPON')

    def output_off(self):
        self.sndt("OPOF")
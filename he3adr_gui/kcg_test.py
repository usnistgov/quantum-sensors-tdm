import random
import time

charcoal_heatswitch_closed = False

class lakeshore():
    def __init__(self):
        self.heater_out = 10

    def getTemperature(self, channel):
        if channel == 3:
            return 2.0 + random.random()*0.1 - 0.05
        elif channel == 4:
            return 0.1 + random.random()*0.02 - 0.01
            
    def getManualHeaterOut(self):
        return self.heater_out

    def setManualHeaterOut(self, heater_out):
        self.heater_out = heater_out

    def MagUpSetup(self, heater_resistance=0):
        pass

    def getControlMode(self):
        return 'off'
    


class cryocon():
    def __init__(self):
        self.control = False
        self.loops = { 'A': 1, 'B': 2}
        self.temps = { 1 : 30, 2: 55}

    def getTemperature(self, channel):
        if channel == 'A':
            return 65 + random.random()*1 - 0.5
        elif channel == 'B':
            if not charcoal_heatswitch_closed and self.control:
                return self.temps[2] + random.random()*0.2 - 0.1
            else:
                return 3 + random.random()*0.2 - 0.1
                
        elif channel == 'C':
            return 3 + random.random()*0.2 - 0.1
        elif channel == 'D':
            return 0.3 + random.random()*0.05 - 0.025

    def CONTrol(self):
        self.control = True

    def STOP(self):
        self.control = False

    def getControl(self):
        if self.control:
            return 'ON'
        else:
            return 'OFF'

    def setLoopThermometer(self, loop, channel):
        self.loops[channel] = loop

    def setTemperature(self, loop, temp):
        self.temps[loop] = temp

    def getTemperatureSetpoint(self, loop):
        return self.temps[loop]
    
    


class labjack():
    def __init__(self):
        pass

    def getAnalogInput(self, channel):
       return 0 + random.random()*0.1 - 0.04

    def setDigIOState(self, channel, state):
        pass

    def pulse_digitalState(self, channel):
        time.sleep(3)

    def setRelayToControl(channel):
        pass


class zaber():
    def __init__(self):
        pass

    def CloseHeatSwitch(self):
        count = 0
        while count < 5:
            time.sleep(1)
            print("Closing Charcoal HS")
            count += 1
        charcoal_heatswitch_closed = True

    def OpenHeatSwitch(self):
        count = 0
        while count < 5:
            time.sleep(1)
            print("Opening Charcoal HS")
            count += 1
        charcoal_heatswitch_closed = False

    def MoveRelative(self,steps):
        print('Charcoal MoveRelative', steps)

class email():
    def __init__(self):
        pass
    

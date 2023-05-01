
from instruments import Cryocon24c_ser

cc = Cryocon24c_ser()
for ch in range(1,5):
    print(ch)
    print(f"getHeaterPower {cc.getHeaterPower(ch)}")
    print(f"getControl {cc.getControl()}")
    print(f"getTemperatureSetpoint {cc.getTemperatureSetpoint(ch)}")
    print(f"getTemperature {cc.getTemperature(ch)}")
cc.setLoopThermometer(1,'A')
cc.setTemperature(1, 3)
cc.setLoopThermometer(2, 'B')
cc.setTemperature(2, 5)
for ch in range(1,5):
    print(ch)
    print(f"getHeaterPower {cc.getHeaterPower(ch)}")
    print(f"getControl {cc.getControl()}")
    print(f"getTemperatureSetpoint {cc.getTemperatureSetpoint(ch)}")
    print(f"getTemperature {cc.getTemperature(ch)}")

from instruments import Cryocon22  
ccon = Cryocon22(port='cryocon')
loop_channel=1
print(ccon.getControlTemperature(loop_channel))
ccon.setControlTemperature(4.0,loop_channel)
#print(ccon.serial.readline())
#ccon.write('loop '+str(loop_channel)+ ':setpt '+str(4.0))
#ccon.write('loop '+str(loop_channel)+ ':setpt '+str(4.0))
#ccon.serial.flushOutput()
print(ccon.getControlTemperature(loop_channel))

#print(ccon.getControlSource(1))
#ccon.controlLoopSetup(loop_channel=1,control_temp=4.0,t_channel='a',PID=[1,5,0],heater_range='low')

# ii=0
# while 1:
#     print(ii,ccon.getTemperature(channel='both'))
#     #print(ii,ccon.getControlSource(1))
#     ii=ii+1


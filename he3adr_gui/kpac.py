import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import time
from instruments import Cryocon22
from instruments import Lakeshore370
from instruments import  AgilentE3644A
from instruments import  labjack
from instruments import Cryocon24c_ser
import numpy
import time
import datetime
from task import Task

# Decorator to help wrap a function with signals. When decorating
# a function with this, sig_start is emitted at the beginning of
# the function, and sig_done at the end.
#
# See http://thecodeship.com/patterns/guide-to-python-function-decorators/
#
# This is not working, apparently because of this:
#
# http://stackoverflow.com/questions/17578428/pyqt5-signals-and-slots-qobject-has-no-attribute-error
#
# xxx - what about exceptions raised? Should still emit the done
# signal? something else?

def start_done_emit(sig_start, sig_done):
    def decorator(func):
        def wrapper(*args, **kwargs):
            sig_start.emit()
            func(*args, **kwargs)
            sig_done.emit()
        return wrapper
    return decorator

class KPAC(QObject):

    def __init__(self):
        QObject.__init__(self)

        # initial time
        self.t0 = time.time()

        # objects for communication with the instruments
        args = QCoreApplication.arguments()
        if len(args) > 1 and args[1] == 'test':
            print('Running in test mode')
            import kcg_test
            self.cc = kcg_test.cryocon()
            self.lsh = kcg_test.lakeshore()
            # self.email = tes_mail.TESMail()
        else:
            self.cc = Cryocon24c_ser(port='cryocon1', shared=True)
            try:
                self.cc2 = Cryocon24c_ser(port='cryocon2', shared=True)
            except Exception as ex:
                print(ex)
                self.cc2 = None
            try:
                self.ps = AgilentE3644A(port='heater')
            except Exception as ex:
                print(ex)
                self.ps = None
            self.lsh = Lakeshore370()
            self.lj = labjack.Labjack()
            # self.email = tes_mail.TESMail()

        print(f"{self.cc.shared=}")
        print(f"{self.lsh.shared=}")
        # Times at which we have polled for temp/voltage/current/pressure
        self.poll_times = []

        # array of temp/voltage/current/pressure measurements
        self.upper_temps = []
        self.charcoal_temps = []
        self.plate_3k_temps = []
        self.he3_temps = []
        if self.cc2 != None:
            self.diode1_temps = []
            self.diode2_temps = []
        else:
            self.diode1_temps = None
            self.diode2_temps = None
        if self.ps != None:
            self.heater_voltage = []
        else:
            self.heater_voltage = None
        self.ring_temps = []
        self.adr_temps = []
        self.he3_pressure = []
        self.kepco_voltage = []
        self.kepco_current = []
        self.cc_loop1_output = []
        self.cc_loop2_output = []

        # states of heat switches. None means that we don't know
        # (which is the case on startup)
        self.hs_adr_closed = None
        self.hs_charcoal_closed = None
        self.hs_pot_closed = None

        self.charcoal_setpoint = None


    sig_charcoal_above_setpoint = pyqtSignal()
    
    def update_temps(self, directory):
        self.poll_times.append((time.time() - self.t0)/60.0)
        self.upper_temps.append(self.cc.getTemperature('A'))
        self.charcoal_temps.append(self.cc.getTemperature('B'))
        self.plate_3k_temps.append(self.cc.getTemperature('C'))
        self.he3_temps.append(self.cc.getTemperature('D'))
        if self.cc2 != None:
            self.diode1_temps.append(self.cc2.getTemperature('A'))
            self.diode2_temps.append(self.cc2.getTemperature('B'))
        if self.ps != None:
            self.heater_voltage.append(self.ps.measureVoltage())
        self.ring_temps.append(self.lsh.getTemperature(3))
        self.adr_temps.append(self.lsh.getTemperature(4))
        self.kepco_voltage.append(self.lj.getAnalogInput(0)*2)
        self.kepco_current.append(self.lj.getAnalogInput(2)*2)
        self.he3_pressure.append(self.lj.getAnalogInput(1)*50.0*0.0689) # Pressure converted from V to bar, LabJack
        self.cc_loop1_output.append(self.cc.getHeaterPower(1))
        self.cc_loop2_output.append(self.cc.getHeaterPower(2))

        if self.charcoal_setpoint != None and self.charcoal_temps[-1] > self.charcoal_setpoint:
            self.sig_charcoal_above_setpoint.emit()

        args = QCoreApplication.arguments()
        if len(args) > 1 and args[1] == 'test':
            pass
        else:
            # Log to a new file each day
            if len(directory) == 0 or directory[-1] != '/':
                directory = directory + '/'
            filename = directory + time.strftime('%Y-%m-%d')
            f = open(filename, 'a')
            f.write('%f\t%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t' %
                    (self.poll_times[-1]*60 + self.t0,
                     datetime.datetime.fromtimestamp(self.poll_times[-1]*60 + self.t0).strftime('%Y-%m-%dT%H:%M:%S'),
                     self.upper_temps[-1],
                     self.charcoal_temps[-1],
                     self.plate_3k_temps[-1],
                     self.he3_temps[-1],
                     self.ring_temps[-1],
                     self.adr_temps[-1],
                     self.kepco_voltage[-1],
                     self.kepco_current[-1],
                     self.he3_pressure[-1],
                     self.cc_loop1_output[-1],
                     self.cc_loop2_output[-1],
                    ))
            if self.cc2 != None:
                f.write('%f\t%f\t' % 
                        (self.diode1_temps[-1],
                         self.diode2_temps[-1],
                        ))
            if self.ps != None:
                f.write('%f\t' % 
                        (self.heater_voltage[-1],
                        ))
            f.write('\n')
            f.close()
        
    sig_adr_hs_close_start = pyqtSignal()
    sig_adr_hs_close_done = pyqtSignal()
    def closeADRHeatSwitch(self):
        '''Closes the ADR heat switch through the labjack.'''
        self.sig_adr_hs_close_start.emit()
        self.lj_pulse_digitalState(10)
        self.sig_adr_hs_close_done.emit()

    sig_adr_hs_open_start = pyqtSignal()
    sig_adr_hs_open_done = pyqtSignal()
    def openADRHeatSwitch(self):
        self.sig_adr_hs_open_start.emit()
        self.lj_pulse_digitalState(8)
        self.sig_adr_hs_open_done.emit()

    def lj_pulse_digitalState(self, ch, sleep_s=0.1, retries=3):  
        for i in range(retries):
            try:         
                if i > 0:
                    print("lj_pulse_digitalState retrying set high")
                self.lj.setDigIOState(ch, 'high')
                break
            except Exception:
                continue
        time.sleep(sleep_s)
        for i in range(retries):
            try:         
                if i > 0:
                    print("lj_pulse_digitalState retrying set low")
                self.lj.setDigIOState(ch, 'low')
                break
            except Exception:
                continue

    sig_pot_hs_close_start = pyqtSignal()
    sig_pot_hs_close_done = pyqtSignal()
    def closePotHeatSwitch(self):
        self.sig_pot_hs_close_start.emit()
        self.lj_pulse_digitalState(11)
        self.sig_pot_hs_close_done.emit()

    sig_pot_hs_open_start = pyqtSignal()
    sig_pot_hs_open_done = pyqtSignal()
    def openPotHeatSwitch(self):
        self.sig_pot_hs_open_start.emit()
        self.lj_pulse_digitalState(12)
        self.sig_pot_hs_open_done.emit()

    sig_charcoal_hs_close_start = pyqtSignal()
    sig_charcoal_hs_close_done = pyqtSignal()
    def closeCharcoalHeatSwitch(self):
        self.sig_charcoal_hs_close_start.emit()
        self.lj_pulse_digitalState(17)
        self.sig_charcoal_hs_close_done.emit()

    sig_charcoal_hs_open_start = pyqtSignal()
    sig_charcoal_hs_open_done = pyqtSignal()
    def openCharcoalHeatSwitch(self):
        self.sig_charcoal_hs_open_start.emit()
        self.lj_pulse_digitalState(19)
        self.sig_charcoal_hs_open_done.emit()

    def enable_cryocon_control(self):
        self.cc.CONTrol()

    def disable_cryocon_control(self):
        self.cc.STOP()

    def write_cryocon_control(self, upperTemp, charcoalTemp):
        self.cc.setLoopThermometer(1,'A')
        self.cc.setTemperature(1, upperTemp)
        self.cc.setLoopThermometer(2, 'B')
        self.cc.setTemperature(2, charcoalTemp)
        self.charcoal_setpoint = charcoalTemp

    def read_cryocon_control(self):
        control = self.cc.getControl()
        upper_setpt = self.cc.getTemperatureSetpoint(2)
        charcoal_setpt = self.cc.getTemperatureSetpoint(1)
        return (control, upper_setpt, charcoal_setpt)

    RELAY_RAMP = 1
    RELAY_CONTROL = 2
        
    def set_relay(self, state):
        if state == self.RELAY_RAMP:
            self.lj.setRelayToControl(io_channel = 16)
        elif state == self.RELAY_CONTROL:
            self.lj.setRelayToControl(io_channel = 18)
        else:
            pass # xxx handle error
    
    def stop_adr_regulate(self):
        self.lsh.setControlMode('off')
        
    def start_adr_regulate(self, temp):
        '''Sets the lakeshore to temperature regulation mode without putting in the control resistor. The ADR does not go to 0 field for this regulation. '''
        
        self.lsh.setControlMode(controlmode = 'closed')
        self.lsh.setReadChannelSetup(channel=4, mode='current', exciterange=3.16e-10)
        self.lsh.setTemperatureControlSetup(channel = 4, maxrange = 1, htrres = 100)
        self.lsh.setTemperatureSetPoint(setpoint=temp)
        self.lsh.setHeaterRange(range=1)
        self.lj.getLabjackCalibration()
        self.lj.setDACVoltage(0, 0, bits=16)
        
    def get_regulate_state(self):
        return self.lsh.getControlMode()

    def calc_mag_steps(self, start, end, time, pausetime):
        num_steps = int(time // pausetime)
        return numpy.linspace(start, end, num_steps).tolist()
        
    
    def get_ramp_current_level(self):
        return self.lsh.getManualHeaterOut()

    sig_ramp_wait_start = pyqtSignal()
    sig_ramp_wait_done = pyqtSignal()
    sig_ramp_value = pyqtSignal(float)

    sig_ramp_up_start = pyqtSignal()
    sig_ramp_up_done = pyqtSignal()
    sig_ramp_down_start = pyqtSignal()
    sig_ramp_down_done = pyqtSignal()
    sig_soak_start = pyqtSignal()
    sig_soak_done = pyqtSignal()

    def magnet_ramp_careful(self, i_end, mag_time, pausetime, start_temp, max_temp, soak_time, alerts_emails):
        '''Ramp ADR magnet to a final value, with checks for quenching and whether current is actually changing.'''

        #wait for 3K stage temp to get below 3.2K
        self.sig_ramp_wait_start.emit()
        while self.cc.getTemperature('C') >= start_temp:
            time.sleep(5)
        self.sig_ramp_wait_done.emit()

        self.lsh.MagUpSetup(heater_resistance = 100)
        time.sleep(3)

        #Always start at current output
        i_start = self.lsh.getManualHeaterOut() 
        
        # initial current
        IOut = self.lj.getAnalogInput(2)*2 		
        
        i_values = self.calc_mag_steps(i_start, i_end, mag_time, pausetime)
        
        if i_end > i_start:
            self.sig_ramp_up_start.emit()
        else:
            self.sig_ramp_down_start.emit()
        
        # start ADR magup
        for index,ivalue in enumerate(i_values):
            self.lsh.setManualHeaterOut(ivalue)
            self.sig_ramp_value.emit(ivalue)
            time.sleep(pausetime)
            
            # Check to make sure that 3 K plate is not too hot
            # This could imply the charcoal heat switch is stuck
            curr_3k_temp = self.cc.getTemperature('C')
            if curr_3k_temp > max_temp:
                # Send out warning emails
                print(f'3 K stage is too hot ({curr_3k_temp}) and in danger of quenching the magnet. Ramping down from {i_values} % to prevent a quench.')

                # Ramp down to stop quench
                self.ramp_magnet(0, (mag_time-index*pausetime), pausetime)
                self.cc.STOP()			
                self.closeCharcoalHeatSwitch()
            
                return

            # Check if Kepco is ramping. At 1% the Isense current
            # should be around 0.25 A. Making threshold 10 mA plus any offset!
            '''
            xxx need to add this back
            if ivalue > 1.0 and self.Isense < (IOut + 0.010):
                print('Kepco does not appear to be ramping correctly!')
                if self.rampFailureCheck is True:
                    print('Ramp condition has falied twice, aborting ramp!')
                    # Send out warning emails
                    self.email_warn.sendScriptErrorEmailKPAC('Ramp condition has falied twice, aborting ramp!')
                    # Take some action if wanted
                    self.closeCharcoalHeatSwitch()
                    #raise an error that will abort the program
                    raise RuntimeError('Kepco Ramp Error Occured')
                    self.rampFailureCheck = True
            '''
                        
        if i_end > i_start:
            self.sig_ramp_up_done.emit()
        else:
            self.sig_ramp_down_done.emit()

        # Hold for one hour after the magup is complete and plot temps
        # xxx beckerd - we might need a way to interrupt this ...
        self.sig_soak_start.emit()
        time.sleep(soak_time)
        self.sig_soak_done.emit()


    def ramp_magnet(self, i_end, mag_ramp_time, mag_step_time):
        ''' Ramp ADR magnet to a final value, with no checks for quenching or whether current is flowing. '''
        
        self.lsh.MagUpSetup(heater_resistance = 100)
        time.sleep(3)

        # Always start at current output
        i_start = self.lsh.getManualHeaterOut() 
        
        i_values = self.calc_mag_steps(i_start, i_end, mag_ramp_time, mag_step_time)
        print(("Ramp Values: ", i_values))

        if i_end > i_start:
            self.sig_ramp_up_start.emit()
        else:
            self.sig_ramp_down_start.emit()
        
        for index,ivalue in enumerate(i_values):
            self.lsh.setManualHeaterOut(ivalue)
            self.sig_ramp_value.emit(ivalue)
            time.sleep(mag_step_time)

        if i_end > i_start:
            self.sig_ramp_up_done.emit()
        else:
            self.sig_ramp_down_done.emit()
        
    def ramp_down(self, mag_precool_temp, mag_ramp_target, mag_ramp_time, mag_step_time, settle_time):

        # first wait for the correct pre-cool temperature
        self.sig_mag_down_temp_wait.emit()
        while (self.lsh.getTemperature(channel='4') > mag_precool_temp):
            time.sleep(10) # xxx this could certainly be 60 s
            
        # open ADR heat switch and wait 120 s for things to settle
        # down
        self.openADRHeatSwitch()
        self.sig_mag_down_open_wait.emit()
        time.sleep(settle_time)

        self.ramp_magnet(mag_ramp_target, mag_ramp_time, mag_step_time)

        
    sig_charcoal_heating_start = pyqtSignal()
    sig_charcoal_heating_done = pyqtSignal()
            

    def ramp_down_is_done(self):
        self.ramp_down_done = True
    
    def he3_test(self, hold_time, apply_heat, heat_start_temp, heat_end_temp, heater_voltage):
        print(('Running He3 test: %f %d %f %f %f ' % (hold_time, apply_heat, heat_start_temp, heat_end_temp, heater_voltage)))

        # Cycle fridge
        self.closeADRHeatSwitch()
        self.closePotHeatSwitch()
        self.openCharcoalHeatSwitch()
        self.enable_cryocon_control()
        self.sig_charcoal_heating_start.emit()

        # wait for hold_time seconds
        print(('waiting for', hold_time, 'seconds'))
        time.sleep(hold_time)

        self.openPotHeatSwitch()
        time.sleep(10)

        self.disable_cryocon_control()
        self.sig_charcoal_heating_done.emit()
        self.closeCharcoalHeatSwitch()

        if apply_heat and self.ps != None:
            # wait for temperature to fall below start temp
            while (self.cc.getTemperature('C') > heat_start_temp):
                print(('Still warm: %f K > %f K' % (self.cc.getTemperature('C'), heat_start_temp)))
                time.sleep(30)
    
            # turn on heat
            self.ps.setVoltage(heater_voltage)
            self.ps.outputOn()

            # wait for temperature to rise above end temp
            while (self.cc.getTemperature('D') < heat_end_temp):
                print(('Still cold: %f K < %f K' % (self.cc.getTemperature('D'), heat_end_temp)))
                time.sleep(30)

            # turn off heat
            print('turn off heat')
            self.ps.setVoltage(0.0)
            self.ps.outputOff()

    def full_cycle(self,
                   mag_up_target, mag_up_time, mag_up_step_time, mag_up_start_temp, mag_up_max_temp, mag_up_soak_time,
                   char_fast_turns, char_stage1_turns, char_stage1_temp, char_stage2_temp, char_full_close_temp,
                   char_step_time, mag_precool_temp, mag_ramp_target, mag_ramp_time, mag_step_time, settle_time, alert_emails):
        
        self.closeADRHeatSwitch()
        self.closePotHeatSwitch()
        self.openCharcoalHeatSwitch()
        self.write_cryocon_control(30, 55)
        self.enable_cryocon_control()
        self.sig_charcoal_heating_start.emit()
        
        self.magnet_ramp_careful(mag_up_target, mag_up_time, mag_up_step_time, mag_up_start_temp, mag_up_max_temp, mag_up_soak_time, alert_emails)

        self.openPotHeatSwitch()

        self.disable_cryocon_control()
        self.sig_charcoal_heating_done.emit()

        self.ramp_down_done = False


        task_ramp_down = Task(lambda: self.ramp_down(mag_precool_temp,
                                                     mag_ramp_target,
                                                     mag_ramp_time,
                                                     mag_step_time,
                                                     settle_time),
                              self.ramp_down_is_done)

        self.closeCharcoalHeatSwitch()
        ## here we're just closing the charcoal heatswitch instead of slow closing it
        ## will we need something more fancy?
        for i in range(10):
            print("TASK START!!!!")
        task_ramp_down.start()


    sig_mag_down_temp_wait = pyqtSignal()
    sig_mag_down_open_wait = pyqtSignal()
    
 
   
#!/usr/bin/env python

import sys, string
from kpac import KPAC
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import PyQt5.uic
import os
from task import Task

def as_bold(s):
    return f"<b>{s}</b>"

def as_bold_14pt(s):
    return f"""<span style="font-size:14pt">{as_bold(s)}</span>"""

def setup_heatswitch_widgets(pb_open, pb_close, label, kpac_open_func, kpac_close_func):
    '''Helper function to setup the widgets and behavior for the heat switches.'''
    
    def closed_func():
        pb_open.setEnabled(True)
        pb_close.setEnabled(True)
        label.setText(as_bold('Closed'))

    def opened_func():
        pb_open.setEnabled(True)
        pb_close.setEnabled(True)
        label.setText(as_bold('Open'))

    open_task = Task(kpac_open_func, opened_func)
    close_task = Task(kpac_close_func, closed_func)

    def open_func():
        pb_open.setEnabled(False)
        pb_close.setEnabled(False)
        label.setText(as_bold('Opening'))
        open_task.start()

    def close_func():
        pb_open.setEnabled(False)
        pb_close.setEnabled(False)
        label.setText(as_bold('Closing'))
        close_task.start()

    pb_open.clicked.connect(open_func)
    pb_close.clicked.connect(close_func)

class KPACGui(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)

        PyQt5.uic.loadUi(os.path.join(os.path.dirname(__file__),"kcg_ui.ui"), self)


        self.KPAC = KPAC()
        self.charcoal_position = 0

        self.dte_start_time.setDate(QDate.currentDate())
        self.dte_start_time.setTime(QTime(1,0))

        self.actionQuit.triggered.connect(qApp.quit)

        #
        # Setup plots
        #

        self.plot1.set_labels('Time (min)', 'Temperature (K)', 'Upper Stage')
        self.plot1.addToolbar()
        self.curve_upper = self.plot1.addLine(name='Upper Stage', xdata=self.KPAC.poll_times, ydata=self.KPAC.upper_temps)

        self.plot2.set_labels('Time (min)', 'Temperature (K)', 'Mid Temps')
        self.plot2.addToolbar()
        self.curve_charcoal = self.plot2.addLine(name='Charcoal', xdata=self.KPAC.poll_times, ydata=self.KPAC.charcoal_temps)
        self.curve_3k       = self.plot2.addLine(name='3K Plate', xdata=self.KPAC.poll_times, ydata=self.KPAC.plate_3k_temps)
        self.plot_al_ring   = self.plot2.addLine(name='Al Ring',  xdata=self.KPAC.poll_times, ydata=self.KPAC.ring_temps)
        if self.KPAC.diode1_temps != None:
            self.curve_diode1 = self.plot2.addLine(name='Diode 1', xdata=self.KPAC.poll_times, ydata=self.KPAC.diode1_temps)
            self.curve_diode2 = self.plot2.addLine(name='Diode 2', xdata=self.KPAC.poll_times, ydata=self.KPAC.diode2_temps)
            
        self.plot2.showLegend(fontsize=6)

        self.plot3.set_labels('Time (min)', 'Temperature (K)', 'Low Temps')
        self.plot3.addToolbar()
        self.curve_he3 = self.plot3.addLine(name='He3 Pot', xdata=self.KPAC.poll_times, ydata=self.KPAC.he3_temps)
        self.curve_adr = self.plot3.addLine(name='ADR',     xdata=self.KPAC.poll_times, ydata=self.KPAC.adr_temps)
        self.plot3.showLegend(fontsize=6)

        self.plot_he3.set_labels('Time (min)', 'Pressure (bar)', 'He3 Pressure')
        self.plot_he3.addToolbar()
        self.curve_he3_pressure = self.plot_he3.addLine(name='Pressure', xdata=self.KPAC.poll_times, ydata=self.KPAC.he3_pressure)

        self.plot_kepco_voltage.set_labels('Time (min)', 'Voltage (V)', 'Kepco Voltage')
        self.plot_kepco_voltage.addToolbar()
        self.curve_kepco_voltage = self.plot_kepco_voltage.addLine(name='Voltage', xdata=self.KPAC.poll_times, ydata=self.KPAC.kepco_voltage)

        self.plot_kepco_current.set_labels('Time (min)', 'Current (A)', 'Kepco Current')
        self.plot_kepco_current.addToolbar()
        self.curve_kepco_current = self.plot_kepco_current.addLine(name='Current', xdata=self.KPAC.poll_times, ydata=self.KPAC.kepco_current)

        #
        # Timer for plotting
        #
        self.timer = QTimer()
        self.timer.timeout.connect(self.my_update)
        self.update_polling_interval()

        self.le_polling_interval.editingFinished.connect(self.update_polling_interval)

        #
        # Timer for starting a cycle
        #
        self.timer_cycle = QTimer()
        self.timer_cycle.timeout.connect(self.start_full_cycle)
        self.auto_cycling = False
        
        #
        # Setup tasks for long-running work that should not freeze the GUI
        #
        self.task_mag_up = Task(self.call_magnet_ramp_careful, self.ramp_done)
        self.task_ramp_magnet = Task(self.call_ramp_magnet, self.ramp_done)
        self.task_ramp_down = Task(self.call_ramp_down, self.ramp_done)
        #self.task_cycle  = TaskWaking(5, self.call_full_cycle, self.cycle_done)
        self.task_cycle  = Task(self.call_full_cycle, self.cycle_done)
        self.task_he3_test  = Task(self.call_he3_test, self.he3_test_done)
        self.task_slow_charcoal_close = Task(self.call_slow_charcoal_close, self.slow_charcoal_close_done)

        #
        # Read status of certain components
        #
        ans = self.KPAC.read_cryocon_control()
        self.l_cryocon_control_status.setText(as_bold_14pt(ans[0]))
        self.le_upper_setpoint.setText(str(ans[2]))
        self.le_charcoal_setpoint.setText(str(ans[1]))

        self.l_ramp_current_level.setText(str(self.KPAC.get_ramp_current_level()))

        self.l_regulate_status.setText(str(self.KPAC.get_regulate_state()))

        #
        # Setup basic heatswitch control
        #
        setup_heatswitch_widgets(self.pb_open_adr_hs, self.pb_close_adr_hs, self.l_adr_hs_status,
                                      self.KPAC.openADRHeatSwitch, self.KPAC.closeADRHeatSwitch)
        setup_heatswitch_widgets(self.pb_open_pot_hs, self.pb_close_pot_hs, self.l_pot_hs_status,
                                      self.KPAC.openPotHeatSwitch, self.KPAC.closePotHeatSwitch)
        setup_heatswitch_widgets(self.pb_open_charcoal_hs, self.pb_close_charcoal_hs, self.l_charcoal_hs_status,
                                      self.KPAC.openCharcoalHeatSwitch, self.KPAC.closeCharcoalHeatSwitch)

        # Would be good to move these into setup_heatswitch_widgets,
        # but not quite ready for that 
        self.KPAC.sig_adr_hs_open_start.connect(lambda: self.l_adr_hs_status.setText(as_bold('Opening')))
        self.KPAC.sig_adr_hs_open_done.connect(lambda: self.l_adr_hs_status.setText(as_bold('Open')))
        self.KPAC.sig_adr_hs_close_start.connect(lambda: self.l_adr_hs_status.setText(as_bold('Closing')))
        self.KPAC.sig_adr_hs_close_done.connect(lambda: self.l_adr_hs_status.setText(as_bold('Close')))
        self.KPAC.sig_pot_hs_open_start.connect(lambda: self.l_pot_hs_status.setText(as_bold('Opening')))
        self.KPAC.sig_pot_hs_open_done.connect(lambda: self.l_pot_hs_status.setText(as_bold('Open')))
        self.KPAC.sig_pot_hs_close_start.connect(lambda: self.l_pot_hs_status.setText(as_bold('Closing')))
        self.KPAC.sig_pot_hs_close_done.connect(lambda: self.l_pot_hs_status.setText(as_bold('Close')))
        self.KPAC.sig_charcoal_hs_open_start.connect(lambda: self.l_charcoal_hs_status.setText(as_bold('Opening')))
        self.KPAC.sig_charcoal_hs_open_done.connect(lambda: self.l_charcoal_hs_status.setText(as_bold('Open')))
        self.KPAC.sig_charcoal_hs_close_start.connect(lambda: self.l_charcoal_hs_status.setText(as_bold('Closing')))
        self.KPAC.sig_charcoal_hs_close_done.connect(lambda: self.l_charcoal_hs_status.setText(as_bold('Close')))

        #
        # Setup detailed charcoal heatswitch control
        # 
        self.pb_charcoal_hs_move_relative.clicked.connect(self.move_charcoal_relative)
        self.pb_charcoal_position_reset.clicked.connect(self.reset_charcoal_position)
        self.KPAC.sig_charcoal_hs_open_done.connect(self.reset_charcoal_position)
        self.KPAC.sig_charcoal_hs_close_done.connect(self.reset_charcoal_position)
        
        #
        # Setup other signals
        #
        self.pb_write_cryocon_control.clicked.connect(self.write_cryocon_control)
        self.pb_enable_cryocon_control.clicked.connect(self.enable_cryocon_control)
        self.pb_disable_cryocon_control.clicked.connect(self.disable_cryocon_control)

        self.pb_relay_ramp.clicked.connect(self.set_relay_ramp)
        self.pb_relay_control.clicked.connect(self.set_relay_control)

        self.pb_ramp_careful.clicked.connect(self.ramp_careful)
        self.KPAC.sig_ramp_wait_start.connect( lambda: self.l_ramp_status.setText(as_bold_14pt('Waiting for start temp')))
        self.KPAC.sig_ramp_up_start.connect(lambda: self.l_ramp_status.setText(as_bold_14pt('Ramping')))
        self.KPAC.sig_soak_start.connect( lambda: self.l_ramp_status.setText(as_bold_14pt('Soaking')))
        self.KPAC.sig_ramp_value.connect(lambda x: self.l_ramp_current_level.setText(str(x)))
        self.KPAC.sig_soak_done.connect( lambda: self.l_ramp_status.setText(as_bold_14pt('')))

        self.pb_ramp_down.clicked.connect(self.ramp_down)

        self.pb_charcoal_close_start.clicked.connect(self.slow_charcoal_close)

        self.pb_ramp.clicked.connect(self.ramp_magnet)

        self.pb_start_cycle_now.clicked.connect(self.start_full_cycle)
        #self.pb_stop_cycle.clicked.connect(self.task_cycle.stop)
        self.KPAC.sig_pot_hs_close_done.connect(lambda: self.l_cycle_status_close_pot_hs.setText('Done'))
        self.KPAC.sig_adr_hs_close_done.connect(lambda: self.l_cycle_status_close_adr_hs.setText('Done'))
        self.KPAC.sig_charcoal_hs_open_done.connect(lambda: self.l_cycle_status_open_charcoal_hs.setText('Done'))
        self.KPAC.sig_charcoal_heating_start.connect(lambda: self.l_cycle_status_heat_charcoal.setText('In Progress'))
        self.KPAC.sig_charcoal_above_setpoint.connect(lambda: self.l_cycle_status_heat_charcoal.setText('Done'))
        self.KPAC.sig_ramp_up_start.connect(lambda: self.l_cycle_status_ramp_up.setText('In Progress'))
        self.KPAC.sig_ramp_up_done.connect(lambda: self.l_cycle_status_ramp_up.setText('Done'))
        self.KPAC.sig_soak_start.connect(lambda: self.l_cycle_status_soak.setText('In Progress'))
        self.KPAC.sig_soak_done.connect(lambda: self.l_cycle_status_soak.setText('Done'))

        self.KPAC.sig_ramp_wait_start.connect(lambda: self.l_cycle_status_wait_cool_plate.setText('In Progress'))
        self.KPAC.sig_ramp_wait_done.connect(lambda: self.l_cycle_status_wait_cool_plate.setText('Done'))

        self.KPAC.sig_ramp_up_start.connect(lambda: self.l_cycle_status_ramp_up.setText('In Progress'))
        self.KPAC.sig_ramp_up_done.connect(lambda: self.l_cycle_status_ramp_up.setText('Done'))

        self.KPAC.sig_pot_hs_open_start.connect(lambda: self.l_cycle_status_open_pot_hs.setText('In Progress'))
        self.KPAC.sig_pot_hs_open_done.connect(lambda: self.l_cycle_status_open_pot_hs.setText('Done'))

        self.KPAC.sig_charcoal_heating_done.connect(lambda: self.l_cycle_status_turn_off_charcoal.setText('Done'))

        self.KPAC.sig_charcoal_careful_close_ok.connect(lambda: self.l_cycle_status_close_charcoal.setText('In Progress'))
        self.KPAC.sig_charcoal_careful_close_wait.connect(lambda: self.l_cycle_status_close_charcoal.setText('Waiting to cool'))
        self.KPAC.sig_charcoal_careful_close_done.connect(lambda: self.l_cycle_status_close_charcoal.setText('Done'))

        self.KPAC.sig_adr_hs_open_start.connect(lambda: self.l_cycle_status_open_adr_hs.setText('In Progress'))
        self.KPAC.sig_adr_hs_open_done.connect(lambda: self.l_cycle_status_open_adr_hs.setText('Done'))
        
        self.KPAC.sig_mag_down_temp_wait.connect(lambda: self.l_cycle_status_ramp_down_magnet.setText('Waiting for pre-cool'))
        self.KPAC.sig_mag_down_open_wait.connect(lambda: self.l_cycle_status_ramp_down_magnet.setText('Settle after HS open'))
        self.KPAC.sig_ramp_down_start.connect(lambda: self.l_cycle_status_ramp_down_magnet.setText('In Progress'))
        self.KPAC.sig_ramp_down_done.connect(lambda: self.l_cycle_status_ramp_down_magnet.setText('Done'))

        self.pb_regulate_start.clicked.connect(lambda: self.KPAC.start_adr_regulate(float(self.le_regulate_temp.text())/1000))
        self.pb_regulate_stop.clicked.connect(lambda: self.KPAC.stop_adr_regulate())

        self.pb_start_cycle_later.clicked.connect(self.start_full_cycle_later)

        self.pb_he3_test_start.clicked.connect(self.he3_test)

    def alert_emails(self):
        return ''
	#emails_string = self.le_email_alerts.text()
        #return string.split(emails_string, ',')
    #
    # Charcoal Heat Switch
    #
    def move_charcoal_relative(self):
        num_turns = float(self.le_charcoal_num_turns.text())
        self.KPAC.move_charcoal_relative(num_turns)
        self.charcoal_position += num_turns
        self.l_charcoal_position.setText('Position: %f' % self.charcoal_position)
        
    def reset_charcoal_position(self):
        self.charcoal_position = 0
        self.l_charcoal_position.setText('Position: %s' % self.charcoal_position)

    def slow_charcoal_close(self):
        self.gb_charcoal_close.setEnabled(False)
        self.l_slow_charcoal_close_status.setText('In Progress')
        self.task_slow_charcoal_close.start()

    def call_slow_charcoal_close(self):
        self.KPAC.slow_charcoal_close(float(self.le_charcoal_close_fast_turns.text()),
                                      float(self.le_charcoal_close_stage1_turns.text()),
                                      float(self.le_charcoal_close_stage1_temp.text()),
                                      float(self.le_charcoal_close_stage2_temp.text()),
                                      float(self.le_charcoal_close_full_close_temp.text()),
                                      float(self.le_charcoal_close_step_time.text()))
        
    def slow_charcoal_close_done(self):
        self.gb_charcoal_close.setEnabled(True)
        self.l_slow_charcoal_close_status.setText('Done')
    
    # 
    # Ramping
    #
    def ramp_careful(self):
        self.pb_ramp_careful.setEnabled(False)
        self.pb_ramp.setEnabled(False)
        self.gb_ramp.setEnabled(False)
        self.l_ramp_status.setText(as_bold_14pt('Ramp in progress'))
        self.task_mag_up.start()

    def call_magnet_ramp_careful(self):
        self.KPAC.magnet_ramp_careful(float(self.le_ramp_target_level.text()),
                                      float(self.le_ramp_time.text())*60,
                                      float(self.le_ramp_step_time.text()),
                                      float(self.le_ramp_start_temp.text()),
                                      float(self.le_ramp_max_temp.text()),
                                      float(self.le_ramp_soak_time.text())*60,
                                      self.alert_emails())
    
    def ramp_magnet(self):
        self.pb_ramp_careful.setEnabled(False)
        self.pb_ramp.setEnabled(False)
        self.gb_ramp.setEnabled(False)
        self.task_ramp_magnet.start()

    def call_ramp_magnet(self):
        self.KPAC.ramp_magnet(float(self.le_ramp_target_level.text()),
                              float(self.le_ramp_time.text())*60,
                              float(self.le_ramp_step_time.text()))
    def ramp_down(self):
        self.pb_ramp_careful.setEnabled(False)
        self.pb_ramp.setEnabled(False)
        self.gb_ramp.setEnabled(False)
        self.gb_ramp_down.setEnabled(False)
        self.task_ramp_down.start()

    def call_ramp_down(self):
        self.KPAC.ramp_down(float(self.le_mag_ramp_down_precool_temp.text()),
                            float(self.le_mag_ramp_down_target_level.text()),
                            float(self.le_mag_ramp_down_ramp_time.text())*60,
                            float(self.le_mag_ramp_down_step_time.text()),
                            float(self.le_mag_ramp_down_settle_time.text()))

    def ramp_done(self):
        self.pb_ramp_careful.setEnabled(True)
        self.pb_ramp.setEnabled(True)
        self.gb_ramp.setEnabled(True)
        self.gb_ramp_down.setEnabled(True)
        self.l_ramp_status.setText(as_bold_14pt('Not ramping'))

    #
    # Run a test on He3 system
    #
    def he3_test(self):
        self.gb_he3_test.setEnabled(False)
        self.gb_charcoal_close.setEnabled(False)
        self.task_he3_test.start()

    def call_he3_test(self):
        self.KPAC.he3_test(float(self.le_he3_test_hold_time.text())*60,
                           self.cb_he3_test_apply_heat.isChecked(),
                           float(self.le_he3_test_heat_start_temp.text()),
                           float(self.le_he3_test_heat_stop_temp.text()),
                           float(self.le_he3_test_voltage.text()))
    
    def he3_test_done(self):
        self.gb_he3_test.setEnabled(True)
        self.gb_charcoal_close.setEnabled(True)

    # 
    # Running a Cycle
    # 
    
    def start_full_cycle_later(self):
        start_dt = self.dte_start_time.dateTime()
        interval = QDateTime.currentDateTime().msecsTo(start_dt)
        if interval < 0:
            mbox = QMessageBox()
            mbox.setText('Cycle start time must be in the future')
            mbox.exec_()
        else:
            self.pb_start_cycle_later.setEnabled(False)
            self.pb_start_cycle_later.setEnabled(False)
            self.dte_start_time.setEnabled(False)
            
            self.auto_cycling = True
            
            start_dt = self.dte_start_time.dateTime()
            interval = QDateTime.currentDateTime().msecsTo(start_dt)
            self.timer_cycle.setSingleShot(True)
            self.timer_cycle.start(interval)
    
    def start_full_cycle(self):
        self.l_cycle_status_close_pot_hs.setText('')
        self.l_cycle_status_close_adr_hs.setText('')
        self.l_cycle_status_open_charcoal_hs.setText('')
        self.l_cycle_status_heat_charcoal.setText('')
        self.l_cycle_status_ramp_up.setText('')
        self.l_cycle_status_soak.setText('')
        self.l_cycle_status_wait_cool_plate.setText('')
        self.l_cycle_status_open_pot_hs.setText('')
        self.l_cycle_status_turn_off_charcoal.setText('')
        self.l_cycle_status_close_charcoal.setText('')
        self.l_cycle_status_open_adr_hs.setText('')
        self.l_cycle_status_ramp_down_magnet.setText('')
        self.pb_start_cycle_now.setEnabled(False)
        self.pb_start_cycle_later.setEnabled(False)
        self.task_cycle.start()

    def cycle_done(self):
        print('cycle_done')
        #if self.auto_cycling and self.cb_is_daily_cycle.isChecked():
        #    self.timer_cycle.setSingleShot(True)
        #else:
        #    #self.timer_cycle.start(23*60*60*1000)
        #    self.timer_cycle.start(5*60*1000)
        #    self.pb_start_cycle_now.setEnabled(True)
        #    self.pb_start_cycle_later.setEnabled(True)
        #    self.dte_start_time.setEnabled(True)
        #    self.auto_cycling = False

    def call_full_cycle(self): #, state):
        print('call_full_cycle')
        self.KPAC.full_cycle(float(self.le_ramp_target_level.text()),
                             float(self.le_ramp_time.text())*60,
                             float(self.le_ramp_step_time.text()),
                             float(self.le_ramp_start_temp.text()),
                             float(self.le_ramp_max_temp.text()),
                             float(self.le_ramp_soak_time.text())*60,
                             float(self.le_charcoal_close_fast_turns.text()),
                             float(self.le_charcoal_close_stage1_turns.text()),
                             float(self.le_charcoal_close_stage1_temp.text()),
                             float(self.le_charcoal_close_stage2_temp.text()),
                             float(self.le_charcoal_close_full_close_temp.text()),
                             float(self.le_charcoal_close_step_time.text()),
                             float(self.le_mag_ramp_down_precool_temp.text()),
                             float(self.le_mag_ramp_down_target_level.text()),
                             float(self.le_mag_ramp_down_ramp_time.text())*60,
                             float(self.le_mag_ramp_down_step_time.text()),
                             float(self.le_mag_ramp_down_settle_time.text()),
                             self.alert_emails())
    
    #
    # Cryocon Stuff
    #

    def write_cryocon_control(self):
        self.KPAC.write_cryocon_control(float(self.le_upper_setpoint.text()),
                                        float(self.le_charcoal_setpoint.text()))

    def disable_cryocon_control(self):
        self.KPAC.disable_cryocon_control()
        self.l_cryocon_control_status.setText(as_bold('OFF'))
        
    def enable_cryocon_control(self):
        self.KPAC.enable_cryocon_control()
        self.l_cryocon_control_status.setText(as_bold('ON'))

    #
    # Relay Stuff
    #

    def set_relay_ramp(self):
        self.KPAC.set_relay(KPAC.RELAY_RAMP)
        self.l_relay_status.setText(as_bold('RAMP'))

    def set_relay_control(self):
        self.KPAC.set_relay(KPAC.RELAY_CONTROL)
        self.l_relay_status.setText(as_bold('CONTROL'))
    
    #
    # Polling stuff
    #

    def update_polling_interval(self):
        interval = int(float(self.le_polling_interval.text())*1000)
        self.timer.start(interval)

    def my_update(self):
        self.KPAC.update_temps(self.le_data_directory.text())
        self.curve_upper.update_data(self.KPAC.poll_times, self.KPAC.upper_temps)
        self.curve_charcoal.update_data(self.KPAC.poll_times, self.KPAC.charcoal_temps)
        if self.KPAC.diode1_temps != None:
            self.curve_diode1.update_data(self.KPAC.poll_times, self.KPAC.diode1_temps)
            self.curve_diode2.update_data(self.KPAC.poll_times, self.KPAC.diode2_temps)
        self.curve_3k.update_data(self.KPAC.poll_times, self.KPAC.plate_3k_temps)
        self.curve_he3.update_data(self.KPAC.poll_times, self.KPAC.he3_temps)
        self.plot_al_ring.update_data(self.KPAC.poll_times, self.KPAC.ring_temps)
        self.curve_adr.update_data(self.KPAC.poll_times, self.KPAC.adr_temps)
        self.curve_he3_pressure.update_data(self.KPAC.poll_times, self.KPAC.he3_pressure)
        self.curve_kepco_voltage.update_data(self.KPAC.poll_times, self.KPAC.kepco_voltage)
        self.curve_kepco_current.update_data(self.KPAC.poll_times, self.KPAC.kepco_current)

        self.plot1.update()
        self.plot2.update()
        self.plot3.update()
        self.plot_he3.update()
        self.plot_kepco_voltage.update()
        self.plot_kepco_current.update()

styleSheet = '''
QGroupBox {
    border: 2px solid black;
    border-radius: 9px;
    margin-top: 0.5em;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 3px 0 3px;
}
'''
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(styleSheet)
    window = KPACGui()
    window.show()
    sys.exit(app.exec_())


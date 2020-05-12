import sys
import optparse
import struct

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import named_serial
from .badchn_builder import badChn
from cringe.shared import terminal_colors as tc


class badrap(QWidget):

# 	def __init__(self, parent=None, **kwargs):
# 		print kwargs
    def __init__(self, parent=None, addr=None, slot=None, seqln=None, lsync=32):

        super(badrap, self).__init__()

        self.serialport = named_serial.Serial(port='rack', shared = True)

        self.chns = 16

# 		if parent == None:
        self.parent = parent
        self.address = addr
        self.slot = slot
        self.seqln = seqln
        self.lsync = lsync
# 		else:
# 			self.address = parent.addr
# 			self.slot = parent.slot
# 			self.seqln = parent.seqln
# 			self.lsync = parent.lsync

        self.frame = self.lsync * self.seqln

        self.delay = 0
        self.led = 0
        self.status = 0
        self.init = 0

        self.dwell_val = 0
        self.dwellDACunits = float(0)
        self.range_val = 1
        self.rangeDACunits = float(2)
        self.step_val = 1
        self.stepDACunits = float(1)
        self.tri_idx = 0

        self.mode = 1


        self.chn_vectors = []
        self.allChannels = {}
# 		self.enb = [0,0,0,0,0,0,0]
# 		self.cal_coeffs = [0,0,0,0,0,0,0]
# 		self.appTrim =[0,0,0,0,0,0,0]

        self.setWindowTitle("BADRAP")	# Phase Offset Widget
        self.setGeometry(30,30,1200,800)
        self.setContentsMargins(0,0,0,0)

        self.layout_widget = QWidget(self)
        self.layout = QVBoxLayout(self)

        '''
        build widget for GLOBALS header
        '''
        if parent == None:
            self.class_interface_widget = QGroupBox(self)
            self.class_interface_widget.setFixedWidth(1035)
            self.class_interface_widget.setFocusPolicy(QtCore.Qt.NoFocus)
            self.class_interface_widget.setTitle("CLASS INTERFACE CONTROLS")

            self.controls_layout = QGridLayout(self.class_interface_widget)
            self.controls_layout.setContentsMargins(5,5,5,5)
            self.controls_layout.setSpacing(5)

    # 		self.controls_widget = QWidget(self.layout_widget)
    # 		self.globals_layout = QGridLayout(self.globals_widget)


            self.addr_indicator = QLineEdit()
            self.addr_indicator.setReadOnly(True)
            self.addr_indicator.setText(str(addr))
            self.addr_indicator.setAlignment(QtCore.Qt.AlignRight)
            self.addr_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
            self.controls_layout.addWidget(self.addr_indicator,0,0,QtCore.Qt.AlignRight)

            self.addr_label = QLabel("card address")
            self.controls_layout.addWidget(self.addr_label,0,1,QtCore.Qt.AlignLeft)

            self.seqln_indicator = QLineEdit()
            self.seqln_indicator.setReadOnly(True)
            self.seqln_indicator.setText('%3d'%seqln)
            self.seqln_indicator.setAlignment(QtCore.Qt.AlignRight)
            self.seqln_indicator.setFixedSize(self.seqln_indicator.sizeHint())
            self.seqln_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
            self.controls_layout.addWidget(self.seqln_indicator,0,2,1,1)

            self.seqln_lbl = QLabel("sequence length")
    # 		self.seqln_lbl.setAlignment(QtCore.Qt.AlignLeft)
            self.controls_layout.addWidget(self.seqln_lbl,0,3,1,1,QtCore.Qt.AlignLeft)

            self.loadsetup = QPushButton(self, text = "load setup")
            self.controls_layout.addWidget(self.loadsetup,0,6,QtCore.Qt.AlignTop)

            self.savesetup = QPushButton(self, text = "save setup")
            self.controls_layout.addWidget(self.savesetup,0,7,QtCore.Qt.AlignTop)

            self.sendALLchns = QPushButton(self, text = "send setup")
            self.controls_layout.addWidget(self.sendALLchns,0,8,QtCore.Qt.AlignTop)

            self.slot_indicator = QLineEdit()
            self.slot_indicator.setReadOnly(True)
    # 		self.addr_indicator.setFixedWidth(40)
            self.slot_indicator.setText('%2d'%slot)
            self.slot_indicator.setAlignment(QtCore.Qt.AlignRight)
            self.slot_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
            self.controls_layout.addWidget(self.slot_indicator,1,0,QtCore.Qt.AlignRight)

            self.slot_label = QLabel("card slot")
            self.controls_layout.addWidget(self.slot_label,1,1,QtCore.Qt.AlignLeft)

            self.lsync_indicator = QLineEdit()
            self.lsync_indicator.setReadOnly(True)
            self.lsync_indicator.setText('%4d'%lsync)
            self.lsync_indicator.setAlignment(QtCore.Qt.AlignRight)
            self.lsync_indicator.setFixedSize(self.seqln_indicator.sizeHint())
            self.lsync_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
            self.controls_layout.addWidget(self.lsync_indicator,1,2,1,1)

            self.seqln_lbl = QLabel("line period")
    # 		self.seqln_lbl.setAlignment(QtCore.Qt.AlignLeft)
            self.controls_layout.addWidget(self.seqln_lbl,1,3,1,1,QtCore.Qt.AlignLeft)

            self.filename_label = QLabel("file")
            self.controls_layout.addWidget(self.filename_label,1,5,QtCore.Qt.AlignRight)

            self.filenameEdit = QLineEdit()
            self.filenameEdit.setReadOnly(True)
            self.controls_layout.addWidget(self.filenameEdit,1,6,1,3)

            self.layout.addWidget(self.class_interface_widget)

            '''
            build widget for GLOBAL VARIABLE CONTROL
            '''
            self.glb_var_widget = QGroupBox(self)
            self.glb_var_widget.setTitle("CARD GLOBAL VARIABLES")
            self.glb_var_layout = QGridLayout(self.glb_var_widget)
            self.glb_var_layout.setContentsMargins(5,5,10,5)
            self.glb_var_layout.setSpacing(5)

            self.card_delay = QLineEdit()
            self.card_delay.setFocusPolicy(QtCore.Qt.NoFocus)
            self.card_delay.setText('%2d'%self.delay)
            self.card_delay.setAlignment(QtCore.Qt.AlignRight)
            self.glb_var_layout.addWidget(self.card_delay,0,0,1,1,QtCore.Qt.AlignRight)

            self.card_delay_lbl = QLabel("card delay")
            self.glb_var_layout.addWidget(self.card_delay_lbl,0,1,1,1,QtCore.Qt.AlignLeft)

            self.LED_button = QToolButton(self, text = 'ON')
            self.LED_button.setFixedHeight(25)
            self.LED_button.setCheckable(1)
            self.LED_button.setChecked(self.led)
            self.LED_button.setStyleSheet("background-color: #" + tc.green + ";")
            self.glb_var_layout.addWidget(self.LED_button,0,2,1,1)
            self.LED_button.toggled.connect(self.LED_changed)

            self.led_lbl = QLabel("LED control")
            self.glb_var_layout.addWidget(self.led_lbl,0,3,1,1,QtCore.Qt.AlignLeft)

            self.status_button = QToolButton(self, text = 'ST')
            self.status_button.setFixedHeight(25)
            self.status_button.setCheckable(1)
            self.status_button.setChecked(self.status)
            self.status_button.setStyleSheet("background-color: #" + tc.red + ";")
            self.glb_var_layout.addWidget(self.status_button,0,4,1,1)
            self.status_button.toggled.connect(self.status_changed)

            self.status_lbl = QLabel("status bit")
            self.glb_var_layout.addWidget(self.status_lbl,0,5,1,3,QtCore.Qt.AlignLeft)

            self.glb_send = QPushButton(self, text = "send globals")
            self.glb_send.setFixedHeight(25)
    # 		self.glb_send.setFixedWidth(160)
            self.glb_var_layout.addWidget(self.glb_send,0,10,1,2,QtCore.Qt.AlignRight)
            self.glb_send.clicked.connect(self.send_globals)

            self.layout.addWidget(self.glb_var_widget)


            '''
            build widget for Triangle Waveform Generator
            '''
            self.tri_wvfm_widget = QGroupBox(self)
    # 		self.tri_wvfm_widget.setFixedHeight(25)
            self.tri_wvfm_widget.setTitle("TRIANGLE WAVEFORM GENERATOR")
            self.tri_wvfm_widget.setCheckable(1)
            self.tri_wvfm_widget.toggled.connect(self.tri_widget_toggle)

            self.tri_wvfm_layout = QGridLayout(self.tri_wvfm_widget)
            self.tri_wvfm_layout.setContentsMargins(5,5,5,5)
            self.tri_wvfm_layout.setSpacing(5)

            self.dwell = QSpinBox()
            self.dwell.setRange(0, 15)
            self.dwell.setFixedHeight(25)
            self.dwell.setSingleStep(1)
            self.dwell.setKeyboardTracking(0)
            self.dwell.setFocusPolicy(QtCore.Qt.StrongFocus)
            self.tri_wvfm_layout.addWidget(self.dwell,0,0,1,1,QtCore.Qt.AlignRight)
            self.dwell.valueChanged.connect(self.dwell_changed)

            self.dwell_lbl = QLabel("dwell (2^N)")
            self.tri_wvfm_layout.addWidget(self.dwell_lbl,0,1,1,1,QtCore.Qt.AlignLeft)

            self.dwell_indicator = QLineEdit()
            self.dwell_indicator.setReadOnly(True)
            self.dwell_indicator.setFixedHeight(25)
            self.dwell_indicator.setText('%5i'%2**(self.dwell_val))
            self.dwell_indicator.setAlignment(QtCore.Qt.AlignRight)
            self.dwell_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
            self.tri_wvfm_layout.addWidget(self.dwell_indicator, 1,0,1,1,QtCore.Qt.AlignRight)

            self.range_indicator_lbl = QLabel("dwell")
            self.tri_wvfm_layout.addWidget(self.range_indicator_lbl,1,1,1,1,QtCore.Qt.AlignLeft)

            self.range = QSpinBox()
            self.range.setRange(1, 14)
            self.range.setFixedHeight(25)
            self.range.setSingleStep(1)
            self.range.setKeyboardTracking(0)
            self.range.setFocusPolicy(QtCore.Qt.StrongFocus)
            self.tri_wvfm_layout.addWidget(self.range,0,2,1,1,QtCore.Qt.AlignRight)
            self.range.valueChanged.connect(self.range_changed)

            self.range_lbl = QLabel("steps (2^N)")
            self.tri_wvfm_layout.addWidget(self.range_lbl,0,3,1,1,QtCore.Qt.AlignLeft)

            self.range_indicator = QLineEdit()
            self.range_indicator.setReadOnly(True)
    # 		self.range_indicator.setFixedWidth(60)
            self.range_indicator.setText(str(2**(self.range_val)))
            self.range_indicator.setAlignment(QtCore.Qt.AlignRight)
            self.range_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
            self.tri_wvfm_layout.addWidget(self.range_indicator,1,2,1,1,QtCore.Qt.AlignRight)

            self.range_indicator_lbl = QLabel("steps")
            self.tri_wvfm_layout.addWidget(self.range_indicator_lbl,1,3,1,1,QtCore.Qt.AlignLeft)

            self.step = QSpinBox()
            self.step.setRange(1, 16383)
            self.step.setFixedHeight(25)
            self.step.setSingleStep(1)
            self.step.setKeyboardTracking(0)
            self.step.setFocusPolicy(QtCore.Qt.StrongFocus)
            self.tri_wvfm_layout.addWidget(self.step,0,4,1,1,QtCore.Qt.AlignRight)
            self.step.valueChanged.connect(self.step_changed)

            self.step_lbl = QLabel("step size")
            self.tri_wvfm_layout.addWidget(self.step_lbl,0,5,1,1,QtCore.Qt.AlignLeft)

            self.period_indicator = QLineEdit()
            self.period_indicator.setReadOnly(True)
    # 		self.period_indicator.setFixedWidth(120)
            self.period_indicator.setText(str(2*(2**self.dwell_val)*(2**self.range_val)))
            self.period_indicator.setAlignment(QtCore.Qt.AlignRight)
            self.period_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
            self.tri_wvfm_layout.addWidget(self.period_indicator,0,6,1,1,QtCore.Qt.AlignRight)

            self.period_indicator_lbl = QLabel("period")
            self.tri_wvfm_layout.addWidget(self.period_indicator_lbl,0,7,1,1,QtCore.Qt.AlignLeft)

            self.period_eng_indicator = QLineEdit()
            self.period_eng_indicator.setReadOnly(True)
    # 		self.period_eng_indicator.setFixedWidth(120)
    # 		self.period_eng_indicator.setText(str(2*(2**self.dwell_val)*(2**self.range_val)))
            self.period_eng_indicator.setText(str(int(self.period_indicator.text())*self.lsync*0.008))
            self.period_eng_indicator.setAlignment(QtCore.Qt.AlignRight)
            self.period_eng_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
            self.tri_wvfm_layout.addWidget(self.period_eng_indicator,1,6,1,1,QtCore.Qt.AlignRight)

            self.period_eng_indicator_lbl = QLabel("period [us]")
            self.tri_wvfm_layout.addWidget(self.period_eng_indicator_lbl,1,7,1,1,QtCore.Qt.AlignLeft)

            self.amp_indicator = QLineEdit()
            self.amp_indicator.setReadOnly(True)
    # 		self.amp_indicator.setFixedWidth(80)
            self.amp_indicator.setText(str((2**self.range_val)*self.step_val))
            self.amp_indicator.setAlignment(QtCore.Qt.AlignRight)
            self.amp_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
            self.tri_wvfm_layout.addWidget(self.amp_indicator,0,8,1,1,QtCore.Qt.AlignRight)

            self.amp_indicator_lbl = QLabel("amplitude")
            self.tri_wvfm_layout.addWidget(self.amp_indicator_lbl,0,9,1,1,QtCore.Qt.AlignLeft)

            self.amp_eng_indicator = QLineEdit()
            self.amp_eng_indicator.setReadOnly(True)
    # 		self.amp_eng_indicator.setFixedWidth(80)
            self.amp_eng_indicator.setText(str(int(self.amp_indicator.text())/16.383)[:6])
            self.amp_eng_indicator.setAlignment(QtCore.Qt.AlignRight)
            self.amp_eng_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
            self.tri_wvfm_layout.addWidget(self.amp_eng_indicator,1,8,1,1,QtCore.Qt.AlignRight)

            self.amp_eng_indicator_lbl = QLabel("amplitude [mV]")
            self.tri_wvfm_layout.addWidget(self.amp_eng_indicator_lbl,1,9,1,1,QtCore.Qt.AlignLeft)

            self.tri_idx_button = QToolButton(self, text = 'LSYNC')
            self.tri_idx_button.setFixedHeight(25)
            self.tri_idx_button.setCheckable(1)
            self.tri_idx_button.setChecked(self.tri_idx)
            self.tri_idx_button.setStyleSheet("background-color: #" + tc.red + ";")
            self.tri_wvfm_layout.addWidget(self.tri_idx_button,1,4,1,1,QtCore.Qt.AlignLeft)
            self.tri_idx_button.toggled.connect(self.tri_idx_changed)

            self.tri_idx_lbl = QLabel("timebase")
            self.tri_wvfm_layout.addWidget(self.tri_idx_lbl,1,5,1,1,QtCore.Qt.AlignLeft)

            self.tri_send = QPushButton(self, text = "send triangle")
            self.tri_send.setFixedHeight(25)
    # 		self.tri_send.setFixedWidth(160)
            self.tri_wvfm_layout.addWidget(self.tri_send,0,10,1,2, QtCore.Qt.AlignVCenter)
            self.tri_send.clicked.connect(self.send_wreg1)

            self.freq_eng_indicator = QLineEdit()
            self.freq_eng_indicator.setReadOnly(True)
    # 		self.freq_eng_indicator.setFixedWidth(80)
            self.freq_eng_indicator.setText(str(1000/float(self.period_eng_indicator.text()))[:6])
            self.freq_eng_indicator.setAlignment(QtCore.Qt.AlignRight)
            self.freq_eng_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
            self.tri_wvfm_layout.addWidget(self.freq_eng_indicator,1,10,1,1,QtCore.Qt.AlignRight)

            self.amp_eng_indicator_lbl = QLabel("freq [kHz]")
            self.tri_wvfm_layout.addWidget(self.amp_eng_indicator_lbl,1,11,1,1,QtCore.Qt.AlignLeft)

            self.layout.addWidget(self.tri_wvfm_widget)

        '''
        build widget for MASTER CONTROL VECTOR: these controls effect all channels on a card
        '''
        self.master_ctrl_widget = QGroupBox(self)
        self.master_ctrl_widget.setTitle("MASTER CONTROL VECTOR")
        self.master_ctrl_layout = QGridLayout(self.master_ctrl_widget)

        self.master_vector = badChn(self, self.master_ctrl_layout, chn=-1, cardaddr=self.address, serialport=self.serialport, master = 'master')
        self.master_vector.counter_label.setText("all")
        self.master_vector.chn_send.setText("send ALL channels")

        self.scale_factor = self.master_vector.width()

        self.layout.addWidget(self.master_ctrl_widget)

        '''
        build widget for arrayed channel parameters
        '''
        self.arrayframe = QWidget(self.layout_widget)
        self.array_layout = QVBoxLayout(self.arrayframe)
        self.array_layout.setSpacing(5)
        self.array_layout.setContentsMargins(10,10,10,10)

        for idx in range(self.chns):
            self.chn_vectors.append(badChn(self, self.array_layout, chn=idx, cardaddr=self.address, serialport=self.serialport))

        self.scrollarea = QScrollArea(self.layout_widget)
        self.scrollarea.setWidget(self.arrayframe)
        self.layout.addWidget(self.scrollarea)
# 		self.show()
# 		print self.arrayframe.width()
        self.master_ctrl_widget.setFixedWidth(self.arrayframe.width()+0)
        if parent == None:
            self.glb_var_widget.setFixedWidth(self.arrayframe.width()+0)
            self.tri_wvfm_widget.setFixedWidth(self.arrayframe.width()+0)

    '''
    child called methods
    '''
    def tri_widget_toggle(self):
        self.tri_widget_state = self.tri_wvfm_widget.isChecked()
        self.tri_wvfm_widget.setHidden(not(self.tri_widget_state))

    def dc_changed(self, state):
        for idx in range(self.chns):
            self.chn_vectors[idx].dc_button.setChecked(state)
        if state == 1:
            self.master_vector.dc_button.setStyleSheet("background-color: #" + tc.green + ";")
        else:
            self.master_vector.dc_button.setStyleSheet("background-color: #" + tc.red + ";")

    def LoHi_changed(self, state):
        for idx in range(self.chns):
            self.chn_vectors[idx].LoHi_button.setChecked(state)
        if state == 1:
            self.master_vector.LoHi_button.setStyleSheet("background-color: #" + tc.green + ";")
            self.master_vector.LoHi_button.setText('HI')
        else:
            self.master_vector.LoHi_button.setStyleSheet("background-color: #" + tc.red + ";")
            self.master_vector.LoHi_button.setText('LO')

    def tri_changed(self, state):
        for idx in range(self.chns):
            self.chn_vectors[idx].Tri_button.setChecked(state)
        if state == 1:
            self.master_vector.Tri_button.setStyleSheet("background-color: #" + tc.green + ";")
        else:
            self.master_vector.Tri_button.setStyleSheet("background-color: #" + tc.red + ";")

    def d2a_lo_spin_changed(self, level):
        for idx in range(self.chns):
            self.chn_vectors[idx].d2a_lo_slider.setValue(level)
        self.master_vector.d2a_lo_slider.setValue(level)

    def d2a_lo_slider_changed(self, level):
        for idx in range(self.chns):
            self.chn_vectors[idx].d2a_lo_slider.setValue(level)
        self.master_vector.d2a_lo_spin.setValue(level)

    def d2a_lo_setMin(self):
        for idx in range(self.chns):
            self.chn_vectors[idx].d2a_lo_setMin()
        self.master_vector.d2a_lo_slider.setValue(0)

    def d2a_lo_setMax(self):
        for idx in range(self.chns):
            self.chn_vectors[idx].d2a_lo_setMax()
        self.master_vector.d2a_lo_slider.setValue(16383)

    def d2a_hi_spin_changed(self, level):
        for idx in range(self.chns):
            self.chn_vectors[idx].d2a_hi_slider.setValue(level)
        self.master_vector.d2a_hi_slider.setValue(level)

    def d2a_hi_slider_changed(self, level):
        for idx in range(self.chns):
            self.chn_vectors[idx].d2a_hi_slider.setValue(level)
        self.master_vector.d2a_hi_spin.setValue(level)

    def d2a_hi_setMin(self):
        for idx in range(self.chns):
            self.chn_vectors[idx].d2a_hi_setMin()
        self.master_vector.d2a_hi_slider.setValue(0)

    def d2a_hi_setMax(self):
        for idx in range(self.chns):
            self.chn_vectors[idx].d2a_hi_setMax()
        self.master_vector.d2a_hi_slider.setValue(16383)

    '''
    self called methods
    '''

    def send_channel(self):
        for idx in range(self.chns):
            self.chn_vectors[idx].send_channel()

    def lock_channel(self, state):
        for idx in range(self.chns):
            self.chn_vectors[idx].lock_button.setChecked(state)
        if state == 1:
            self.master_vector.lock_button.setStyleSheet("background-color: #" + tc.green + ";")
            self.master_vector.lock_button.setText('dynamic')
        else:
            self.master_vector.lock_button.setStyleSheet("background-color: #" + tc.red + ";")
            self.master_vector.lock_button.setText('static')

    def card_delay_changed(self, newDelay):
        self.delay = newDelay
        if self.mode == 1:
            self.send_wreg0()

    def LED_changed(self):
        self.led = self.LED_button.isChecked()
        if self.led ==1:
            self.LED_button.setStyleSheet("background-color: #" + tc.red + ";")
            self.LED_button.setText('OFF')
        else:
            self.LED_button.setStyleSheet("background-color: #" + tc.green + ";")
            self.LED_button.setText('ON')
#		 if self.unlocked == 1:
        self.send_wreg0()

    def status_changed(self):
        self.status = self.status_button.isChecked()
        if self.status ==1:
            self.status_button.setStyleSheet("background-color: #" + tc.green + ";")
        else:
            self.status_button.setStyleSheet("background-color: #" + tc.red + ";")
        self.send_wreg0()

    def send_globals(self):
        print(tc.FCTCALL + "send BAD16 globals:", tc.ENDC)
        self.send_wreg0()
# 		self.send_wreg1()
        print()

    def dwell_changed(self):
        self.dwell_val = self.dwell.value()
        self.dwellDACunits = 2**(self.dwell_val)
        self.dwell_indicator.setText('%5i'%self.dwellDACunits)
        self.period_changed()
        if self.mode == 1:
            self.send_wreg1()

    def range_changed(self):
        self.range_val = self.range.value()
        self.rangeDACunits = 2**self.range_val
        self.range_indicator.setText('%5i'%self.rangeDACunits)
        self.amp_changed()
        self.period_changed()
# 		periodDACunits = 2*2**self.dwell_val*2**self.range_val
# 		self.period_indicator.setText('%11i'%periodDACunits)
# 		self.period_eng_indicator.setText('%12.4d'%periodDACunits*self.lsync*0.008)
# 		self.freq_eng_indicator.setText(str(1000/float(self.period_eng_indicator.text()))[:6])
        if self.mode == 1:
            self.send_wreg1()

    def step_changed(self):
        self.step_val = self.step.value()
        self.stepDACunits = self.step_val
        self.amp_changed()
# 		self.period_changed()
# 		self.amp_indicator.setText(str((2**self.range_val)*self.step_val))
# 		self.amp_eng_indicator.setText(str(int(self.amp_indicator.text())/16.383)[:6])
        if self.mode == 1:
            self.send_wreg1()

    def amp_changed(self):
        print("amp_changed")
        self.ampDACunits = self.rangeDACunits * self.stepDACunits
        print(self.ampDACunits)
        if self.ampDACunits > 16383:
            self.ampDACunits = 16383
        self.amp_indicator.setText('%5i'%self.ampDACunits)
        mV = 1000*self.ampDACunits/16383.0
        print(mV, str(mV))
        self.amp_eng_indicator.setText('%4.3f'%mV)
# 		self.amp_eng_indicator.setText('%6.3d'%volts)
        print()

    def period_changed(self):
        print("period changed")
        self.periodDACunits = float(2*self.dwellDACunits*self.rangeDACunits)
        self.period_indicator.setText('%12i'%self.periodDACunits)
        uSecs = self.periodDACunits*self.lsync*0.008
        print(uSecs)
        kHz = 1000/uSecs
        print(kHz)
        self.period_eng_indicator.setText('%8.4f'%uSecs)
        self.freq_eng_indicator.setText('%6.3f'%kHz)
        print()


    def tri_idx_changed(self):
        self.tri_idx = self.tri_idx_button.isChecked()
        if self.tri_idx ==1:
            self.tri_idx_button.setStyleSheet("background-color: #" + tc.green + ";")
            self.tri_idx_button.setText('FRAME')
        else:
            self.tri_idx_button.setStyleSheet("background-color: #" + tc.red + ";")
            self.tri_idx_button.setText('LSYNC')
        self.send_wreg1()

    def send_wreg0(self):
        print("BAD16: WREG0: legacy globals")
        wreg = 0 << 25
        wregval = wreg | (self.status << 16) | (self.led << 14) | (self.delay << 10) | (self.init << 8)| self.seqln
# 		wregval = wreg | (self.led << 24) | (self.status << 16) | (self.delay << 10) | (self.seqln << 1)
        self.sendReg(wregval)
        print()

    def send_wreg1(self):
        print("BAD16:WREG1: triangle parameters")
        wreg = 1 << 25
        wregval = wreg + (self.tri_idx << 24) + (self.dwell_val << 20) + (self.range_val << 16) + self.step_val
        self.sendReg(wregval)
        print()

    def sendReg(self, wregval):
        print(tc.COMMAND + "send to address", self.address, ":", tc.BOLD, wregval, tc.ENDC)
        b0 = (wregval & 0x7f ) << 1			# 1st 7 bits shifted up 1
        b1 = ((wregval >> 7) & 0x7f) <<  1	 # 2nd 7 bits shifted up 1
        b2 = ((wregval >> 14) & 0x7f) << 1	 # 3rd 7 bits shifted up 1
        b3 = ((wregval >> 21) & 0x7f) << 1	 # 4th 7 bits shifted up 1
        b4 = (self.address << 1) + 1		   # Address shifted up 1 bit with address bit set
        msg = struct.pack('BBBBB', b0, b1, b2, b3, b4)

        self.serialport.write(msg)

    def packMasterVector(self):
        self.MasterState	=	{
            'dc'			:	self.master_vector.dc_button.isChecked(),
            'LoHi'			:	self.master_vector.LoHi_button.isChecked(),
            'tri'			:	self.master_vector.Tri_button.isChecked(),
            'd2a_lo'		:	self.master_vector.d2a_lo_spin.value(),
            'd2a_hi'		:	self.master_vector.d2a_hi_spin.value(),
                                }

    def unpackMasterVector(self, masterVector):
            self.master_vector.dc_button.setChecked(masterVector['dc'])
            self.master_vector.LoHi_button.setChecked(masterVector['LoHi'])
            self.master_vector.Tri_button.setChecked(masterVector['tri'])
            self.master_vector.d2a_lo_spin.setValue(masterVector['d2a_lo'])
            self.master_vector.d2a_hi_spin.setValue(masterVector['d2a_hi'])

    def packChannels(self):
        for idx in range(self.chns):
            self.chn_vectors[idx].packChannel()
            self.allChannels['channel%i'%idx] = self.chn_vectors[idx].ChannelVector

    def unpackChannels(self, badAllChannels):
        for idx in range(self.chns):
            self.chn_vectors[idx].unpackChannel(badAllChannels['channel%i'%idx])

def main():

    app = QApplication(sys.argv)
    app.setStyle("plastique")
    app.setStyleSheet("""	QPushbutton{font: 10px; padding: 6px}
                            QToolButton{font: 10px; padding: 6px}""")
    win = badrap(addr=addr, slot=slot, seqln=seqln)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    p = optparse.OptionParser()
# 	p.add_option('-C','--card_type', action='store', dest='ctype', type='str',
# 				 help='Type of card to calibrate (default=DFBx2).')
    p.add_option('-A','--card_address', action='store', dest='addr', type='int',
                 help='Hardware address of card (default=32).')
    p.add_option('-S','--slot', action='store', dest='slot', type='int',
                 help='Host slot in crate (default=9)')
    p.add_option('-L','--length', action='store', dest='seqln', type='int',
                 help='Number of states in sequence (default=4')
# 	p.set_defaults(ctype="DFBx2")
    p.set_defaults(addr=40)
    p.set_defaults(slot=9)
    p.set_defaults(seqln=4)
    opt, args = p.parse_args()
# 	ctype = opt.ctype
    addr = opt.addr
    slot = opt.slot
    seqln = opt.seqln
    main()



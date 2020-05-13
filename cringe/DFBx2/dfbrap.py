import sys
import optparse
import struct
import time

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import named_serial
from .dfbchn import dfbChn
from cringe.shared import terminal_colors as tc


class dfbrap(QWidget):

# 	def __init__(self, parent=None, **kwargs):
# 		print kwargs
	def __init__(self, parent=None, addr=None, slot=None, column=1, seqln=None, lsync=32):

		super(dfbrap, self).__init__()

		self.parent = parent
		self.address = addr
		self.slot = slot
		self.lsync = lsync
		self.col = column

		self.chID = str(slot) + "/" + str(addr) + "/" + str(column)

		self.serialport = named_serial.Serial(port='rack', shared = True)

		self.states = 64

# 		self.delay = 0

		'''global booleans'''

		self.GR = True

		self.LED = False
		self.ST = False
		self.PS = False
		self.CLK = False

		'''global variables'''

		self.XPT = 0
		self.NSAMP = 4
		self.prop_delay = 0
		self.card_delay = 0
		self.seqln = seqln
		self.SETT = 12

		'''ARL parameters'''

		self.ARLsense = 0
		self.RLDpos = 0
		self.RLDneg = 0

		self.frame_period = self.lsync * self.seqln * 0.008


		'''triangle parameters'''

		self.dwell_val = 0
		self.dwellDACunits = float(0)
		self.range_val = 1
		self.rangeDACunits = float(2)
		self.step_val = 1
		self.stepDACunits = float(1)
		self.tri_idx = 0

		self.MVTX = 0
		self.MVRX = 1

# 		self.frame = self.lsync * self.seqln
		self.mode = 1
		self.wreg4 = 134905864

		self.state_vectors = []
		self.allStates = {}
# 		self.enb = [0,0,0,0,0,0,0]
# 		self.cal_coeffs = [0,0,0,0,0,0,0]
# 		self.appTrim =[0,0,0,0,0,0,0]

		self.setWindowTitle("DFBRAP")	# Phase Offset Widget
		self.setGeometry(30,30,1200,800)
		self.setContentsMargins(0,0,0,0)

		self.layout_widget = QWidget(self)
		self.layout = QVBoxLayout(self)

		'''
		build widget for GLOBALS header
		'''
		if parent == None:
			'''
			build widget for file management controls
			'''
			self.file_mgmt_widget = QGroupBox(self)
	# 		self.file_mgmt_widget.setFlat(1)
			self.file_mgmt_widget.setFixedWidth(1080)
			self.file_mgmt_widget.setFocusPolicy(QtCore.Qt.NoFocus)
			self.file_mgmt_widget.setTitle("FILE MANAGEMENT INTERFACE")

			self.file_mgmt_layout = QGridLayout(self.file_mgmt_widget)
			self.file_mgmt_layout.setContentsMargins(5,5,5,5)
			self.file_mgmt_layout.setSpacing(5)

			self.loadsetup = QPushButton(self, text = "load setup")
			self.loadsetup.setFixedHeight(25)
			self.file_mgmt_layout.addWidget(self.loadsetup,0,0,1,1,QtCore.Qt.AlignLeft)

			self.savesetup = QPushButton(self, text = "save setup")
			self.savesetup.setFixedHeight(25)
			self.file_mgmt_layout.addWidget(self.savesetup,0,1,1,1,QtCore.Qt.AlignLeft)

			self.sendALLchns = QPushButton(self, text = "send setup")
			self.sendALLchns.setFixedHeight(25)
			self.file_mgmt_layout.addWidget(self.sendALLchns,0,2,1,1,QtCore.Qt.AlignLeft)

			self.filenameEdit = QLineEdit()
			self.filenameEdit.setReadOnly(True)
			self.file_mgmt_layout.addWidget(self.filenameEdit,0,4,1,4)

			self.filename_label = QLabel("file")
			self.file_mgmt_layout.addWidget(self.filename_label,0,3,1,1,QtCore.Qt.AlignRight)

			self.layout.addWidget(self.file_mgmt_widget)

			'''
			build widget for SYSTEM GLOBALS header
			'''
			self.sys_glob_hdr_widget = QGroupBox(self)
			self.sys_glob_hdr_widget.setFixedWidth(1080)
			self.sys_glob_hdr_widget.setFocusPolicy(QtCore.Qt.NoFocus)
			self.sys_glob_hdr_widget.setTitle("SYSTEM GLOBALS")

			self.sys_glob_layout = QGridLayout(self.sys_glob_hdr_widget)
			self.sys_glob_layout.setContentsMargins(5,5,5,5)
			self.sys_glob_layout.setSpacing(5)

			self.seqln_indicator = QLineEdit()
			self.seqln_indicator.setReadOnly(True)
			self.seqln_indicator.setText('%3d'%seqln)
			self.seqln_indicator.setAlignment(QtCore.Qt.AlignRight)
			self.seqln_indicator.setFixedSize(self.seqln_indicator.sizeHint())
			self.seqln_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
			self.sys_glob_layout.addWidget(self.seqln_indicator,0,0,1,1)

			self.seqln_lbl = QLabel("sequence length")
	# 		self.seqln_lbl.setAlignment(QtCore.Qt.AlignLeft)
			self.sys_glob_layout.addWidget(self.seqln_lbl,0,1,1,1,QtCore.Qt.AlignLeft)

			self.lsync_indicator = QLineEdit()
			self.lsync_indicator.setReadOnly(True)
			self.lsync_indicator.setText('%4d'%lsync)
			self.lsync_indicator.setAlignment(QtCore.Qt.AlignRight)
			self.lsync_indicator.setFixedSize(self.seqln_indicator.sizeHint())
			self.lsync_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
			self.sys_glob_layout.addWidget(self.lsync_indicator,0,2,1,1)

			self.seqln_lbl = QLabel("line period")
			self.sys_glob_layout.addWidget(self.seqln_lbl,0,3,1,5,QtCore.Qt.AlignLeft)

			self.layout.addWidget(self.sys_glob_hdr_widget)

			'''
			build widget for CLASS GLOBALS header
			'''
			self.class_glob_hdr_widget = QGroupBox(self)
			self.class_glob_hdr_widget.setFixedWidth(1080)
			self.class_glob_hdr_widget.setFocusPolicy(QtCore.Qt.NoFocus)
			self.class_glob_hdr_widget.setTitle("CLASS GLOBALS")

			self.class_glob_layout = QGridLayout(self.class_glob_hdr_widget)
			self.class_glob_layout.setContentsMargins(5,5,5,5)
			self.class_glob_layout.setSpacing(5)

			self.card_delay_spin = QSpinBox()
			self.card_delay_spin.setRange(0, 15)
	# 		self.card_delay_spin.setFixedWidth(45)
			self.card_delay_spin.setSingleStep(1)
			self.card_delay_spin.setKeyboardTracking(0)
			self.card_delay_spin.setFocusPolicy(QtCore.Qt.StrongFocus)
			self.card_delay_spin.setValue(self.card_delay)
			self.card_delay_spin.setAlignment(QtCore.Qt.AlignRight)
			self.class_glob_layout.addWidget(self.card_delay_spin,0,0,1,1)
			self.card_delay_spin.valueChanged.connect(self.card_delay_changed)

			self.card_delay_lbl = QLabel("card delay")
			self.class_glob_layout.addWidget(self.card_delay_lbl,0,1,1,7,QtCore.Qt.AlignLeft)

			self.prop_delay_spin = QSpinBox()
			self.prop_delay_spin.setRange(0,15)
			self.prop_delay_spin.setSingleStep(1)
			self.prop_delay_spin.setKeyboardTracking(0)
			self.prop_delay_spin.setFocusPolicy(QtCore.Qt.StrongFocus)
			self.prop_delay_spin.setValue(self.prop_delay)
			self.prop_delay_spin.setAlignment(QtCore.Qt.AlignRight)
			self.class_glob_layout.addWidget(self.prop_delay_spin,0,2,1,1)
			self.prop_delay_spin.valueChanged.connect(self.prop_delay_changed)

			self.prop_delay_lbl = QLabel("prop delay")
			self.class_glob_layout.addWidget(self.prop_delay_lbl,0,3,1,1,QtCore.Qt.AlignLeft)

			self.xpt_mode = QComboBox()
			self.xpt_mode.setFixedHeight(25)
			self.xpt_mode.addItem('0: A-C-B-D')
			self.xpt_mode.addItem('1: C-A-D-B')
			self.xpt_mode.addItem('2: B-D-A-C')
			self.xpt_mode.addItem('3: D-B-C-A')
			self.xpt_mode.addItem('4: A-B-C-D')
			self.xpt_mode.addItem('5: C-D-A-B')
			self.xpt_mode.addItem('6: B-A-D-C')
			self.xpt_mode.addItem('7: D-C-B-A')
			self.class_glob_layout.addWidget(self.xpt_mode,0,4,1,1)
			self.xpt_mode.currentIndexChanged.connect(self.XPT_changed)

			self.status_lbl = QLabel("crosspoint mode")
			self.class_glob_layout.addWidget(self.status_lbl,0,5,1,3,QtCore.Qt.AlignLeft)

			self.class_glb_send = QPushButton(self, text = "send DFBx2 class globals")
			self.class_glb_send.setFixedHeight(25)
	# 		self.glb_send.setFixedWidth(160)
			self.class_glob_layout.addWidget(self.class_glb_send,0,8,1,1,QtCore.Qt.AlignRight)
			self.class_glb_send.clicked.connect(self.send_class_globals)

			self.NSAMP_spin = QSpinBox()
			self.NSAMP_spin.setRange(0, 255)
	# 		self.NSAMP_spin.setFixedWidth(45)
			self.NSAMP_spin.setSingleStep(1)
			self.NSAMP_spin.setKeyboardTracking(0)
			self.NSAMP_spin.setFocusPolicy(QtCore.Qt.StrongFocus)
			self.NSAMP_spin.setAlignment(QtCore.Qt.AlignRight)
			self.NSAMP_spin.setValue(self.NSAMP)
			self.class_glob_layout.addWidget(self.NSAMP_spin,1,0,1,1)
			self.NSAMP_spin.valueChanged.connect(self.NSAMP_changed)

			self.NSAMP_spin_lbl = QLabel("NSAMP")
			self.class_glob_layout.addWidget(self.NSAMP_spin_lbl,1,1,1,7,QtCore.Qt.AlignLeft)

			self.SETT_spin = QSpinBox()
			self.SETT_spin.setRange(0,255)
			self.SETT_spin.setSingleStep(1)
			self.SETT_spin.setKeyboardTracking(0)
			self.SETT_spin.setFocusPolicy(QtCore.Qt.StrongFocus)
			self.SETT_spin.setValue(self.SETT)
			self.SETT_spin.setAlignment(QtCore.Qt.AlignRight)
			self.class_glob_layout.addWidget(self.SETT_spin,1,2,1,1)
			self.SETT_spin.valueChanged.connect(self.SETT_changed)

			self.SETT_spin_lbl = QLabel("SETT")
			self.class_glob_layout.addWidget(self.SETT_spin_lbl,1,3,1,1,QtCore.Qt.AlignLeft)

			self.PS_button = QToolButton(self, text = 'PS')
			self.PS_button.setFixedHeight(25)
			self.PS_button.setCheckable(1)
			self.PS_button.setChecked(self.PS)
			self.PS_button.setStyleSheet("background-color: #" + tc.red + ";")
			self.class_glob_layout.addWidget(self.PS_button,1,4,1,1,QtCore.Qt.AlignRight)
			self.PS_button.toggled.connect(self.PS_changed)

			self.status_lbl = QLabel("parallel stream")
			self.class_glob_layout.addWidget(self.status_lbl,1,5,1,3,QtCore.Qt.AlignLeft)

			self.layout.addWidget(self.class_glob_hdr_widget)

			'''
			build widget for ARL control
			'''
			self.arl_widget = QGroupBox(self)
	# 		self.tri_wvfm_widget.setFixedHeight(25)
			self.arl_widget.setTitle("AUTO RELOCK CONTROL")
			self.arl_layout = QGridLayout(self.arl_widget)
			self.arl_layout.setContentsMargins(5,5,5,5)
			self.arl_layout.setSpacing(5)

			self.ARLsense_spin = QSpinBox()
			self.ARLsense_spin.setRange(0, 13)
			self.ARLsense_spin.setFixedHeight(25)
			self.ARLsense_spin.setSingleStep(1)
			self.ARLsense_spin.setKeyboardTracking(0)
			self.ARLsense_spin.setFocusPolicy(QtCore.Qt.StrongFocus)
			self.arl_layout.addWidget(self.ARLsense_spin,0,0,1,1,QtCore.Qt.AlignRight)
			self.ARLsense_spin.valueChanged.connect(self.ARLsense_changed)

			self.ARLsense_lbl = QLabel("2^N index")
			self.arl_layout.addWidget(self.ARLsense_lbl,0,1,1,1,QtCore.Qt.AlignLeft)

			self.ARLsense_indicator = QLineEdit()
			self.ARLsense_indicator.setReadOnly(True)
			self.ARLsense_indicator.setFixedHeight(25)
			self.ARLsense_indicator.setText('%5i'%2**(self.ARLsense))
			self.ARLsense_indicator.setAlignment(QtCore.Qt.AlignRight)
			self.ARLsense_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
			self.arl_layout.addWidget(self.ARLsense_indicator, 1,0,1,1,QtCore.Qt.AlignRight)

			self.ARLsense_indicator_lbl = QLabel("flux sensitivity [DAC units]")
			self.arl_layout.addWidget(self.ARLsense_indicator_lbl,1,1,1,1,QtCore.Qt.AlignLeft)

			self.ARLsense_eng_indicator = QLineEdit()
			self.ARLsense_eng_indicator.setReadOnly(True)
			self.ARLsense_eng_indicator.setFixedHeight(25)
			self.ARLsense_eng_indicator.setText(str(2**(self.ARLsense)/16.383)[:6])
			self.ARLsense_eng_indicator.setAlignment(QtCore.Qt.AlignRight)
			self.ARLsense_eng_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
			self.arl_layout.addWidget(self.ARLsense_eng_indicator, 2,0,1,1,QtCore.Qt.AlignRight)

			self.ARLsense_eng_indicator_lbl = QLabel("flux sensitivity [mV]")
			self.arl_layout.addWidget(self.ARLsense_eng_indicator_lbl,2,1,1,1,QtCore.Qt.AlignLeft)

			self.RLDpos_spin = QSpinBox()
			self.RLDpos_spin.setRange(0, 15)
			self.RLDpos_spin.setFixedHeight(25)
			self.RLDpos_spin.setSingleStep(1)
			self.RLDpos_spin.setKeyboardTracking(0)
			self.RLDpos_spin.setFocusPolicy(QtCore.Qt.StrongFocus)
			self.arl_layout.addWidget(self.RLDpos_spin,0,2,1,1,QtCore.Qt.AlignRight)
			self.RLDpos_spin.valueChanged.connect(self.RLDpos_changed)

			self.RLDpos_lbl = QLabel("2^N index")
			self.arl_layout.addWidget(self.RLDpos_lbl,0,3,1,1,QtCore.Qt.AlignLeft)

			self.RLDpos_indicator = QLineEdit()
			self.RLDpos_indicator.setReadOnly(True)
	# 		self.range_indicator.setFixedWidth(60)
			self.RLDpos_indicator.setText(str(2**(self.RLDpos)))
			self.RLDpos_indicator.setAlignment(QtCore.Qt.AlignRight)
			self.RLDpos_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
			self.arl_layout.addWidget(self.RLDpos_indicator,1,2,1,1,QtCore.Qt.AlignRight)

			self.RLDpos_indicator_lbl = QLabel("(+) relock delay")
			self.arl_layout.addWidget(self.RLDpos_indicator_lbl,1,3,1,1,QtCore.Qt.AlignLeft)

			self.RLDpos_eng_indicator = QLineEdit()
			self.RLDpos_eng_indicator.setReadOnly(True)
			self.RLDpos_eng_indicator.setFixedHeight(25)
			self.RLDpos_eng_indicator.setText(str(2**(self.RLDpos)*self.frame_period)[:6])
			self.RLDpos_eng_indicator.setAlignment(QtCore.Qt.AlignRight)
			self.RLDpos_eng_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
			self.arl_layout.addWidget(self.RLDpos_eng_indicator, 2,2,1,1,QtCore.Qt.AlignRight)

			self.RLDpos_eng_indicator_lbl = QLabel("positive relock delay [us]")
			self.arl_layout.addWidget(self.RLDpos_eng_indicator_lbl,2,3,1,1,QtCore.Qt.AlignLeft)

			self.RLDneg_spin = QSpinBox()
			self.RLDneg_spin.setRange(0, 15)
			self.RLDneg_spin.setFixedHeight(25)
			self.RLDneg_spin.setSingleStep(1)
			self.RLDneg_spin.setKeyboardTracking(0)
			self.RLDneg_spin.setFocusPolicy(QtCore.Qt.StrongFocus)
			self.arl_layout.addWidget(self.RLDneg_spin,0,4,1,1,QtCore.Qt.AlignRight)
			self.RLDneg_spin.valueChanged.connect(self.RLDneg_changed)

			self.RLDneg_lbl = QLabel("2^N index")
			self.arl_layout.addWidget(self.RLDneg_lbl,0,5,1,1,QtCore.Qt.AlignLeft)

			self.RLDneg_indicator = QLineEdit()
			self.RLDneg_indicator.setReadOnly(True)
	# 		self.range_indicator.setFixedWidth(60)
			self.RLDneg_indicator.setText(str(2**(self.RLDneg)))
			self.RLDneg_indicator.setAlignment(QtCore.Qt.AlignRight)
			self.RLDneg_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
			self.arl_layout.addWidget(self.RLDneg_indicator,1,4,1,1,QtCore.Qt.AlignRight)

			self.RLDneg_indicator_lbl = QLabel("(-) relock delay")
			self.arl_layout.addWidget(self.RLDneg_indicator_lbl,1,5,1,1,QtCore.Qt.AlignLeft)

			self.RLDneg_eng_indicator = QLineEdit()
			self.RLDneg_eng_indicator.setReadOnly(True)
			self.RLDneg_eng_indicator.setFixedHeight(25)
			self.RLDneg_eng_indicator.setText(str(2**(self.RLDneg)*self.frame_period)[:6])
			self.RLDneg_eng_indicator.setAlignment(QtCore.Qt.AlignRight)
			self.RLDneg_eng_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
			self.arl_layout.addWidget(self.RLDneg_eng_indicator, 2,4,1,1,QtCore.Qt.AlignRight)

			self.RLDneg_eng_indicator_lbl = QLabel("negative relock delay [us]")
			self.arl_layout.addWidget(self.RLDneg_eng_indicator_lbl,2,5,1,1,QtCore.Qt.AlignLeft)

			self.layout.addWidget(self.arl_widget)

			'''
			build widget for Triangle Waveform Generator
			'''
			self.tri_wvfm_widget = QGroupBox(self)
	# 		self.tri_wvfm_widget.setFixedHeight(25)
			self.tri_wvfm_widget.setTitle("TRIANGLE WAVEFORM GENERATOR")
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
			self.tri_wvfm_layout.addWidget(self.tri_idx_button,1,4,1,1,QtCore.Qt.AlignRight)
			self.tri_idx_button.toggled.connect(self.tri_idx_changed)

			self.tri_idx_lbl = QLabel("timebase")
			self.tri_wvfm_layout.addWidget(self.tri_idx_lbl,1,5,1,1,QtCore.Qt.AlignLeft)

			self.tri_send = QPushButton(self, text = "send triangle")
			self.tri_send.setFixedHeight(25)
	# 		self.tri_send.setFixedWidth(160)
			self.tri_wvfm_layout.addWidget(self.tri_send,0,10,1,2, QtCore.Qt.AlignVCenter)
			self.tri_send.clicked.connect(self.send_wreg4(self.wreg4))

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
			build widget for CARD INTERFACE PARAMETERS header
			'''
			self.class_interface_widget = QGroupBox(self)
			self.class_interface_widget.setFixedWidth(1080)
			self.class_interface_widget.setFocusPolicy(QtCore.Qt.NoFocus)
			self.class_interface_widget.setTitle("CARD INTERFACE PARAMETERS")

			self.controls_layout = QGridLayout(self.class_interface_widget)
			self.controls_layout.setContentsMargins(5,5,5,5)
			self.controls_layout.setSpacing(5)

			self.addr_indicator = QLineEdit()
			self.addr_indicator.setReadOnly(True)
			self.addr_indicator.setText(str(addr))
			self.addr_indicator.setAlignment(QtCore.Qt.AlignRight)
			self.addr_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
			self.controls_layout.addWidget(self.addr_indicator,0,0,1,1,QtCore.Qt.AlignRight)

			self.addr_label = QLabel("card address")
			self.controls_layout.addWidget(self.addr_label,0,1,1,1,QtCore.Qt.AlignLeft)

			self.slot_indicator = QLineEdit()
			self.slot_indicator.setReadOnly(True)
	# 		self.addr_indicator.setFixedWidth(40)
			self.slot_indicator.setText('%2d'%slot)
			self.slot_indicator.setAlignment(QtCore.Qt.AlignRight)
			self.slot_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
			self.controls_layout.addWidget(self.slot_indicator,0,2,1,1,QtCore.Qt.AlignRight)

			self.slot_label = QLabel("card slot")
			self.controls_layout.addWidget(self.slot_label,0,3,1,5,QtCore.Qt.AlignLeft)

			self.layout.addWidget(self.class_interface_widget)

		'''
		build widget for CHANNEL GLOBAL VARIABLE control
		'''
		self.glb_var_widget = QGroupBox(self)
		self.glb_var_widget.setTitle("CHANNEL GLOBAL VARIABLES")
		self.glb_var_layout = QGridLayout(self.glb_var_widget)
		self.glb_var_layout.setContentsMargins(5,5,10,5)
		self.glb_var_layout.setSpacing(5)

		self.MSTR_TX = QToolButton(self, text = 'OFF')
		self.MSTR_TX.setFixedHeight(25)
		self.MSTR_TX.setCheckable(1)
		self.MSTR_TX.setChecked(self.MVTX)
		self.MSTR_TX.setStyleSheet("background-color: #" + tc.red + ";")
		self.glb_var_layout.addWidget(self.MSTR_TX,0,0,1,1)
		self.MSTR_TX.toggled.connect(self.MSTR_TX_changed)

		self.MSTR_TX_lbl = QLabel("MASTER VECTOR Broadcast")
		self.glb_var_layout.addWidget(self.MSTR_TX_lbl,0,1,1,1,QtCore.Qt.AlignLeft)

		self.MSTR_RX = QToolButton(self, text = 'RX')
		self.MSTR_RX.setFixedHeight(25)
		self.MSTR_RX.setCheckable(1)
		self.MSTR_RX.setChecked(self.MVRX)
		self.MSTR_RX.setStyleSheet("background-color: #" + tc.green + ";")
		self.glb_var_layout.addWidget(self.MSTR_RX,0,2,1,1)
		self.MSTR_RX.toggled.connect(self.MSTR_RX_changed)

		self.MSTR_RX_lbl = QLabel("MASTER VECTOR Echo")
		self.glb_var_layout.addWidget(self.MSTR_RX_lbl,0,3,1,4,QtCore.Qt.AlignLeft)

		self.GR_button = QToolButton(self, text = 'ENB')
		self.GR_button.setFixedHeight(25)
		self.GR_button.setCheckable(1)
		self.GR_button.setChecked(self.GR)
		self.GR_button.setStyleSheet("background-color: #" + tc.green + ";")
		self.glb_var_layout.addWidget(self.GR_button,0,8,1,1)
		self.GR_button.toggled.connect(self.GR_changed)

		self.led_lbl = QLabel("channel lock enable")
		self.glb_var_layout.addWidget(self.led_lbl,0,7,1,1,QtCore.Qt.AlignRight)

		self.glb_send = QPushButton(self, text = "send CHANNEL globals")
		self.glb_send.setFixedHeight(25)
		self.glb_send.setFixedWidth(200)
		self.glb_var_layout.addWidget(self.glb_send,0,10,1,2,QtCore.Qt.AlignRight)
		self.glb_send.clicked.connect(self.send_channel_globals)

		self.layout.addWidget(self.glb_var_widget)

		'''
		build widget for MASTER CONTROL VECTOR: these controls effect all channels on a card
		'''
		self.master_ctrl_widget = QGroupBox(self)
		self.master_ctrl_widget.setTitle("MASTER CONTROL VECTOR")
		self.master_ctrl_layout = QGridLayout(self.master_ctrl_widget)

		self.master_vector = dfbChn(self, self.master_ctrl_layout, state=-1, chn=0, cardaddr=self.address, serialport=self.serialport, master = 'master')
		self.master_vector.counter_label.setText("all")
		self.master_vector.chn_send.setText("send ALL")

		self.layout.addWidget(self.master_ctrl_widget)

		'''
		build widget for arrayed channel parameters
		'''
		self.arrayframe = QWidget(self.layout_widget)
		self.array_layout = QVBoxLayout(self.arrayframe)
		self.array_layout.setSpacing(5)
		self.array_layout.setContentsMargins(10,10,10,10)

		for idx in range(self.states):
			self.state_vectors.append(dfbChn(self, self.array_layout, state=idx, chn=column, cardaddr=self.address, serialport=self.serialport))
#
		self.scrollarea = QScrollArea(self.layout_widget)
		self.scrollarea.setWidget(self.arrayframe)
		self.layout.addWidget(self.scrollarea)
# 		self.show()
# 		print self.arrayframe.width()

		self.master_ctrl_widget.setFixedWidth(self.arrayframe.width()+0)
		self.glb_var_widget.setFixedWidth(self.arrayframe.width()+0)
# 		self.class_interface_widget.setFixedWidth(self.arrayframe.width()+0)

	def __str__(self):
		return "dfbrap: slot/addr %g/%g channel %g"%(self.slot, self.address, self.col)

	def __repr__(self):
		return self.__str__()

	'''
	child called methods
	'''
	def triA_changed(self, state, *bc):
		var = "triA"
		if (not(bc) and self.MVTX) == True:
			self.parent.parent.broadcast_channel(state, var)
		else:
			for idx in range(self.states):
				self.state_vectors[idx].TriA_button.setChecked(state)
				if state == 1:
					self.master_vector.TriA_button.setStyleSheet("background-color: #" + tc.green + ";")
				else:
					self.master_vector.TriA_button.setStyleSheet("background-color: #" + tc.red + ";")

	def triB_changed(self, state, *bc):
		var = "triB"
		if (not(bc) and self.MVTX) == True:
			self.parent.parent.broadcast_channel(state, var)
		else:
			for idx in range(self.states):
				self.state_vectors[idx].TriB_button.setChecked(state)
			if state == 1:
				self.master_vector.TriB_button.setStyleSheet("background-color: #" + tc.green + ";")
			else:
				self.master_vector.TriB_button.setStyleSheet("background-color: #" + tc.red + ";")

	def a2d_lockpt_spin_changed(self, level, *bc):
		var = "a2d_lp_spin"
		if (not(bc) and self.MVTX) == True:
			self.parent.parent.broadcast_channel(level, var)
		else:
			for idx in range(self.states):
				self.state_vectors[idx].a2d_lockpt_slider.setValue(level)
			self.master_vector.a2d_lockpt_slider.setValue(level)

	def a2d_lockpt_slider_changed(self, level, *bc):
		var = "a2d_lp_slider"
		if (not(bc) and self.MVTX) == True:
			self.parent.parent.broadcast_channel(level, var)
		else:
			for idx in range(self.states):
				self.state_vectors[idx].a2d_lockpt_slider.setValue(level)
			self.master_vector.a2d_lockpt_spin.setValue(level)

	def d2a_A_spin_changed(self, level, *bc):
		var = "d2a_A_spin"
		if (not(bc) and self.MVTX) == True:
			self.parent.parent.broadcast_channel(level, var)
		else:
			for idx in range(self.states):
				self.state_vectors[idx].d2a_A_slider.setValue(level)
			self.master_vector.d2a_A_slider.setValue(level)

	def d2a_A_slider_changed(self, level, *bc):
		var = "d2a_A_slider"
		if (not(bc) and self.MVTX) == True:
			self.parent.parent.broadcast_channel(level, var)
		else:
			for idx in range(self.states):
				self.state_vectors[idx].d2a_A_slider.setValue(level)
			self.master_vector.d2a_A_spin.setValue(level)

	def d2a_B_spin_changed(self, level, *bc):
		var = "d2a_B_spin"
		if (not(bc) and self.MVTX) == True:
			self.parent.parent.broadcast_channel(level, var)
		else:
			for idx in range(self.states):
				self.state_vectors[idx].d2a_B_slider.setValue(level)
			self.master_vector.d2a_B_slider.setValue(level)

	def d2a_B_slider_changed(self, level, *bc):
		var = "d2a_B_slider"
		if (not(bc) and self.MVTX) == True:
			self.parent.parent.broadcast_channel(level, var)
		else:
			for idx in range(self.states):
				self.state_vectors[idx].d2a_B_slider.setValue(level)
			self.master_vector.d2a_B_spin.setValue(level)

	def data_packet_changed(self, index, *bc):
		var = "SM"
		if (not(bc) and self.MVTX) == True:
			self.parent.parent.broadcast_channel(index, var)
		else:
			for idx in range(self.states):
				self.state_vectors[idx].data_packet.setCurrentIndex(index)
			self.master_vector.data_packet.setCurrentIndex(index)

	def P_spin_changed(self, level, *bc):
		var = "P"
		if (not(bc) and self.MVTX) == True:
			self.parent.parent.broadcast_channel(level, var)
		else:
			for idx in range(self.states):
				self.state_vectors[idx].P_spin.setValue(level)
			self.master_vector.P_spin.setValue(level)

	def I_spin_changed(self, level, *bc):
		var = "I"
		if (not(bc) and self.MVTX) == True:
			self.parent.parent.broadcast_channel(level, var)
		else:
			for idx in range(self.states):
				self.state_vectors[idx].I_spin.setValue(level)
			self.master_vector.I_spin.setValue(level)

	def FBA_changed(self, state, *bc):
		var = "FBa"
		if (not(bc) and self.MVTX) == True:
			self.parent.parent.broadcast_channel(state, var)
		else:
			for idx in range(self.states):
				self.state_vectors[idx].FBA_button.setChecked(state)
			if state == 1:
				self.master_vector.FBA_button.setStyleSheet("background-color: #" + tc.green + ";")
				self.master_vector.FBB_button.setChecked(0)
			else:
				self.master_vector.FBA_button.setStyleSheet("background-color: #" + tc.red + ";")

	def FBB_changed(self, state, *bc):
		var = "FBb"
		if (not(bc) and self.MVTX) == True:
			self.parent.parent.broadcast_channel(state, var)
		else:
			for idx in range(self.states):
				self.state_vectors[idx].FBB_button.setChecked(state)
			if state == 1:
				self.master_vector.FBB_button.setStyleSheet("background-color: #" + tc.green + ";")
				self.master_vector.FBA_button.setChecked(0)
			else:
				self.master_vector.FBB_button.setStyleSheet("background-color: #" + tc.red + ";")

	def ARL_changed(self, state, *bc):
		var = "ARL"
		if (not(bc) and self.MVTX) == True:
			self.parent.parent.broadcast_channel(state, var)
		else:
			for idx in range(self.states):
				self.state_vectors[idx].ARL_button.setChecked(state)
			if state == 1:
				self.master_vector.ARL_button.setStyleSheet("background-color: #" + tc.green + ";")
			else:
				self.master_vector.ARL_button.setStyleSheet("background-color: #" + tc.red + ";")

	def send_channel(self, *bc):
		var = "send"
		if (not(bc) and self.MVTX) == True:
			self.parent.parent.broadcast_channel(0, var)
		else:
			for idx in range(self.states):
				self.state_vectors[idx].send_channel()

	def lock_channel(self, state, *bc):
		var = "lock"
		if (not(bc) and self.MVTX) == True:
			self.parent.parent.broadcast_channel(state, var)
		else:
			for idx in range(self.states):
				self.state_vectors[idx].lock_button.setChecked(state)
			if state == 1:
				self.master_vector.lock_button.setStyleSheet("background-color: #" + tc.green + ";")
				self.master_vector.lock_button.setText('dynamic')
			else:
				self.master_vector.lock_button.setStyleSheet("background-color: #" + tc.red + ";")
				self.master_vector.lock_button.setText('static')

	'''
	self called methods
	'''

	def card_delay_changed(self):
		self.card_delay = self.card_delay_spin.value()
		if self.mode == 1:
			self.send_wreg7()

	def prop_delay_changed(self):
		self.prop_delay = self.prop_delay_spin.value()
		if self.mode == 1:
			self.send_wreg7()

	def XPT_changed(self):
		self.XPT = self.xpt_mode.currentIndex()
		if self.mode == 1:
			self.send_wreg6()

	def NSAMP_changed(self):
		self.NSAMP = self.NSAMP_spin.value()
		if self.mode == 1:
			self.send_wreg6()

	def SETT_changed(self):
		self.SETT = self.SETT_spin.value()
		if self.mode == 1:
			self.send_wreg7()

	def PS_changed(self):
		self.PS = self.PS_button.isChecked()
		if self.PS ==1:
			self.PS_button.setStyleSheet("background-color: #" + tc.green + ";")
		else:
			self.PS_button.setStyleSheet("background-color: #" + tc.red + ";")
		self.send_wreg6()

	def ARLsense_changed(self):
		self.ARLsense = self.ARLsense_spin.value()
		self.ARLsense_indicator.setText("%5i"%(2**self.ARLsense))
		self.ARLsense_eng_indicator.setText(str((2**self.ARLsense)/16.383)[:6])
		self.send_wreg6()

	def RLDpos_changed(self):
		self.RLDpos = self.RLDpos_spin.value()
		self.RLDpos_indicator.setText("%5i"%(2**self.RLDpos))
		self.RLDpos_eng_indicator.setText(str((2**self.RLDpos)*self.frame_period)[:6])
		self.send_wreg6()

	def RLDneg_changed(self):
		self.RLDneg = self.RLDneg_spin.value()
		self.RLDneg_indicator.setText("%5i"%(2**self.RLDneg))
		self.RLDneg_eng_indicator.setText(str((2**self.RLDneg)*self.frame_period)[:6])
		self.send_wreg6()

	def LED_changed(self):
		self.LED = self.LED_button.isChecked()
		if self.LED ==1:
			self.LED_button.setStyleSheet("background-color: #" + tc.red + ";")
			self.LED_button.setText('OFF')
		else:
			self.LED_button.setStyleSheet("background-color: #" + tc.green + ";")
			self.LED_button.setText('ON')
#		 if self.unlocked == 1:
		self.send_wreg7()

	def status_changed(self):
		self.ST = self.status_button.isChecked()
		if self.ST ==1:
			self.status_button.setStyleSheet("background-color: #" + tc.green + ";")
		else:
			self.status_button.setStyleSheet("background-color: #" + tc.red + ";")
		self.send_wreg7()

	def GR_changed(self):
		print(tc.FCTCALL + "send global relock enable to DFB channel", self.col, tc.ENDC)
		self.GR = self.GR_button.isChecked()
		if self.GR == 1:
			self.GR_button.setStyleSheet("background-color: #" + tc.green + ";")
			self.GR_button.setText('ENB')
		else:
			self.GR_button.setStyleSheet("background-color: #" + tc.red + ";")
			self.GR_button.setText('OFF')
		self.send_channel_globals()
# 		self.send_wreg0()
# 		self.send_wreg4()
		

	def MSTR_TX_changed(self):
		self.MVTX = self.MSTR_TX.isChecked()
		print(tc.FCTCALL + "set Master Vector Broadcast for DFB channel", self.col, ":", bool(self.MVTX), tc.ENDC)
		
		if self.MVTX == 1:
			self.MSTR_TX.setStyleSheet("background-color: #" + tc.green + ";")
			self.MSTR_TX.setText('TX')
			self.MSTR_RX.setChecked(1)
		else:
			self.MSTR_TX.setStyleSheet("background-color: #" + tc.red + ";")
			self.MSTR_TX.setText('OFF')

	def MSTR_RX_changed(self):
		self.MVRX = self.MSTR_RX.isChecked()
		print(tc.FCTCALL + "set Master Vector Echo for DFB channel", self.col, ":", bool(self.MVRX), tc.ENDC)
		
		if self.MVRX == 1:
			self.MSTR_RX.setStyleSheet("background-color: #" + tc.green + ";")
			self.MSTR_RX.setText('RX')
		else:
			self.MSTR_RX.setStyleSheet("background-color: #" + tc.red + ";")
			self.MSTR_RX.setText('OFF')
			self.MSTR_TX.setChecked(0)

	def send_class_globals(self):
		print(tc.FCTCALL + "send DFB class globals:", tc.ENDC)
		self.send_wreg0()
		self.send_wreg4()
		self.send_wreg6()
		self.send_wreg7()
		

	def send_channel_globals(self):
		print(tc.FCTCALL + "send DFB channel globals:", tc.ENDC)
		self.send_wreg0()
		self.send_wreg4(self.wreg4)
# 		self.send_wreg7()
		

# 	def mode_changed(self):
# 		self.mode = self.mode_button.isChecked()
# 		if self.mode ==1:
# 			self.mode_button.setStyleSheet("background-color: #" + tc.green + ";")
# 			self.mode_button.setText('dynamic')
# 		else:
# 			self.mode_button.setStyleSheet("background-color: #" + tc.red + ";")
# 			self.mode_button.setText('static')

	def dwell_changed(self):
		self.dwell_val = self.dwell.value()
		self.dwellDACunits = 2**(self.dwell_val)
		self.dwell_indicator.setText('%5i'%self.dwellDACunits)
		self.period_changed()
		if self.mode == 1:
			self.send_wreg0()
			self.send_wreg4()

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
			self.send_wreg0()
			self.send_wreg4()

	def step_changed(self):
		self.step_val = self.step.value()
		self.stepDACunits = self.step_val
		self.amp_changed()
# 		self.period_changed()
# 		self.amp_indicator.setText(str((2**self.range_val)*self.step_val))
# 		self.amp_eng_indicator.setText(str(int(self.amp_indicator.text())/16.383)[:6])
		if self.mode == 1:
			self.send_wreg0()
			self.send_wreg4()

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
		


	def tri_idx_changed(self):
		self.tri_idx = self.tri_idx_button.isChecked()
		if self.tri_idx ==1:
			self.tri_idx_button.setStyleSheet("background-color: #" + tc.green + ";")
			self.tri_idx_button.setText('FRAME')
		else:
			self.tri_idx_button.setStyleSheet("background-color: #" + tc.red + ";")
			self.tri_idx_button.setText('LSYNC')
		self.send_wreg0()
		self.send_wreg4()

	def send_wreg0(self):
		print("DFB:WREG0: page register: COL", self.col)
		wreg = 0 << 25
		wregval = wreg + (self.col << 6)
		self.sendReg(wregval)
		

	def send_wreg4(self, wreg4):
		if self.parent != None:
			self.parent.parent.send_dfb_wreg4(self.GR, self.address)
		else:
# 		print "WREG4: triangle parameters & GR"
			self.wreg4 = wreg4
# 		self.wreg4 = parent.parent.dfb_wreg4()
			cmd_reg = bin(self.wreg4)[5:].zfill(25)
			dwell = int(cmd_reg[1:5], base=2)
			steps = int(cmd_reg[5:9], base=2)
			step = int(cmd_reg[11:], base=2)
			print("DFB:WREG4: triangle parameters DWELL, STEPS, STEP SIZE (& global relock):", dwell, steps, step, "(",self.GR,")")
			self.sendReg((self.wreg4 & 0xFFF7FFF) | (self.GR << 15))
			

	def sendReg(self, wregval):
		print(tc.COMMAND + "send to address", self.address, ":", tc.BOLD, wregval, tc.ENDC)
		b0 = (wregval & 0x7f ) << 1			# 1st 7 bits shifted up 1
		b1 = ((wregval >> 7) & 0x7f) <<  1	 # 2nd 7 bits shifted up 1
		b2 = ((wregval >> 14) & 0x7f) << 1	 # 3rd 7 bits shifted up 1
		b3 = ((wregval >> 21) & 0x7f) << 1	 # 4th 7 bits shifted up 1
		b4 = (self.address << 1) + 1		   # Address shifted up 1 bit with address bit set
		msg = struct.pack('BBBBB', b0, b1, b2, b3, b4)
		self.serialport.write(msg)
		time.sleep(0.001)

	def packCHglobals(self):
		self.CHglobals = 	{
			'enb'			:	self.GR_button.isChecked(),
			'mvtx'			:	self.MSTR_TX.isChecked(),
			'mvrx'			:	self.MSTR_RX.isChecked(),
							}

	def unpackCHglobals(self, CHglobals):
		self.GR_button.setChecked(CHglobals['enb'])
		self.MSTR_TX.setChecked(CHglobals['mvtx'])
		self.MSTR_RX.setChecked(CHglobals['mvrx'])

	def packMasterVector(self):
		self.MasterState	=	{
			'triA'			:	self.master_vector.TriA_button.isChecked(),
			'triB'			:	self.master_vector.TriB_button.isChecked(),
			'a2d_lockpt'	:	self.master_vector.a2d_lockpt_spin.value(),
			'd2a_A'			:	self.master_vector.d2a_A_spin.value(),
			'd2a_B'			:	self.master_vector.d2a_B_spin.value(),
			'SM'			:	self.master_vector.data_packet.currentIndex(),
			'P'				:	self.master_vector.P_spin.value(),
			'I'				:	self.master_vector.I_spin.value(),
			'FBA'			:	self.master_vector.FBA_button.isChecked(),
			'FBB'			:	self.master_vector.FBB_button.isChecked(),
			'ARL'			:	self.master_vector.ARL_button.isChecked()
								}

	def unpackMasterVector(self, masterVector):
			self.master_vector.TriA_button.setChecked(masterVector['triA'])
			self.master_vector.TriB_button.setChecked(masterVector['triB'])
			self.master_vector.a2d_lockpt_spin.setValue(masterVector['a2d_lockpt'])
			self.master_vector.d2a_A_spin.setValue(masterVector['d2a_A'])
			self.master_vector.d2a_B_spin.setValue(masterVector['d2a_B'])
			self.master_vector.data_packet.setCurrentIndex(masterVector['SM'])
			self.master_vector.P_spin.setValue(masterVector['P'])
			self.master_vector.I_spin.setValue(masterVector['I'])
			self.master_vector.FBA_button.setChecked(masterVector['FBA'])
			self.master_vector.FBB_button.setChecked(masterVector['FBB'])
			self.master_vector.ARL_button.setChecked(masterVector['ARL'])

	def packStates(self):
		for idx in range(self.states):
			self.state_vectors[idx].packState()
			self.allStates['state%i'%idx] = self.state_vectors[idx].stateVector

	def unpackStates(self, dfbAllStates):
		for idx in range(self.states):
			self.state_vectors[idx].unpackState(dfbAllStates['state%i'%idx])


def main():

	app = QApplication(sys.argv)
	app.setStyle("plastique")
	app.setStyleSheet("""	QPushbutton{font: 10px; padding: 6px}
							QToolButton{font: 10px; padding: 6px}""")
	win = dfbrap(addr=addr, slot=slot, seqln=seqln)
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
	p.set_defaults(addr=32)
	p.set_defaults(slot=9)
	p.set_defaults(seqln=4)
	opt, args = p.parse_args()
# 	ctype = opt.ctype
	addr = opt.addr
	slot = opt.slot
	seqln = opt.seqln
	main()

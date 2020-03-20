#-*- coding: utf-8 -*-
import sys
import optparse
import struct
import time

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt, SIGNAL
from PyQt4.QtGui import QFileDialog, QPalette, QSpinBox, QToolButton

import named_serial
from . import dfbrap
from . import dprcal
# from dprcal import dprcal

class dfbx2(QtGui.QWidget):

    def __init__(self, parent=None, addr=None, slot=None, seqln=None, lsync=32):

        super(dfbx2, self).__init__()


        self.COMMAND = '\033[95m'
        self.FCTCALL = '\033[94m'
        self.INIT = '\033[92m'
        self.WARNING = '\033[93m'
        self.FAIL = '\033[91m'
        self.ENDC = '\033[0m'
        self.BOLD = "\033[1m"

        self.green = "90EE90"
        self.red ="F08080"
        self.yellow = "FFFFCC"
        self.grey = "808080"
        self.white = "FFFFFF"
        self.grey = "808080"

# 		print QtGui.QApplication.palette()	
# 		self.color_list = QtGui.QColor.colorNames()
# 		print self.color_list
# 		self.tt_palette = QtGui.QPalette(self)
# 		self.tt_palette.setColor(2, 18, Qt.white)
# 		self.tt_palette.setColor(2, 19, Qt.black)
# 		self.tt_palette.setColor(2,10,Qt.yellow)
# 		self.tt_palette.setColor(2,6,Qt.black)

        self.serialport = named_serial.Serial(port='rack', shared = True)

        self.states = 32

        self.address = addr
        self.slot = slot
        self.seqln = seqln
        self.lsync = 32
# 		self.frame = self.lsync * self.seqln

        '''global booleans'''

        self.LED = 0
        self.ST = 0
        self.PS = 0
        self.CLK = 0
        self.GR = 0

        '''global variables'''

        self.XPT = 0
        self.NSAMP = 4
        self.prop_delay = 0
        self.card_delay = 0
        self.seqln = seqln
        self.SETT = 12

        '''ARL default parameters'''

        self.ARLsense = 10
        self.RLDpos = 6
        self.RLDneg = 2

        self.frame_period = self.lsync * self.seqln * 0.008

        '''triangle default parameters'''

        self.dwell_val = 0
        self.dwellDACunits = float(1)
        self.range_val = 10
        self.rangeDACunits = float(1024)
        self.step_val = 8
        self.stepDACunits = float(256)
        self.tri_idx = 0

        '''card global default variables'''

        self.mode = 1


        self.chn_vectors = []
# 		self.enb = [0,0,0,0,0,0,0]
# 		self.cal_coeffs = [0,0,0,0,0,0,0]
# 		self.appTrim =[0,0,0,0,0,0,0]

        self.setWindowTitle("DFBx2: %d/%d"%(slot, addr))	# Phase Offset Widget
        self.setGeometry(30,30,800,1000)
        self.setContentsMargins(0,0,0,0)

        self.layout_widget = QtGui.QWidget(self)
        self.layout = QtGui.QGridLayout(self)
# 		self.layout = QtGui.QVBoxLayout(self)

        '''
        build widget for file management controls
        '''
        self.file_mgmt_widget = QtGui.QGroupBox(self)
# 		self.file_mgmt_widget.setFlat(1)
        self.file_mgmt_widget.setFixedWidth(1080)
        self.file_mgmt_widget.setFocusPolicy(Qt.NoFocus)
        self.file_mgmt_widget.setTitle("FILE MANAGEMENT INTERFACE")

        self.file_mgmt_layout = QtGui.QGridLayout(self.file_mgmt_widget)
        self.file_mgmt_layout.setContentsMargins(5,5,5,5)
        self.file_mgmt_layout.setSpacing(5)

        self.loadsetup = QtGui.QPushButton(self, text = "load setup")
        self.loadsetup.setFixedHeight(25)
        self.file_mgmt_layout.addWidget(self.loadsetup,0,0,1,1,QtCore.Qt.AlignLeft)

        self.savesetup = QtGui.QPushButton(self, text = "save setup")
        self.savesetup.setFixedHeight(25)
        self.file_mgmt_layout.addWidget(self.savesetup,0,1,1,1,QtCore.Qt.AlignLeft)

        self.sendALLchns = QtGui.QPushButton(self, text = "send setup")
        self.sendALLchns.setFixedHeight(25)
        self.file_mgmt_layout.addWidget(self.sendALLchns,0,2,1,1,QtCore.Qt.AlignLeft)

        self.filenameEdit = QtGui.QLineEdit()
        self.filenameEdit.setReadOnly(True)
        self.file_mgmt_layout.addWidget(self.filenameEdit,0,4,1,4)

        self.filename_label = QtGui.QLabel("file")
        self.file_mgmt_layout.addWidget(self.filename_label,0,3,1,1,QtCore.Qt.AlignRight)

        self.layout.addWidget(self.file_mgmt_widget, 0,0,1,2)

        '''
        build widget for SYSTEM GLOBALS header
        '''
        self.sys_glob_hdr_widget = QtGui.QGroupBox(self)
        self.sys_glob_hdr_widget.setFixedWidth(1080)
        self.sys_glob_hdr_widget.setFocusPolicy(Qt.NoFocus)
        self.sys_glob_hdr_widget.setTitle("SYSTEM GLOBALS")

        self.sys_glob_layout = QtGui.QGridLayout(self.sys_glob_hdr_widget)
        self.sys_glob_layout.setContentsMargins(5,5,5,5)
        self.sys_glob_layout.setSpacing(5)

        self.seqln_spin = QSpinBox()
        self.seqln_spin.setRange(4, 32)
# 		self.seqln_spin.setFixedWidth(45)
        self.seqln_spin.setSingleStep(1)
        self.seqln_spin.setKeyboardTracking(0)
        self.seqln_spin.setFocusPolicy(Qt.StrongFocus)
        self.seqln_spin.setValue(self.seqln)
        self.seqln_spin.setAlignment(QtCore.Qt.AlignRight)
        self.sys_glob_layout.addWidget(self.seqln_spin,0,0,1,1)
        self.seqln_spin.valueChanged.connect(self.seqln_changed)

        self.seqln_lbl = QtGui.QLabel("sequence length")
# 		self.seqln_lbl.setAlignment(QtCore.Qt.AlignLeft)
        self.sys_glob_layout.addWidget(self.seqln_lbl,0,1,1,1,QtCore.Qt.AlignLeft)

        self.lsync_spin = QSpinBox()
        self.lsync_spin.setRange(20,256)
# 		self.lsync_spin.setFixedWidth(45)
        self.lsync_spin.setSingleStep(1)
        self.lsync_spin.setKeyboardTracking(0)
        self.lsync_spin.setFocusPolicy(Qt.StrongFocus)
        self.lsync_spin.setValue(self.lsync)
        self.lsync_spin.setAlignment(QtCore.Qt.AlignRight)
        self.sys_glob_layout.addWidget(self.lsync_spin,0,2,1,1)
        self.lsync_spin.valueChanged.connect(self.lsync_changed)

        self.seqln_lbl = QtGui.QLabel("line period")
        self.sys_glob_layout.addWidget(self.seqln_lbl,0,3,1,5,QtCore.Qt.AlignLeft)

        self.sys_glob_send = QtGui.QPushButton(self, text = "send system globals")
        self.sys_glob_send.setFixedHeight(25)
# 		self.sys_glob_send.setFixedWidth(160)
        self.sys_glob_layout.addWidget(self.sys_glob_send,0,9,1,2, QtCore.Qt.AlignRight)
# 		self.sys_glob_send.clicked.connect(self.stateEnable)

        self.layout.addWidget(self.sys_glob_hdr_widget,1,0,1,2)

        '''
        build widget for CLASS GLOBALS header: OLD
        '''
# 		self.class_glob_hdr_widget = QtGui.QGroupBox(self)
# 		self.class_glob_hdr_widget.setFixedWidth(1080)
# 		self.class_glob_hdr_widget.setFocusPolicy(Qt.NoFocus)
# 		self.class_glob_hdr_widget.setTitle("CLASS GLOBALS")
# 		
# 		self.class_glob_layout = QtGui.QGridLayout(self.class_glob_hdr_widget)
# 		self.class_glob_layout.setContentsMargins(5,5,5,5)
# 		self.class_glob_layout.setSpacing(5)
# 
# 		self.card_delay = QSpinBox()
# 		self.card_delay.setRange(0, 15)
# # 		self.card_delay.setFixedWidth(45)
# 		self.card_delay.setSingleStep(1)
# 		self.card_delay.setKeyboardTracking(0)
# 		self.card_delay.setFocusPolicy(Qt.StrongFocus)
# 		self.class_glob_layout.addWidget(self.card_delay,0,0,1,1)
# 		self.card_delay.valueChanged.connect(self.card_delay_changed)
# 
# 		self.card_delay_lbl = QtGui.QLabel("card delay")
# 		self.class_glob_layout.addWidget(self.card_delay_lbl,0,1,1,7,QtCore.Qt.AlignLeft)
# 		
# 		self.layout.addWidget(self.class_glob_hdr_widget)

        '''
        build widget for CLASS GLOBALS header
        '''
        self.class_glob_hdr_widget = QtGui.QGroupBox(self)
        self.class_glob_hdr_widget.setFixedWidth(1080)
        self.class_glob_hdr_widget.setFocusPolicy(Qt.NoFocus)
        self.class_glob_hdr_widget.setTitle("DFB CLASS GLOBALS")

        self.class_glob_layout = QtGui.QGridLayout(self.class_glob_hdr_widget)
        self.class_glob_layout.setContentsMargins(5,5,5,5)
        self.class_glob_layout.setSpacing(5)

        self.card_delay_spin = QSpinBox()
        self.card_delay_spin.setRange(0, 15)
# 		self.card_delay_spin.setFixedWidth(45)
        self.card_delay_spin.setSingleStep(1)
        self.card_delay_spin.setKeyboardTracking(0)
        self.card_delay_spin.setFocusPolicy(Qt.StrongFocus)
        self.card_delay_spin.setValue(self.card_delay)
        self.card_delay_spin.setAlignment(QtCore.Qt.AlignRight)
        self.class_glob_layout.addWidget(self.card_delay_spin,0,0,1,1)
        self.card_delay_spin.valueChanged.connect(self.card_delay_changed)

        self.card_delay_lbl = QtGui.QLabel("card delay")
        self.class_glob_layout.addWidget(self.card_delay_lbl,0,1,1,7,QtCore.Qt.AlignLeft)

        self.prop_delay_spin = QtGui.QSpinBox()
        self.prop_delay_spin.setRange(0,15)
        self.prop_delay_spin.setSingleStep(1)
        self.prop_delay_spin.setKeyboardTracking(0)
        self.prop_delay_spin.setFocusPolicy(Qt.StrongFocus)
        self.prop_delay_spin.setValue(self.prop_delay)
        self.prop_delay_spin.setAlignment(QtCore.Qt.AlignRight)
        self.class_glob_layout.addWidget(self.prop_delay_spin,0,2,1,1)
        self.prop_delay_spin.valueChanged.connect(self.prop_delay_changed)

        self.prop_delay_lbl = QtGui.QLabel("prop delay")
        self.class_glob_layout.addWidget(self.prop_delay_lbl,0,3,1,1,QtCore.Qt.AlignLeft)

        self.xpt_mode = QtGui.QComboBox()
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

        self.status_lbl = QtGui.QLabel("crosspoint mode")
        self.class_glob_layout.addWidget(self.status_lbl,0,5,1,3,QtCore.Qt.AlignLeft)

        self.tp_mode = QtGui.QComboBox()
        self.tp_mode.setFixedHeight(25)
        self.tp_mode.addItem('DEADBEEF')
        self.tp_mode.addItem('55555555')
        self.tp_mode.addItem('AAAAAAAA')
        self.tp_mode.addItem('33333333')
        self.tp_mode.addItem('0F0F0F0F')
        self.tp_mode.addItem('00FF00FF')
        self.tp_mode.addItem('0000FFFF')
        self.tp_mode.addItem('00000000')
        self.tp_mode.addItem('FFFFFFFF')
# 		self.tp_mode.addItem('B16B00B5')		
        self.tp_mode.addItem('8BADF00D')
        self.class_glob_layout.addWidget(self.tp_mode,0,6,1,1)
        self.tp_mode.currentIndexChanged.connect(self.TP_changed)

        self.status_lbl = QtGui.QLabel("test pattern")
        self.class_glob_layout.addWidget(self.status_lbl,0,7,1,3,QtCore.Qt.AlignLeft)

        self.class_glb_send = QtGui.QPushButton(self, text = "send DFBx2 class globals")
        self.class_glb_send.setFixedHeight(25)
        self.class_glb_send.setFixedWidth(200)
        self.class_glob_layout.addWidget(self.class_glb_send,0,9,1,1,QtCore.Qt.AlignRight)
        self.class_glb_send.clicked.connect(self.send_class_globals)

        self.NSAMP_spin = QSpinBox()
        self.NSAMP_spin.setRange(0, 255)
# 		self.NSAMP_spin.setFixedWidth(45)
        self.NSAMP_spin.setSingleStep(1)
        self.NSAMP_spin.setKeyboardTracking(0)
        self.NSAMP_spin.setFocusPolicy(Qt.StrongFocus)
        self.NSAMP_spin.setAlignment(QtCore.Qt.AlignRight)
        self.NSAMP_spin.setValue(self.NSAMP)
        self.class_glob_layout.addWidget(self.NSAMP_spin,1,0,1,1)
        self.NSAMP_spin.valueChanged.connect(self.NSAMP_changed)

        self.NSAMP_spin_lbl = QtGui.QLabel("NSAMP")
        self.class_glob_layout.addWidget(self.NSAMP_spin_lbl,1,1,1,7,QtCore.Qt.AlignLeft)

        self.SETT_spin = QtGui.QSpinBox()
        self.SETT_spin.setRange(0,255)
        self.SETT_spin.setSingleStep(1)
        self.SETT_spin.setKeyboardTracking(0)
        self.SETT_spin.setFocusPolicy(Qt.StrongFocus)
        self.SETT_spin.setValue(self.SETT)
        self.SETT_spin.setAlignment(QtCore.Qt.AlignRight)
        self.class_glob_layout.addWidget(self.SETT_spin,1,2,1,1)
        self.SETT_spin.valueChanged.connect(self.SETT_changed)

        self.SETT_spin_lbl = QtGui.QLabel("SETT")
        self.class_glob_layout.addWidget(self.SETT_spin_lbl,1,3,1,1,QtCore.Qt.AlignLeft)

        self.PS_button = QToolButton(self, text = 'PS')
        self.PS_button.setFixedHeight(25)
        self.PS_button.setCheckable(1)
        self.PS_button.setChecked(self.PS)
        self.PS_button.setStyleSheet("background-color: #" + self.red + ";")
        self.class_glob_layout.addWidget(self.PS_button,1,4,1,1,QtCore.Qt.AlignRight)
        self.PS_button.toggled.connect(self.PS_changed)

        self.status_lbl = QtGui.QLabel("parallel stream")
        self.class_glob_layout.addWidget(self.status_lbl,1,5,1,3,QtCore.Qt.AlignLeft)

        self.layout.addWidget(self.class_glob_hdr_widget,2,0,1,2)

        '''
        build widget for ARL control
        '''
        self.arl_widget = QtGui.QGroupBox(self)
# 		self.tri_wvfm_widget.setFixedHeight(25)
        self.arl_widget.setTitle("AUTO RELOCK CONTROL")
        self.arl_layout = QtGui.QGridLayout(self.arl_widget)
        self.arl_layout.setContentsMargins(5,5,5,5)
        self.arl_layout.setSpacing(5)

        self.ARLsense_title = QtGui.QLabel("flux jump threshold")
        self.arl_layout.addWidget(self.ARLsense_title,0,0,1,1, QtCore.Qt.AlignRight)

        self.ARLsense_spin = QSpinBox()
        self.ARLsense_spin.setRange(0, 13)
        self.ARLsense_spin.setFixedHeight(25)
        self.ARLsense_spin.setSingleStep(1)
        self.ARLsense_spin.setKeyboardTracking(0)
        self.ARLsense_spin.setFocusPolicy(Qt.StrongFocus)
        self.ARLsense_spin.setValue(self.ARLsense)
        self.arl_layout.addWidget(self.ARLsense_spin,1,0,1,1,QtCore.Qt.AlignRight)
        self.ARLsense_spin.valueChanged.connect(self.ARLsense_changed)

        self.ARLsense_lbl = QtGui.QLabel("2^N index")
        self.arl_layout.addWidget(self.ARLsense_lbl,1,1,1,1,QtCore.Qt.AlignLeft)

        self.ARLsense_indicator = QtGui.QLineEdit()
        self.ARLsense_indicator.setReadOnly(True)
        self.ARLsense_indicator.setFixedHeight(25)
        self.ARLsense_indicator.setText('%5i'%2**(self.ARLsense))
        self.ARLsense_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.ARLsense_indicator.setFocusPolicy(Qt.NoFocus)
        self.arl_layout.addWidget(self.ARLsense_indicator, 2,0,1,1,QtCore.Qt.AlignRight)

        self.ARLsense_indicator_lbl = QtGui.QLabel("DAC units")
        self.arl_layout.addWidget(self.ARLsense_indicator_lbl,2,1,1,1,QtCore.Qt.AlignLeft)

        self.ARLsense_eng_indicator = QtGui.QLineEdit()
        self.ARLsense_eng_indicator.setReadOnly(True)
        self.ARLsense_eng_indicator.setFixedHeight(25)
        self.ARLsense_eng_indicator.setText(str(2**(self.ARLsense)/16.383)[:6])
        self.ARLsense_eng_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.ARLsense_eng_indicator.setFocusPolicy(Qt.NoFocus)
        self.arl_layout.addWidget(self.ARLsense_eng_indicator, 3,0,1,1,QtCore.Qt.AlignRight)

        self.ARLsense_eng_indicator_lbl = QtGui.QLabel("mV")
        self.arl_layout.addWidget(self.ARLsense_eng_indicator_lbl,3,1,1,1,QtCore.Qt.AlignLeft)

        self.RLDpos_title = QtGui.QLabel("[+] event reset delay")
        self.arl_layout.addWidget(self.RLDpos_title,0,2,1,1, QtCore.Qt.AlignRight)

        self.RLDpos_spin = QSpinBox()
        self.RLDpos_spin.setRange(0, 15)
        self.RLDpos_spin.setFixedHeight(25)
        self.RLDpos_spin.setSingleStep(1)
        self.RLDpos_spin.setKeyboardTracking(0)
        self.RLDpos_spin.setFocusPolicy(Qt.StrongFocus)
        self.RLDpos_spin.setValue(self.RLDpos)
        self.arl_layout.addWidget(self.RLDpos_spin,1,2,1,1,QtCore.Qt.AlignRight)
        self.RLDpos_spin.valueChanged.connect(self.RLDpos_changed)

        self.RLDpos_lbl = QtGui.QLabel("2^N index")
        self.arl_layout.addWidget(self.RLDpos_lbl,1,3,1,1,QtCore.Qt.AlignLeft)

        self.RLDpos_indicator = QtGui.QLineEdit()
        self.RLDpos_indicator.setReadOnly(True)
# 		self.range_indicator.setFixedWidth(60)
        self.RLDpos_indicator.setText(str(2**(self.RLDpos)))
        self.RLDpos_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.RLDpos_indicator.setFocusPolicy(Qt.NoFocus)
        self.arl_layout.addWidget(self.RLDpos_indicator,2,2,1,1,QtCore.Qt.AlignRight)

        self.RLDpos_indicator_lbl = QtGui.QLabel("FRM units")
        self.arl_layout.addWidget(self.RLDpos_indicator_lbl,2,3,1,1,QtCore.Qt.AlignLeft)

        self.RLDpos_eng_indicator = QtGui.QLineEdit()
        self.RLDpos_eng_indicator.setReadOnly(True)
        self.RLDpos_eng_indicator.setFixedHeight(25)
        self.RLDpos_eng_indicator.setText(str(2**(self.RLDpos)*self.frame_period)[:6])
        self.RLDpos_eng_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.RLDpos_eng_indicator.setFocusPolicy(Qt.NoFocus)
        self.arl_layout.addWidget(self.RLDpos_eng_indicator, 3,2,1,1,QtCore.Qt.AlignRight)

        self.RLDpos_eng_indicator_lbl = QtGui.QLabel("\u00B5s")
        self.arl_layout.addWidget(self.RLDpos_eng_indicator_lbl,3,3,1,1,QtCore.Qt.AlignLeft)

        self.RLDneg_title = QtGui.QLabel("[-] event reset delay")
        self.arl_layout.addWidget(self.RLDneg_title,0,4,1,1, QtCore.Qt.AlignRight)

        self.RLDneg_spin = QSpinBox()
        self.RLDneg_spin.setRange(0, 15)
        self.RLDneg_spin.setFixedHeight(25)
        self.RLDneg_spin.setSingleStep(1)
        self.RLDneg_spin.setKeyboardTracking(0)
        self.RLDneg_spin.setFocusPolicy(Qt.StrongFocus)
        self.RLDneg_spin.setValue(self.RLDneg)
        self.arl_layout.addWidget(self.RLDneg_spin,1,4,1,1,QtCore.Qt.AlignRight)
        self.RLDneg_spin.valueChanged.connect(self.RLDneg_changed)

        self.RLDneg_lbl = QtGui.QLabel("2^N index")
        self.arl_layout.addWidget(self.RLDneg_lbl,1,5,1,1,QtCore.Qt.AlignLeft)

        self.RLDneg_indicator = QtGui.QLineEdit()
        self.RLDneg_indicator.setReadOnly(True)
# 		self.range_indicator.setFixedWidth(60)
        self.RLDneg_indicator.setText(str(2**(self.RLDneg)))
        self.RLDneg_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.RLDneg_indicator.setFocusPolicy(Qt.NoFocus)
        self.arl_layout.addWidget(self.RLDneg_indicator,2,4,1,1,QtCore.Qt.AlignRight)

        self.RLDneg_indicator_lbl = QtGui.QLabel("FRM units")
        self.arl_layout.addWidget(self.RLDneg_indicator_lbl,2,5,1,1,QtCore.Qt.AlignLeft)

        self.RLDneg_eng_indicator = QtGui.QLineEdit()
        self.RLDneg_eng_indicator.setReadOnly(True)
        self.RLDneg_eng_indicator.setFixedHeight(25)
        self.RLDneg_eng_indicator.setText(str(2**(self.RLDneg)*self.frame_period)[:6])
        self.RLDneg_eng_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.RLDneg_eng_indicator.setFocusPolicy(Qt.NoFocus)
        self.arl_layout.addWidget(self.RLDneg_eng_indicator, 3,4,1,1,QtCore.Qt.AlignRight)

        self.RLDneg_eng_indicator_lbl = QtGui.QLabel("\u00B5s")
        self.arl_layout.addWidget(self.RLDneg_eng_indicator_lbl,3,5,1,1,QtCore.Qt.AlignLeft)

        self.layout.addWidget(self.arl_widget,3,0,1,1)

        '''
        build widget for Triangle Waveform Generator
        '''
        self.tri_wvfm_widget = QtGui.QGroupBox(self)
# 		self.tri_wvfm_widget.setFixedHeight(25)
        self.tri_wvfm_widget.setTitle("TRIANGLE WAVEFORM GENERATOR")
        self.tri_wvfm_layout = QtGui.QGridLayout(self.tri_wvfm_widget)
        self.tri_wvfm_layout.setContentsMargins(5,5,5,5)
        self.tri_wvfm_layout.setSpacing(5)

        self.dwell = QSpinBox()
        self.dwell.setRange(0, 15)
        self.dwell.setFixedHeight(25)
        self.dwell.setSingleStep(1)
        self.dwell.setKeyboardTracking(0)
        self.dwell.setFocusPolicy(Qt.StrongFocus)
        self.dwell.setValue(self.dwell_val)
        self.tri_wvfm_layout.addWidget(self.dwell,0,0,1,1,QtCore.Qt.AlignRight)
        self.dwell.valueChanged.connect(self.dwell_changed)

        self.dwell_lbl = QtGui.QLabel("dwell (2^N)")
        self.tri_wvfm_layout.addWidget(self.dwell_lbl,0,1,1,1,QtCore.Qt.AlignLeft)

# 		self.dwell_indicator = QtGui.QLineEdit()
# 		self.dwell_indicator.setReadOnly(True)
# 		self.dwell_indicator.setFixedHeight(25)
# 		self.dwell_indicator.setText('%5i'%2**(self.dwell_val))
# 		self.dwell_indicator.setAlignment(QtCore.Qt.AlignRight)
# 		self.dwell_indicator.setFocusPolicy(Qt.NoFocus)
# 		self.tri_wvfm_layout.addWidget(self.dwell_indicator, 1,0,1,1,QtCore.Qt.AlignRight)
# 		
# 		self.range_indicator_lbl = QtGui.QLabel("dwell")
# 		self.tri_wvfm_layout.addWidget(self.range_indicator_lbl,1,1,1,1,QtCore.Qt.AlignLeft)

        self.range = QSpinBox()
        self.range.setRange(1, 14)
        self.range.setFixedHeight(25)
        self.range.setSingleStep(1)
        self.range.setKeyboardTracking(0)
        self.range.setFocusPolicy(Qt.StrongFocus)
        self.range.setValue(self.range_val)
        self.tri_wvfm_layout.addWidget(self.range,0,2,1,1,QtCore.Qt.AlignRight)
        self.range.valueChanged.connect(self.range_changed)

        self.range_lbl = QtGui.QLabel("steps (2^N)")
        self.tri_wvfm_layout.addWidget(self.range_lbl,0,3,1,1,QtCore.Qt.AlignLeft)

# 		self.range_indicator = QtGui.QLineEdit()
# 		self.range_indicator.setReadOnly(True)
# 		self.range_indicator.setText(str(2**(self.range_val)))
# 		self.range_indicator.setAlignment(QtCore.Qt.AlignRight)
# 		self.range_indicator.setFocusPolicy(Qt.NoFocus)
# 		self.tri_wvfm_layout.addWidget(self.range_indicator,1,2,1,1,QtCore.Qt.AlignRight)
# 		
# 		self.range_indicator_lbl = QtGui.QLabel("steps")
# 		self.tri_wvfm_layout.addWidget(self.range_indicator_lbl,1,3,1,1,QtCore.Qt.AlignLeft)

        self.step = QSpinBox()
        self.step.setRange(1, 16383)
        self.step.setFixedHeight(25)
        self.step.setSingleStep(1)
        self.step.setKeyboardTracking(0)
        self.step.setFocusPolicy(Qt.StrongFocus)
        self.step.setValue(self.step_val)
        self.tri_wvfm_layout.addWidget(self.step,0,4,1,1,QtCore.Qt.AlignRight)
        self.step.valueChanged.connect(self.step_changed)

        self.step_lbl = QtGui.QLabel("step size")
        self.tri_wvfm_layout.addWidget(self.step_lbl,0,5,1,1,QtCore.Qt.AlignLeft)

        self.period_indicator = QtGui.QLineEdit()
        self.period_indicator.setReadOnly(True)
# 		self.period_indicator.setFixedWidth(120)
        self.period_indicator.setText(str(2*(2**self.dwell_val)*(2**self.range_val)))
        self.period_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.period_indicator.setFocusPolicy(Qt.NoFocus)
        self.tri_wvfm_layout.addWidget(self.period_indicator,2,0,1,1,QtCore.Qt.AlignRight)

        self.period_indicator_lbl = QtGui.QLabel("period")
        self.tri_wvfm_layout.addWidget(self.period_indicator_lbl,2,1,1,1,QtCore.Qt.AlignLeft)

        self.period_eng_indicator = QtGui.QLineEdit()
        self.period_eng_indicator.setReadOnly(True)
# 		self.period_eng_indicator.setFixedWidth(120)
# 		self.period_eng_indicator.setText(str(2*(2**self.dwell_val)*(2**self.range_val)))
        self.period_eng_indicator.setText(str(int(self.period_indicator.text())*self.lsync*0.008))
        self.period_eng_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.period_eng_indicator.setFocusPolicy(Qt.NoFocus)
        self.tri_wvfm_layout.addWidget(self.period_eng_indicator,3,0,1,1,QtCore.Qt.AlignRight)

        self.period_eng_indicator_lbl = QtGui.QLabel("period [""\u00B5s]")
        self.tri_wvfm_layout.addWidget(self.period_eng_indicator_lbl,3,1,1,1,QtCore.Qt.AlignLeft)

        self.amp_indicator = QtGui.QLineEdit()
        self.amp_indicator.setReadOnly(True)
# 		self.amp_indicator.setFixedWidth(80)
        self.amp_indicator.setText(str((2**self.range_val)*self.step_val))
        self.amp_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.amp_indicator.setFocusPolicy(Qt.NoFocus)
        self.tri_wvfm_layout.addWidget(self.amp_indicator,2,2,1,1,QtCore.Qt.AlignRight)

        self.amp_indicator_lbl = QtGui.QLabel("amplitude")
        self.tri_wvfm_layout.addWidget(self.amp_indicator_lbl,2,3,1,1,QtCore.Qt.AlignLeft)

        self.amp_eng_indicator = QtGui.QLineEdit()
        self.amp_eng_indicator.setReadOnly(True)
# 		self.amp_eng_indicator.setFixedWidth(80)
        self.amp_eng_indicator.setText(str(int(self.amp_indicator.text())/16.383)[:6])
        self.amp_eng_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.amp_eng_indicator.setFocusPolicy(Qt.NoFocus)
        self.tri_wvfm_layout.addWidget(self.amp_eng_indicator,3,2,1,1,QtCore.Qt.AlignRight)

        self.amp_eng_indicator_lbl = QtGui.QLabel("amplitude [mV]")
        self.tri_wvfm_layout.addWidget(self.amp_eng_indicator_lbl,3,3,1,1,QtCore.Qt.AlignLeft)

        self.tri_idx_button = QToolButton(self, text = 'LSYNC')
        self.tri_idx_button.setFixedHeight(25)
        self.tri_idx_button.setCheckable(1)
        self.tri_idx_button.setChecked(self.tri_idx)
        self.tri_idx_button.setStyleSheet("background-color: #" + self.red + ";")
        self.tri_wvfm_layout.addWidget(self.tri_idx_button,0,6,1,1,QtCore.Qt.AlignRight)
        self.tri_idx_button.toggled.connect(self.tri_idx_changed)

        self.tri_idx_lbl = QtGui.QLabel("timebase")
        self.tri_wvfm_layout.addWidget(self.tri_idx_lbl,0,7,1,1,QtCore.Qt.AlignLeft)

        self.tri_send = QtGui.QPushButton(self, text = "send triangle")
        self.tri_send.setFixedHeight(25)
        self.tri_send.setFixedWidth(200)
        self.tri_wvfm_layout.addWidget(self.tri_send,2,4,1,4, QtCore.Qt.AlignRight)
        self.tri_send.clicked.connect(self.send_triangle)

        self.freq_eng_indicator = QtGui.QLineEdit()
        self.freq_eng_indicator.setReadOnly(True)
# 		self.freq_eng_indicator.setFixedWidth(80)
        self.freq_eng_indicator.setText(str(1000/float(self.period_eng_indicator.text()))[:6])
        self.freq_eng_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.freq_eng_indicator.setFocusPolicy(Qt.NoFocus)
        self.tri_wvfm_layout.addWidget(self.freq_eng_indicator,3,4,1,1,QtCore.Qt.AlignRight)

        self.amp_eng_indicator_lbl = QtGui.QLabel("freq [kHz]")
        self.tri_wvfm_layout.addWidget(self.amp_eng_indicator_lbl,3,5,1,1,QtCore.Qt.AlignLeft)

        self.layout.addWidget(self.tri_wvfm_widget,3,1,1,1,QtCore.Qt.AlignRight)

        '''
        build widget for CARD GLOBAL VARIABLE control
        '''
        self.card_glb_widget = QtGui.QGroupBox(self)
        self.card_glb_widget.setTitle("CARD GLOBAL VARIABLES")
        self.card_glb_layout = QtGui.QGridLayout(self.card_glb_widget)
        self.card_glb_layout.setContentsMargins(5,5,10,5)
        self.card_glb_layout.setSpacing(5)

        self.LED_button = QToolButton(self, text = 'ON')
        self.LED_button.setFixedHeight(25)
        self.LED_button.setCheckable(1)
        self.LED_button.setChecked(self.LED)
        self.LED_button.setStyleSheet("background-color: #" + self.green + ";")
        self.card_glb_layout.addWidget(self.LED_button,0,0,1,1)
        self.LED_button.toggled.connect(self.LED_changed)
        self.LED_button.setEnabled(1)

        self.led_lbl = QtGui.QLabel("LED control")
        self.card_glb_layout.addWidget(self.led_lbl,0,1,1,1,QtCore.Qt.AlignLeft)

        self.status_button = QToolButton(self, text = 'ST')
        self.status_button.setFixedHeight(25)
        self.status_button.setCheckable(1)
        self.status_button.setChecked(self.ST)
        self.status_button.setStyleSheet("background-color: #" + self.red + ";")
        self.card_glb_layout.addWidget(self.status_button,0,2,1,1)
        self.status_button.toggled.connect(self.status_changed)

        self.status_lbl = QtGui.QLabel("status bit")
        self.card_glb_layout.addWidget(self.status_lbl,0,3,1,1,QtCore.Qt.AlignLeft)

        self.card_glb_send = QtGui.QPushButton(self, text = "send CARD globals")
        self.card_glb_send.setFixedHeight(25)
        self.card_glb_send.setFixedWidth(200)
        self.card_glb_layout.addWidget(self.card_glb_send,0,4,1,1,QtCore.Qt.AlignRight)
        self.card_glb_send.clicked.connect(self.send_card_globals)

        self.layout.addWidget(self.card_glb_widget,4,0,1,1)

        '''
        build widget for CARD INTERFACE PARAMETERS header
        '''
        self.class_interface_widget = QtGui.QGroupBox(self)
        self.class_interface_widget.setFixedWidth(1080)
        self.class_interface_widget.setFocusPolicy(Qt.NoFocus)
        self.class_interface_widget.setTitle("CARD INTERFACE PARAMETERS")

        self.controls_layout = QtGui.QGridLayout(self.class_interface_widget)
        self.controls_layout.setContentsMargins(5,5,5,5)
        self.controls_layout.setSpacing(5)

        self.addr_indicator = QtGui.QLineEdit()
        self.addr_indicator.setReadOnly(True)
        self.addr_indicator.setText(str(addr))
        self.addr_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.addr_indicator.setFocusPolicy(Qt.NoFocus)
        self.controls_layout.addWidget(self.addr_indicator,0,0,1,1,QtCore.Qt.AlignRight)

        self.addr_label = QtGui.QLabel("card address")
        self.controls_layout.addWidget(self.addr_label,0,1,1,1,QtCore.Qt.AlignLeft)

        self.slot_indicator = QtGui.QLineEdit()
        self.slot_indicator.setReadOnly(True)
# 		self.addr_indicator.setFixedWidth(40)
        self.slot_indicator.setText('%2d'%slot)
        self.slot_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.slot_indicator.setFocusPolicy(Qt.NoFocus)
        self.controls_layout.addWidget(self.slot_indicator,0,2,1,1,QtCore.Qt.AlignRight)

        self.slot_label = QtGui.QLabel("card slot")
        self.controls_layout.addWidget(self.slot_label,0,3,1,5,QtCore.Qt.AlignLeft)

        self.layout.addWidget(self.class_interface_widget,4,1,1,1,QtCore.Qt.AlignRight)

        '''
        create TAB widget for embedding BAD16 functional widgets
        '''
        self.dfbx2_widget = QtGui.QTabWidget(self)

        self.dfbx2_widget1 = dfbrap.dfbrap(parent=self, addr=addr, slot=slot, column=1, seqln=seqln, lsync=lsync)
        self.dfbx2_widget.addTab(self.dfbx2_widget1, "CH1")

        self.dfbx2_widget2 = dfbrap.dfbrap(parent=self, addr=addr, slot=slot, column=2, seqln=seqln, lsync=lsync)
        self.dfbx2_widget.addTab(self.dfbx2_widget2, "CH2")

        dfbx2_widget3 = dprcal.dprcal(ctype="DFBx2", addr=addr, slot=slot)
        self.dfbx2_widget.addTab(dfbx2_widget3, "phase")

        self.layout.addWidget(self.dfbx2_widget,5,0,1,2)

        '''
        resize widgets for relative, platform dependent variability
        '''
        rm = 45
        self.file_mgmt_widget.setFixedWidth(self.dfbx2_widget1.width()+rm)
        self.sys_glob_hdr_widget.setFixedWidth(self.dfbx2_widget1.width()+rm)
        self.class_glob_hdr_widget.setFixedWidth(self.dfbx2_widget1.width()+rm)
        self.arl_widget.setFixedWidth(self.dfbx2_widget1.width()/2+10)
        self.tri_wvfm_widget.setFixedWidth(self.dfbx2_widget1.width()/2+10)
        self.card_glb_widget.setFixedWidth(self.dfbx2_widget1.width()/2+10)
        self.class_interface_widget.setFixedWidth(self.dfbx2_widget1.width()/2+10)


# 	def card_delay_changed(self):
        '''
        not sure what the best structure is for class global commanding
        at child level need to change control to indicator (QSpinBOx to QLineEdit)
        1 - can preserve function call triggered from within child as 'QLineEdit.textChanged'
        or
        2 - can pass parent.QSpinBox.value() to child function as parameter
        '''
# 		self.dfbx2_widget1.card_delay.setText(str("%2d"%self.card_delay.value()))
# 		self.dfbx2_widget1.card_delay_changed(self.card_delay.value())

    def seqln_changed(self):
        self.seqln = self.seqln_spin.value()
        self.send_wreg7()
        if self.tri_idx == 1:
            self.period_changed()
        for idx in range(self.states):
            if idx < self.seqln:
                self.dfbx2_widget1.state_vectors[idx].setEnabled(1)
                self.dfbx2_widget2.state_vectors[idx].setEnabled(1)
            else:
                self.dfbx2_widget1.state_vectors[idx].setEnabled(0)
                self.dfbx2_widget2.state_vectors[idx].setEnabled(0)

    def lsync_changed(self):
        self.lsync = self.lsync_spin.value()
        self.period_changed()

    def card_delay_changed(self):
        self.DFBx2class_glb_chg_msg()
        self.card_delay = self.card_delay_spin.value()
        if self.mode == 1:
            self.send_wreg7()
        print()

    def prop_delay_changed(self):
        self.DFBx2class_glb_chg_msg()
        self.prop_delay = self.prop_delay_spin.value()
        if self.mode == 1:
            self.send_wreg7()
        print()

    def XPT_changed(self):
        self.DFBx2class_glb_chg_msg()
        self.XPT = self.xpt_mode.currentIndex()
        if self.mode == 1:
            self.send_wreg6()
        print()

    def TP_changed(self):
        self.DFBx2class_glb_chg_msg()
        self.TP = self.tp_mode.currentIndex()
        if self.mode == 1:
            if self.TP != 0:
                self.send_GPI5()
                self.send_GPI6()
            self.send_GPI4()
        print()
    def NSAMP_changed(self):
        self.DFBx2class_glb_chg_msg()
        self.NSAMP = self.NSAMP_spin.value()
        if self.mode == 1:
            self.send_wreg6()
        print()

    def SETT_changed(self):
        self.DFBx2class_glb_chg_msg()
        self.SETT = self.SETT_spin.value()
        if self.mode == 1:
            self.send_wreg7()
        print()

    def PS_changed(self):
        self.DFBx2class_glb_chg_msg()
        self.PS = self.PS_button.isChecked()
        if self.PS ==1:
            self.PS_button.setStyleSheet("background-color: #" + self.green + ";")
        else:
            self.PS_button.setStyleSheet("background-color: #" + self.red + ";")
        self.send_wreg6()
        print()

    def DFBx2class_glb_chg_msg(self):
        print()
        print(self.FCTCALL + "DFBx2 CLASS global changed:", self.ENDC)


    def ARLsense_changed(self):
        print()
        print(self.FCTCALL + "ARL parameter changed:", self.ENDC)
        self.ARLsense = self.ARLsense_spin.value()
        self.ARLsense_indicator.setText("%5i"%(2**self.ARLsense))
        self.ARLsense_eng_indicator.setText(str((2**self.ARLsense)/16.383)[:6])
        self.send_wreg6()
        print()

    def RLDpos_changed(self):
        print()
        print(self.FCTCALL + "ARL parameter changed:", self.ENDC)
        self.RLDpos = self.RLDpos_spin.value()
        self.RLDpos_indicator.setText("%5i"%(2**self.RLDpos))
        self.RLDpos_eng_indicator.setText(str((2**self.RLDpos)*self.frame_period)[:6])
        self.send_wreg6()
        print()

    def RLDneg_changed(self):
        print()
        print(self.FCTCALL + "ARL parameter changed:", self.ENDC)
        self.RLDneg = self.RLDneg_spin.value()
        self.RLDneg_indicator.setText("%5i"%(2**self.RLDneg))
        self.RLDneg_eng_indicator.setText(str((2**self.RLDneg)*self.frame_period)[:6])
        self.send_wreg6()
        print()

    def dwell_changed(self):
        print()
        print(self.FCTCALL + "triangle step dwell changed:", self.ENDC)
        self.dwell_val = self.dwell.value()
        self.dwellDACunits = 2**(self.dwell_val)
# 		self.dwell_indicator.setText('%5i'%self.dwellDACunits)
        self.period_changed()
        if self.mode == 1:
            self.send_wreg0(1)
            self.send_wreg4()
            self.send_wreg0(2)
            self.send_wreg4()
        print()

    def range_changed(self):
        print()
        print(self.FCTCALL + "triangle number of steps changed:", self.ENDC)
        self.range_val = self.range.value()
        self.rangeDACunits = 2**(self.range_val)
# 		self.range_indicator.setText('%5i'%self.rangeDACunits)
        self.amp_changed()
        self.period_changed()
# 		periodDACunits = 2*2**self.dwell_val*2**self.range_val
# 		self.period_indicator.setText('%11i'%periodDACunits)
# 		self.period_eng_indicator.setText('%12.4d'%periodDACunits*self.lsync*0.008)
# 		self.freq_eng_indicator.setText(str(1000/float(self.period_eng_indicator.text()))[:6])
        if self.mode == 1:
            self.send_wreg0(1)
            self.send_wreg4()
            self.send_wreg0(2)
            self.send_wreg4()
        print()

    def step_changed(self):
        print()
        print(self.FCTCALL + "triangle step size changed:", self.ENDC)
        self.step_val = self.step.value()
        self.stepDACunits = self.step_val
        self.amp_changed()
# 		self.period_changed()
# 		self.amp_indicator.setText(str((2**self.range_val)*self.step_val))
# 		self.amp_eng_indicator.setText(str(int(self.amp_indicator.text())/16.383)[:6])
        if self.mode == 1:
            self.send_wreg0(1)
            self.send_wreg4()
            self.send_wreg0(2)
            self.send_wreg4()
        print()

    def amp_changed(self):
        print("amplitude changed")
        self.ampDACunits = self.rangeDACunits * self.stepDACunits
# 		print self.ampDACunits
        if self.ampDACunits > 16383:
            self.ampDACunits = 16383
        self.amp_indicator.setText('%5i'%self.ampDACunits)
        mV = 1000*self.ampDACunits/16383.0
# 		print mV, str(mV)
        self.amp_eng_indicator.setText('%4.3f'%mV)
# 		self.amp_eng_indicator.setText('%6.3d'%volts)

    def period_changed(self):
        print("period changed")
        self.periodDACunits = float(2*self.dwellDACunits*self.rangeDACunits)
        self.period_indicator.setText('%12i'%self.periodDACunits)
        if self.tri_idx == 0:
            uSecs = self.periodDACunits*self.lsync*0.008
        else:
            uSecs = self.periodDACunits*self.lsync*self.seqln*0.008
# 		print uSecs
        kHz = 1000/uSecs
# 		print kHz
        self.period_eng_indicator.setText('%8.4f'%uSecs)
        self.freq_eng_indicator.setText('%6.3f'%kHz)

    def tri_idx_changed(self):
        print()
        print(self.FCTCALL + "triangle timebase changed:", self.ENDC)
        self.tri_idx = self.tri_idx_button.isChecked()
        self.period_changed()
        if self.tri_idx ==1:
            self.tri_idx_button.setStyleSheet("background-color: #" + self.green + ";")
            self.tri_idx_button.setText('FRAME')
        else:
            self.tri_idx_button.setStyleSheet("background-color: #" + self.red + ";")
            self.tri_idx_button.setText('LSYNC')
        self.send_wreg0(1)
        self.send_wreg4()
        self.send_wreg0(2)
        self.send_wreg4()
        print()

    def LED_changed(self):
        self.LED = self.LED_button.isChecked()
        if self.LED ==1:
            self.LED_button.setStyleSheet("background-color: #" + self.red + ";")
            self.LED_button.setText('OFF')
        else:
            self.LED_button.setStyleSheet("background-color: #" + self.green + ";")
            self.LED_button.setText('ON')
#		 if self.unlocked == 1:
        self.send_wreg7()

    def status_changed(self):
        self.ST = self.status_button.isChecked()
        if self.ST ==1:
            self.status_button.setStyleSheet("background-color: #" + self.green + ";")
        else:
            self.status_button.setStyleSheet("background-color: #" + self.red + ";")
        self.send_wreg7()

    def send_class_globals(self):
        print()
        print(self.FCTCALL + "send class globals:", self.ENDC)
# 		self.send_wreg0()
# 		self.send_wreg4()
        self.send_wreg6()
        self.send_wreg7()
        print()

    def send_triangle(self):
        print()
        print(self.FCTCALL + "send triangle parameters to both channels:", self.ENDC)
        self.send_wreg0(1)
        self.send_wreg4()
        self.send_wreg0(2)
        self.send_wreg4()
        print()

    def send_card_globals(self):
        print()
        print(self.FCTCALL + "send card globals:", self.ENDC)
        self.send_wreg7()
        print()

    def send_wreg0(self, col):
        print("WREG0: page register")
        wreg = 0 << 25
        wregval = wreg + (col << 6)
        self.sendReg(wregval)

    def send_wreg4(self):
        print("WREG4: triangle parameters & GR")
        wreg = 4 << 25
        wregval = wreg + (self.tri_idx << 24) + (self.dwell_val << 20) + (self.range_val << 16) + self.GR + self.step_val
        self.sendReg(wregval)

    def send_wreg6(self):
        print("WREG6: global parameters: PS, XPT, ARL, NSAMP")
        wreg = 6 << 25
        wregval = wreg + (self.PS << 24) + (self.XPT << 21) + (self.RLDpos << 16) + (self.ARLsense << 12) +(self.RLDneg << 8) + self.NSAMP
        self.sendReg(wregval)

    def send_wreg7(self):
        print("WREG7: global parameters: LED, ST, delays, sequence length, SETT")
        wreg = 7 << 25
        wreg = wreg + (self.LED << 23) + (self.ST << 22) + (self.prop_delay << 18) + (self.card_delay << 14)
        wregval = wreg + (self.seqln << 8) + self.SETT
        self.sendReg(wregval)

    def send_GPI4(self):
        print("GPI4: test mode select")
        wreg = 4 << 17
        wregval = wreg + self.TP
        self.sendReg(wregval)

    def send_GPI5(self):
        print("GPI5: test pattern hi-bytes [31..16]")
        if self.TP == 1:
            hibytes = 0x5555
        if self.TP ==2:
            hibytes = 0xaaaa
        if self.TP == 3:
            hibytes = 0x3333
        if self.TP == 4:
            hibytes = 0x0f0f
        if self.TP == 5:
            hibytes = 0x00ff
        if self.TP == 6:
            hibytes = 0x0000
        if self.TP == 7:
            hibytes = 0x0000
        if self.TP == 8:
            hibytes = 0xffff
        if self.TP == 9:
            hibytes = 0xcafe
        wreg = 5 << 17
        wregval = wreg + hibytes
        self.sendReg(wregval)

    def send_GPI6(self):
        print("GPI6: test pattern lo-bytes [15..0]")
        if self.TP == 1:
            lobytes = 0x5555
        if self.TP ==2:
            lobytes = 0xaaaa
        if self.TP == 3:
            lobytes = 0x3333
        if self.TP == 4:
            lobytes = 0x0f0f
        if self.TP == 5:
            lobytes = 0x00ff
        if self.TP == 6:
            lobytes = 0xffff
        if self.TP == 7:
            lobytes = 0x0000
        if self.TP == 8:
            lobytes = 0xffff
        if self.TP == 9:
            lobytes = 0xf00d
        wreg = 6 << 17
        wregval = wreg + lobytes
        self.sendReg(wregval)

    def sendReg(self, wregval):
        print(self.COMMAND + "send", self.BOLD, wregval, self.ENDC)
        b0 = (wregval & 0x7f ) << 1			# 1st 7 bits shifted up 1
        b1 = ((wregval >> 7) & 0x7f) <<  1	 # 2nd 7 bits shifted up 1
        b2 = ((wregval >> 14) & 0x7f) << 1	 # 3rd 7 bits shifted up 1
        b3 = ((wregval >> 21) & 0x7f) << 1	 # 4th 7 bits shifted up 1
        b4 = (self.address << 1) + 1		   # Address shifted up 1 bit with address bit set
        msg = struct.pack('BBBBB', b0, b1, b2, b3, b4)
        self.serialport.write(msg)
        time.sleep(0.001)


def main():

    app = QtGui.QApplication(sys.argv)
    app.setStyle("plastique")
    app.setStyleSheet("""	QPushbutton{font: 10px; padding: 6px}
                            QToolButton{font: 10px; padding: 6px}
                            QLineEdit {background-color: #FFFFCC;}""")
    win = dfbx2(addr=addr, slot=slot, seqln=seqln)
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
    p.set_defaults(addr=3)
    p.set_defaults(slot=3)
    p.set_defaults(seqln=4)
    opt, args = p.parse_args()
# 	ctype = opt.ctype
    addr = opt.addr
    slot = opt.slot
    seqln = opt.seqln
    main()


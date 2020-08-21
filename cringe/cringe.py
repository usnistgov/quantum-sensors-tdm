# -*- coding: utf-8 -*-
import sys
import optparse
import argparse
import struct
import time
import pickle
import json
import os
import zmq

import IPython  # ADDED JG

from PyQt5 import QtGui, QtCore, QtWidgets, QtNetwork
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import named_serial

from .DFBx2.dfbcard import dfbcard
from .BADASS.badcard import badcard
from .DFBx2.dfbclkcard import dfbclkcard
from .emu_card import EMU_Card
from .DFBx2.dfbscard import dfbscard
from .tune.tunetab import TuneTab
from .tower import towerwidget
from .calibration.caltab import CalTab
from cringe.shared import terminal_colors as tc
from cringe.shared import log, logging
from cringe import zmq_rep


class Cringe(QWidget):
    '''CRate Interface for NextGen Electronics'''

    def __init__(self, parent=None, addr_vector=None, slot_vector=None, class_vector=None,
                 seqln=30, lsync=40, tower_vector=None, argfilename=None, calibrationtab=False):

        super(Cringe, self).__init__()
        self.setWindowIcon(QIcon("cringe_img.jpg"))

        self.serialport = named_serial.Serial(port='rack', shared=True)
        self.seqln_timer = None
        self.lsync_timer = None
        self.dfb_delay_timer = None
        self.bad_delay_timer = None
        self.prop_delay_timer = None
        self.NSAMP_delay_timer = None
        self.SETT_delay_timer = None
        self.ARLsense_timer = None
        self.RLDpos_timer = None
        self.RLDneg_timer = None

        self.states = 64

        # 		self.address = addr
        # 		self.slot = slot
        self.seqln = seqln
        self.lsync = lsync
        self.last_lsync = lsync
        self.frame_period = self.lsync * self.seqln * 0.008
        self.locked = False
        self.scale_factor = 1400  # will be redefined later, hopefully this value doesnt matter

        self.saveGlobals = {}
        self.saveClassParameters = {}
        self.loadGlobals = {}
        # 		self.saveFilename = None

        '''global booleans'''

        self.PS = False

        # 		self.LED = 0
        # 		self.ST = 0
        # 		self.GR = 0

        '''global variables'''

        self.dfbclk_XPT = 5
        self.dfbx2_XPT = 0
        self.NSAMP = 4
        self.prop_delay = 3
        self.dfb_delay = 0
        self.seqln = seqln
        self.SETT = 12
        self.TP = 0

        '''ARL default parameters'''

        self.ARLsense = 1024
        self.RLDpos = 64
        self.RLDneg = 4
        self.RLD_track_state = True
        self.RLDpos_delay = self.RLDpos * self.frame_period
        self.RLDneg_delay = self.RLDneg * self.frame_period

        '''triangle default parameters'''

        self.dwell_val = 0
        self.dwellDACunits = float(1)
        self.range_val = 10
        self.rangeDACunits = float(1024)
        self.step_val = 8
        self.stepDACunits = float(256)
        self.tri_idx = False

        '''BAD16 class globals'''

        self.bad_delay = 5
        # 		self.bad_wreg0 = (0 << 25) | (self.ST << 16) | (self.LED << 14) | (self.bad_delay << 10) | self.seqln

        '''DFB class global registers'''
        self.dfb_wreg4 = 0
        self.dfb_wreg6 = 0
        self.dfb_wreg7 = 0

        '''card global default variables'''

        self.mode = 1
        self.CLK = True

        self.slot_vector = slot_vector
        self.addr_vector = addr_vector
        self.class_vector = class_vector
        self.tower_vector = tower_vector
        # 		self.slot_vector = [1,3,10]
        # 		self.addr_vector = [1,3,32]
        # 		self.class_vector = ['DFBCLK', 'DFBx2','BAD16']
        # 		self.slot_vector = [1,3,4,5,6,10,11]
        # 		self.addr_vector = [1,3,5,7,9,32,33]
        # 		self.class_vector = ['DFBCLK', 'DFBx2', 'DFBx2', 'DFBx2', 'DFBx2', 'BAD16', 'BAD16']
        self.crate_widgets = []
        # 		self.enb = [0,0,0,0,0,0,0]
        # 		self.cal_coeffs = [0,0,0,0,0,0,0]
        # 		self.appTrim =[0,0,0,0,0,0,0]

        self.setWindowTitle("CRINGE")  # Phase Offset Widget
        self.setGeometry(30, 30, 800, 1000)
        self.setContentsMargins(0, 0, 0, 0)
        # 		self.setFixedWidth(1400)

        self.layout_widget = QWidget(self)
        self.layout = QGridLayout(self)

        log.debug(tc.INIT + tc.BOLD + "building GUI" + tc.ENDC)

        '''
        build widget for file management controls
        '''
        self.file_mgmt_widget = QGroupBox(self)
        # 		self.file_mgmt_widget.setFlat(1)
        self.file_mgmt_widget.setFixedWidth(1080)
        self.file_mgmt_widget.setFocusPolicy(QtCore.Qt.NoFocus)
        self.file_mgmt_widget.setTitle("FILE MANAGEMENT INTERFACE")

        self.file_mgmt_layout = QGridLayout(self.file_mgmt_widget)
        self.file_mgmt_layout.setContentsMargins(5, 5, 5, 5)
        self.file_mgmt_layout.setSpacing(5)

        self.loadsetup = QPushButton(self, text="load setup")
        self.loadsetup.setFixedHeight(25)
        self.loadsetup.setEnabled(1)
        self.file_mgmt_layout.addWidget(
            self.loadsetup, 0, 0, 1, 1, QtCore.Qt.AlignLeft)
        self.loadsetup.clicked.connect(self.loadSettings)

        self.savesetup = QPushButton(self, text="save setup")
        self.savesetup.setFixedHeight(25)
        self.savesetup.setEnabled(1)
        self.file_mgmt_layout.addWidget(
            self.savesetup, 0, 1, 1, 1, QtCore.Qt.AlignLeft)
        self.savesetup.clicked.connect(self.saveSettings)

        self.sendsetup = QPushButton(self, text="re-assert setup")
        self.sendsetup.setFixedHeight(25)
        self.sendsetup.setEnabled(1)
        self.file_mgmt_layout.addWidget(
            self.sendsetup, 0, 2, 1, 1, QtCore.Qt.AlignLeft)
        self.sendsetup.clicked.connect(self.assertSettings)

        self.filenameEdit = QLineEdit()
        self.filenameEdit.setReadOnly(True)
        self.file_mgmt_layout.addWidget(self.filenameEdit, 0, 4, 1, 4)

        self.filename_label = QLabel("file")
        self.file_mgmt_layout.addWidget(
            self.filename_label, 0, 3, 1, 1, QtCore.Qt.AlignRight)

        self.layout.addWidget(self.file_mgmt_widget, 0, 0, 1, 2)

        '''
        build widget for SYSTEM GLOBALS header
        '''
        self.sys_glob_hdr_widget = QGroupBox(self)
        self.sys_glob_hdr_widget.setFixedWidth(1080)
        self.sys_glob_hdr_widget.setFocusPolicy(QtCore.Qt.NoFocus)
        self.sys_glob_hdr_widget.setTitle("SYSTEM GLOBALS")

        self.sys_glob_layout = QGridLayout(self.sys_glob_hdr_widget)
        self.sys_glob_layout.setContentsMargins(5, 5, 5, 5)
        self.sys_glob_layout.setSpacing(5)

        self.seqln_spin = QSpinBox()
        self.seqln_spin.setRange(4, 64)
        # 		self.seqln_spin.setFixedWidth(45)
        self.seqln_spin.setSingleStep(1)
        self.seqln_spin.setKeyboardTracking(0)
        self.seqln_spin.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.seqln_spin.setValue(self.seqln)
        self.seqln_spin.setAlignment(QtCore.Qt.AlignRight)
        self.sys_glob_layout.addWidget(self.seqln_spin, 0, 0, 1, 1)
        self.seqln_spin.valueChanged.connect(self.seqln_changed)

        self.seqln_lbl = QLabel("sequence length")
        # 		self.seqln_lbl.setAlignment(QtCore.Qt.AlignLeft)
        self.sys_glob_layout.addWidget(
            self.seqln_lbl, 0, 1, 1, 1, QtCore.Qt.AlignLeft)

        self.lsync_spin = QSpinBox()
        self.lsync_spin.setRange(20, 256)
        # 		self.lsync_spin.setFixedWidth(45)
        self.lsync_spin.setSingleStep(1)
        self.lsync_spin.setKeyboardTracking(0)
        self.lsync_spin.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.lsync_spin.setValue(self.lsync)
        self.lsync_spin.setAlignment(QtCore.Qt.AlignRight)
        self.sys_glob_layout.addWidget(self.lsync_spin, 0, 2, 1, 1)
        self.lsync_spin.valueChanged.connect(self.lsync_changed)

        self.seqln_lbl = QLabel("line period")
        self.sys_glob_layout.addWidget(
            self.seqln_lbl, 0, 3, 1, 2, QtCore.Qt.AlignLeft)

        if log.verbosity >= logging.VERBOSITY_DEBUG:
            self.debug_full_tune_button = QPushButton("debug: extern tune")
            self.debug_full_tune_button.setFixedHeight(25)
            self.sys_glob_layout.addWidget(
                self.debug_full_tune_button, 0, 3, 1, 2, QtCore.Qt.AlignLeft)
            self.debug_full_tune_button.clicked.connect(self.extern_tune)

        self.sys_glob_send = QPushButton(self, text="send system globals")
        self.sys_glob_send.setFixedHeight(25)
        # 		self.sys_glob_send.setFixedWidth(160)
        self.sys_glob_layout.addWidget(
            self.sys_glob_send, 0, 5, 1, 2, QtCore.Qt.AlignRight)
        self.sys_glob_send.clicked.connect(self.send_all_sys_globals)

        self.layout.addWidget(self.sys_glob_hdr_widget, 1,
                              0, 1, 1, QtCore.Qt.AlignLeft)

        '''
        build widget for SYSTEM CONTROL header
        '''
        self.sys_control_hdr_widget = QGroupBox(self)
        self.sys_control_hdr_widget.setFixedWidth(1080)
        self.sys_control_hdr_widget.setFocusPolicy(QtCore.Qt.NoFocus)
        self.sys_control_hdr_widget.setTitle("SYSTEM CONTROL")

        self.sys_control_layout = QGridLayout(self.sys_control_hdr_widget)
        self.sys_control_layout.setContentsMargins(5, 5, 5, 5)
        self.sys_control_layout.setSpacing(5)

        self.crate_power = QToolButton(self, text='crate power ON')
        self.crate_power.setFixedHeight(25)
        self.crate_power.setCheckable(1)
        self.crate_power.setChecked(1)
        self.crate_power.setStyleSheet("background-color: #" + tc.green + ";")
        # 		self.crate_power.setFixedWidth(160)
        self.sys_control_layout.addWidget(
            self.crate_power, 0, 0, 1, 1, QtCore.Qt.AlignLeft)
        # 		self.crate_power.setEnabled(0)
        self.crate_power.toggled.connect(self.cratePower)

        self.server_lock = QToolButton(self, text="server LOCK OFF")
        self.server_lock.setToolTip(
            "engage when SERVER is running to prevent changing of critical parameters")
        self.server_lock.setFixedHeight(25)
        self.server_lock.setCheckable(1)
        self.server_lock.setChecked(self.locked)
        self.server_lock.setStyleSheet("background-color: #" + tc.green + ";")
        # 		self.resync_system.setFixedWidth(160)
        self.sys_control_layout.addWidget(
            self.server_lock, 0, 1, 1, 1)  # , QtCore.Qt.AlignLeft)
        self.server_lock.toggled.connect(self.lockServer)

        self.send_all_globals = QPushButton(self, text="send globals")
        self.send_all_globals.setFixedHeight(25)
        # 		self.send_all_globals.setFixedWidth(160)
        self.sys_control_layout.addWidget(
            self.send_all_globals, 0, 2, 1, 1)  # , QtCore.Qt.AlignLeft)
        self.send_all_globals.clicked.connect(self.send_ALL_globals)

        self.send_all_states_chns = QPushButton(self, text="send arrayed")
        self.send_all_states_chns.setFixedHeight(25)
        # 		self.send_all_states_chns.setFixedWidth(160)
        self.sys_control_layout.addWidget(
            self.send_all_states_chns, 0, 3, 1, 1)  # , QtCore.Qt.AlignLeft)
        self.send_all_states_chns.clicked.connect(self.send_ALL_states_chns)

        self.cal_system = QPushButton(self, text="CALIBRATE")
        self.cal_system.setFixedHeight(25)
        # 		self.cal_system.setFixedWidth(160)
        self.sys_control_layout.addWidget(
            self.cal_system, 0, 4, 1, 1)  # , QtCore.Qt.AlignLeft)
        self.cal_system.clicked.connect(self.phcal_system)

        self.resync_system = QPushButton(self, text="RESYNC")
        self.resync_system.setFixedHeight(25)
        # 		self.resync_system.setFixedWidth(160)
        self.sys_control_layout.addWidget(
            self.resync_system, 0, 5, 1, 1)  # , QtCore.Qt.AlignLeft)
        self.resync_system.clicked.connect(self.system_resync)

        self.full_init_button = QPushButton(self, text="init")
        self.full_init_button.setFixedHeight(25)
        self.sys_control_layout.addWidget(self.full_init_button, 0, 6, 1, 1)
        self.full_init_button.clicked.connect(self.full_crate_init)

        self.layout.addWidget(self.sys_control_hdr_widget,
                              1, 1, 1, 1, QtCore.Qt.AlignRight)

        '''
        build widget for CLASS GLOBALS header
        '''
        self.class_glob_hdr_widget = QGroupBox(self)
        self.class_glob_hdr_widget.setFixedWidth(1080)
        self.class_glob_hdr_widget.setFocusPolicy(QtCore.Qt.NoFocus)
        self.class_glob_hdr_widget.setTitle("DFB/BAD CLASS GLOBALS")

        self.class_glob_layout = QGridLayout(self.class_glob_hdr_widget)
        self.class_glob_layout.setContentsMargins(5, 5, 5, 5)
        self.class_glob_layout.setSpacing(5)

        self.SETT_spin = QSpinBox()
        self.SETT_spin.setRange(0, 255)
        self.SETT_spin.setSingleStep(1)
        self.SETT_spin.setKeyboardTracking(0)
        self.SETT_spin.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.SETT_spin.setValue(self.SETT)
        self.SETT_spin.setAlignment(QtCore.Qt.AlignRight)
        self.class_glob_layout.addWidget(self.SETT_spin, 0, 0, 1, 1)
        self.SETT_spin.valueChanged.connect(self.SETT_changed)

        self.SETT_spin_lbl = QLabel("SETT")
        self.class_glob_layout.addWidget(
            self.SETT_spin_lbl, 0, 1, 1, 1, QtCore.Qt.AlignLeft)

        self.NSAMP_spin = QSpinBox()
        self.NSAMP_spin.setRange(0, 255)
        self.NSAMP_spin.setSingleStep(1)
        self.NSAMP_spin.setKeyboardTracking(0)
        self.NSAMP_spin.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.NSAMP_spin.setAlignment(QtCore.Qt.AlignRight)
        self.NSAMP_spin.setValue(self.NSAMP)
        self.class_glob_layout.addWidget(self.NSAMP_spin, 1, 0, 1, 1)
        self.NSAMP_spin.valueChanged.connect(self.NSAMP_changed)

        self.NSAMP_spin_lbl = QLabel("NSAMP (SETT+NSAMP+2<=line period)")
        self.class_glob_layout.addWidget(
            self.NSAMP_spin_lbl, 1, 1, 1, 1, QtCore.Qt.AlignLeft)

        self.prop_delay_spin = QSpinBox()
        self.prop_delay_spin.setRange(0, 15)
        self.prop_delay_spin.setSingleStep(1)
        self.prop_delay_spin.setKeyboardTracking(0)
        self.prop_delay_spin.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.prop_delay_spin.setValue(self.prop_delay)
        self.prop_delay_spin.setAlignment(QtCore.Qt.AlignRight)
        self.class_glob_layout.addWidget(self.prop_delay_spin, 0, 2, 1, 1)
        self.prop_delay_spin.valueChanged.connect(self.prop_delay_changed)

        self.prop_delay_lbl = QLabel("DFB prop delay")
        self.class_glob_layout.addWidget(
            self.prop_delay_lbl, 0, 3, 1, 1, QtCore.Qt.AlignLeft)

        self.dfb_delay_spin = QSpinBox()
        self.dfb_delay_spin.setRange(0, 15)
        self.dfb_delay_spin.setSingleStep(1)
        self.dfb_delay_spin.setKeyboardTracking(0)
        self.dfb_delay_spin.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.dfb_delay_spin.setValue(self.dfb_delay)
        self.dfb_delay_spin.setAlignment(QtCore.Qt.AlignRight)
        self.class_glob_layout.addWidget(self.dfb_delay_spin, 0, 4, 1, 1)
        self.dfb_delay_spin.valueChanged.connect(self.dfb_delay_changed)

        self.card_delay_lbl = QLabel("DFB card delay")
        self.class_glob_layout.addWidget(
            self.card_delay_lbl, 0, 5, 1, 1, QtCore.Qt.AlignLeft)

        self.bad_delay_spin = QSpinBox()
        self.bad_delay_spin.setRange(0, 15)
        self.bad_delay_spin.setSingleStep(1)
        self.bad_delay_spin.setKeyboardTracking(0)
        self.bad_delay_spin.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.bad_delay_spin.setValue(self.bad_delay)
        self.bad_delay_spin.setAlignment(QtCore.Qt.AlignRight)
        self.class_glob_layout.addWidget(self.bad_delay_spin, 1, 4, 1, 1)
        self.bad_delay_spin.valueChanged.connect(self.bad_delay_changed)

        self.card_delay_lbl = QLabel("BAD16 card delay")
        self.class_glob_layout.addWidget(
            self.card_delay_lbl, 1, 5, 1, 1, QtCore.Qt.AlignLeft)

        self.dfbx2_xpt_mode = QComboBox()
        self.dfbx2_xpt_mode.setFixedHeight(25)
        self.dfbx2_xpt_mode.addItem('0: A-C-B-D')
        self.dfbx2_xpt_mode.addItem('1: C-A-D-B')
        self.dfbx2_xpt_mode.addItem('2: B-D-A-C')
        self.dfbx2_xpt_mode.addItem('3: D-B-C-A')
        self.dfbx2_xpt_mode.addItem('4: A-B-C-D')
        self.dfbx2_xpt_mode.addItem('5: C-D-A-B')
        self.dfbx2_xpt_mode.addItem('6: B-A-D-C')
        self.dfbx2_xpt_mode.addItem('7: D-C-B-A')
        self.class_glob_layout.addWidget(self.dfbx2_xpt_mode, 0, 6, 1, 1)
        self.dfbx2_xpt_mode.currentIndexChanged.connect(self.dfbx2_XPT_changed)

        self.status_lbl = QLabel("DFBx2 XPT mode")
        self.class_glob_layout.addWidget(
            self.status_lbl, 0, 7, 1, 1, QtCore.Qt.AlignLeft)

        self.tp_mode = QComboBox()
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
        self.tp_mode.addItem('8BADF00D')
        self.class_glob_layout.addWidget(self.tp_mode, 0, 8, 1, 1)
        self.tp_mode.currentIndexChanged.connect(self.TP_changed)

        self.status_lbl = QLabel("test pattern")
        self.class_glob_layout.addWidget(
            self.status_lbl, 0, 9, 1, 1, QtCore.Qt.AlignLeft)

        self.class_glb_send = QPushButton(self, text="send class globals")
        self.class_glb_send.setFixedHeight(25)
        self.class_glb_send.setFixedWidth(200)
        self.class_glob_layout.addWidget(
            self.class_glb_send, 0, 10, 1, 1, QtCore.Qt.AlignLeft)
        self.class_glb_send.clicked.connect(self.send_all_class_globals)

        self.dfbclk_xpt_mode = QComboBox()
        self.dfbclk_xpt_mode.setFixedHeight(25)
        self.dfbclk_xpt_mode.setToolTip(
            "MCLK & LSYNC are hard wired to data pipes 1 & 2, choose mode 5 to get CH1 data on A/B")
        self.dfbclk_xpt_mode.addItem('0: FANOUT')
        self.dfbclk_xpt_mode.addItem('1: BKPLN')
        self.dfbclk_xpt_mode.addItem('2: SYNC25')
        self.dfbclk_xpt_mode.addItem('3: SYNC10')
        self.dfbclk_xpt_mode.addItem('4: M-L-C-D')
        self.dfbclk_xpt_mode.addItem('5: M-L-A-B')
        self.dfbclk_xpt_mode.addItem('6: M-L-D-C')
        self.dfbclk_xpt_mode.addItem('7: M-L-B-A')
        self.dfbclk_xpt_mode.setToolTip(
            "FANOUT = M-L-M-L, BKPLN = M-L-LSYNC-FRSYNC, SYNC25 = M-L-25-25, SYNC10 = M-L-10-10")
        self.dfbclk_xpt_mode.setCurrentIndex(5)
        self.class_glob_layout.addWidget(self.dfbclk_xpt_mode, 1, 6, 1, 1)
        self.dfbclk_xpt_mode.currentIndexChanged.connect(
            self.dfbclk_XPT_changed)

        self.status_lbl = QLabel("DFBCLK XPT mode")
        self.class_glob_layout.addWidget(
            self.status_lbl, 1, 7, 1, 1, QtCore.Qt.AlignLeft)

        self.PS_button = QToolButton(self, text='PS')
        self.PS_button.setFixedHeight(25)
        self.PS_button.setCheckable(1)
        self.PS_button.setChecked(self.PS)
        self.PS_button.setStyleSheet("background-color: #" + tc.red + ";")
        self.class_glob_layout.addWidget(
            self.PS_button, 1, 8, 1, 1, QtCore.Qt.AlignRight)
        self.PS_button.toggled.connect(self.PS_changed)

        self.status_lbl = QLabel("parallel stream")
        self.class_glob_layout.addWidget(
            self.status_lbl, 1, 9, 1, 1, QtCore.Qt.AlignLeft)

        self.layout.addWidget(self.class_glob_hdr_widget, 2, 0, 1, 2)

        '''
        build widget for ARL control
        '''
        self.arl_widget = QGroupBox(self)
        # 		self.tri_wvfm_widget.setFixedHeight(25)
        self.arl_widget.setTitle("DFB AUTO RELOCK CONTROL")
        self.arl_layout = QGridLayout(self.arl_widget)
        self.arl_layout.setContentsMargins(5, 5, 5, 5)
        self.arl_layout.setSpacing(5)

        ''' flux jump threshold '''

        self.ARLsense_title = QLabel("flux jump threshold")
        self.arl_layout.addWidget(
            self.ARLsense_title, 0, 0, 1, 1, QtCore.Qt.AlignRight)

        # 		self.ARLsense_vernier = QRadioButton(self, text = "vernier")
        # 		self.ARLsense_vernier.setCheckable(1)
        # 		self.ARLsense_vernier.setChecked(0)
        # 		self.arl_layout.addWidget(self.ARLsense_vernier,1,0,1,1,QtCore.Qt.AlignLeft)

        self.ARLsense_spin = QSpinBox()
        self.ARLsense_spin.setRange(0, 16383)
        self.ARLsense_spin.setFixedHeight(25)
        self.ARLsense_spin.setSingleStep(256)
        self.ARLsense_spin.setKeyboardTracking(0)
        self.ARLsense_spin.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.ARLsense_spin.setValue(self.ARLsense)
        self.arl_layout.addWidget(
            self.ARLsense_spin, 1, 0, 1, 1, QtCore.Qt.AlignRight)
        self.ARLsense_spin.valueChanged.connect(self.ARLsense_changed)

        # 		self.ARLsense_lbl = QLabel("2^N index")
        # 		self.arl_layout.addWidget(self.ARLsense_lbl,1,2,1,1,QtCore.Qt.AlignLeft)

        # 		self.ARLsense_indicator = QLineEdit()
        # 		self.ARLsense_indicator.setReadOnly(True)
        # 		self.ARLsense_indicator.setFixedHeight(25)
        # 		self.ARLsense_indicator.setText('%5i'%(self.ARLsense))
        # 		self.ARLsense_indicator.setAlignment(QtCore.Qt.AlignRight)
        # 		self.ARLsense_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
        # 		self.arl_layout.addWidget(self.ARLsense_indicator, 2,0,1,2,QtCore.Qt.AlignRight)

        self.ARLsense_indicator_lbl = QLabel("DAC units")
        self.arl_layout.addWidget(
            self.ARLsense_indicator_lbl, 1, 1, 1, 1, QtCore.Qt.AlignLeft)

        self.ARLsense_eng_indicator = QLineEdit()
        self.ARLsense_eng_indicator.setReadOnly(True)
        self.ARLsense_eng_indicator.setFixedHeight(25)
        self.ARLsense_eng_indicator.setText(str((self.ARLsense)/16.383)[:6])
        self.ARLsense_eng_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.ARLsense_eng_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
        self.arl_layout.addWidget(
            self.ARLsense_eng_indicator, 2, 0, 1, 1, QtCore.Qt.AlignRight)

        self.ARLsense_eng_indicator_lbl = QLabel("mV")
        self.arl_layout.addWidget(
            self.ARLsense_eng_indicator_lbl, 2, 1, 1, 1, QtCore.Qt.AlignLeft)

        ''' [+] event reset delay '''

        self.RLDpos_title = QLabel("[+] event reset delay")
        self.arl_layout.addWidget(
            self.RLDpos_title, 0, 2, 1, 1, QtCore.Qt.AlignRight)

        self.RLDpos_spin = QSpinBox()
        self.RLDpos_spin.setRange(1, 65535)
        self.RLDpos_spin.setFixedHeight(25)
        self.RLDpos_spin.setSingleStep(16)
        self.RLDpos_spin.setKeyboardTracking(0)
        self.RLDpos_spin.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.RLDpos_spin.setValue(self.RLDpos)
        self.arl_layout.addWidget(
            self.RLDpos_spin, 1, 2, 1, 1, QtCore.Qt.AlignRight)
        self.RLDpos_spin.valueChanged.connect(self.RLDpos_changed)

        self.RLDpos_indicator_lbl = QLabel("FRM units")
        self.arl_layout.addWidget(
            self.RLDpos_indicator_lbl, 1, 3, 1, 1, QtCore.Qt.AlignLeft)

        self.RLDpos_eng_indicator = QLineEdit()
        self.RLDpos_eng_indicator.setReadOnly(True)
        self.RLDpos_eng_indicator.setFixedHeight(25)
        self.RLDpos_eng_indicator.setText(
            str((self.RLDpos)*self.frame_period)[:6])
        self.RLDpos_eng_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.RLDpos_eng_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
        self.arl_layout.addWidget(
            self.RLDpos_eng_indicator, 2, 2, 1, 1, QtCore.Qt.AlignRight)
        # 		self.RLDpos_eng_indicator.textChanged.connect(self.RLDwarning)

        self.RLDpos_eng_indicator_lbl = QLabel("\u00B5s")
        self.arl_layout.addWidget(
            self.RLDpos_eng_indicator_lbl, 2, 3, 1, 1, QtCore.Qt.AlignLeft)

        ''' [-] event reset delay '''

        self.RLDneg_title = QLabel("[-] event reset delay")
        self.arl_layout.addWidget(
            self.RLDneg_title, 0, 4, 1, 1, QtCore.Qt.AlignRight)

        self.RLDneg_spin = QSpinBox()
        self.RLDneg_spin.setRange(1, 65535)
        self.RLDneg_spin.setFixedHeight(25)
        self.RLDneg_spin.setSingleStep(16)
        self.RLDneg_spin.setKeyboardTracking(0)
        self.RLDneg_spin.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.RLDneg_spin.setValue(self.RLDneg)
        self.arl_layout.addWidget(
            self.RLDneg_spin, 1, 4, 1, 1, QtCore.Qt.AlignRight)
        self.RLDneg_spin.valueChanged.connect(self.RLDneg_changed)

        self.RLDneg_indicator_lbl = QLabel("FRM units")
        self.arl_layout.addWidget(
            self.RLDneg_indicator_lbl, 1, 5, 1, 1, QtCore.Qt.AlignLeft)

        self.RLDneg_eng_indicator = QLineEdit()
        self.RLDneg_eng_indicator.setReadOnly(True)
        self.RLDneg_eng_indicator.setFixedHeight(25)
        self.RLDneg_eng_indicator.setText(
            str(self.RLDneg*self.frame_period)[:6])
        self.RLDneg_eng_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.RLDneg_eng_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
        self.arl_layout.addWidget(
            self.RLDneg_eng_indicator, 2, 4, 1, 1, QtCore.Qt.AlignRight)
        # 		self.RLDneg_eng_indicator.textChanged.connect(self.RLDwarning)

        self.RLDneg_eng_indicator_lbl = QLabel("\u00B5s")
        self.arl_layout.addWidget(
            self.RLDneg_eng_indicator_lbl, 2, 5, 1, 1, QtCore.Qt.AlignLeft)

        ''' track control '''

        self.RLD_track = QLabel("track")
        self.arl_layout.addWidget(
            self.RLD_track, 0, 6, 1, 1, QtCore.Qt.AlignLeft)

        self.RLD_frame = QRadioButton(self, text="frame")
        self.RLD_frame.setCheckable(1)
        self.RLD_frame.setChecked(self.RLD_track_state)
        self.arl_layout.addWidget(
            self.RLD_frame, 1, 6, 1, 1, QtCore.Qt.AlignLeft)
        self.RLD_frame.clicked.connect(self.track_changed)

        self.RLD_time = QRadioButton(self, text="time")
        self.RLD_time.setCheckable(1)
        self.RLD_time.setChecked(not(self.RLD_track_state))
        self.arl_layout.addWidget(
            self.RLD_time, 2, 6, 1, 1, QtCore.Qt.AlignLeft)
        self.RLD_time.clicked.connect(self.track_changed)

        self.layout.addWidget(self.arl_widget, 3, 0, 1, 1)

        '''
        build widget for Triangle Waveform Generator
        '''
        self.tri_wvfm_widget = QGroupBox(self)
        self.tri_wvfm_widget.setTitle("DFB/BAD TRIANGLE WAVEFORM GENERATOR")
        self.tri_wvfm_layout = QGridLayout(self.tri_wvfm_widget)
        self.tri_wvfm_layout.setContentsMargins(5, 5, 5, 5)
        self.tri_wvfm_layout.setSpacing(5)

        self.dwell = QSpinBox()
        self.dwell.setRange(0, 15)
        self.dwell.setFixedHeight(25)
        self.dwell.setSingleStep(1)
        self.dwell.setKeyboardTracking(0)
        self.dwell.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.dwell.setValue(self.dwell_val)
        self.tri_wvfm_layout.addWidget(
            self.dwell, 0, 0, 1, 1, QtCore.Qt.AlignRight)
        self.dwell.valueChanged.connect(self.dwell_changed)

        self.dwell_lbl = QLabel("dwell (2^N)")
        self.tri_wvfm_layout.addWidget(
            self.dwell_lbl, 0, 1, 1, 1, QtCore.Qt.AlignLeft)

        self.range = QSpinBox()
        self.range.setRange(1, 14)
        self.range.setFixedHeight(25)
        self.range.setSingleStep(1)
        self.range.setKeyboardTracking(0)
        self.range.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.range.setValue(self.range_val)
        self.tri_wvfm_layout.addWidget(
            self.range, 0, 2, 1, 1, QtCore.Qt.AlignRight)
        self.range.valueChanged.connect(self.range_changed)

        self.range_lbl = QLabel("steps (2^N)")
        self.tri_wvfm_layout.addWidget(
            self.range_lbl, 0, 3, 1, 1, QtCore.Qt.AlignLeft)

        self.step = QSpinBox()
        self.step.setRange(1, 16383)
        self.step.setFixedHeight(25)
        self.step.setSingleStep(1)
        self.step.setKeyboardTracking(0)
        self.step.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.step.setValue(self.step_val)
        self.tri_wvfm_layout.addWidget(
            self.step, 0, 4, 1, 1, QtCore.Qt.AlignRight)
        self.step.valueChanged.connect(self.step_changed)

        self.step_lbl = QLabel("step size")
        self.tri_wvfm_layout.addWidget(
            self.step_lbl, 0, 5, 1, 1, QtCore.Qt.AlignLeft)

        self.period_indicator = QLineEdit()
        self.period_indicator.setReadOnly(True)
        # 		self.period_indicator.setFixedWidth(120)
        self.period_indicator.setText(
            str(2*(2**self.dwell_val)*(2**self.range_val)))
        self.period_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.period_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
        self.tri_wvfm_layout.addWidget(
            self.period_indicator, 2, 0, 1, 1, QtCore.Qt.AlignRight)

        self.period_indicator_lbl = QLabel("period")
        self.tri_wvfm_layout.addWidget(
            self.period_indicator_lbl, 2, 1, 1, 1, QtCore.Qt.AlignLeft)

        self.period_eng_indicator = QLineEdit()
        self.period_eng_indicator.setReadOnly(True)
        # 		self.period_eng_indicator.setFixedWidth(120)
        # 		self.period_eng_indicator.setText(str(2*(2**self.dwell_val)*(2**self.range_val)))
        self.period_eng_indicator.setText(
            str(int(self.period_indicator.text())*self.lsync*0.008))
        self.period_eng_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.period_eng_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
        self.tri_wvfm_layout.addWidget(
            self.period_eng_indicator, 3, 0, 1, 1, QtCore.Qt.AlignRight)

        self.period_eng_indicator_lbl = QLabel("period [""\u00B5s]")
        self.tri_wvfm_layout.addWidget(
            self.period_eng_indicator_lbl, 3, 1, 1, 1, QtCore.Qt.AlignLeft)

        self.amp_indicator = QLineEdit()
        self.amp_indicator.setReadOnly(True)
        # 		self.amp_indicator.setFixedWidth(80)
        self.amp_indicator.setText(str((2**self.range_val)*self.step_val))
        self.amp_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.amp_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
        self.tri_wvfm_layout.addWidget(
            self.amp_indicator, 2, 2, 1, 1, QtCore.Qt.AlignRight)

        self.amp_indicator_lbl = QLabel("amplitude")
        self.tri_wvfm_layout.addWidget(
            self.amp_indicator_lbl, 2, 3, 1, 1, QtCore.Qt.AlignLeft)

        self.amp_eng_indicator = QLineEdit()
        self.amp_eng_indicator.setReadOnly(True)
        # 		self.amp_eng_indicator.setFixedWidth(80)
        self.amp_eng_indicator.setText(
            str(int(self.amp_indicator.text())/16.383)[:6])
        self.amp_eng_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.amp_eng_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
        self.tri_wvfm_layout.addWidget(
            self.amp_eng_indicator, 3, 2, 1, 1, QtCore.Qt.AlignRight)

        self.amp_eng_indicator_lbl = QLabel("amplitude [mV]")
        self.tri_wvfm_layout.addWidget(
            self.amp_eng_indicator_lbl, 3, 3, 1, 1, QtCore.Qt.AlignLeft)

        self.tri_idx_button = QToolButton(self, text='LSYNC')
        self.tri_idx_button.setFixedHeight(25)
        self.tri_idx_button.setCheckable(1)
        self.tri_idx_button.setChecked(self.tri_idx)
        self.tri_idx_button.setStyleSheet("background-color: #" + tc.red + ";")
        self.tri_wvfm_layout.addWidget(
            self.tri_idx_button, 0, 6, 1, 1, QtCore.Qt.AlignRight)
        self.tri_idx_button.toggled.connect(self.tri_idx_changed)

        self.tri_idx_lbl = QLabel("timebase")
        self.tri_wvfm_layout.addWidget(
            self.tri_idx_lbl, 0, 7, 1, 1, QtCore.Qt.AlignLeft)

        self.tri_send = QPushButton(self, text="send triangle")
        self.tri_send.setFixedHeight(25)
        self.tri_send.setFixedWidth(200)
        self.tri_wvfm_layout.addWidget(
            self.tri_send, 2, 4, 1, 4, QtCore.Qt.AlignRight)
        self.tri_send.clicked.connect(self.send_triangle)

        self.freq_eng_indicator = QLineEdit()
        self.freq_eng_indicator.setReadOnly(True)
        # 		self.freq_eng_indicator.setFixedWidth(80)
        self.freq_eng_indicator.setText(
            str(1000/float(self.period_eng_indicator.text()))[:6])
        self.freq_eng_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.freq_eng_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
        self.tri_wvfm_layout.addWidget(
            self.freq_eng_indicator, 3, 4, 1, 1, QtCore.Qt.AlignRight)

        self.amp_eng_indicator_lbl = QLabel("freq [kHz]")
        self.tri_wvfm_layout.addWidget(
            self.amp_eng_indicator_lbl, 3, 5, 1, 1, QtCore.Qt.AlignLeft)

        self.layout.addWidget(self.tri_wvfm_widget, 3, 1,
                              1, 1, QtCore.Qt.AlignRight)

        '''
        build tab widget for crate cards
        '''
        self.crate_widget = QTabWidget(self)
        self.crate_widget.setTabShape(1)

        for idx, val in enumerate(self.class_vector):
            if val == "DFBCLK":
                self.card_widget = dfbclkcard(
                    parent=self, addr=self.addr_vector[idx], slot=self.slot_vector[idx], seqln=self.seqln, lsync=self.lsync)
                tab_lbl = " DFBx1CLK: " + \
                    str(self.slot_vector[idx]) + "/" + \
                    str(self.addr_vector[idx]) + " "
                self.scale_factor = self.card_widget.dfbclk_widget1.state_vectors[0].width(
                )
            if val == "DFBx2":
                self.card_widget = dfbcard(
                    parent=self, addr=self.addr_vector[idx], slot=self.slot_vector[idx], seqln=self.seqln, lsync=self.lsync)
                # 				self.card_widget.setStyleSheet("background-color: #" + tc.grey + ";color : #" + tc.white)
                tab_lbl = " DFBx2: " + \
                    str(self.slot_vector[idx]) + "/" + \
                    str(self.addr_vector[idx]) + " "
            # 				tab_lbl.setStyleSheet("background-color: #" + tc.grey + ";color : #" + tc.white)
            # 				self.scale_factor = self.card_widget.dfbx2_widget1.state_vectors[0].width()
            if val == "BAD16":
                self.card_widget = badcard(
                    parent=self, addr=self.addr_vector[idx], slot=self.slot_vector[idx], seqln=self.seqln, lsync=self.lsync)
                tab_lbl = " BAD16: " + \
                    str(self.slot_vector[idx]) + "/" + \
                    str(self.addr_vector[idx]) + " "
            if val == "DFBs":
                self.card_widget = dfbscard(
                    parent=self, addr=self.addr_vector[idx], slot=self.slot_vector[idx], lsync=self.lsync)
                tab_lbl = " DFBscream: " + \
                    str(self.slot_vector[idx]) + "/" + \
                    str(self.addr_vector[idx]) + " "
            self.crate_widgets.append(self.card_widget)
            self.crate_widget.addTab(self.card_widget, tab_lbl)
        self.tune_widget = TuneTab(self)
        self.crate_widget.addTab(self.tune_widget, "Tune")
        self.crate_widgets.append(self.tune_widget)
        if calibrationtab:
            self.cal_widget = CalTab(self)
            self.crate_widget.addTab(self.cal_widget, "Calibration")
        self.crate_widgets.append(self.tune_widget)

        if not self.tower_vector is None:
            log.debug("building tower widget")
            self.tower_widget = towerwidget.TowerWidget(
                parent=self, nameaddrlist=self.tower_vector)
            self.scroll = QScrollArea(self)
            self.scroll.setWidgetResizable(True)
            self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
            self.scroll.setWidget(self.tower_widget)
            self.crate_widget.addTab(self.scroll, "Tower")
        else:
            self.tower_widget = None

        log.debug("code checkpoint 1")

        # 		'''
        # 		create TABs for embedding DFBx1CLK functional widget
        # 		'''
        # 		self.card_widget = dfbclkcard(parent=self, addr=1, slot=1, seqln=self.seqln, lsync=self.lsync)
        # 		self.crate_widgets.append(self.card_widget)
        # 		self.crate_widget.addTab(self.card_widget, "DFBx1CLK: 1/1")
        #
        # 		'''
        # 		create TABs for embedding DFBx2 functional widgets
        # 		'''
        # 		self.card_widget = dfbcard(parent=self, addr=3, slot=3, seqln=self.seqln, lsync=self.lsync)
        # # 		self.scale_factor = self.card_widget.dfbx2_widget1.width()
        # 		self.scale_factor = self.card_widget.dfbx2_widget1.state_vectors[0].width()
        # 		self.crate_widgets.append(self.card_widget)
        # 		self.crate_widget.addTab(self.card_widget, "DFBx2:3/3")
        #
        # 		self.card_widget = dfbcard(parent=self, addr=5, slot=4, seqln=self.seqln, lsync=self.lsync)
        # 		self.crate_widgets.append(self.card_widget)
        # 		self.crate_widget.addTab(self.card_widget, "DFBx2:4/5")
        #
        # 		'''
        # 		create TAB for embedding BAD16 functional widgets
        # 		'''
        # 		self.card_widget = badcard(parent=self, addr=33, slot=10, seqln=self.seqln, lsync=self.lsync)
        # 		self.crate_widgets.append(self.card_widget)
        # 		self.crate_widget.addTab(self.card_widget, "BAD16:10/33")

        '''build crate TAB widget'''
        self.layout.addWidget(self.crate_widget, 4, 0, 1, 2)

        '''
        resize widgets for relative, platform dependent variability
        '''
        # 		self.scale_factor = self.dfbcard_widget1.dfbx2_widget1.width()
        # 		print self.scale_factor
        rm = 90
        self.file_mgmt_widget.setFixedWidth(int(self.scale_factor + rm))
        self.sys_glob_hdr_widget.setFixedWidth(int(self.scale_factor/2 + rm/3))
        self.sys_control_hdr_widget.setFixedWidth(
            int(self.scale_factor/2 + rm/3))
        self.class_glob_hdr_widget.setFixedWidth(int(self.scale_factor + rm))
        self.arl_widget.setFixedWidth(int(self.scale_factor/2+rm/3))
        self.tri_wvfm_widget.setFixedWidth(int(self.scale_factor/2+rm/3))
        # 		self.card_glb_widget.setFixedWidth(self.dfbx2card_widget1.width()/2+10)
        # 		self.class_interface_widget.setFixedWidth(self.dfbx2card_widget1.width()/2+10)
        self.setFixedWidth(int(self.scale_factor + rm + 20))
        self.setFixedHeight(1000)

        self.emu = EMU_Card()

        log.debug("before loadSettings")

        if argfilename is not None:
            self.loadSettings(argfilename)

        log.debug("after loadSettings")

        self.control_socket = zmq_rep.ZmqRep(self, "tcp://*:5509")
        self.control_socket.gotMessage.connect(self.handleMessage)

    def handleMessage(self, message):
        llog = log.child("handleMessage")
        llog.info(message)
        d = {"SETUP_CRATE": self.full_crate_init,
             "FULL_TUNE": self.extern_tune}
        if message in d.keys():
            f = d[message]
            llog.info(f"calling: {f}")
            try:
                success, extra_info = f()
            except Exception as ex:
                success = False
                extra_info = f"Exception: {ex}"
        else:
            success = False
            extra_info = f"`{message}` invalid, must be one of {list(d.keys())}"
        self.control_socket.resolve_message(success, extra_info)

    def full_crate_init(self):
        llog = log.child("full_crate_init")
        llog.info("started")
        crate_sleep_s = 1.0
        crate_sleep_final_s = 2.0
        if not self.crate_power.isChecked():
            self.crate_power.click()  # turn on crate
            time.sleep(crate_sleep_s)
        self.crate_power.click()  # turn off crate
        time.sleep(crate_sleep_s)
        llog.info("crate power turned off")
        self.crate_power.click()  # turn on crate
        time.sleep(crate_sleep_final_s)
        llog.info("crate power turned on")
        self.send_all_sys_globals()  # send globals system globals button
        llog.info("sent all sys globals")
        self.send_all_class_globals()  # 2nd half of send globals button
        llog.info("sent all class globals")
        self.send_ALL_states_chns(resync=False)  # send arrayed button
        llog.info("sent arrayed")
        self.phcal_system(resync=False)  # CALIBRATE button
        llog.info("sent calibration")
        llog.info("begin resync")
        self.system_resync()
        llog.info("done")
        return True, ""

    def extern_tune(self):
        llog = log.child("extern_tune")
        llog.debug("start")
        connected = self.tune_widget.vphidemo.c.startclient()
        if not connected:
            return False, "tune client failed to connect, is dastard lancero source running?"
        self.tune_widget.vphidemo.fullTune()
        llog.debug("done")
        return True, ""

    def seqln_changed(self):
        if self.seqln_timer == None:
            self.seqln_timer = QtCore.QTimer()
            self.seqln_timer.timeout.connect(self.change_seqln)
        self.seqln_timer.start(750)

    def change_seqln(self):
        log.debug(tc.WARNING + "SEQLN changed:", tc.ENDC)

        log.debug(tc.FCTCALL + "send SEQLN parameter to all cards:", tc.ENDC)
        self.seqln = self.seqln_spin.value()
        for idx, val in enumerate(self.class_vector):
            self.card_addr = self.addr_vector[idx]
            if val == "DFBCLK":
                self.crate_widgets[idx].seqln_changed(self.seqln)
            if val == "DFBx2":
                self.crate_widgets[idx].seqln_changed(self.seqln)
            if val == "BAD16":
                self.crate_widgets[idx].seqln_changed(self.seqln)
        if self.tri_idx == 1:
            self.tri_period_changed()
        self.frame_period_changed()
        self.system_resync()
        self.seqln_timer.stop()
        self.seqln_timer = None

    def lsync_changed(self):
        if self.lsync_timer == None:
            self.lsync_timer = QtCore.QTimer()
            self.lsync_timer.timeout.connect(self.change_lsync)
        self.lsync_timer.start(750)

    def change_lsync(self):
        self.lsync_timer.stop()
        self.lsync_timer = None
        self.last_lsync = self.lsync
        self.lsync = self.lsync_spin.value()
        if self.lsync == self.last_lsync:
            return
        log.debug(tc.WARNING + "Line period changed:",
                  self.lsync * 8, "ns", tc.ENDC)

        # when the clock widget sends wreg2 it actually sends lsync to the clock card
        self.crate_widgets[0].dfbclk_widget2.lsync_indicator.setText(
            str(self.lsync))
        if self.lsync < 40:
            # 			print tc.FCTCALL + "parallel stream engaged" + tc.ENDC
            # 			print
            self.PS_button.setChecked(1)
        self.tri_period_changed()
        self.frame_period_changed()
        self.system_resync()

    def frame_period_changed(self):
        self.frame_period = self.lsync * self.seqln * 0.008
        log.debug(tc.WARNING + "frame period changed:",
                  self.frame_period, "\u00B5s", tc.ENDC)

        if self.RLD_frame.isChecked() == True:
            self.RLDwarning()
            self.RLDpos_eng_indicator.setText(
                str((self.RLDpos)*self.frame_period)[:6])
            self.RLDneg_eng_indicator.setText(
                str((self.RLDneg)*self.frame_period)[:6])
        if self.RLD_time.isChecked() == True:
            self.RLDpos_spin.setValue(int(self.RLDpos_delay/self.frame_period))
            self.RLDneg_spin.setValue(int(self.RLDneg_delay/self.frame_period))

    def dfb_delay_changed(self):
        if self.dfb_delay_timer == None:
            self.dfb_delay_timer = QtCore.QTimer()
            self.dfb_delay_timer.timeout.connect(self.change_dfb_delay)
        self.dfb_delay_timer.start(750)

    def change_dfb_delay(self):
        log.debug(tc.FCTCALL + "send card delay to all DFB cards:" + tc.ENDC)

        self.dfb_delay = self.dfb_delay_spin.value()
        for idx, val in enumerate(self.class_vector):
            self.card_addr = self.addr_vector[idx]
            if val == "DFBCLK":
                LED = self.crate_widgets[idx].LED
                ST = self.crate_widgets[idx].ST
                self.send_dfb_wreg7(LED, ST, self.card_addr)
            if val == "DFBx2":
                LED = self.crate_widgets[idx].LED
                ST = self.crate_widgets[idx].ST
                self.send_dfb_wreg7(LED, ST, self.card_addr)
            if val == "DFBs":
                log.debug(
                    tc.FCTCALL + "send CARD_DELAY parameter to SCREAM card:", tc.ENDC)

                self.crate_widgets[idx].send_global("CARD", self.dfb_delay)
        if self.locked == 0:
            self.system_resync()
        self.dfb_delay_timer.stop()
        self.dfb_delay_timer = None

    def bad_delay_changed(self):
        if self.bad_delay_timer == None:
            self.bad_delay_timer = QtCore.QTimer()
            self.bad_delay_timer.timeout.connect(self.change_bad_delay)
        self.bad_delay_timer.start(750)

    def change_bad_delay(self):
        log.debug(tc.FCTCALL + "send card delay to all BAD16 cards:" + tc.ENDC)
        self.bad_delay = self.bad_delay_spin.value()
        # 		mask = 0xfffc3ff
        # 		self.bad_wreg0 = (self.bad_wreg0 & mask) | (self.bad_delay << 10)
        for idx, val in enumerate(self.class_vector):
            self.card_addr = self.addr_vector[idx]
            if val == "BAD16":
                LED = self.crate_widgets[idx].LED
                ST = self.crate_widgets[idx].ST
                INT = self.crate_widgets[idx].INT
                self.send_bad16_wreg0(ST, LED, INT, self.card_addr)
        if self.locked == 0:
            self.system_resync()
        self.bad_delay_timer.stop()
        self.bad_delay_timer = None

    def prop_delay_changed(self):
        if self.prop_delay_timer == None:
            self.prop_delay_timer = QtCore.QTimer()
            self.prop_delay_timer.timeout.connect(self.change_prop_delay)
        self.prop_delay_timer.start(750)

    def change_prop_delay(self):
        log.debug(tc.FCTCALL + "send propagation delay to all DFB cards:" + tc.ENDC)

        self.prop_delay = self.prop_delay_spin.value()
        for idx, val in enumerate(self.class_vector):
            self.card_addr = self.addr_vector[idx]
            if val == "DFBCLK":
                LED = self.crate_widgets[idx].LED
                ST = self.crate_widgets[idx].ST
                self.send_dfb_wreg7(LED, ST, self.card_addr)
            if val == "DFBx2":
                LED = self.crate_widgets[idx].LED
                ST = self.crate_widgets[idx].ST
                self.send_dfb_wreg7(LED, ST, self.card_addr)
            if val == "DFBs":
                log.debug(
                    tc.FCTCALL + "send PROP_DELAY parameter to SCREAM card:", tc.ENDC)

                self.crate_widgets[idx].send_global("PROP", self.prop_delay)
        if self.locked == 0:
            self.system_resync()
        self.prop_delay_timer.stop()
        self.prop_delay_timer = None

    def dfbclk_XPT_changed(self):
        log.debug(
            tc.FCTCALL + "send crosspoint switch setting to DFBx1CLK card:" + tc.ENDC)
        self.dfbclk_XPT = self.dfbclk_xpt_mode.currentIndex()
        for idx, val in enumerate(self.class_vector):
            self.card_addr = self.addr_vector[idx]
            if val == "DFBCLK":
                self.send_dfbclk_wreg6(self.card_addr)

    def dfbx2_XPT_changed(self):
        log.debug(
            tc.FCTCALL + "send crosspoint switch setting to all DFBx2 cards:" + tc.ENDC)

        self.dfbx2_XPT = self.dfbx2_xpt_mode.currentIndex()
        for idx, val in enumerate(self.class_vector):
            self.card_addr = self.addr_vector[idx]
            if val == "DFBx2":
                self.send_dfbx2_wreg6(self.card_addr)
            if val == "DFBs":
                log.debug(
                    tc.FCTCALL + "send XPT parameter to SCREAM card:", tc.ENDC)

                self.crate_widgets[idx].send_global("XPT", self.dfbx2_XPT)

    def TP_changed(self):
        log.debug(
            tc.FCTCALL + "send test pattern parameters to all DFB cards:" + tc.ENDC)

        self.TP = self.tp_mode.currentIndex()
        for idx, val in enumerate(self.class_vector):
            self.card_addr = self.addr_vector[idx]
            if self.TP != 0:
                if val == "DFBCLK":
                    self.send_GPI5()
                    self.send_GPI6()
                if val == "DFBx2":
                    self.send_GPI5()
                    self.send_GPI6()
            if val == "DFBCLK":
                self.send_GPI4()
            if val == "DFBx2":
                self.send_GPI4()
            if val == "DFBs":
                log.debug(
                    tc.FCTCALL + "send Test Pattern Mode parameters to SCREAM card:", tc.ENDC)

                self.crate_widgets[idx].send_global("TP", self.TP)

    def NSAMP_changed(self):
        if self.NSAMP_delay_timer == None:
            self.NSAMP_delay_timer = QtCore.QTimer()
            self.NSAMP_delay_timer.timeout.connect(self.change_NSAMP)
        self.NSAMP_delay_timer.start(750)

    def change_NSAMP(self):
        log.debug(tc.FCTCALL + "send NSAMP to all DFB cards:" + tc.ENDC)

        self.NSAMP = self.NSAMP_spin.value()
        for idx, val in enumerate(self.class_vector):
            self.card_addr = self.addr_vector[idx]
            if val == "DFBCLK":
                self.send_dfbclk_wreg6(self.card_addr)
            if val == "DFBx2":
                self.send_dfbx2_wreg6(self.card_addr)
            if val == "DFBs":
                log.debug(
                    tc.FCTCALL + "send NSAMP parameter to SCREAM card:", tc.ENDC)

                self.crate_widgets[idx].send_global("NSAMP", self.NSAMP)
            # 		self.system_resync()
        self.NSAMP_delay_timer.stop()
        self.NSAMP_delay_timer = None

    def SETT_changed(self):
        if self.SETT_delay_timer == None:
            self.SETT_delay_timer = QtCore.QTimer()
            self.SETT_delay_timer.timeout.connect(self.change_SETT)
        self.SETT_delay_timer.start(750)

    def change_SETT(self):
        log.debug(tc.FCTCALL + "send SETT to all DFB cards:" + tc.ENDC)

        self.SETT = self.SETT_spin.value()
        for idx, val in enumerate(self.class_vector):
            self.card_addr = self.addr_vector[idx]
            if val == "DFBCLK":
                LED = self.crate_widgets[idx].LED
                ST = self.crate_widgets[idx].ST
                self.send_dfb_wreg7(LED, ST, self.card_addr)
            if val == "DFBx2":
                LED = self.crate_widgets[idx].LED
                ST = self.crate_widgets[idx].ST
                self.send_dfb_wreg7(LED, ST, self.card_addr)
            if val == "DFBs":
                log.debug(
                    tc.FCTCALL + "send SETTLE parameter to SCREAM card:", tc.ENDC)

                self.crate_widgets[idx].send_global("SETT", self.SETT)

            # 		self.system_resync()
        self.SETT_delay_timer.stop()
        self.SETT_delay_timer = None

    def PS_changed(self):
        self.PS = self.PS_button.isChecked()
        if self.PS == 1:
            self.PS_button.setStyleSheet(
                "background-color: #" + tc.green + ";")
            log.debug(
                tc.FCTCALL + "send parallel stream to all DFB cards: parallel stream engaged" + tc.ENDC)

        else:
            self.PS_button.setStyleSheet("background-color: #" + tc.red + ";")
            log.debug(
                tc.FCTCALL + "send parallel stream to all DFB cards: parallel stream disengaged" + tc.ENDC)

        for idx, val in enumerate(self.class_vector):
            self.card_addr = self.addr_vector[idx]
            if val == "DFBCLK":
                self.send_dfbclk_wreg6(self.card_addr)
            if val == "DFBx2":
                self.send_dfbx2_wreg6(self.card_addr)
            if val == "DFBs":
                log.debug(
                    tc.FCTCALL + "send Parallel Stream Boolean to SCREAM card:", tc.ENDC)

                self.crate_widgets[idx].send_global("PS", self.PS)

    def DFBx2class_glb_chg_msg(self):
        log.debug(tc.FCTCALL + "DFBx2 CLASS global changed:", tc.ENDC)

    def ARLsense_changed(self):
        if self.ARLsense_timer == None:
            self.ARLsense_timer = QtCore.QTimer()
            self.ARLsense_timer.timeout.connect(self.change_ARLsense)
        self.ARLsense_timer.start(750)

    def change_ARLsense(self):
        log.debug(
            tc.FCTCALL + "send ARL sensitivity parameter to all DFB cards:", tc.ENDC)

        self.ARLsense = self.ARLsense_spin.value()
        # 		self.ARLsense_indicator.setText("%5i"%(self.ARLsense))
        self.ARLsense_eng_indicator.setText(str((self.ARLsense)/16.383)[:6])
        for idx, val in enumerate(self.class_vector):
            self.card_addr = self.addr_vector[idx]
            if val == "DFBCLK" or val == "DFBx2":
                log.debug(
                    tc.FCTCALL + "send ARL sensitivity parameter to", val, "card:", tc.ENDC)

                self.send_dfb_GPI16()
            if val == "DFBs":
                log.debug(
                    tc.FCTCALL + "send ARL sensitivity parameter to SCREAM card:", tc.ENDC)

                self.crate_widgets[idx].send_ARL("ARLsense", self.ARLsense)
        self.ARLsense_timer.stop()
        self.ARLsense_timer = None

    def RLDpos_changed(self):
        if self.RLDpos_timer == None:
            self.RLDpos_timer = QtCore.QTimer()
            self.RLDpos_timer.timeout.connect(self.change_RLDpos)
        self.RLDpos_timer.start(750)

    def change_RLDpos(self):
        log.debug(
            tc.FCTCALL + "send ARL positive relock delay parameter to all DFB cards:", tc.ENDC)

        self.RLDpos = self.RLDpos_spin.value()
        self.RLDpos_delay = self.frame_period * self.RLDpos
        # 		self.RLDpos_indicator.setText("%5i"%(self.RLDpos))
        self.RLDpos_eng_indicator.setText(
            str((self.RLDpos)*self.frame_period)[:6])
        for idx, val in enumerate(self.class_vector):
            self.card_addr = self.addr_vector[idx]
            if (val == "DFBCLK" or val == "DFBx2"):
                log.debug(
                    tc.FCTCALL + "send RLD positive delay parameter to", val, "card:", tc.ENDC)

                self.send_dfb_GPI17()
            if val == "DFBs":
                log.debug(
                    tc.FCTCALL + "send RLD positive delay parameter to SCREAM card:", tc.ENDC)

                self.crate_widgets[idx].send_ARL("RLDpos", self.RLDpos)
        self.RLDpos_timer.stop()
        self.RLDpos_timer = None

    def RLDneg_changed(self):
        if self.RLDneg_timer == None:
            self.RLDneg_timer = QtCore.QTimer()
            self.RLDneg_timer.timeout.connect(self.change_RLDneg)
        self.RLDneg_timer.start(750)

    def change_RLDneg(self):
        log.debug(
            tc.FCTCALL + "send ARL negative relock delay parameter to all DFB cards:", tc.ENDC)

        self.RLDneg = self.RLDneg_spin.value()
        self.RLDneg_delay = self.frame_period * self.RLDneg
        # 		self.RLDneg_indicator.setText("%5i"%(self.RLDneg))
        self.RLDneg_eng_indicator.setText(
            str((self.RLDneg)*self.frame_period)[:6])
        for idx, val in enumerate(self.class_vector):
            self.card_addr = self.addr_vector[idx]
            if (val == "DFBCLK" or val == "DFBx2"):
                log.debug(
                    tc.FCTCALL + "send RLD negative delay parameter to", val, "card:", tc.ENDC)

                self.send_dfb_GPI18()
            if val == "DFBs":
                log.debug(
                    tc.FCTCALL + "send RLD negative delay parameter to SCREAM card:", tc.ENDC)

                self.crate_widgets[idx].send_ARL("RLDneg", self.RLDneg)
        self.RLDneg_timer.stop()
        self.RLDneg_timer = None

    def track_changed(self):
        self.RLD_track_state = self.RLD_frame.isChecked()

    def RLDwarning(self):
        log.debug(
            tc.WARNING + "ARL physical relock delay changed: this may impact relocking", tc.ENDC)

    '''triangle methods'''

    def dwell_changed(self):
        log.debug(tc.FCTCALL + "triangle step dwell changed:", tc.ENDC)
        self.dwell_val = self.dwell.value()
        self.dwellDACunits = 2**(self.dwell_val)
        self.tri_period_changed()
        self.send_triangle()

    def range_changed(self):
        log.debug(tc.FCTCALL + "triangle number of steps changed:", tc.ENDC)
        self.range_val = self.range.value()
        self.rangeDACunits = 2**(self.range_val)
        self.tri_amp_changed()
        self.tri_period_changed()
        self.send_triangle()

    def step_changed(self):
        log.debug(tc.FCTCALL + "triangle step size changed:", tc.ENDC)
        self.step_val = self.step.value()
        self.stepDACunits = self.step_val
        self.tri_amp_changed()
        self.send_triangle()

    def tri_amp_changed(self):
        self.ampDACunits = self.rangeDACunits * self.stepDACunits
        if self.ampDACunits > 16383:
            self.ampDACunits = 16383
        self.amp_indicator.setText('%5i' % self.ampDACunits)
        mV = 1000*self.ampDACunits/16383.0
        log.debug(tc.WARNING + "triangle amplitude changed:", mV, "mV", tc.ENDC)
        self.amp_eng_indicator.setText('%4.3f' % mV)

    def tri_period_changed(self):
        self.periodDACunits = float(2*self.dwellDACunits*self.rangeDACunits)
        self.period_indicator.setText('%6i' % self.periodDACunits)
        if self.tri_idx == 0:
            uSecs = self.periodDACunits*self.lsync*0.008
        else:
            uSecs = self.periodDACunits*self.lsync*self.seqln*0.008
        kHz = 1000/uSecs
        self.period_eng_indicator.setText('%7.3f' % uSecs)
        self.freq_eng_indicator.setText('%6.3f' % kHz)
        log.debug(tc.WARNING + "triangle period changed:",
                  '%7.3f' % uSecs, "\u00B5s", tc.ENDC)

    def tri_idx_changed(self):
        log.debug(tc.FCTCALL + "triangle time base changed:", tc.ENDC)
        self.tri_idx = self.tri_idx_button.isChecked()
        self.tri_period_changed()
        self.send_triangle()
        if self.tri_idx == 1:
            self.tri_idx_button.setStyleSheet(
                "background-color: #" + tc.green + ";")
            self.tri_idx_button.setText('FRAME')
        else:
            self.tri_idx_button.setStyleSheet(
                "background-color: #" + tc.red + ";")
            self.tri_idx_button.setText('LSYNC')

    '''system global methods'''

    def cratePower(self, sleep_s=0.1):
        self.power_state = self.crate_power.isChecked()
        self.send_all_globals.setEnabled(self.power_state)
        self.send_all_states_chns.setEnabled(self.power_state)
        self.cal_system.setEnabled(self.power_state)
        self.resync_system.setEnabled(self.power_state)
        if self.power_state == 1:

            log.info(
                tc.INIT + "cycle power to crate through EMU: power ON", tc.ENDC)

            self.crate_power.setStyleSheet(
                "background-color: #" + tc.green + ";")
            self.crate_power.setText('crate power ON')
            self.emu.powerOn()
            log.debug(
                tc.FAIL + "calibration has been lost as result of power cycle:", tc.ENDC)

            log.debug(
                tc.INIT + "reset ALL phase offsets for DFB/BAD cards:", tc.ENDC)

            for idx, val in enumerate(self.class_vector):
                self.card_addr = self.addr_vector[idx]
                if val == "DFBCLK":
                    log.debug(tc.FCTCALL + "reset phase offsets for DFBCLK card address",
                              self.addr_vector[idx], tc.ENDC)

                    self.crate_widgets[idx].dfbclk_widget3.resetALLphase()
                if val == "DFBx2":
                    log.debug(tc.FCTCALL + "reset phase offsets for DFBx2 card address",
                              self.addr_vector[idx], tc.ENDC)

                    self.crate_widgets[idx].dfbx2_widget3.resetALLphase()
                if val == "BAD16":
                    log.debug(tc.FCTCALL + "reset phase offsets for BAD16 card address",
                              self.addr_vector[idx], tc.ENDC)

                    self.crate_widgets[idx].badrap_widget3.resetALLphase()
        else:

            log.info(
                tc.INIT + "cycle power to crate through EMU: power OFF", tc.ENDC)

            self.crate_power.setStyleSheet(
                "background-color: #" + tc.red + ";")
            self.crate_power.setText('crate power OFF')
            self.emu.powerOff()
        time.sleep(sleep_s)
        QtCore.QCoreApplication.processEvents()

    def send_ALL_globals(self):

        log.debug(tc.INIT + tc.BOLD +
                  "send ALL globals to ALL cards:", tc.ENDC)

        self.send_all_sys_globals()
        self.send_all_class_globals()

    def send_ALL_states_chns(self, resync=False):

        log.debug(tc.INIT + tc.BOLD +
                  "send ALL states & channels to DFB/BAD cards:", tc.ENDC)

        for idx, val in enumerate(self.class_vector):
            self.card_addr = self.addr_vector[idx]
            if val == "DFBCLK":

                log.debug(tc.INIT + "send ALL states to DFBCLK card address",
                          self.addr_vector[idx], ":", tc.ENDC)

                self.crate_widgets[idx].dfbclk_widget1.master_vector.chn_send.click(
                )
            if val == "DFBx2":

                log.debug(tc.INIT + "send ALL states to DFBx2 card address",
                          self.addr_vector[idx], ", channel 1:", tc.ENDC)
                self.crate_widgets[idx].dfbx2_widget1.master_vector.chn_send.click(
                )

                log.debug(tc.INIT + "send ALL states to DFBx2 card address",
                          self.addr_vector[idx], ", channel 2:", tc.ENDC)

                self.crate_widgets[idx].dfbx2_widget2.master_vector.chn_send.click(
                )
            if val == "DFBs":

                log.debug(tc.INIT + "send ALL states to SCREAM card address",
                          self.addr_vector[idx], ", channel 1:", tc.ENDC)
                self.crate_widgets[idx].dfbs_widget1.master_vector.chn_send.click(
                )

                log.debug(tc.INIT + "send ALL states to SCREAM card address",
                          self.addr_vector[idx], ", channel 2:", tc.ENDC)

                self.crate_widgets[idx].dfbs_widget2.master_vector.chn_send.click(
                )
            if val == "BAD16":

                log.debug(tc.INIT + "send channels to BAD16 card address",
                          self.addr_vector[idx], ":", tc.ENDC)
                self.crate_widgets[idx].badrap_widget1.master_vector.chn_send.click(
                )

                log.debug(tc.INIT + "send ALL states to BAD16 card address",
                          self.addr_vector[idx], ":", tc.ENDC)

                self.crate_widgets[idx].badrap_widget2.SendAllStates()
        if resync:
            self.system_resync()

    def phcal_system(self, resync=True):

        log.debug(tc.INIT + tc.BOLD +
                  "auto phase calibrate all DFB/BAD cards:", tc.ENDC)

        for idx, val in enumerate(self.class_vector):
            self.card_addr = self.addr_vector[idx]
            if val == "DFBCLK":
                # 				print
                # 				print tc.INIT + "auto calibrate DFBCLK card address",self.addr_vector[idx],":", tc.ENDC
                self.crate_widgets[idx].dfbclk_widget3.autocal.click()
            if val == "DFBx2":
                # 				print
                # 				print tc.INIT + "auto calibrate DFBx2 card address",self.addr_vector[idx], tc.ENDC
                self.crate_widgets[idx].dfbx2_widget3.autocal.click()
            if val == "DFBs":
                # 				print
                # 				print tc.INIT + "auto calibrate DFB SCREAM card address",self.addr_vector[idx], tc.ENDC
                self.crate_widgets[idx].dfbs_widget3.autocal.click()
            if val == "BAD16":
                # 				print
                # 				print tc.INIT + "auto calibrate BAD16 card address",self.addr_vector[idx],":", tc.ENDC
                self.crate_widgets[idx].badrap_widget3.autocal.click()
        if resync:
            self.system_resync()

    def system_resync(self):

        log.debug(tc.INIT + "resynchronize ALL cards AND SENDING LSYNC:", tc.ENDC)

        self.send_CLK_globals()
        for idx, val in enumerate(self.class_vector):
            self.card_addr = self.addr_vector[idx]
            if val == "DFBCLK":
                self.crate_widgets[idx].dfbclk_widget2.resync_button.click()
        self.writeGlobalsToDotCringeDirectory()

    def lockServer(self):
        self.locked = self.server_lock.isChecked()
        if self.locked == 0:
            message = "restore"
            self.server_lock.setStyleSheet(
                "background-color: #" + tc.green + ";")
            self.server_lock.setText('server LOCK OFF')

            log.debug(tc.INIT + "critical parameter lock out disengaged:")

            log.debug(
                tc.WARNING + "auto re-sync engaged for delay parameter changes:", tc.ENDC)

            log.debug(
                tc.FAIL + "changing SYSTEM globals, re-sync, or send mode may crash SERVER (if running):", tc.ENDC)

        else:
            message = "limit"
            self.server_lock.setStyleSheet(
                "background-color: #" + tc.red + ";")
            self.server_lock.setText('server LOCK ON')

            log.debug(
                tc.INIT + "critical parameter lock out engaged for SERVER keep alive:", tc.ENDC)

            log.debug(
                tc.WARNING + "auto re-sync disengaged for delay parameter changes:", tc.ENDC)

        self.sys_glob_hdr_widget.setEnabled(not(self.locked))
        self.sendsetup.setEnabled(not(self.locked))
        self.NSAMP_spin.setEnabled(not(self.locked))
        self.dfbclk_xpt_mode.setEnabled(not(self.locked))
        self.dfbx2_xpt_mode.setEnabled(not(self.locked))
        for idx, val in enumerate(self.class_vector):
            self.card_addr = self.addr_vector[idx]
            if val == "DFBCLK":
                log.debug(tc.FCTCALL + message, "SEND MODE on", val, "card:",
                          self.slot_vector[idx], "/", self.card_addr, tc.ENDC)

                # 				self.crate_widgets[idx].dfbclk_widget1.master_vector.data_packet.setEnabled(not(self.locked))
                if self.locked == 1:
                    self.crate_widgets[idx].dfbclk_widget1.master_vector.data_packet.setCurrentIndex(
                        0)
                    self.crate_widgets[idx].dfbclk_widget1.master_vector.data_packet.removeItem(
                        3)
                    self.crate_widgets[idx].dfbclk_widget1.master_vector.data_packet.removeItem(
                        2)
                    for i in range(self.seqln):
                        self.crate_widgets[idx].dfbclk_widget1.state_vectors[i].data_packet.removeItem(
                            3)
                        self.crate_widgets[idx].dfbclk_widget1.state_vectors[i].data_packet.removeItem(
                            2)
                else:
                    self.crate_widgets[idx].dfbclk_widget1.master_vector.data_packet.addItem(
                        'FBB, FBA')
                    self.crate_widgets[idx].dfbclk_widget1.master_vector.data_packet.addItem(
                        'test pattern')
                    for i in range(self.seqln):
                        self.crate_widgets[idx].dfbclk_widget1.state_vectors[i].data_packet.addItem(
                            'FBB, FBA')
                        self.crate_widgets[idx].dfbclk_widget1.state_vectors[i].data_packet.addItem(
                            'test pattern')
                    # 						self.crate_widgets[idx].dfbclk_widget1.state_vectors[i].data_packet.setEnabled(not(self.locked))
            if val == "DFBx2":
                log.debug(tc.FCTCALL + message, "SEND MODE on both channels of",
                          val, "card:", self.slot_vector[idx], "/", self.card_addr, tc.ENDC)

                if self.locked == 1:
                    self.crate_widgets[idx].dfbx2_widget1.master_vector.data_packet.setCurrentIndex(
                        0)
                    self.crate_widgets[idx].dfbx2_widget1.master_vector.data_packet.removeItem(
                        3)
                    self.crate_widgets[idx].dfbx2_widget1.master_vector.data_packet.removeItem(
                        2)
                    self.crate_widgets[idx].dfbx2_widget2.master_vector.data_packet.setCurrentIndex(
                        0)
                    self.crate_widgets[idx].dfbx2_widget2.master_vector.data_packet.removeItem(
                        3)
                    self.crate_widgets[idx].dfbx2_widget2.master_vector.data_packet.removeItem(
                        2)
                    for i in range(self.seqln):
                        self.crate_widgets[idx].dfbx2_widget1.state_vectors[i].data_packet.removeItem(
                            3)
                        self.crate_widgets[idx].dfbx2_widget1.state_vectors[i].data_packet.removeItem(
                            2)
                        self.crate_widgets[idx].dfbx2_widget2.state_vectors[i].data_packet.removeItem(
                            3)
                        self.crate_widgets[idx].dfbx2_widget2.state_vectors[i].data_packet.removeItem(
                            2)
                else:
                    self.crate_widgets[idx].dfbx2_widget1.master_vector.data_packet.addItem(
                        'FBB, FBA')
                    self.crate_widgets[idx].dfbx2_widget1.master_vector.data_packet.addItem(
                        'test pattern')
                    self.crate_widgets[idx].dfbx2_widget2.master_vector.data_packet.addItem(
                        'FBB, FBA')
                    self.crate_widgets[idx].dfbx2_widget2.master_vector.data_packet.addItem(
                        'test pattern')
                    for i in range(self.seqln):
                        self.crate_widgets[idx].dfbx2_widget1.state_vectors[i].data_packet.addItem(
                            'FBB, FBA')
                        self.crate_widgets[idx].dfbx2_widget1.state_vectors[i].data_packet.addItem(
                            'test pattern')
                        self.crate_widgets[idx].dfbx2_widget2.state_vectors[i].data_packet.addItem(
                            'FBB, FBA')
                        self.crate_widgets[idx].dfbx2_widget2.state_vectors[i].data_packet.addItem(
                            'test pattern')
            if val == "DFBs":
                log.debug(tc.FCTCALL + message, "SEND MODE on both channels of",
                          val, "card:", self.slot_vector[idx], "/", self.card_addr, tc.ENDC)

                if self.locked == 1:
                    self.crate_widgets[idx].dfbs_widget1.master_vector.data_packet.setCurrentIndex(
                        0)
                    self.crate_widgets[idx].dfbs_widget1.master_vector.data_packet.removeItem(
                        3)
                    self.crate_widgets[idx].dfbs_widget1.master_vector.data_packet.removeItem(
                        2)
                    self.crate_widgets[idx].dfbs_widget2.master_vector.data_packet.setCurrentIndex(
                        0)
                    self.crate_widgets[idx].dfbs_widget2.master_vector.data_packet.removeItem(
                        3)
                    self.crate_widgets[idx].dfbs_widget2.master_vector.data_packet.removeItem(
                        2)
                else:
                    self.crate_widgets[idx].dfbs_widget1.master_vector.data_packet.addItem(
                        'FBB, FBA')
                    self.crate_widgets[idx].dfbs_widget1.master_vector.data_packet.addItem(
                        'test pattern')
                    self.crate_widgets[idx].dfbs_widget2.master_vector.data_packet.addItem(
                        'FBB, FBA')
                    self.crate_widgets[idx].dfbs_widget2.master_vector.data_packet.addItem(
                        'test pattern')

    def send_all_sys_globals(self):

        log.debug(tc.INIT + tc.BOLD +
                  "send system globals to all cards:", tc.ENDC)

        self.send_CLK_globals()
        log.debug(
            tc.INIT + "send system globals to DFBCLK (DFB CH 1), DFB, and BAD16 cards:", tc.ENDC)

        for idx, val in enumerate(self.class_vector):
            self.card_addr = self.addr_vector[idx]
            if (val == "DFBCLK"):
                log.debug(
                    tc.FCTCALL + "send SEQLN parameter to DFB CH1 on", val, "card:", tc.ENDC)

                LED = self.crate_widgets[idx].LED
                ST = self.crate_widgets[idx].ST
                self.send_dfb_wreg7(LED, ST, self.card_addr)
            if (val == "DFBx2"):
                log.debug(tc.FCTCALL + "send SEQLN parameter to",
                          val, "card:", tc.ENDC)

                LED = self.crate_widgets[idx].LED
                ST = self.crate_widgets[idx].ST
                self.send_dfb_wreg7(LED, ST, self.card_addr)
            if val == ("BAD16"):
                log.debug(tc.FCTCALL + "send SEQLN parameter to",
                          val, "card:", tc.ENDC)

                LED = self.crate_widgets[idx].LED
                ST = self.crate_widgets[idx].ST
                INT = self.crate_widgets[idx].INT
                self.send_bad16_wreg0(ST, LED, INT, self.card_addr)
            # 				self.crate_widgets[idx].send_class_globals(self.bad_wreg0)

    def send_CLK_globals(self):

        log.debug(
            tc.INIT + "send system globals to DFBCLK card (clock controller):", tc.ENDC)

        log.debug(tc.FCTCALL + "send LSYNC parameter to CLK:", tc.ENDC)
        self.crate_widgets[0].dfbclk_widget2.send_wreg2()
        log.debug(tc.FCTCALL + "send SEQLN parameter to CLK:", tc.ENDC)
        self.crate_widgets[0].dfbclk_widget2.send_wreg7()

    def writeGlobalsToDotCringeDirectory(self):
        dirname = os.path.expanduser("~/.cringe")
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        globals = {"lsync": self.lsync,
                   "SETT": self.SETT,
                   "seqln": self.seqln,
                   "NSAMP": self.NSAMP,
                   "propagationdelay": self.prop_delay,
                   "carddelay": self.dfb_delay,
                   "XPT": self.dfbx2_XPT,
                   "testpattern": self.TP}
        for k, v in list(globals.items()):
            filename = os.path.join(dirname, k)
            log.debug("Writing {}={} to: {}".format(k, v, filename))
            with open(filename, "w") as f:
                f.write(str(v))
        jsonFilename = os.path.join(dirname, "cringeGlobals.json")
        log.debug("Writing all values to {}".format(jsonFilename))
        with open(jsonFilename, "w") as f:
            json.dump(globals, f, indent=4)

    '''class global methods'''

    def send_all_class_globals(self):

        log.debug(tc.INIT + tc.BOLD +
                  "send class globals to all cards:", tc.ENDC)

        self.send_dfb_class_globals()
        self.send_bad_class_globals()
        self.send_triangle()
        self.send_ARL()
        self.send_TP()

    def send_dfb_class_globals(self):
        # 		print
        # 		print tc.INIT + "send class globals to DFB cards:", tc.ENDC
        # 		self.dfbclk_wreg6 = (6 << 25) | (self.PS << 24) | (self.dfbclk_XPT << 21) | (self.CLK << 20) | self.NSAMP
        # 		self.dfbx2_wreg6 = (6 << 25) | (self.PS << 24) | (self.dfbx2_XPT << 21) | self.NSAMP
        # 		self.dfbclk_wreg6 = (6 << 25) | (self.PS << 24) | (self.dfbclk_XPT << 21) | (self.CLK << 20) | (self.RLDpos << 16) | (self.ARLsense << 12) |(self.RLDneg << 8) | self.NSAMP
        # 		self.dfbx2_wreg6 = (6 << 25) | (self.PS << 24) | (self.dfbx2_XPT << 21) | (self.RLDpos << 16) | (self.ARLsense << 12) |(self.RLDneg << 8) | self.NSAMP
        # 		self.dfb_wreg7 = (7 << 25) | (self.prop_delay << 18) | (self.dfb_delay << 14) | (self.seqln << 8) | self.SETT
        for idx, val in enumerate(self.class_vector):
            self.card_addr = self.addr_vector[idx]
            if val == "DFBCLK":
                log.debug(
                    tc.FCTCALL + "send DFB class globals to DFBCLK card:", tc.ENDC)

                LED = self.crate_widgets[idx].LED
                ST = self.crate_widgets[idx].ST
                self.send_dfbclk_wreg6(self.card_addr)
                self.send_dfb_wreg7(LED, ST, self.card_addr)
            if val == "DFBx2":
                log.debug(
                    tc.FCTCALL + "send DFB class globals to DFBx2 card:", tc.ENDC)

                LED = self.crate_widgets[idx].LED
                ST = self.crate_widgets[idx].ST
                self.send_dfbx2_wreg6(self.card_addr)
                self.send_dfb_wreg7(LED, ST, self.card_addr)
            if val == "DFBs":
                log.debug(
                    tc.FCTCALL + "send DFB class globals to DFBscream card:", tc.ENDC)

                self.crate_widgets[idx].send_global("XPT", self.dfbx2_XPT)
                self.crate_widgets[idx].send_global("TP", self.TP)
                self.crate_widgets[idx].send_global("NSAMP", self.NSAMP)
                self.crate_widgets[idx].send_global("SETT", self.SETT)
                self.crate_widgets[idx].send_global("PROP", self.prop_delay)
                self.crate_widgets[idx].send_global("CARD", self.dfb_delay)
                self.crate_widgets[idx].send_card_globals()
                self.crate_widgets[idx].send_channel_globals()

    def send_bad_class_globals(self):
        # 		print
        # 		print tc.INIT + "send BAD16 class globals to all BAD16 cards:", tc.ENDC
        for idx, val in enumerate(self.class_vector):
            self.card_addr = self.addr_vector[idx]
            if val == "BAD16":
                log.debug(
                    tc.FCTCALL + "send BAD class globals to BAD16 card:", tc.ENDC)

                LED = self.crate_widgets[idx].LED
                ST = self.crate_widgets[idx].ST
                INT = self.crate_widgets[idx].INT
                self.send_bad16_wreg0(ST, LED, INT, self.card_addr)

    def send_triangle(self):
        # 		print
        # 		print tc.INIT + "send triangle parameters to DFB, and BAD16 cards:", tc.ENDC
        for idx, val in enumerate(self.class_vector):
            self.card_addr = self.addr_vector[idx]
            if val == ("DFBCLK"):
                log.debug(
                    tc.FCTCALL + "send triangle parameters to DFB CH1 on DFBCLK card:", tc.ENDC)

                self.send_wreg0(1)
                GR = self.crate_widgets[idx].dfbclk_widget1.GR
                self.send_dfb_wreg4(GR, self.card_addr)
            if val == ("DFBx2"):
                log.debug(
                    tc.FCTCALL + "send triangle parameters to DFB CH1 on DFBx2 card:", tc.ENDC)
                self.send_wreg0(1)
                GR = self.crate_widgets[idx].dfbx2_widget1.GR
                self.send_dfb_wreg4(GR, self.card_addr)
                log.debug(
                    tc.FCTCALL + "send triangle parameters to DFB CH2 on DFBx2 card:", tc.ENDC)

                self.send_wreg0(2)
                GR = self.crate_widgets[idx].dfbx2_widget2.GR
                self.send_dfb_wreg4(GR, self.card_addr)
            if val == ("DFBs"):
                log.debug(
                    tc.FCTCALL + "send triangle parameters to SCREAM card:", tc.ENDC)

                self.crate_widgets[idx].send_triangle("dwell", self.dwell_val)
                self.crate_widgets[idx].send_triangle("range", self.range_val)
                self.crate_widgets[idx].send_triangle("step", self.step_val)
            if val == ("BAD16"):
                log.debug(
                    tc.FCTCALL + "send triangle parameters to BAD16 card:", tc.ENDC)

                self.send_bad16_wreg1()

    def send_ARL(self):
        # 		print
        # 		print tc.INIT + "send ARL parameters to DFB cards:", tc.ENDC
        for idx, val in enumerate(self.class_vector):
            self.card_addr = self.addr_vector[idx]
            if (val == "DFBCLK" or val == "DFBx2"):
                log.debug(tc.FCTCALL + "send ARL parameters to",
                          val, "card:", tc.ENDC)

                self.send_dfb_GPI16()
                self.send_dfb_GPI17()
                self.send_dfb_GPI18()
            if val == "DFBs":
                log.debug(
                    tc.FCTCALL + "send ARL parameters to SCREAM card:", tc.ENDC)

                self.crate_widgets[idx].send_ARL("ARLsense", self.ARLsense)
                self.crate_widgets[idx].send_ARL("RLDpos", self.RLDpos)
                self.crate_widgets[idx].send_ARL("RLDneg", self.RLDneg)

    def send_TP(self):
        # 		print
        # 		print tc.INIT + "send test pattern parameters to DFB cards:", tc.ENDC
        for idx, val in enumerate(self.class_vector):
            self.card_addr = self.addr_vector[idx]
            if (val == "DFBCLK" or val == "DFBx2"):
                log.debug(tc.FCTCALL + "send test pattern parameters to",
                          val, "card:", tc.ENDC)

                self.send_GPI5()
                self.send_GPI6()
                self.send_GPI4()
            if val == "DFBs":
                log.debug(
                    tc.FCTCALL + "send test pattern parameters to SCREAM card:", tc.ENDC)

                self.crate_widgets[idx].send_global("TP", self.TP)

    '''	child called methods '''

    def broadcast_channel(self, state, var):
        # 		print "BROADCAST CHANNEL:"
        # 		print
        for idx, val in enumerate(self.class_vector):
            addr = str(self.addr_vector[idx])
            slot = str(self.slot_vector[idx])
            # 			if (val == "DFBCLK"):
            # 				print tc.FCTCALL + "broadcast to", val,"card:", tc.ENDC
            # 				print
            # 				self.crate_widgets[idx].dfbclk_widget1.triA_changed(state, True)
            if val == "DFBCLK":
                if self.crate_widgets[idx].dfbclk_widget1.MVRX:
                    log.debug(tc.FCTCALL + "broadcast to", val, "card", slot +
                              "/" + addr + "/1", "(slot/addr/ch):", var, tc.ENDC)

                    if var == "triA":
                        self.crate_widgets[idx].dfbclk_widget1.triA_changed(
                            state, True)
                    if var == "triB":
                        self.crate_widgets[idx].dfbclk_widget1.triB_changed(
                            state, True)
                    if var == "a2d_lp_spin":
                        self.crate_widgets[idx].dfbclk_widget1.a2d_lockpt_spin_changed(
                            state, True)
                    if var == "a2d_lp_slider":
                        self.crate_widgets[idx].dfbclk_widget1.a2d_lockpt_slider_changed(
                            state, True)
                    if var == "d2a_A_spin":
                        self.crate_widgets[idx].dfbclk_widget1.d2a_A_spin_changed(
                            state, True)
                    if var == "d2a_A_slider":
                        self.crate_widgets[idx].dfbclk_widget1.d2a_A_slider_changed(
                            state, True)
                    if var == "d2a_B_spin":
                        self.crate_widgets[idx].dfbclk_widget1.d2a_B_spin_changed(
                            state, True)
                    if var == "d2a_B_slider":
                        self.crate_widgets[idx].dfbclk_widget1.d2a_B_slider_changed(
                            state, True)
                    if var == "SM":
                        self.crate_widgets[idx].dfbclk_widget1.data_packet_changed(
                            state, True)
                    if var == "P":
                        self.crate_widgets[idx].dfbclk_widget1.P_spin_changed(
                            state, True)
                    if var == "I":
                        self.crate_widgets[idx].dfbclk_widget1.I_spin_changed(
                            state, True)
                    if var == "FBa":
                        self.crate_widgets[idx].dfbclk_widget1.FBA_changed(
                            state, True)
                    if var == "FBb":
                        self.crate_widgets[idx].dfbclk_widget1.FBB_changed(
                            state, True)
                    if var == "ARL":
                        self.crate_widgets[idx].dfbclk_widget1.ARL_changed(
                            state, True)
                    if var == "send":
                        self.crate_widgets[idx].dfbclk_widget1.send_channel(
                            True)
                    if var == "lock":
                        self.crate_widgets[idx].dfbclk_widget1.lock_channel(
                            state, True)
            if val == "DFBx2":
                if self.crate_widgets[idx].dfbx2_widget1.MVRX:
                    log.debug(tc.FCTCALL + "broadcast to", val, "card", slot +
                              "/" + addr + "/1", "(slot/addr/ch):", var, tc.ENDC)

                    if var == "triA":
                        self.crate_widgets[idx].dfbx2_widget1.triA_changed(
                            state, True)
                    if var == "triB":
                        self.crate_widgets[idx].dfbx2_widget1.triB_changed(
                            state, True)
                    if var == "a2d_lp_spin":
                        self.crate_widgets[idx].dfbx2_widget1.a2d_lockpt_spin_changed(
                            state, True)
                    if var == "a2d_lp_slider":
                        self.crate_widgets[idx].dfbx2_widget1.a2d_lockpt_slider_changed(
                            state, True)
                    if var == "d2a_A_spin":
                        self.crate_widgets[idx].dfbx2_widget1.d2a_A_spin_changed(
                            state, True)
                    if var == "d2a_A_slider":
                        self.crate_widgets[idx].dfbx2_widget1.d2a_A_slider_changed(
                            state, True)
                    if var == "d2a_B_spin":
                        self.crate_widgets[idx].dfbx2_widget1.d2a_B_spin_changed(
                            state, True)
                    if var == "d2a_B_slider":
                        self.crate_widgets[idx].dfbx2_widget1.d2a_B_slider_changed(
                            state, True)
                    if var == "SM":
                        self.crate_widgets[idx].dfbx2_widget1.data_packet_changed(
                            state, True)
                    if var == "P":
                        self.crate_widgets[idx].dfbx2_widget1.P_spin_changed(
                            state, True)
                    if var == "I":
                        self.crate_widgets[idx].dfbx2_widget1.I_spin_changed(
                            state, True)
                    if var == "FBa":
                        self.crate_widgets[idx].dfbx2_widget1.FBA_changed(
                            state, True)
                    if var == "FBb":
                        self.crate_widgets[idx].dfbx2_widget1.FBB_changed(
                            state, True)
                    if var == "ARL":
                        self.crate_widgets[idx].dfbx2_widget1.ARL_changed(
                            state, True)
                    if var == "send":
                        self.crate_widgets[idx].dfbx2_widget1.send_channel(
                            True)
                    if var == "lock":
                        self.crate_widgets[idx].dfbx2_widget1.lock_channel(
                            state, True)
                if self.crate_widgets[idx].dfbx2_widget2.MVRX:
                    log.debug(tc.FCTCALL + "broadcast to", val, "card", slot +
                              "/" + addr + "/2", "(slot/addr/ch):", var, tc.ENDC)

                    if var == "triA":
                        self.crate_widgets[idx].dfbx2_widget2.triA_changed(
                            state, True)
                    if var == "triB":
                        self.crate_widgets[idx].dfbx2_widget2.triB_changed(
                            state, True)
                    if var == "a2d_lp_spin":
                        self.crate_widgets[idx].dfbx2_widget2.a2d_lockpt_spin_changed(
                            state, True)
                    if var == "a2d_lp_slider":
                        self.crate_widgets[idx].dfbx2_widget2.a2d_lockpt_slider_changed(
                            state, True)
                    if var == "d2a_A_spin":
                        self.crate_widgets[idx].dfbx2_widget2.d2a_A_spin_changed(
                            state, True)
                    if var == "d2a_A_slider":
                        self.crate_widgets[idx].dfbx2_widget2.d2a_A_slider_changed(
                            state, True)
                    if var == "d2a_B_spin":
                        self.crate_widgets[idx].dfbx2_widget2.d2a_B_spin_changed(
                            state, True)
                    if var == "d2a_B_slider":
                        self.crate_widgets[idx].dfbx2_widget2.d2a_B_slider_changed(
                            state, True)
                    if var == "SM":
                        self.crate_widgets[idx].dfbx2_widget2.data_packet_changed(
                            state, True)
                    if var == "P":
                        self.crate_widgets[idx].dfbx2_widget2.P_spin_changed(
                            state, True)
                    if var == "I":
                        self.crate_widgets[idx].dfbx2_widget2.I_spin_changed(
                            state, True)
                    if var == "FBa":
                        self.crate_widgets[idx].dfbx2_widget2.FBA_changed(
                            state, True)
                    if var == "FBb":
                        self.crate_widgets[idx].dfbx2_widget2.FBB_changed(
                            state, True)
                    if var == "ARL":
                        self.crate_widgets[idx].dfbx2_widget2.ARL_changed(
                            state, True)
                    if var == "send":
                        self.crate_widgets[idx].dfbx2_widget2.send_channel(
                            True)
                    if var == "lock":
                        self.crate_widgets[idx].dfbx2_widget2.lock_channel(
                            state, True)

    '''	commanding methods '''

    def send_wreg0(self, col):
        log.debug("DFB:WREG0: page register: col", col)
        wreg = 0 << 25
        wregval = wreg | (col << 6)
        self.sendReg(wregval, self.card_addr)

    def send_dfb_wreg4(self, GR, addr):
        log.debug("DFB:WREG4: triangle parameters; time base, dwell, range, step: global relock boolean:",
                  self.tri_idx, self.dwell_val, self.range_val, self.step_val, GR)
        self.dfb_wreg4 = (4 << 25) | (self.tri_idx << 24) | (self.dwell_val << 20) | (
            self.range_val << 16) | (GR << 15) | self.step_val
        # 		wregval = (4 << 25) | (self.tri_idx << 24) | (self.dwell_val << 20) \
        # 			| (self.range_val << 16) | self.GR | self.step_val
        self.sendReg(self.dfb_wreg4, addr)

    def send_dfbclk_wreg6(self, addr):
        log.debug("DFB:WREG6: global parameters: PS, DFBCLK_XPT, CLK, NSAMP:",
                  self.PS, self.dfbclk_XPT, self.CLK, self.NSAMP)
        wregval = (6 << 25) | (self.PS << 24) | (
            self.dfbclk_XPT << 21) | (self.CLK << 20) | self.NSAMP
        self.sendReg(wregval, addr)

    def send_dfbx2_wreg6(self, addr):
        log.debug("DFB:WREG6: global parameters: PS, DFBx2_XPT, NSAMP:",
                  self.PS, self.dfbx2_XPT, self.NSAMP)
        wregval = (6 << 25) | (self.PS << 24) | (
            self.dfbx2_XPT << 21) | self.NSAMP
        self.sendReg(wregval, addr)

    def send_dfb_wreg7(self, LED, ST, addr):
        log.debug("DFB:WREG7: global parameters: LED, ST, prop delay, dfb delay, sequence length, SETT:", LED, ST, self.prop_delay,
                  self.dfb_delay, self.seqln, self.SETT)
        wregval = (7 << 25) | (LED << 23) | (ST << 22) | (self.prop_delay << 18) \
            | (self.dfb_delay << 14) | (self.seqln << 8) | self.SETT
        self.sendReg(wregval, addr)

    def send_bad16_wreg0(self, ST, LED, INT, addr):
        log.debug("BAD16:WREG0: ST, LED, card delay, INIT, sequence length:",
                  ST, LED, self.bad_delay, INT, self.seqln)
        # 		mask = 0xffebeff
        wregval = (0 << 25) | (ST << 16) | (LED << 14) | (
            self.bad_delay << 10) | (INT << 8) | self.seqln
        self.sendReg(wregval, addr)

    def send_bad16_wreg1(self):
        log.debug("BAD16:WREG1: triangle parameters time base, dwell, range, step:",
                  self.tri_idx, self.dwell_val, self.range_val, self.step_val)
        self.bad_wreg1 = (1 << 25) | (self.tri_idx << 24) | (
            self.dwell_val << 20) | (self.range_val << 16) | self.step_val
        self.sendReg(self.bad_wreg1, self.card_addr)

    def send_GPI4(self):
        log.debug("DFB:GPI4: test mode select:", self.TP)
        wreg = 4 << 17
        if self.TP != 0:
            wregval = wreg | 1
        else:
            wregval = wreg | 0
        self.sendReg(wregval, self.card_addr)

    def send_GPI5(self):
        lobytes, hibytes = self.lohibytes()
        log.debug(
            "DFB:GPI5: test pattern hi-bytes [31..16]:", hex(hibytes)[2:].zfill(4))
        wreg = 5 << 17
        wregval = wreg | hibytes
        self.sendReg(wregval, self.card_addr)

    def lohibytes(self):
        if self.TP == 0:
            lobytes = 0xDEAD
            hibytes = 0xBEEF
        if self.TP == 1:
            lobytes = 0x5555
        if self.TP == 2:
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
        if self.TP == 1:
            hibytes = 0x5555
        if self.TP == 2:
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
            hibytes = 0x8bad
        return lobytes, hibytes

    def send_GPI6(self):
        lobytes, hibytes = self.lohibytes()
        log.debug(
            "DFB:GPI6: test pattern lo-bytes [15..0]:", hex(lobytes)[2:].zfill(4))
        wreg = 6 << 17
        wregval = wreg | lobytes
        self.sendReg(wregval, self.card_addr)

    def send_dfb_GPI16(self):
        log.debug("DFB:GPI16: ARL sensitivity level:", self.ARLsense)
        wreg = 16 << 17
        wregval = wreg | self.ARLsense
        self.sendReg(wregval, self.card_addr)

    def send_dfb_GPI17(self):
        log.debug("DFB:GPI17: ARL positive relock delay:", self.RLDpos)
        wreg = 17 << 17
        wregval = wreg | self.RLDpos
        self.sendReg(wregval, self.card_addr)

    def send_dfb_GPI18(self):
        log.debug("DFB:GPI18: ARL negative relock delay:", self.RLDneg)
        wreg = 18 << 17
        wregval = wreg | self.RLDneg
        self.sendReg(wregval, self.card_addr)

    def sendReg(self, wregval, addr):
        log.debug(tc.COMMAND + "send to address",
                  addr, ":", tc.BOLD, wregval, tc.ENDC)
        b0 = (wregval & 0x7f) << 1			# 1st 7 bits shifted up 1
        b1 = ((wregval >> 7) & 0x7f) << 1	 # 2nd 7 bits shifted up 1
        b2 = ((wregval >> 14) & 0x7f) << 1	 # 3rd 7 bits shifted up 1
        b3 = ((wregval >> 21) & 0x7f) << 1	 # 4th 7 bits shifted up 1
        # Address shifted up 1 bit with address bit set
        b4 = (addr << 1) + 1
        msg = struct.pack('BBBBB', b0, b1, b2, b3, b4)
        self.serialport.write(msg)
        time.sleep(0.001)

    def saveSettings(self):
        log.debug(tc.FCTCALL + "saving settings in pickle file:", tc.ENDC)
        filename = str(QFileDialog.getSaveFileName()[0])
        if len(filename) > 0:
            # 		if (filename != []):
            if filename[-4:] == '.pkl':
                savename = filename
            else:
                savename = filename + '.pkl'
            log.debug("saving settings file:", filename)

        else:
            log.debug(tc.FAIL + "save file cancelled:", tc.ENDC)

            return
        self.filenameEdit.setText(savename)
        self.packCrateConfig()
        self.packGlobals()
        self.loadGlobals = self.saveGlobals
        self.packClassParameters()
        self.loadClassParameters = self.saveClassParameters
        self.packTower()
        self.loadTower = self.saveTower
        self.packTune()
        self.loadTune = self.saveTune
        currentState = {'CrateConfig': self.saveCrateConfig, 'globals': self.saveGlobals, 'classParameters': self.saveClassParameters,
                        "Tower": self.saveTower, "Tune": self.saveTune}

        f = open(savename, "wb")

        log.debug(
            tc.FCTCALL + ("Saving current settings in pickle format to %s" % savename), tc.ENDC)

        pickle.dump(currentState, f, protocol=0)
        f.close()

    def loadSettings(self, filename=None):
        log.debug(tc.FCTCALL + "load pickle settings:", tc.ENDC)

        if filename is None or filename == False:
            self.load_filename = str(QFileDialog.getOpenFileName())
        else:
            self.load_filename = filename
        self.filenameEdit.setText(self.load_filename)
        log.debug("loading file: [%s]" % self.load_filename)

        f = open(self.load_filename, "rb")
        load_sys_config = pickle.load(f)
        f.close()

        '''test here for matching crate dimensions
        Vectors are used for build, they represent the structure (cards, addresses, names, location)
        Parameters are settings like a DAC value, or LYSNC or ARL on/off
        '''
        self.loadCrateConfig = load_sys_config['CrateConfig']
        LoadSlotVector = self.loadCrateConfig['SlotVector']
        LoadAddressVector = self.loadCrateConfig['AddressVector']
        LoadClassVector = self.loadCrateConfig['ClassVector']

        log.debug(tc.INIT + "crate configuration settings (from file):")

        log.debug("number of cards in crate: %i" % len(LoadAddressVector))
        log.debug("Type	Address	Slot")
        for idx, val in enumerate(LoadClassVector):
            log.debug(
                val, "	", LoadAddressVector[idx], "	", LoadSlotVector[idx])
        log.debug(tc.ENDC)
        if LoadSlotVector != self.slot_vector or LoadAddressVector != self.addr_vector or LoadClassVector != self.class_vector:
            log.debug(
                tc.FAIL + "load crate configuration does NOT match instantiated crate configuration")
            log.debug("load configuration aborted" + tc.ENDC)

            return

        self.loadGlobals = load_sys_config['globals']
        self.loadClassParameters = load_sys_config['classParameters']
        self.loadTower = load_sys_config["Tower"]
        if "Tune" in list(load_sys_config.keys()):
            self.loadTune = load_sys_config["Tune"]
        self.assertSettings()

    def assertSettings(self):
        log.debug(tc.FCTCALL + ("asserting loaded or last saved settings"), tc.ENDC)

        self.unpackGlobals()
        self.unpackClassParameters()
        self.unpackTower()
        self.unpackTune()

    def packTower(self):
        self.saveTower = {}
        self.saveTower['TowerVector'] = self.tower_vector
        if self.tower_widget is not None:
            self.saveTower["TowerParameters"] = self.tower_widget.packState()

    def packTune(self):
        self.saveTune = {}
        if self.tune_widget is not None:
            self.saveTune["TuneParameters"] = self.tune_widget.packState()

    def packCrateConfig(self):
        self.saveCrateConfig = {
            'SlotVector'		:	self.slot_vector,
            'AddressVector'		:	self.addr_vector,
            'ClassVector'		:	self.class_vector,
        }

    def packGlobals(self):
        self.saveGlobals = {
            'SEQLN'			:	self.seqln,
            'LSYNC'			:	self.lsync,
            'SETT'			:	self.SETT,
            'NSAMP'			:	self.NSAMP,
            'PROP_DELAY'	:	self.prop_delay,
            'DFB_DELAY'		:	self.dfb_delay,
            'DFBx2_XPT'		:	self.dfbx2_XPT,
            'DFBCLK_XPT'	:	self.dfbclk_XPT,
            'TP'			:	self.TP,
            'PS'			:	self.PS,
            'ARL_SENSE'		:	self.ARLsense,
            'RLD_POS'		:	self.RLDpos,
            'RLD_NEG'		:	self.RLDneg,
            'RLD_TRACK'		:	self.RLD_track_state,
            'TRI_DWELL'		:	self.dwell_val,
            'TRI_RANGE'		:	self.range_val,
            'TRI_STEP'		:	self.step_val,
            'TRI_IDX'		:	self.tri_idx
        }

    def unpackGlobals(self):
        self.seqln_spin.setValue(self.loadGlobals['SEQLN'])
        self.lsync_spin.setValue(self.loadGlobals['LSYNC'])
        self.SETT_spin.setValue(self.loadGlobals['SETT'])
        self.NSAMP_spin.setValue(self.loadGlobals['NSAMP'])
        self.prop_delay_spin.setValue(self.loadGlobals['PROP_DELAY'])
        self.dfb_delay_spin.setValue(self.loadGlobals['DFB_DELAY'])
        self.dfbx2_xpt_mode.setCurrentIndex(self.loadGlobals['DFBx2_XPT'])
        self.dfbclk_xpt_mode.setCurrentIndex(self.loadGlobals['DFBCLK_XPT'])
        self.tp_mode.setCurrentIndex(self.loadGlobals['TP'])
        self.PS_button.setChecked(self.loadGlobals['PS'])
        self.ARLsense_spin.setValue(self.loadGlobals['ARL_SENSE'])
        self.RLDpos_spin.setValue(self.loadGlobals['RLD_POS'])
        self.RLDneg_spin.setValue(self.loadGlobals['RLD_NEG'])
        self.RLD_frame.setChecked(self.loadGlobals['RLD_TRACK'])
        self.dwell.setValue(self.loadGlobals['TRI_DWELL'])
        self.range.setValue(self.loadGlobals['TRI_RANGE'])
        self.step.setValue(self.loadGlobals['TRI_STEP'])
        self.tri_idx_button.setChecked(self.loadGlobals['TRI_IDX'])

    def packClassParameters(self):
        for idx, val in enumerate(self.class_vector):
            self.crate_widgets[idx].packClass()
            self.saveClassParameters['classParameters%i' %
                                     idx] = self.crate_widgets[idx].classParameters

    def unpackClassParameters(self):
        for idx, val in enumerate(self.class_vector):
            self.crate_widgets[idx].unpackClass(
                self.loadClassParameters['classParameters%i' % idx])

    def unpackTower(self):
        if self.tower_widget is not None and "TowerParameters" in self.loadTower:
            self.tower_widget.unpackState(self.loadTower["TowerParameters"])

    def unpackTune(self):
        self.tune_widget.unpackState(self.loadTune["TuneParameters"])


def main():

    app = QApplication(sys.argv)
    app.setStyle("plastique")
    app.setStyleSheet("""	QPushbutton{font: 10px; padding: 6px}
                            QToolButton{font: 10px; padding: 6px}
                            QLineEdit {background-color: #FFFFCC;}
                            QToolTip {background-color: #FFFFCC}""")

    class MyParser(argparse.ArgumentParser):
        def error(self, message, help=True):
            """override error to print out the help"""
            if help:
                self.print_help()
            sys.stderr.write(message+"\n")
            sys.exit(2)
    p = MyParser(
        description='Enter or import crate card details for GUI build: default 8x32 standard configuration')
    p.add_argument('-A', '--card_address', action='store', dest='addr_vector', type=int, nargs='+',
                   help='List of hardware addresses of cards, example: -A 1 3 5 7 9 32 33.')
    p.add_argument('-S', '--slot', action='store', dest='slot_vector', type=int, nargs='+',
                   help='List of slot positions in crate, example: -S 1 3 4 5 6 10 11')
    p.add_argument('-C', '--type', action='store', dest='class_vector', type=str, nargs='+',
                   help='List of card types, example: DFBCLK DFBx2 DFBx2 DFBx2 DFBx2 BAD16 BAD16')
    p.add_argument('-T', '--tower', action="store", dest="tower_vector", type=str, nargs='+',
                   help="for tower provide a list of names and addresses, example: -T DB1 13 SAb 4 SQ1b 12")
    p.add_argument('-F', '--file', action='store', dest='setup_filename', type=str, nargs=1, default="",
                   help='Setup file from which to extract CRATE configuration, example: -F cringe_save.pkl')
    p.add_argument('-L', '--load', action='store_true',
                   help='Use file dialog to load file from which to extract CRATE configuration')
    p.add_argument('-i', '--interactive', action='store_true', dest='interactive',
                   help='Drop into an interactive IPython window')  # ADDED JG
    p.add_argument('-r', '--raw', action="store_true", dest="raw",
                   help="add the Calibration tab (experimental)")
    p.add_argument('-D', '--debug', action="store_true",
                   help="enable debug log level, aka print out EVERYTHING like what values are sent to what registers")

    args = p.parse_args()

    if args.debug:
        log.set_debug()
    log.info("cringe.main with args={}".format(args))

    if not any(vars(args).values()):
        # this exits
        p.error(tc.BOLD+"No arguments provided. You probably want -L or -F."+tc.ENDC)

    def noneLen(x):
        if x is None:
            return 0
        return len(x)

    if not noneLen(args.addr_vector) == noneLen(args.slot_vector) == noneLen(args.class_vector):
        p.error(
            tc.BOLD+"-A, -S and -C must all have the same number of arguments"+tc.ENDC)

    # -F gives setup_file which takes a filename from the command line
    # -L gets a filename from a open file dialog
    if args.setup_filename != "" or args.load:
        load_file = None
        if args.load:
            load_filename = str(QFileDialog.getOpenFileName(caption="choose cringe file",
                                                            directory=os.path.expanduser("~/cringe_config"), filter="(*.pkl)")[0])
        else:
            load_filename = args.setup_filename
        load_file = open(load_filename, "rb")
        log.info("build GUI from file:")
        log.info(f"load filename: {load_filename}")
        log.info(f"load file: {load_file}")

        load_sys_config = pickle.load(load_file)
        load_file.close()
        load_on_launch = True
        log.info("failed to interpret file %s" % load_filename)
        # these are global variables accessed later
        tower_vector = load_sys_config['Tower']['TowerVector']
        loadCrateConfig = load_sys_config['CrateConfig']
        slot_vector = loadCrateConfig['SlotVector']
        addr_vector = loadCrateConfig['AddressVector']
        class_vector = loadCrateConfig['ClassVector']
        log.info("slot vector from file:")
        log.info(slot_vector)

        log.info("address vector from file:")
        log.info(addr_vector)

        log.info("class vector from file:")
        log.info(class_vector)

    else:
        load_on_launch = False

    if not load_on_launch:
        addr_vector = args.addr_vector
        slot_vector = args.slot_vector
        class_vector = args.class_vector
        tower_vector = args.tower_vector
    else:
        if not args.tower_vector is None:
            log.info("using tower vector from command line, not from file")
            tower_vector = args.tower_vector
            log.info(tower_vector)

    # old main below here
    app.processEvents()

    # later it will run load settings if argfilename is not None
    if load_on_launch:
        argfilename = load_filename
    else:
        argfilename = None

    win = Cringe(addr_vector=addr_vector, slot_vector=slot_vector, class_vector=class_vector, tower_vector=tower_vector,
                 argfilename=argfilename, calibrationtab=args.raw)

    win.show()
    if args.interactive:
        IPython.embed()

    app.exec_()


if __name__ == '__main__':
    main()

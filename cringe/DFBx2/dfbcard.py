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

class dfbcard(QtGui.QWidget):

    def __init__(self, parent=None, addr=None, slot=None, seqln=None, lsync=32):

        super(dfbcard, self).__init__()


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

        self.serialport = named_serial.Serial(port='rack', shared = True)

        self.states = 64

        self.parent = parent
        self.address = addr
        self.slot = slot
        self.seqln = seqln
        self.lsync = lsync
# 		self.frame = self.lsync * self.seqln

        '''global booleans'''

        self.LED = False
        self.ST = False
        self.CLK = False

        self.PS = False
        self.GR = True

        '''global variables'''

        self.XPT = 0
        self.NSAMP = 4
        self.prop_delay = 0
        self.card_delay = 0
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
        self.wreg6 = 201761284
        self.wreg7 = 235668492


        self.chn_vectors = []
# 		self.enb = [0,0,0,0,0,0,0]
# 		self.cal_coeffs = [0,0,0,0,0,0,0]
# 		self.appTrim =[0,0,0,0,0,0,0]

        self.setWindowTitle("DFBx2: %d/%d"%(slot, addr))	# Phase Offset Widget
        self.setGeometry(30,30,1300,1000)
        self.setContentsMargins(0,0,0,0)

        self.layout_widget = QtGui.QWidget(self)
        self.layout = QtGui.QGridLayout(self)

        print(self.INIT + "building DFBx2 card: slot", self.slot, "/ address", self.address, self.ENDC)
        print()

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
        self.controls_layout.addWidget(self.addr_indicator,0,2,1,1,QtCore.Qt.AlignRight)

        self.addr_label = QtGui.QLabel("card address")
        self.controls_layout.addWidget(self.addr_label,0,3,1,1,QtCore.Qt.AlignLeft)

        self.slot_indicator = QtGui.QLineEdit()
        self.slot_indicator.setReadOnly(True)
# 		self.addr_indicator.setFixedWidth(40)
        self.slot_indicator.setText('%2d'%slot)
        self.slot_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.slot_indicator.setFocusPolicy(Qt.NoFocus)
        self.controls_layout.addWidget(self.slot_indicator,0,0,1,1,QtCore.Qt.AlignRight)

        self.slot_label = QtGui.QLabel("card slot")
        self.controls_layout.addWidget(self.slot_label,0,1,1,1,QtCore.Qt.AlignLeft)

        self.layout.addWidget(self.class_interface_widget,4,1,1,1,QtCore.Qt.AlignRight)

        '''
        create TAB widget for embedding BAD16 functional widgets
        '''
        self.dfbx2_widget = QtGui.QTabWidget(self)

        self.dfbx2_widget1 = dfbrap.dfbrap(parent=self, addr=addr, slot=slot, column=1, seqln=seqln, lsync=lsync)
        self.dfbx2_widget.addTab(self.dfbx2_widget1, " CH1 ")

        self.dfbx2_widget2 = dfbrap.dfbrap(parent=self, addr=addr, slot=slot, column=2, seqln=seqln, lsync=lsync)
        self.dfbx2_widget.addTab(self.dfbx2_widget2, " CH2 ")

        self.dfbx2_widget3 = dprcal.dprcal(ctype="DFBx2", addr=addr, slot=slot)
        self.dfbx2_widget.addTab(self.dfbx2_widget3, " phase ")

        self.layout.addWidget(self.dfbx2_widget,5,0,1,2)

        '''
        resize widgets for relative, platform dependent variability
        '''
        rm = 45
# 		self.file_mgmt_widget.setFixedWidth(self.dfbx2_widget1.width()+rm)
# 		self.sys_glob_hdr_widget.setFixedWidth(self.dfbx2_widget1.width()+rm)
# 		self.class_glob_hdr_widget.setFixedWidth(self.dfbx2_widget1.width()+rm)
# 		self.arl_widget.setFixedWidth(self.dfbx2_widget1.width()/2+10)
# 		self.tri_wvfm_widget.setFixedWidth(self.dfbx2_widget1.width()/2+10)
        self.card_glb_widget.setFixedWidth(self.dfbx2_widget1.width()/2+10)
        self.class_interface_widget.setFixedWidth(self.dfbx2_widget1.width()/2+10)

        '''
        initialization
        '''
        for idx in range(self.states):
            if idx < self.seqln:
                self.dfbx2_widget1.state_vectors[idx].setEnabled(1)
                self.dfbx2_widget2.state_vectors[idx].setEnabled(1)
            else:
                self.dfbx2_widget1.state_vectors[idx].setEnabled(0)
                self.dfbx2_widget2.state_vectors[idx].setEnabled(0)

    def seqln_changed(self, seqln):
        print(self.FCTCALL + "send SEQLN to DFBx2 card:", self.ENDC)
        self.seqln = seqln
# 		self.wreg7 = ((self.wreg7 & 0xfffc0ff) | (self.seqln << 8))
        self.send_dfb_wreg7()
# 		if self.tri_idx == 1:
# 			self.period_changed()
        for idx in range(self.states):
            if idx < self.seqln:
                self.dfbx2_widget1.state_vectors[idx].setEnabled(1)
                self.dfbx2_widget2.state_vectors[idx].setEnabled(1)
            else:
                self.dfbx2_widget1.state_vectors[idx].setEnabled(0)
                self.dfbx2_widget2.state_vectors[idx].setEnabled(0)
        print()

    def LED_changed(self):
        print(self.FCTCALL + "send LED boolean (True = OFF) to DFBx2 card:" + self.ENDC)
        self.LED = self.LED_button.isChecked()
        if self.LED ==1:
            self.LED_button.setStyleSheet("background-color: #" + self.red + ";")
            self.LED_button.setText('OFF')
        else:
            self.LED_button.setStyleSheet("background-color: #" + self.green + ";")
            self.LED_button.setText('ON')
#		 if self.unlocked == 1:
        self.send_dfb_wreg7()
        print()

    def status_changed(self):
        print(self.FCTCALL + "send ST boolean to DFBx2 card:" + self.ENDC)
        self.ST = self.status_button.isChecked()
        if self.ST ==1:
            self.status_button.setStyleSheet("background-color: #" + self.green + ";")
        else:
            self.status_button.setStyleSheet("background-color: #" + self.red + ";")
        self.send_dfb_wreg7()
        self.dfbx2_widget3.enbDiagnostic(self.ST)
        print()

    def send_triangle(self, wreg4):
        print(self.FCTCALL + "send triangle parameters to DFB CH1 on DFBx2 card:", self.ENDC)
        print("DFB:WREG0: page register: CH 1")
        self.sendReg(1 << 6)
        self.dfbx2_widget1.send_wreg4(wreg4)
        print()
        print(self.FCTCALL + "send triangle parameters to DFB CH2 on DFBx2 card:", self.ENDC)
        print("DFB:WREG0: page register: CH 2")
        self.sendReg(2 << 6)
        self.dfbx2_widget2.send_wreg4(wreg4)
        print()

###	child called methods

# 	def broadcast_channel(self, state):
# 		print "BROADCAST CHANNEL:", state

    def send_class_globals(self, wreg6, wreg7):
        print(self.FCTCALL + "send class globals to DFBx2 card:", self.ENDC)
        self.wreg6 = wreg6
        self.wreg7 = wreg7
        self.send_global_regs()
        print()

    def send_card_globals(self):
        print(self.FCTCALL + "send card globals to DFBx2 card:", self.ENDC)
        self.send_dfbx2_wreg6()
        self.send_dfb_wreg7()
        print()

    def send_global_regs(self):
        cmd_reg = bin(self.wreg6)[5:].zfill(25)
        PS = cmd_reg[0]
        XPT = int(cmd_reg[1:4], base=2)
        NSAMP = int(cmd_reg[17:], base=2)
        print("DFB:WREG6: global parameters: PS, DFBx2_XPT, NSAMP:", PS, XPT, NSAMP)
        self.sendReg(self.wreg6)
        cmd_reg = bin(self.wreg7)[5:].zfill(25)
        PD = int(cmd_reg[3:7], base=2)
        CD = int(cmd_reg[7:11], base=2)
        SL = int(cmd_reg[11:17], base=2)
        SE = int(cmd_reg[17:], base=2)
        print("DFB:WREG7: global parameters: LED, ST, Prop Delay, Card Delay, sequence length, SETT:", self.LED, self.ST, PD, CD, SL, SE)
        self.sendReg((self.wreg7 & 0xF3FFFFF) | (self.LED << 23) | (self.ST << 22))
        print()

    def send_dfbx2_wreg6(self):
        if self.parent != None:
            self.parent.send_dfbx2_wreg6(self.address)
        else:
            print("DFB:WREG6: global parameters: PS, DFBx2_XPT, NSAMP:", self.PS, self.dfbx2_XPT, self.NSAMP)
            print()
            self.wreg6 = (6 << 25) | (self.PS << 24) | (self.dfbx2_XPT << 21) | self.NSAMP
            self.sendReg(self.wreg6)

    def send_dfb_wreg7(self):
        if self.parent != None:
            self.parent.send_dfb_wreg7(self.LED, self.ST, self.address)
        else:
            print("DFB:WREG7: global parameters: LED, ST, prop delay, dfb delay, sequence length, SETT:", self.LED, self.ST, self.prop_delay, \
                self.dfb_delay, self.seqln, self.SETT)
            self.wreg7 = (7 << 25) | (self.LED << 23) | (self.ST << 22) | (self.prop_delay << 18) \
                | (self.dfb_delay << 14) | (self.seqln << 8) | self.SETT
            self.sendReg(self.wreg7)
            print()

# 	def send_wreg7(self):
# 		print "WREG7: global parameters: LED, ST, delays, sequence length, SETT"
# 		self.sendReg(self.wreg7 | (self.LED << 23) | (self.ST << 22))

    def send_GPI4(self):
        print("DFB:GPI4: test mode select")
        wreg = 4 << 17
        wregval = wreg | self.TP
        self.sendReg(wregval)
        print()

    def send_GPI5(self):
        print("DFB:GPI5: test pattern hi-bytes [31..16]")
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
        wregval = wreg | hibytes
        self.sendReg(wregval)
        print()

    def send_GPI6(self):
        print("DFB:GPI6: test pattern lo-bytes [15..0]")
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
        wregval = wreg | lobytes
        self.sendReg(wregval)
        print()

    def sendReg(self, wregval):
        print(self.COMMAND + "send to address", self.address, ":", self.BOLD, wregval, self.ENDC)
        b0 = (wregval & 0x7f ) << 1			# 1st 7 bits shifted up 1
        b1 = ((wregval >> 7) & 0x7f) <<  1	 # 2nd 7 bits shifted up 1
        b2 = ((wregval >> 14) & 0x7f) << 1	 # 3rd 7 bits shifted up 1
        b3 = ((wregval >> 21) & 0x7f) << 1	 # 4th 7 bits shifted up 1
        b4 = (self.address << 1) + 1		   # Address shifted up 1 bit with address bit set
        msg = struct.pack('BBBBB', b0, b1, b2, b3, b4)
        self.serialport.write(msg)
        time.sleep(0.001)

    def packCARDglobals(self):
        self.CARDglobals = {	'LED'	:	self.LED_button.isChecked(),
                                'ST'	:	self.status_button.isChecked()}

    def unpackCARDglobals(self, CARDglobals):
        self.LED_button.setChecked(CARDglobals['LED'])
        self.status_button.setChecked(CARDglobals['ST'])

    def packClass(self):
        self.packCARDglobals()
        self.dfbx2_widget1.packCHglobals()
        self.dfbx2_widget1.packMasterVector()
        self.dfbx2_widget1.packStates()
        self.dfbx2_widget2.packCHglobals()
        self.dfbx2_widget2.packMasterVector()
        self.dfbx2_widget2.packStates()
        self.dfbx2_widget3.packCal()
        self.classParameters = {	'CARDglobals'		:	self.CARDglobals,
                                    'CHglobals1'		:	self.dfbx2_widget1.CHglobals,
                                    'CHglobals2'		:	self.dfbx2_widget2.CHglobals,
                                    'dfbMasterVector1'	:	self.dfbx2_widget1.MasterState,
                                    'dfbMasterVector2'	:	self.dfbx2_widget2.MasterState,
                                    'dfbAllStates1'		:	self.dfbx2_widget1.allStates,
                                    'dfbAllStates2'		:	self.dfbx2_widget2.allStates,
                                    'CARDphase'			:	self.dfbx2_widget3.CalCoeffs,
                                }

    def unpackClass(self, classParameters):
        CARDglobals = classParameters['CARDglobals']
        self.unpackCARDglobals(CARDglobals)
        CHglobals1 = classParameters['CHglobals1']
        self.dfbx2_widget1.unpackCHglobals(CHglobals1)
        masterVector1 = classParameters['dfbMasterVector1']
        self.dfbx2_widget1.unpackMasterVector(masterVector1)
        dfbAllStates1 = classParameters['dfbAllStates1']
        self.dfbx2_widget1.unpackStates(dfbAllStates1)
        CHglobals2 = classParameters['CHglobals2']
        self.dfbx2_widget2.unpackCHglobals(CHglobals2)
        masterVector2 = classParameters['dfbMasterVector2']
        self.dfbx2_widget2.unpackMasterVector(masterVector2)
        dfbAllStates2 = classParameters['dfbAllStates2']
        self.dfbx2_widget2.unpackStates(dfbAllStates2)
        CARDphase = classParameters['CARDphase']
        self.dfbx2_widget3.unpackCal(CARDphase)


def main():

    app = QtGui.QApplication(sys.argv)
    app.setStyle("plastique")
    app.setStyleSheet("""	QPushbutton{font: 10px; padding: 6px}
                            QToolButton{font: 10px; padding: 6px}
                            QLineEdit {background-color: #FFFFCC;}""")
    win = dfbcard(addr=addr, slot=slot, seqln=seqln)
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


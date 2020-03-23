#-*- coding: utf-8 -*-
import sys
import optparse
import struct
import time

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import named_serial
from . import scream
from . import dprS
# from dprcal import dprcal

class dfbscard(QWidget):

	def __init__(self, parent=None, addr=None, slot=None, lsync=32):
		
		super(dfbscard, self).__init__()
		

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
# 		self.seqln = seqln
		self.lsync = lsync

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
		
# 		self.frame_period = self.lsync * self.seqln * 0.008
					
		'''triangle default parameters'''
		
		self.TriDwell = 0
		self.TriRange = 10
		self.TriStep = 8

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
		
		self.layout_widget = QWidget(self)
		self.layout = QGridLayout(self)
		
		print(self.INIT + "building DFBscream card: slot", self.slot, "/ address", self.address, self.ENDC)
		print()
		
		'''
		build widget for CARD GLOBAL VARIABLE control
		'''
		self.card_glb_widget = QGroupBox(self)
		self.card_glb_widget.setTitle("CARD GLOBAL VARIABLES")
		self.card_glb_layout = QGridLayout(self.card_glb_widget)
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

		self.led_lbl = QLabel("LED control")
		self.card_glb_layout.addWidget(self.led_lbl,0,1,1,1,QtCore.Qt.AlignLeft)

		self.status_button = QToolButton(self, text = 'ST')
		self.status_button.setFixedHeight(25)
		self.status_button.setCheckable(1)
		self.status_button.setChecked(self.ST)
		self.status_button.setStyleSheet("background-color: #" + self.red + ";")
		self.card_glb_layout.addWidget(self.status_button,0,2,1,1)
		self.status_button.toggled.connect(self.status_changed)

		self.status_lbl = QLabel("status bit")
		self.card_glb_layout.addWidget(self.status_lbl,0,3,1,1,QtCore.Qt.AlignLeft)
			
		self.card_glb_send = QPushButton(self, text = "send CARD globals")
		self.card_glb_send.setFixedHeight(25)
		self.card_glb_send.setFixedWidth(200)
		self.card_glb_layout.addWidget(self.card_glb_send,0,4,1,1,QtCore.Qt.AlignRight)
		self.card_glb_send.clicked.connect(self.send_card_globals)

		self.layout.addWidget(self.card_glb_widget,4,0,1,1)

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
		self.controls_layout.addWidget(self.addr_indicator,0,2,1,1,QtCore.Qt.AlignRight)
		
		self.addr_label = QLabel("card address")
		self.controls_layout.addWidget(self.addr_label,0,3,1,1,QtCore.Qt.AlignLeft)

		self.slot_indicator = QLineEdit()
		self.slot_indicator.setReadOnly(True)
# 		self.addr_indicator.setFixedWidth(40)
		self.slot_indicator.setText('%2d'%slot)
		self.slot_indicator.setAlignment(QtCore.Qt.AlignRight)
		self.slot_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
		self.controls_layout.addWidget(self.slot_indicator,0,0,1,1,QtCore.Qt.AlignRight)

		self.slot_label = QLabel("card slot")
		self.controls_layout.addWidget(self.slot_label,0,1,1,1,QtCore.Qt.AlignLeft)

		self.layout.addWidget(self.class_interface_widget,4,1,1,1,QtCore.Qt.AlignRight)

		'''
		create TAB widget for embedding BAD16 functional widgets
		'''
		self.dfbs_widget = QTabWidget(self)

		self.dfbs_widget1 = scream.scream(parent=self, addr=addr, slot=slot, channel=1, lsync=lsync)
		self.dfbs_widget.addTab(self.dfbs_widget1, " CH1 ")
		
		self.dfbs_widget2 = scream.scream(parent=self, addr=addr, slot=slot, channel=2, lsync=lsync)
		self.dfbs_widget.addTab(self.dfbs_widget2, " CH2 ")
		
		self.dfbs_widget3 = dprS.dprS(ctype="DFBs", addr=addr, slot=slot)
		self.dfbs_widget.addTab(self.dfbs_widget3, " phase ")
		
		self.layout.addWidget(self.dfbs_widget,5,0,1,2)
		
		'''
		resize widgets for relative, platform dependent variability
		'''
		rm = 45
# 		self.file_mgmt_widget.setFixedWidth(self.dfbx2_widget1.width()+rm)
# 		self.sys_glob_hdr_widget.setFixedWidth(self.dfbx2_widget1.width()+rm)
# 		self.class_glob_hdr_widget.setFixedWidth(self.dfbx2_widget1.width()+rm)
# 		self.arl_widget.setFixedWidth(self.dfbx2_widget1.width()/2+10)
# 		self.tri_wvfm_widget.setFixedWidth(self.dfbx2_widget1.width()/2+10)
		self.card_glb_widget.setFixedWidth(self.dfbs_widget1.width()/2+10)
		self.class_interface_widget.setFixedWidth(self.dfbs_widget1.width()/2+10)
		
	def LED_changed(self):
		self.LED = self.LED_button.isChecked()
		print("SCREAM LED boolean (True = OFF):", self.LED, self.ENDC)
		if self.LED ==1:
			self.LED_button.setStyleSheet("background-color: #" + self.red + ";")			
			self.LED_button.setText('OFF')
		else:
			self.LED_button.setStyleSheet("background-color: #" + self.green + ";")
			self.LED_button.setText('ON')
#		 if self.unlocked == 1:
		self.send_cmd(2, self.LED)

	def status_changed(self):
		self.ST = self.status_button.isChecked()
		print("SCREAM ST boolean:", self.ST, self.ENDC)
		if self.ST ==1:
			self.status_button.setStyleSheet("background-color: #" + self.green + ";")
		else:
			self.status_button.setStyleSheet("background-color: #" + self.red + ";")			
		self.send_cmd(3, self.ST)
		self.dfbs_widget1.enbDiagnostic(self.ST)
		self.dfbs_widget2.enbDiagnostic(self.ST)
		self.dfbs_widget3.enbDiagnostic(self.ST)
		
	def send_card_globals(self):
		print()
		print(self.FCTCALL + "send card globals to SCREAM card:", self.ENDC)
		self.LED_changed()
		self.status_changed()

	def send_channel_globals(self):
		self.dfbs_widget1.send_channel_globals()
		self.dfbs_widget2.send_channel_globals()
		
	def decode_tp(self):
		if self.TP == 0:
			return
		if self.TP == 1:
			self.lobytes = 0x5555
			self.hibytes = 0x5555
		if self.TP == 2:
			self.lobytes = 0xaaaa
			self.hibytes = 0xaaaa
		if self.TP == 3:
			self.lobytes = 0x3333
			self.hibytes = 0x3333
		if self.TP == 4:
			self.lobytes = 0x0f0f
			self.hibytes = 0x0f0f
		if self.TP == 5:
			self.lobytes = 0x00ff
			self.hibytes = 0x00ff
		if self.TP == 6:
			self.lobytes = 0xffff
			self.hibytes = 0x0000
		if self.TP == 7:
			self.lobytes = 0x0000
			self.hibytes = 0x0000
		if self.TP == 8:
			self.lobytes = 0xffff
			self.hibytes = 0xffff
		if self.TP == 9:
			self.lobytes = 0xf00d
			self.hibytes = 0x8bad
	

	def send_global(self, parameter, value):
		if parameter == "PS":
			self.PS = value
			GPI = 8
			print("SCREAM PS:", value, self.ENDC)
		if parameter == "XPT":
			self.XPT = value
			GPI = 9
			print("SCREAM XPT:", value, self.ENDC)
		if parameter == "TP":
			self.TP = value
			self.TPboolean = 0
			if value != 0:
				self.TPboolean = 1
				self.decode_tp()
				print("SCREAM Test Pattern Hi Byte:", hex(self.hibytes), self.ENDC)
				GPI = 12
				self.send_cmd(GPI, self.hibytes)
				print("SCREAM Test Pattern Lo Byte:", hex(self.lobytes), self.ENDC)
				GPI = 13
				self.send_cmd(GPI, self.lobytes)
			GPI = 11
			value = self.TPboolean
			print("SCREAM Test Pattern Boolean:", self.TPboolean, self.ENDC)
		if parameter == "NSAMP":
			self.NSAMP = value
			GPI = 40
			print("SCREAM NSAMP:", value, self.ENDC)
		if parameter == "SETT":
			self.SETT = value
			GPI = 41
			print("SCREAM SETT:", value, self.ENDC)
		if parameter == "CARD":
			self.card_delay = value
			GPI = 42
			print("SCREAM CARD_DELAY:", value, self.ENDC)
		if parameter == "PROP":
			self.prop_delay = value
			GPI = 43
			print("SCREAM PROP_DELAY:", value, self.ENDC)
		self.send_cmd(GPI, value)

	def send_ARL(self, parameter, value):
		if parameter == "ARLsense":
			self.ARLsense = value
			GPI = 16
			print("SCREAM ARLsense:", value, self.ENDC)
		if parameter == "RLDpos":
			self.RLDpos = value
			GPI = 17
			print("SCREAM RLDpos:", value, self.ENDC)
		if parameter == "RLDneg":
			self.RLDneg = value
			GPI = 18
			print("SCREAM RLDneg:", value, self.ENDC)
		self.send_cmd(GPI, value)

	def send_triangle(self, parameter, value):
		if parameter == "dwell":
			self.TriDwell = value
			print("SCREAM DWELL:", value)
			self.send_cmd(33, value)
		if parameter == "range":
			self.TriRange = value
			print("SCREAM RANGE:", value)
			self.send_cmd(34, value)
		if parameter == "step":
			self.TriStep = value
			print("SCREAM STEP:", value)
			self.send_cmd(35, value)

	def send_cmd(self, GPI, val): 
		wregval = (GPI << 20) | val
		print(self.COMMAND + "send to card address", self.address, "/ GPI", GPI, ":", self.BOLD, wregval, "(", val, ")",self.ENDC)
		b0 = (wregval & 0x7f ) << 1				# 0-6 bits shifted up 1
		b1 = ((wregval >> 7) & 0x7f) <<  1	 	# 7-13 bits shifted up 1
		b2 = ((wregval >> 14) & 0x7f) << 1	 	# 14-19 bits shifted up 1
		b3 = ((wregval >> 21) & 0x7f) << 1	 	# 4th 7 bits shifted up 1
		b4 = (self.address << 1) + 1		  	# Address shifted up 1 bit with address bit set
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
		self.dfbs_widget1.packCHglobals()
		self.dfbs_widget1.packMasterVector()
# 		self.dfbs_widget1.packStates()
		self.dfbs_widget2.packCHglobals()
		self.dfbs_widget2.packMasterVector()
# 		self.dfbs_widget2.packStates()
		self.dfbs_widget3.packCal()
		self.classParameters = {	'CARDglobals'		:	self.CARDglobals,
									'CHglobals1'		:	self.dfbs_widget1.CHglobals,
									'CHglobals2'		:	self.dfbs_widget2.CHglobals,
									'dfbMasterVector1'	:	self.dfbs_widget1.MasterState,
									'dfbMasterVector2'	:	self.dfbs_widget2.MasterState,
									'CARDphase'			:	self.dfbs_widget3.CalCoeffs,
								}

# 									'dfbAllStates1'		:	self.dfbs_widget1.allStates,
# 									'dfbAllStates2'		:	self.dfbs_widget2.allStates,

	def unpackClass(self, classParameters):
		CARDglobals = classParameters['CARDglobals']
		self.unpackCARDglobals(CARDglobals)
		CHglobals1 = classParameters['CHglobals1']
		self.dfbs_widget1.unpackCHglobals(CHglobals1)
		masterVector1 = classParameters['dfbMasterVector1']
		self.dfbs_widget1.unpackMasterVector(masterVector1)
# 		dfbAllStates1 = classParameters['dfbAllStates1']
# 		self.dfbx2_widget1.unpackStates(dfbAllStates1)
		CHglobals2 = classParameters['CHglobals2']
		self.dfbs_widget2.unpackCHglobals(CHglobals2)
		masterVector2 = classParameters['dfbMasterVector2']
		self.dfbs_widget2.unpackMasterVector(masterVector2)
# 		dfbAllStates2 = classParameters['dfbAllStates2']
# 		self.dfbx2_widget2.unpackStates(dfbAllStates2)
		CARDphase = classParameters['CARDphase']
		self.dfbs_widget3.unpackCal(CARDphase)

		
def main():
	
	app = QApplication(sys.argv)
	app.setStyle("plastique")
	app.setStyleSheet("""	QPushbutton{font: 10px; padding: 6px}
							QToolButton{font: 10px; padding: 6px}
							QLineEdit {background-color: #FFFFCC;}""")
	win = dfbscard(addr=addr, slot=slot, seqln=seqln)
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
	

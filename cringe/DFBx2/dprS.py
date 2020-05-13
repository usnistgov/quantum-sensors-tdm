import sys
import optparse
import struct
import time

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

# import state_vector_builder
import named_serial
from .dprS_counter import dprS_counter
from cringe.shared import terminal_colors as tc


class dprS(QWidget):

	def __init__(self, ctype=None, addr=None, slot=None):
		super(dprS, self).__init__()

		self.serialport = named_serial.Serial(port='rack', shared = True)
		
		self.card_type = ctype
		self.address = addr
		self.slot = slot
		self.counters = 7
		
		self.GPI24 = (24 << 20) | self.slot
		
		self.phase_counters = []
# 		self.phase_counters_active = []
# 		self.nstates = 256
		self.enb = [0,0,0,0,0,0,0]
		self.cal_coeffs = [0,0,0,0,0,0,0]
		self.appTrim =[0,0,0,0,0,0,0]
		
		self.CalCoeffs = {}

		self.setWindowTitle("POW")	# Phase Offset Widget
		self.setGeometry(50,50,660,800)
		
		self.layout_widget = QWidget(self)
		self.layout = QVBoxLayout(self)
		
		self.globals_widget = QGroupBox(self.layout_widget)
		self.globals_layout = QGridLayout(self.globals_widget)
		
# 		self.slot_indicator = QLineEdit()
# 		self.slot_indicator.setReadOnly(True)
# 		self.slot_indicator.setFixedWidth(40)
# 		self.slot_indicator.setText(str(slot))
# 		self.slot_indicator.setAlignment(QtCore.Qt.AlignRight)
# 		self.globals_layout.addWidget(self.slot_indicator,0,0,QtCore.Qt.AlignLeft)
# 		
# 		self.slot_label = QLabel("card slot")
# 		self.globals_layout.addWidget(self.slot_label,0,1,1,4,QtCore.Qt.AlignLeft)
		
		self.loadcal = QPushButton(self, text = "load calibration")
		self.globals_layout.addWidget(self.loadcal,0,0,QtCore.Qt.AlignTop)

		self.savecal = QPushButton(self, text = "save calibration")
		self.globals_layout.addWidget(self.savecal,0,1,QtCore.Qt.AlignTop)

		self.autocal = QPushButton(self, text = "auto calibrate")
		self.globals_layout.addWidget(self.autocal,0,2,QtCore.Qt.AlignTop)

		self.null_phase = QPushButton(self, text = "null phase")
		self.null_phase.setEnabled(0)
		self.globals_layout.addWidget(self.null_phase,0,4,QtCore.Qt.AlignTop)

		tc.INIT_slot = QPushButton(self, text = "send slot")
		tc.INIT_slot.setEnabled(0)
		self.globals_layout.addWidget(tc.INIT_slot,1,4,QtCore.Qt.AlignTop)
		
		self.filenameEdit = QLineEdit()
		self.filenameEdit.setReadOnly(True)
		self.globals_layout.addWidget(self.filenameEdit,1,0,1,3)
		
		self.filename_label = QLabel("file")
		self.globals_layout.addWidget(self.filename_label,1,3,QtCore.Qt.AlignLeft)		

		self.arrayframe = QWidget(self.layout_widget)
		self.array_layout = QVBoxLayout(self.arrayframe)
		self.array_layout.setAlignment(QtCore.Qt.AlignHCenter)
		self.array_layout.setSpacing(20)
		self.array_layout.setContentsMargins(10,10,10,10)
		
		if ctype == 'BAD16':
			self.enabled = (0,2,3)
			self.ctr_dsc = ("ALTPLL_A","MCLK_PLL","DAC_CLK_1")
			self.ctr_fct = ("all clocks","master process clock","DAC clock 1/2")
			
		if ctype == 'DFBx1CLK':
			self.enabled = (0,2,3,4,5,6)
			self.cal_coeffs = [0,0,0,0,0,0,0]
			self.ctr_dsc = ("ALTPLL_A","MCLK_PLL","DCLK_PLL_1","DCLK_PLL_2","DCLK_PLL_3","DCLK_PLL_4",)
			self.ctr_fct = ("all clocks","master process clock","master data clock","line clock","data pipe 3","data pipe 4",)
			
		if ctype == 'DFBx2':
			self.enabled = (0,2,3,4,5,6)
			self.ctr_dsc = ("ALTPLL_A","MCLK_PLL","DCLK_PLL_1","DCLK_PLL_2","DCLK_PLL_3","DCLK_PLL_4",)
			self.ctr_fct = ("all clocks","master process clock","data pipe 1","data pipe 2","data pipe 3","data pipe 4",)

		if ctype == 'DFBs':
			self.enabled = (0,2,3,4,5,6)
			self.ctr_dsc = ("ALTPLL_A","MCLK_PLL","DCLK_PLL_1","DCLK_PLL_2","DCLK_PLL_3","DCLK_PLL_4",)
			self.ctr_fct = ("all clocks","master process clock","data pipe 1","data pipe 2","data pipe 3","data pipe 4",)

		for idx, val in enumerate(self.enabled):
			self.counters = idx + 1
			pcs = val
			self.phase_counters.append(dprS_counter(self, self.array_layout, pcs=pcs, idx=idx, slot=self.slot, coeffs=self.cal_coeffs, appTrim=self.appTrim, serialport=self.serialport, cardaddr=self.address))
			
		self.scrollarea = QScrollArea(self.layout_widget)
		self.scrollarea.setWidget(self.arrayframe)
		self.layout.addWidget(self.globals_widget)
		self.layout.addWidget(self.scrollarea)
				

		self.loadcal.clicked.connect(self.loadCalFile)
		self.savecal.clicked.connect(self.saveCalfile)
		self.autocal.clicked.connect(self.CalAllCounters)
		self.null_phase.clicked.connect(self.nullPhase)
		tc.INIT_slot.clicked.connect(self.sendSlot)
		
		self.scale_factor = self.phase_counters[0].width()
		self.globals_widget.setFixedWidth(self.scale_factor*1.1)
		
		'''initialization'''
# 		print tc.INIT + "Initialize:", tc.ENDC
# 		self.sendSlot()

	def resetALLphase(self):
		
		for i in range(self.counters):
# 		if self.enb[i] == 1:
			self.phase_counters[i].resetPhase()


	def loadCalFile(self):
		print(tc.FCTCALL + "load calibration file:", tc.ENDC)
		
		filename = str(QFileDialog.getOpenFileName())
# 		print("filename = [%s]" % filename)
		if len(filename) > 0:
			print("loading file:", filename)
			
			f = open(filename, 'r')
			for idx, val in enumerate(f):
# 				print idx, val
				self.cal_coeffs[idx] = int(val)
				self.phase_counters[idx].loadCal(int(val))				#	13apr16:	chg to call loadCal with passthru coefficient consistent with dprcal mar16 cleanup
# 				self.phase_counters[idx].cal_offset.setText(str(val))
			f.close()
			self.filenameEdit.setText(filename)
		else:
			print(tc.FAIL + "load file cancelled:", tc.ENDC)
			
			return
		
		if (idx + 1) != self.counters:
			print(tc.FAIL + "WARNING: calibration file does not meet card class format" + tc.ENDC)
			
			return
# 		for i in range(self.counters):
# 			self.phase_counters[i].loadCal()
			
	def saveCalfile(self):
		print(tc.FCTCALL + "save calibration file:", tc.ENDC)
		
		filestring = str(self.card_type)+"_S"+str(self.slot)+"A"+str(self.address)+".txt"
		filename = str(QFileDialog.getSaveFileName(self, "save calibration file",filestring))
		if len(filename) > 0:
			self.filenameEdit.setText(filename)
			print("saving calibration file:", filename)
			
			f = open(filename, 'w')
			for idx in range(self.counters):
				f.write(str(self.phase_counters[idx].cal_offset.text() +"\n"))
			f.close()
		else:
			print(tc.FAIL + "save file cancelled:", tc.ENDC)
			
			return

	def CalAllCounters(self):
		
		print(tc.INIT + "auto calibrate: ", self.card_type, "card address", self.address, tc.ENDC)
		for i in range(self.counters):
# 			if self.enb[i] == 1:
			self.phase_counters[i].calcounter()
			
	def enbDiagnostic(self, mode):
		self.null_phase.setEnabled(mode)
		tc.INIT_slot.setEnabled(mode)
			
	def nullPhase(self):
		print(tc.INIT + "null phase:", tc.ENDC)
		
		self.resetALLphase()
		print(tc.FCTCALL + "reset slot & phase trim to 0:", tc.ENDC)
		
		print("GPI24: send slot:", 0)
		self.send_cmd(24, 0)
		print("GPI25: send phase trim coefficient:", 0)
		self.send_cmd(25, 0)
		
		
	def sendSlot(self):
		print(tc.FCTCALL + "GPI24: send slot:", self.slot, tc.ENDC)
		
		self.send_cmd(24, self.slot)
				
	def sendReg(self, wregval):
		print(tc.COMMAND + "send to address", self.address, ":", tc.BOLD, wregval, tc.ENDC)
		b0 = (wregval & 0x7f ) << 1			 # 1st 7 bits shifted up 1
		b1 = ((wregval >> 7) & 0x7f) <<  1	 # 2nd 7 bits shifted up 1
		b2 = ((wregval >> 14) & 0x7f) << 1	 # 3rd 7 bits shifted up 1
		b3 = ((wregval >> 21) & 0x7f) << 1	 # 4th 7 bits shifted up 1
		b4 = (self.address << 1) + 1		 # Address shifted up 1 bit with address bit set

		msg = struct.pack('BBBBB', b0, b1, b2, b3, b4)
		self.serialport.write(msg)
		
	def send_cmd(self, GPI, val): 
		wregval = (GPI << 20) | val
		print(tc.COMMAND + "send to card address", self.address, "/ GPI", GPI, ":", tc.BOLD, wregval, "(", val, ")",tc.ENDC)
		b0 = (wregval & 0x7f ) << 1				# 0-6 bits shifted up 1
		b1 = ((wregval >> 7) & 0x7f) <<  1	 	# 7-13 bits shifted up 1
		b2 = ((wregval >> 14) & 0x7f) << 1	 	# 14-19 bits shifted up 1
		b3 = ((wregval >> 21) & 0x7f) << 1	 	# 4th 7 bits shifted up 1
		b4 = (self.address << 1) + 1		   	# Address shifted up 1 bit with address bit set
		msg = struct.pack('BBBBB', b0, b1, b2, b3, b4)
		self.serialport.write(msg)
		time.sleep(0.001)
		
	def packCal(self):
		for idx in range(self.counters):
			self.CalCoeffs['coeff%i'%idx] = str(self.phase_counters[idx].cal_offset.text())

	def unpackCal(self, CalCoeffs):
		for idx in range(self.counters):
			self.phase_counters[idx].cal_offset.setText(CalCoeffs['coeff%i'%idx])

def main():
	
	app = QApplication(sys.argv)
	win = dprS(ctype=ctype, addr=addr, slot=slot)
	win.show()
	sys.exit(app.exec_())


if __name__ == '__main__':
	p = optparse.OptionParser()
	p.add_option('-C','--card_type', action='store', dest='ctype', type='str',
				 help='Type of card to calibrate (default=DFBx2).')
	p.add_option('-A','--card_address', action='store', dest='addr', type='int',
				 help='Hardware address of card to calibrate (default=3).')
	p.add_option('-S','--slot', action='store', dest='slot', type='int',
				 help='Host slot in crate (default=2)')
	p.set_defaults(ctype="DFBx2")
	p.set_defaults(addr=3)
	p.set_defaults(slot=2)
	opt, args = p.parse_args()
	ctype = opt.ctype
	addr = opt.addr
	slot = opt.slot
	main()
	

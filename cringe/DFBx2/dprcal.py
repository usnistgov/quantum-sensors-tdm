import sys
import optparse
import struct
import time

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt, SIGNAL
from PyQt4.QtGui import QFileDialog, QPalette

# import state_vector_builder
import named_serial
from dpr_counter import dpr_counter

class dprcal(QtGui.QWidget):

	def __init__(self, ctype=None, addr=None, slot=None):
		super(dprcal, self).__init__()
		
		self.COMMAND = '\033[95m'
		self.FCTCALL = '\033[94m'
		self.INIT = '\033[92m'
		self.WARNING = '\033[93m'
		self.FAIL = '\033[91m'
		self.ENDC = '\033[0m'
		self.BOLD = "\033[1m"

		self.serialport = named_serial.Serial(port='rack', shared = True)
		
		self.card_type = ctype
		self.address = addr
		self.slot = slot
		self.counters = 7
		
		self.GPI1 = (1 << 17) | self.slot
		
		self.phase_counters = []
# 		self.phase_counters_active = []
# 		self.nstates = 256
		self.enb = [0,0,0,0,0,0,0]
		self.cal_coeffs = [0,0,0,0,0,0,0]
		self.appTrim =[0,0,0,0,0,0,0]
		
		self.CalCoeffs = {}

		self.setWindowTitle("POW")	# Phase Offset Widget
		self.setGeometry(50,50,660,800)
		
		self.layout_widget = QtGui.QWidget(self)
		self.layout = QtGui.QVBoxLayout(self)
		
		self.globals_widget = QtGui.QGroupBox(self.layout_widget)
		self.globals_layout = QtGui.QGridLayout(self.globals_widget)
		
# 		self.slot_indicator = QtGui.QLineEdit()
# 		self.slot_indicator.setReadOnly(True)
# 		self.slot_indicator.setFixedWidth(40)
# 		self.slot_indicator.setText(str(slot))
# 		self.slot_indicator.setAlignment(QtCore.Qt.AlignRight)
# 		self.globals_layout.addWidget(self.slot_indicator,0,0,QtCore.Qt.AlignLeft)
# 		
# 		self.slot_label = QtGui.QLabel("card slot")
# 		self.globals_layout.addWidget(self.slot_label,0,1,1,4,QtCore.Qt.AlignLeft)
		
		self.loadcal = QtGui.QPushButton(self, text = "load calibration")
		self.globals_layout.addWidget(self.loadcal,0,0,QtCore.Qt.AlignTop)

		self.savecal = QtGui.QPushButton(self, text = "save calibration")
		self.globals_layout.addWidget(self.savecal,0,1,QtCore.Qt.AlignTop)

		self.autocal = QtGui.QPushButton(self, text = "auto calibrate")
		self.globals_layout.addWidget(self.autocal,0,2,QtCore.Qt.AlignTop)

		self.null_phase = QtGui.QPushButton(self, text = "null phase")
		self.null_phase.setEnabled(0)
		self.globals_layout.addWidget(self.null_phase,0,4,QtCore.Qt.AlignTop)

		self.init_slot = QtGui.QPushButton(self, text = "send slot")
		self.init_slot.setEnabled(0)
		self.globals_layout.addWidget(self.init_slot,1,4,QtCore.Qt.AlignTop)
		
		self.filenameEdit = QtGui.QLineEdit()
		self.filenameEdit.setReadOnly(True)
		self.globals_layout.addWidget(self.filenameEdit,1,0,1,3)
		
		self.filename_label = QtGui.QLabel("file")
		self.globals_layout.addWidget(self.filename_label,1,3,QtCore.Qt.AlignLeft)		

		self.arrayframe = QtGui.QWidget(self.layout_widget)
		self.array_layout = QtGui.QVBoxLayout(self.arrayframe)
		self.array_layout.setAlignment(QtCore.Qt.AlignHCenter)
		self.array_layout.setSpacing(20)
		self.array_layout.setMargin(10)
		
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

		for idx, val in enumerate(self.enabled):
			self.counters = idx + 1
# 			self.counters = idx
			pcs = val
			self.phase_counters.append(dpr_counter(self, self.array_layout, pcs=pcs, idx=idx, slot=self.slot, coeffs=self.cal_coeffs, appTrim=self.appTrim, serialport=self.serialport, cardaddr=self.address))
			
		self.scrollarea = QtGui.QScrollArea(self.layout_widget)
		self.scrollarea.setWidget(self.arrayframe)
		self.layout.addWidget(self.globals_widget)
		self.layout.addWidget(self.scrollarea)
				
		self.connect(self.loadcal, SIGNAL("clicked()"), self.loadCalFile)
		self.connect(self.savecal, SIGNAL("clicked()"), self.saveCalfile)
		self.connect(self.autocal, SIGNAL("clicked()"), self.CalAllCounters)
		self.null_phase.clicked.connect(self.nullPhase)
		self.init_slot.clicked.connect(self.sendSlot)
		
		self.scale_factor = self.phase_counters[0].width()
		self.globals_widget.setFixedWidth(self.scale_factor*1.1)
		
		'''initialization'''
# 		print self.INIT + "Initialize:", self.ENDC
# 		self.sendSlot()
		self.phase_counters[1].phase_trim_spin.setEnabled(0)			# initialize to C0 counter spin disabled 8mar16

	def resetALLphase(self):
		for i in range(self.counters):
# 		if self.enb[i] == 1:
			self.phase_counters[i].resetPhase()


	def loadCalFile(self):
		print self.FCTCALL + "load phase calibration file:", self.ENDC
		filename = str(QFileDialog.getOpenFileName())
# 		print("filename = [%s]" % filename)
		if len(filename) > 0:
			print "loading file:", filename
			print
			f = open(filename, 'r')
			for idx, val in enumerate(f):
# 				print idx, val
				self.cal_coeffs[idx] = int(val)
				self.phase_counters[idx].loadCal(int(val))							#	8mar16:	chg to call loadCal with passthru coefficient
# 				self.phase_counters[idx].cal_offset.setText(str(val))							comment out
# 				self.phase_counters[idx].cal_off_deg.setText(str(self.cal_off*9))				comment out
			f.close()
			self.filenameEdit.setText(filename)
		else:
			print self.FAIL + "load file cancelled:", self.ENDC
			print
			return
		
# 		if idx != self.counters:
		if (idx + 1) != self.counters:
			print self.FAIL + "WARNING: calibration file does not meet card class format" + self.ENDC
			print
			return
# 		for i in range(self.counters):
# 			self.phase_counters[i].loadCal()
			
	def saveCalfile(self):
		print self.FCTCALL + "save phase calibration file:", self.ENDC
		filestring = str(self.card_type)+"_S"+str(self.slot)+"A"+str(self.address)+".txt"
		filename = str(QFileDialog.getSaveFileName(self, "save calibration file",filestring))
		if len(filename) > 0:
			self.filenameEdit.setText(filename)
			print "saving calibration file:", filename
			print
			f = open(filename, 'w')
			for idx in range(self.counters):
				f.write(str(self.phase_counters[idx].cal_offset.text() +"\n"))
			f.close()
		else:
			print self.FAIL + "save file cancelled:", self.ENDC
			print
			return

	def CalAllCounters(self):
		print
		print self.INIT + "auto phase calibrate: ", self.card_type, "card address", self.address, self.ENDC
		print
		for i in range(self.counters):
# 			if self.enb[i] == 1:
			self.phase_counters[i].calcounter()
		print
			
	def enbDiagnostic(self, mode):
		self.null_phase.setEnabled(mode)
		self.init_slot.setEnabled(mode)
		self.phase_counters[1].phase_trim_spin.setEnabled(mode)		#	tie C0 spin to enable bit 8mar16
			
	def nullPhase(self):
		print
		print self.INIT + "null phase:", self.ENDC
		print
		self.resetALLphase()
		print self.FCTCALL + "GPI1: reset slot & phase trim to 0:", self.ENDC
		print
		mask = 0xfffe000
		wregval = self.GPI1 & mask
		self.sendReg(wregval)
		
	def sendSlot(self):
		print self.FCTCALL + "send slot:", self.slot, self.ENDC
		print "GPI1: send slot:"
		print
		mask = 0xfffffe0
		wregval = (mask & self.GPI1) | self.slot
		self.sendReg(wregval)
				
	def sendReg(self, wregval):
		print self.COMMAND + "send to address", self.address, ":", self.BOLD, wregval, self.ENDC
		b0 = (wregval & 0x7f ) << 1			 # 1st 7 bits shifted up 1
		b1 = ((wregval >> 7) & 0x7f) <<  1	 # 2nd 7 bits shifted up 1
		b2 = ((wregval >> 14) & 0x7f) << 1	 # 3rd 7 bits shifted up 1
		b3 = ((wregval >> 21) & 0x7f) << 1	 # 4th 7 bits shifted up 1
		b4 = (self.address << 1) + 1		 # Address shifted up 1 bit with address bit set

		msg = struct.pack('BBBBB', b0, b1, b2, b3, b4)
		self.serialport.write(msg)
		
	def packCal(self):
		for idx in range(self.counters):
			self.CalCoeffs['coeff%i'%idx] = str(self.phase_counters[idx].cal_offset.text())

	def unpackCal(self, CalCoeffs):
		for idx in range(self.counters):
			self.phase_counters[idx].cal_offset.setText(CalCoeffs['coeff%i'%idx])

def main():
	
	app = QtGui.QApplication(sys.argv)
	win = dprcal(ctype=ctype, addr=addr, slot=slot)
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
	

import sys
import optparse

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
		self.OKGREEN = '\033[92m'
		self.WARNING = '\033[93m'
		self.FAIL = '\033[91m'
		self.ENDC = '\033[0m'
		self.BOLD = "\033[1m"

		self.serialport = named_serial.Serial(port='rack', shared = True)

		self.addr = addr
		self.slot = slot
		self.counters = 7
		
		self.phase_counters = []
# 		self.phase_counters_active = []
# 		self.nstates = 256
		self.enb = [0,0,0,0,0,0,0]
		self.cal_coeffs = [0,0,0,0,0,0,0]
		self.appTrim =[0,0,0,0,0,0,0]

		self.setWindowTitle("POW")	# Phase Offset Widget
		self.setGeometry(50,50,660,800)
		
		self.layout_widget = QtGui.QWidget(self)
		self.layout = QtGui.QVBoxLayout(self)
		
		self.globals_widget = QtGui.QWidget(self.layout_widget)
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
		
		self.filename_label = QtGui.QLabel("file")
		self.globals_layout.addWidget(self.filename_label,1,4,QtCore.Qt.AlignLeft)		
		
		self.filenameEdit = QtGui.QLineEdit()
		self.filenameEdit.setReadOnly(True)
		self.globals_layout.addWidget(self.filenameEdit,1,0,1,3)

		self.arrayframe = QtGui.QWidget(self.layout_widget)
		self.array_layout = QtGui.QVBoxLayout(self.arrayframe)
		self.array_layout.setSpacing(20)
		self.array_layout.setMargin(10)
		
		if ctype == 'BAD16':
			self.enabled = (0,2,3)
			self.ctr_dsc = ("ALTPLL_A","MCLK_PLL","DAC_CLK_1")
			self.ctr_fct = ("all clocks","master process clock","DAC clock 1/2")
			
		if ctype == 'DFBx1CLK':
			self.enabled = (0,2,3,4,5,6)
			self.ctr_dsc = ("ALTPLL_A","MCLK_PLL","DCLK_PLL_1","DCLK_PLL_2","DCLK_PLL_3","DCLK_PLL_4",)
			self.ctr_fct = ("all clocks","master process clock","master data clock","line clock","data pipe 3","data pipe 4",)
			
		if ctype == 'DFBx2':
			self.enabled = (0,2,3,4,5,6)
			self.ctr_dsc = ("ALTPLL_A","MCLK_PLL","DCLK_PLL_1","DCLK_PLL_2","DCLK_PLL_3","DCLK_PLL_4",)
			self.ctr_fct = ("all clocks","master process clock","data pipe 1","data pipe 2","data pipe 3","data pipe 4",)

		for idx, val in enumerate(self.enabled):
			self.counters = idx + 1
			pcs = val
			self.phase_counters.append(dpr_counter(self, self.array_layout, pcs=pcs, idx=idx, slot=self.slot, coeffs=self.cal_coeffs, appTrim=self.appTrim, serialport=self.serialport, cardaddr=self.addr))
			
		self.scrollarea = QtGui.QScrollArea(self.layout_widget)
		self.scrollarea.setWidget(self.arrayframe)
		self.layout.addWidget(self.globals_widget)
		self.layout.addWidget(self.scrollarea)
		self.show()
				
		self.connect(self.loadcal, SIGNAL("clicked()"), self.loadCalFile)
		self.connect(self.savecal, SIGNAL("clicked()"), self.saveCalfile)
		self.connect(self.autocal, SIGNAL("clicked()"), self.CalAllCounters)

	def resetALLphase(self):
		for i in range(self.counters):
# 		if self.enb[i] == 1:
			self.phase_counters[i].resetPhase()


	def loadCalFile(self):
		print(self.FCTCALL + "load calibration file:", self.ENDC)
		filename = str(QFileDialog.getOpenFileName())
		print(("filename = [%s]" % filename))
		if len(filename) > 0:
			print("loading file")
# 			self.loadcal.setStyleSheet("color: #008000")
			f = open(filename, 'r')
			for idx, val in enumerate(f):
# 				print idx, val
				self.cal_coeffs[idx] = int(val)
			f.close()
		if (idx + 1) != self.counters:
			print(self.FAIL + "WARNING: calibration file does not meet card class format" + self.ENDC)
			return
		for i in range(self.counters):
			self.phase_counters[i].loadCal()
		self.filenameEdit.setText(filename)
		print()
			
	def saveCalfile(self):
		print(self.FCTCALL + "save calibration file:", self.ENDC)
		filename = str(QFileDialog.getSaveFileName())
		self.filenameEdit.setText(filename)
		print("saving calibration file:", filename)
		f = open(filename, 'w')
		for idx in range(self.counters):
			f.write(str(self.cal_coeffs[idx])+"\n")
		f.close()
		print()

	def CalAllCounters(self):
		print(self.FCTCALL + "auto calibrate", self.ENDC)
		for i in range(self.counters):
# 			if self.enb[i] == 1:
			self.phase_counters[i].calcounter()

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
	

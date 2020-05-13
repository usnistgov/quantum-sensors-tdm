import sys
import optparse

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

# from PyUtil.terminal_color import terminal_color

from . import state_vector_builder
from cringe.shared import terminal_colors as tc
import named_serial
import time

class SV_array(QWidget):

    def __init__(self, parent=None, seqln=None, addr=None):
        super(SV_array, self).__init__()

        self.serialport = named_serial.Serial(port='rack', shared = True)



        self.parent = parent
        self.addr = addr
        self.nstates = 64
        self.state_vectors = []
        self.state_vector_val = []
        self.allStates = {}

        self.init_flag = 1

        self.setWindowTitle("PRISM")	# Program for Reconfiguration & Initialization of State Matrices
        self.setGeometry(50,50,700,900)

        self.layout_widget = QWidget(self)
        self.layout = QVBoxLayout(self)

        self.globals_widget = QWidget(self.layout_widget)
        self.globals_layout = QGridLayout(self.globals_widget)

# 		self.seqln_indicator = QLineEdit()
# 		self.seqln_indicator.setReadOnly(True)
# 		self.seqln_indicator.setFixedWidth(40)
# 		self.seqln_indicator.setText(str(seqln))
# 		self.seqln_indicator.setAlignment(QtCore.Qt.AlignRight)
# 		self.globals_layout.addWidget(self.seqln_indicator,0,0,QtCore.Qt.AlignLeft)
# 		
# 		self.seqln_label = QLabel("states in sequence")
# 		self.globals_layout.addWidget(self.seqln_label,0,1,1,4,QtCore.Qt.AlignLeft)

        self.loadseq = QPushButton(self, text = "load sequence")
        self.globals_layout.addWidget(self.loadseq,0,0,QtCore.Qt.AlignTop)

        self.saveseq = QPushButton(self, text = "save sequence")
        self.globals_layout.addWidget(self.saveseq,0,1,QtCore.Qt.AlignTop)

        self.filenameEdit = QLineEdit()
        self.filenameEdit.setReadOnly(True)
        self.globals_layout.addWidget(self.filenameEdit,0,2,1,5)

        self.filename_label = QLabel("file")
        self.globals_layout.addWidget(self.filename_label,0,7,QtCore.Qt.AlignLeft)

        self.setseq = QPushButton(self, text = "send sequence")
        self.setseq.setToolTip("send current state vectors to card")
        self.globals_layout.addWidget(self.setseq,0,8,QtCore.Qt.AlignTop)

        self.initseq = QPushButton(self, text = "initialize sequence")
        self.initseq.setToolTip("set all internal state vectors to 0x0000, hit send sequence to send null states to card")
        self.globals_layout.addWidget(self.initseq,0,9,QtCore.Qt.AlignTop)
        self.initseq.clicked.connect(self.initSeq)

        self.restore_seq = QPushButton(self, text = "restore sequence")
        self.restore_seq.setToolTip("restore all internal state vectors to values from last load file, hit send sequence to send restored states to card")
        self.globals_layout.addWidget(self.restore_seq,0,10,QtCore.Qt.AlignTop)
        self.restore_seq.clicked.connect(self.fillSV)


        self.arrayframe = QWidget(self.layout_widget)
        self.array_layout = QVBoxLayout(self.arrayframe)
        self.array_layout.setSpacing(0)
        self.array_layout.setContentsMargins(0,0,0,0)

        for i in range(self.nstates):
            if i < seqln:
                self.state_vectors.append(state_vector_builder.state_vector_builder(self,self.array_layout,state=i,enb=1, vectors=self.state_vector_val, serialport=self.serialport, cardaddr=self.addr))
            else:
                self.state_vectors.append(state_vector_builder.state_vector_builder(self,self.array_layout,state=i,enb=0, vectors=self.state_vector_val, serialport=self.serialport, cardaddr=self.addr))
            self.state_vector_val.append(0)

        self.scrollarea = QScrollArea(self.layout_widget)
        self.scrollarea.setWidget(self.arrayframe)
        self.layout.addWidget(self.globals_widget)
        self.layout.addWidget(self.scrollarea)
# 		self.show()

        self.loadseq.clicked.connect(self.loadSVfile)
        self.saveseq.clicked.connect(self.saveSVfile)
        self.setseq.clicked.connect(self.SendAllStates)

    def seqln_changed(self, seqln):
        for i in range(self.nstates):
            if i < seqln:
                self.state_vectors[i].setEnabled(1)
            else:
                self.state_vectors[i].setEnabled(0)

    def loadSVfile(self):
        print(tc.FCTCALL + "Load state sequence from file: BAD16 /", self.addr, tc.ENDC)
        self.load_filename = str(QFileDialog.getOpenFileName())
        self.filenameEdit.setText(self.load_filename)
        print(("filename = [%s]" % self.load_filename))
        if len(self.load_filename) > 0:
            print("loading file")
            self.fillSV()
        else:
            print(tc.FAIL + "invalid file" + tc.ENDC)
        

    def fillSV(self):
        if self.load_filename == None:
            print(tc.FAIL + "No file to load/restore states from" + tc.ENDC)
            
            return
        f = open(self.load_filename, 'r')
        for idx, val in enumerate(f):
            if idx < self.nstates:
                binstr = bin(int(val))[2:].zfill(16)
                self.state_vector_val[idx] = int(val)
                for i in range(0, 16):
                    self.state_vectors[idx].buttons[i].setChecked(bool(int(binstr[15-i])))
                self.state_vectors[idx].update_sv()
        f.close()
        self.init_flag = 0


    def saveSVfile(self):
        print(tc.FCTCALL + "Save current state sequence to file: BAD16 /", self.addr, tc.ENDC)
        
        filename = str(QFileDialog.getSaveFileName()[0])
        self.filenameEdit.setText(filename)
        f = open(filename, 'w')
        for idx in range(self.nstates):
            f.write(str(self.state_vector_val[idx])+"\n")
        f.close()
        self.filenameEdit.setText(filename)

    def initSeq(self):
        print(tc.FCTCALL + "Clear BAD16 state sequence memory: Initialize", self.nstates, "state vectors to 0x0000: BAD16 /", self.addr, tc.ENDC)
        
        for i in range(self.nstates):
            for j in range(0, 16):
                self.state_vectors[i].buttons[j].setChecked(False)
            self.state_vectors[i].update_sv()
# 			self.state_vectors[i].send_state(1) 
        self.init_flag = 1

    def SendAllStates(self):
        print(tc.FCTCALL + "Send all", self.nstates, "states: BAD16/", self.addr, tc.ENDC)
        
        for i in range(self.nstates):
            self.state_vectors[i].send_state()
        self.parent.initMem(self.init_flag)

    def packStates(self):
        for idx in range(self.nstates):
            self.state_vectors[idx].packState()
            self.allStates['state%i'%idx] = self.state_vectors[idx].StateVector

    def unpackStates(self, badAllStates):
        for idx in range(self.nstates):
            self.state_vectors[idx].unpackState(badAllStates['state%i'%idx])

# 	def unpackChannels(self, badAllChannels):
# 		for idx in range(self.chns):
# 			self.chn_vectors[idx].unpackChannel(badAllChannels['channel%i'%idx])

    def packState(self):
        s=""
        for idx in range(self.nstates):
            s+=str(self.state_vector_val[idx])+"\n"
        return s

    def unpackState(self,s):
        for idx, val in enumerate(s.split()):
            if idx < self.nstates:
                binstr = bin(int(val))[2:].zfill(16)
                self.state_vector_val[idx] = int(val)
                for i in range(0, 16):
                    self.state_vectors[idx].buttons[i].setChecked(bool(int(binstr[15-i])))
                self.state_vectors[idx].update_sv()
        self.init_flag = 0
        self.SendAllStates()

def main():

    app = QApplication(sys.argv)
    win = SV_array(seqln=seqln, addr=addr)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    p = optparse.OptionParser()
    p.add_option('-A','--address', action='store', dest='addr', type='int',
                 help='Physical hardware address (default=32).')
    p.add_option('-L','--length', action='store', dest='seqln', type='int',
                 help='Number of states in sequence (default=4')
    p.set_defaults(addr=32)
    p.set_defaults(seqln=4)
    opt, args = p.parse_args()
    addr = opt.addr
    seqln = opt.seqln
    main()


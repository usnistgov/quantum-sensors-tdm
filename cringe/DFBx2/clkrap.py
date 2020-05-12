#-*- coding: utf-8 -*-
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


class clkrap(QWidget):

# 	def __init__(self, parent=None, **kwargs):
# 		print kwargs
    def __init__(self, parent=None, addr=0, slot=1, seqln=None, lsync=40):

        super(clkrap, self).__init__()

        self.parent = parent
        self.address = addr
        self.slot = slot
        self.seqln = seqln
        self.lsync = lsync

        self.serialport = named_serial.Serial(port='rack', shared = True)

        '''global booleans'''

        self.ST = 0
        self.CLKstate = 1

        '''global variables'''

        self.lsync = lsync
        self.lsync_minus1 = self.lsync - 1

        self.setWindowTitle("CRAP")	# CLK register address program
        self.setGeometry(30,30,400,200)
        self.setContentsMargins(0,0,0,0)

        self.layout_widget = QWidget(self)
        self.layout = QGridLayout(self)

        self.lp_title = QLabel("LSYNC")
        self.layout.addWidget(self.lp_title,0,0,1,1,QtCore.Qt.AlignLeft)

# 		self.lsync_spin = QSpinBox(self)
# 		self.lsync_spin.setRange(20,256)
# 		self.lsync_spin.setSingleStep(1)
# 		self.lsync_spin.setKeyboardTracking(0)
# 		self.lsync_spin.setFocusPolicy(QtCore.Qt.StrongFocus)
# 		self.lsync_spin.setValue(self.lsync)
# 		self.lsync_spin.setAlignment(QtCore.Qt.AlignRight)
# 		self.layout.addWidget(self.lsync_spin,1,0,1,1)
# 		self.lsync_spin.valueChanged.connect(self.lsync_changed)

        self.lsync_indicator = QLineEdit()
        self.lsync_indicator.setReadOnly(True)
        self.lsync_indicator.setFixedHeight(25)
        self.lsync_indicator.setText(str(self.lsync))
        self.lsync_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.lsync_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
        self.layout.addWidget(self.lsync_indicator, 1,0,1,1,QtCore.Qt.AlignRight)
        self.lsync_indicator.textChanged.connect(self.lsync_changed)

        self.lsync_lbl = QLabel("MCLK cycles")
        self.layout.addWidget(self.lsync_lbl,1,1,1,1,QtCore.Qt.AlignLeft)

        self.lp_title = QLabel("line period")
        self.layout.addWidget(self.lp_title,2,0,1,1,QtCore.Qt.AlignLeft)

        self.line_period_indicator = QLineEdit()
        self.line_period_indicator.setReadOnly(True)
        self.line_period_indicator.setFixedHeight(25)
        self.line_period_indicator.setText(str(8*(self.lsync)))
        self.line_period_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.line_period_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
        self.layout.addWidget(self.line_period_indicator, 3,0,1,1,QtCore.Qt.AlignRight)

        self.line_period_lbl = QLabel("ns")
        self.layout.addWidget(self.line_period_lbl,3,1,1,1,QtCore.Qt.AlignLeft)

        self.lp_title = QLabel("line rate")
        self.layout.addWidget(self.lp_title,4,0,1,1,QtCore.Qt.AlignLeft)

        self.line_freq_indicator = QLineEdit()
        self.line_freq_indicator.setReadOnly(True)
        self.line_freq_indicator.setFixedHeight(25)
        self.line_freq_indicator.setText(str(125/(self.lsync))[:6])
        self.line_freq_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.line_freq_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
        self.layout.addWidget(self.line_freq_indicator, 5,0,1,1,QtCore.Qt.AlignRight)

        self.line_freq_lbl = QLabel("MHz")
        self.layout.addWidget(self.line_freq_lbl,5,1,1,1,QtCore.Qt.AlignLeft)

        self.lp_title = QLabel("frame period")
        self.layout.addWidget(self.lp_title,2,2,1,1,QtCore.Qt.AlignLeft)

        self.frame_period_indicator = QLineEdit()
        self.frame_period_indicator.setReadOnly(True)
        self.frame_period_indicator.setFixedHeight(25)
        self.frame_period_indicator.setText(str(self.seqln*0.008*self.lsync)[:6])
        self.frame_period_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.frame_period_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
        self.layout.addWidget(self.frame_period_indicator,3,2,1,1,QtCore.Qt.AlignRight)

        self.frame_period_lbl = QLabel("\u00B5s")
        self.layout.addWidget(self.frame_period_lbl,3,3,1,1,QtCore.Qt.AlignLeft)

        self.lp_title = QLabel("frame rate")
        self.layout.addWidget(self.lp_title,4,2,1,1,QtCore.Qt.AlignLeft)

        self.frame_freq_indicator = QLineEdit()
        self.frame_freq_indicator.setReadOnly(True)
        self.frame_freq_indicator.setFixedHeight(25)
        self.frame_freq_indicator.setText(str(125000/(self.lsync*self.seqln))[:6])
        self.frame_freq_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.frame_freq_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
        self.layout.addWidget(self.frame_freq_indicator, 5,2,1,1,QtCore.Qt.AlignRight)

        self.frame_freq_lbl = QLabel("kHz")
        self.layout.addWidget(self.frame_freq_lbl,5,3,1,1,QtCore.Qt.AlignLeft)

        self.resync_button = QPushButton(self, text = "RESYNC")
        self.resync_button.setFixedWidth(125)
        self.resync_button.setFixedHeight(25)
        self.resync_button.setStyleSheet("background-color: #" + tc.green + ";")
        self.layout.addWidget(self.resync_button,3,5,1,2,QtCore.Qt.AlignRight)
        self.resync_button.clicked.connect(self.resync)

        self.CLKstate_button = QToolButton(self, text = 'RUN')
        self.CLKstate_button.setFixedHeight(25)
        self.CLKstate_button.setCheckable(1)
        self.CLKstate_button.setChecked(self.CLKstate)
        self.CLKstate_button.setStyleSheet("background-color: #" + tc.green + ";")
        self.layout.addWidget(self.CLKstate_button,5,5,1,1,QtCore.Qt.AlignLeft)
        self.CLKstate_button.toggled.connect(self.CLKstate_changed)

        self.CLKstate_lbl = QLabel("line clock")
        self.layout.addWidget(self.CLKstate_lbl,5,6,1,1,QtCore.Qt.AlignLeft)

    '''
    self called methods
    '''
    def lsync_changed(self):
        self.lsync = int(self.lsync_indicator.text())
# 		print tc.WARNING + "Line period changed:", self.lsync*8, "ns", tc.ENDC
        self.lsync_minus1 = self.lsync - 1
        self.line_period_changed()
        self.frame_period_changed()
        print(tc.FCTCALL + "send LSYNC-1 to (DFB)CLK:", tc.ENDC)
        self.send_wreg2()

    def seqln_changed(self, seqln):
        self.seqln = seqln
        self.frame_period_changed()

    def update_lsync(self):
        wreg = 2 << 25
        wregval = wreg | self.lsync_minus1
        self.sendReg(wregval)

    def CLKstate_changed(self):
        self.CLKstate = self.CLKstate_button.isChecked()
        self.notCLKstate = not(self.CLKstate)
        if self.CLKstate == 1:
            print(tc.FCTCALL + "line clock enabled:", tc.ENDC)
            self.CLKstate_button.setStyleSheet("background-color: #" + tc.green + ";")
            self.CLKstate_button.setText('RUN')
            self.resync_button.setStyleSheet("background-color: #" + tc.green + ";")
        else:
            print(tc.FCTCALL + "line clock disabled:", tc.ENDC)
            self.CLKstate_button.setStyleSheet("background-color: #" + tc.red + ";")
            self.CLKstate_button.setText('STOP')
            self.resync_button.setStyleSheet("background-color: #" + tc.red + ";")
        self.send_wreg1()

    def resync(self):
        if self.CLKstate == 1:
            print(tc.FCTCALL + "resynchronize system:", tc.ENDC)
            print()
            self.CLKstate_button.click()
            time.sleep(1)
            self.CLKstate_button.click()
        else:
            print(tc.FAIL + "line clock must be enabled for RESYNC:", tc.ENDC)
            print()

    def line_period_changed(self):
        self.line_period_indicator.setText(str(8*(self.lsync)))
        self.line_freq_indicator.setText(str(125.0/(self.lsync))[:6])

    def frame_period_changed(self):
        self.frame_period_indicator.setText(str(self.seqln*0.008*self.lsync)[:6])
        self.frame_freq_indicator.setText(str(125000/(self.lsync*self.seqln))[:6])
# 		self.send_wreg7()


    def send_wreg1(self):
        print("CLK:WREG1: clock state:", self.CLKstate)
        wreg = 1 << 25
        wregval = wreg | (self.notCLKstate << 24) | (self.CLKstate << 23)
        self.sendReg(wregval)
        print()

    def send_wreg2(self):
        print("CLK:WREG2: LSYNC-1:", self.lsync_minus1)
        wreg = 2 << 25
        wregval = wreg | self.lsync_minus1
        self.sendReg(wregval)
        print()

    def send_wreg7(self):
        print("CLK:WREG7: sequence length:", self.seqln)
        wreg = 7 << 25
        wregval = wreg | (self.seqln << 8)
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
        time.sleep(0.001)

def main():

    app = QApplication(sys.argv)
    app.setStyle("plastique")
    app.setStyleSheet("""	QPushbutton{font: 10px; padding: 6px}
                            QToolButton{font: 10px; padding: 6px}
                            QLineEdit {background-color: #FFFFCC;}
                            QToolTip {background-color: #FFFFCC;}""")
    win = clkrap(addr=addr, slot=slot, seqln=seqln)
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
    p.set_defaults(addr=0)
    p.set_defaults(slot=1)
    p.set_defaults(seqln=4)
    opt, args = p.parse_args()
# 	ctype = opt.ctype
    addr = opt.addr
    slot = opt.slot
    seqln = opt.seqln
    main()


import sys
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import struct
import time
from cringe.shared import terminal_colors as tc
import logging


class state_vector_builder(QWidget):
    
    def __init__(self, parent=None, layout=None, state=0, enb=1, vectors=[0], serialport=None, cardaddr=32):

        super(state_vector_builder, self).__init__()
        
        self.serialport = serialport
        self.address = cardaddr

        self.parent = parent
        self.layout = layout
        self.state = state
        self.vectors = vectors
        self.row_layout = QHBoxLayout(self)
        self.row_layout.setSpacing(5)
        
        self.state_label = QLineEdit()
        self.state_label.setReadOnly(True)
        self.state_label.setFixedWidth(36)
        self.state_label.setAlignment(QtCore.Qt.AlignRight)
        self.state_label.setText(str(state))
        
        self.row_layout.addWidget(self.state_label)
        
        self.buttons = []
        self.states = []
        
        for i in range(0, 16):
            self.a = QToolButton(self, text = str(i))
            self.a.setFixedWidth(25)
            self.a.setFixedHeight(25)
            self.a.setCheckable(1)
            self.a.setStyleSheet("background-color: #F08080;")
            self.buttons.append(self.a)
            self.row_layout.addWidget(self.buttons[i])
            self.buttons[i].clicked.connect(self.update_sv)

        self.sv_send = QPushButton(self, text = "send state")
        self.sv_send.setFixedHeight(25)
        self.row_layout.addWidget(self.sv_send)
        
        self.sv_send.clicked.connect(self.send_state)
        
        self.setEnabled(enb)        
        
        if self.parent != None:
            self.layout.addWidget(self)
               
#         self.show()
        
    def update_sv(self):
        logging.debug(tc.FCTCALL + "update BAD16 internal state vector", self.state, tc.ENDC)
        sv_str = ""
        sv_dec = 0
        for i in range(0, 16):
            if self.buttons[i].isChecked() == True:
                self.buttons[i].setStyleSheet("background-color: #90EE90;")
                sv_str = '1' + sv_str
                sv_dec = sv_dec + 2**i
            else:
                self.buttons[i].setStyleSheet("background-color: #F08080;")
                sv_str = '0' + sv_str
                sv_dec = sv_dec
                
        self.vectors[self.state] = sv_dec
        logging.debug("state vector binary", sv_str)
        
        
    def send_state(self):
        addr = 3
        wreg = addr << 25
        logging.debug(tc.FCTCALL + "update BAD16 single state vector:", self.state, tc.ENDC)
        logging.debug("BAD16:WREG3: update state index:", self.state)
        wregval = wreg + self.state
        self.sendReg(wregval)
        wregval & 0x0000000                        # blank WREG
        addr = 7
        wregval = addr << 25 
        logging.debug("BAD16:WREG7: send state vector:", hex(self.vectors[self.state]&0xffff))
        wregval = wregval + self.vectors[self.state] 
        self.sendReg(wregval)
        if self.vectors[self.state] != 0:
            self.parent.parent.initMem(0)
        

    def sendReg(self, wregval): 
        logging.debug(tc.COMMAND + "send to address", self.address, ":", tc.BOLD, wregval, tc.ENDC)
        b0 = (wregval & 0x7f ) << 1            # 1st 7 bits shifted up 1
        b1 = ((wregval >> 7) & 0x7f) <<  1     # 2nd 7 bits shifted up 1
        b2 = ((wregval >> 14) & 0x7f) << 1     # 3rd 7 bits shifted up 1
        b3 = ((wregval >> 21) & 0x7f) << 1     # 4th 7 bits shifted up 1
        b4 = (self.address << 1) + 1           # Address shifted up 1 bit with address bit set
        msg = struct.pack('BBBBB', b0, b1, b2, b3, b4)
        self.serialport.write(msg)
        time.sleep(0.001)
        
    def packState(self):
        self.StateVector    =   {
            'sv_dec'        :    self.vectors[self.state],
                                }
        
    def unpackState(self, loadState):
        val = loadState['sv_dec']
        binstr = bin(int(val))[2:].zfill(16)
        for i in range(0, 16):
            self.buttons[i].setChecked(bool(int(binstr[15-i])))
        self.update_sv()
        
                
def main():
     
    app = QApplication(sys.argv)
    ex = state_vector_builder()
    sys.exit(app.exec_())
 
if __name__ == '__main__':
    main()
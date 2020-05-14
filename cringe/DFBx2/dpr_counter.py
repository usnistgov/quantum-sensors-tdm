import sys
import time

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import struct
from cringe.shared import terminal_colors as tc
import logging


class dpr_counter(QWidget):
    
    def __init__(self, parent=None, layout=None, pcs=0, idx=0, slot=1, coeffs=[0], appTrim=[0], serialport=None, cardaddr=3):

        super(dpr_counter, self).__init__()

        self.serialport = serialport
        self.address = cardaddr
        self.cal_off = coeffs[pcs]
#         self.cal_off = 0
        self.slot = slot

        self.parent = parent
        self.layout = layout
        self.counter = pcs
        self.coeffs = coeffs
        self.ptrim = appTrim
        
        self.GPI1 = (1 << 17) | (self.ptrim[0] << 5) | self.slot
        self.slot_mask = 0xfffffe0
        self.phtr_mask = 0xfffe01f

        self.GPI2 = (2 << 17) | (self.counter << 2)
        self.dp_mask = 0xffffffc 
        
        self.GPI3 = (3 << 17)
        self.enb_mask = 0xffffffe 
        
        self.msg = 0
        self.lastSpinVal = 0
        self.lastTotVal = 0
        self.ctr_dsc = self.parent.ctr_dsc[idx]
        self.ctr_fct = self.parent.ctr_fct[idx]
        self.card_type = self.parent.card_type
        self.card_ID = str(self.card_type) + " [" + str(self.slot) + "/" + str(self.address) + "]"

        self.resetFlag = False
        self.commitFlag = False
        
        if pcs == 0:
            self.pcs_str = "ALL counters"
            sep_str = "+"
        else:
            sep_str = "|"
            if pcs == 1:
                self.pcs_str = "M counter"
            else:
                self.pcs_str = "C"+str(pcs-2)+" counter"
                
        self.layout_widget = QGroupBox(self)
        self.layout_widget.setTitle(self.pcs_str)
        self.layout_widget.setStyleSheet("font-size: 14px")
#         self.layout_widget.setStyleSheet("font-style: italic")
        self.row_layout = QGridLayout(self.layout_widget)
        self.row_layout.setSpacing(5)
        
        self.counter_label = QLabel(self.ctr_dsc)
        self.counter_label.setFixedWidth(200)
        self.counter_label.setAlignment(QtCore.Qt.AlignLeft)
                
        self.row_layout.addWidget(self.counter_label,0,0,1,1,QtCore.Qt.AlignCenter)
        
        self.counter_label = QLabel(self.ctr_fct)
        self.counter_label.setFixedWidth(200)
        self.counter_label.setAlignment(QtCore.Qt.AlignLeft)
                
        self.row_layout.addWidget(self.counter_label,1,0,1,1,QtCore.Qt.AlignCenter)
        
        self.cal_offset = QLineEdit()
        self.cal_offset.setReadOnly(True)
        self.cal_offset.setFixedWidth(50)
        self.cal_offset.setAlignment(QtCore.Qt.AlignRight)
        self.cal_offset.setText(str(self.cal_off))
        
        self.row_layout.addWidget(self.cal_offset,0,1,QtCore.Qt.AlignCenter)

        self.seperator_label = QLabel(sep_str)
        self.seperator_label.setFixedWidth(20)
        self.seperator_label.setAlignment(QtCore.Qt.AlignCenter)
                
        self.row_layout.addWidget(self.seperator_label,0,2,QtCore.Qt.AlignCenter)
        
        self.phase_trim_spin = QSpinBox()
        self.phase_trim_spin.setRange(-40, 40)
        self.phase_trim_spin.setFixedWidth(60)
        self.phase_trim_spin.setValue(self.ptrim[self.counter])
        self.phase_trim_spin.setAlignment(QtCore.Qt.AlignRight)
        self.phase_trim_spin.valueChanged.connect(self.newvalue)
 
        self.row_layout.addWidget(self.phase_trim_spin,0,3,QtCore.Qt.AlignRight)

        self.result1_label = QLabel("=")
        self.result1_label.setFixedWidth(20)
        self.result1_label.setAlignment(QtCore.Qt.AlignCenter)
                
        self.row_layout.addWidget(self.result1_label,0,4,QtCore.Qt.AlignCenter)
         
        self.tot_steps = QLineEdit()
        self.tot_steps.setReadOnly(True)
        self.tot_steps.setFixedWidth(50)
        self.tot_steps.setAlignment(QtCore.Qt.AlignRight)
        self.tot_steps.setText(str(self.cal_off+self.phase_trim_spin.value()))
        
        self.row_layout.addWidget(self.tot_steps,0,5,QtCore.Qt.AlignCenter)
        
        self.step_label = QLabel("steps")
        self.step_label.setFixedWidth(50)
         
        self.row_layout.addWidget(self.step_label,0,6,QtCore.Qt.AlignLeft)
        
        self.calibrate = QPushButton(self, text = "Calibrate "+self.pcs_str)
        self.calibrate.setFixedWidth(180)
        self.calibrate.clicked.connect(self.calcounter)

        self.row_layout.addWidget(self.calibrate,0,7,QtCore.Qt.AlignCenter)


        self.cal_off_deg = QLineEdit()
        self.cal_off_deg.setReadOnly(True)
        self.cal_off_deg.setFixedWidth(50)
        self.cal_off_deg.setAlignment(QtCore.Qt.AlignRight)
        self.cal_off_deg.setText(str(self.cal_off*9))
        
        self.row_layout.addWidget(self.cal_off_deg,1,1,QtCore.Qt.AlignCenter)

        self.seperator2_label = QLabel(sep_str)
        self.seperator2_label.setFixedWidth(20)
        self.seperator2_label.setAlignment(QtCore.Qt.AlignCenter)
                
        self.row_layout.addWidget(self.seperator2_label,1,2,QtCore.Qt.AlignCenter)

        self.trim_deg = QLineEdit()
        self.trim_deg.setReadOnly(True)
        self.trim_deg.setFixedWidth(43)
        self.trim_deg.setAlignment(QtCore.Qt.AlignRight)
        self.trim_deg.setText(str(self.phase_trim_spin.value()*9))
        
        self.row_layout.addWidget(self.trim_deg,1,3,QtCore.Qt.AlignLeft)

        self.result2_label = QLabel("=")
        self.result2_label.setFixedWidth(20)
        self.result2_label.setAlignment(QtCore.Qt.AlignCenter)
         
        self.row_layout.addWidget(self.result2_label,1,4,QtCore.Qt.AlignLeft)
         
        self.tot_degs = QLineEdit()
        self.tot_degs.setReadOnly(True)
        self.tot_degs.setFixedWidth(50)
        self.tot_degs.setAlignment(QtCore.Qt.AlignRight)
        self.tot_degs.setText(str((self.cal_off+self.phase_trim_spin.value())*9))
        
        self.row_layout.addWidget(self.tot_degs,1,5,QtCore.Qt.AlignCenter)
 
        self.degrees_label = QLabel("deg")
        self.degrees_label.setFixedWidth(50)
                
        self.row_layout.addWidget(self.degrees_label,1,6,QtCore.Qt.AlignCenter)

        self.commit = QPushButton(self, text = "Commit Calibration")
        self.commit.setFixedWidth(180)
        self.commit.clicked.connect(self.commit_cal)

        self.row_layout.addWidget(self.commit,1,7,QtCore.Qt.AlignHCenter)
        
        if self.parent != None:
            self.layout.addWidget(self.layout_widget)

    def newvalue(self, val):
        if self.resetFlag == True:
            return
        if self.commitFlag == True:
            return
        self.trim_deg.setText(str(self.phase_trim_spin.value()*9))

        if self.counter == 0:         
            self.tot_steps.setText(str(self.cal_off+self.phase_trim_spin.value()))
            logging.debug(tc.FCTCALL + "step phase", self.card_ID,":", self.pcs_str, "from", self.lastTotVal,"to", self.tot_steps.text(), tc.ENDC)
        else:
            self.tot_steps.setText(str(self.phase_trim_spin.value()))
            logging.debug(tc.FCTCALL + "step phase:", self.card_ID,":", self.pcs_str, "from", self.lastSpinVal,"to", val, tc.ENDC)
        

        self.tot_degs.setText(str(int(self.tot_steps.text())*9))
        self.ptrim[self.counter] = int(self.tot_steps.text())

        self.enableDPR()

        logging.debug("send GPI2: phase adjust counter (pll_counter)")
        self.sendReg(self.GPI2)

        if val > self.lastSpinVal:
            logging.debug("increment "+ self.pcs_str)
            self.sendReg(self.GPI2 + 0b01)
        else:
            logging.debug("decrement "+ self.pcs_str)
            self.sendReg(self.GPI2 + 0b10)
            
        logging.debug("reset phase step register (dyn_phase)")
        self.sendReg(self.GPI2)
        
        self.disableDPR()
        

        self.lastSpinVal = val
        self.lastTotVal = int(self.tot_steps.text())

        if self.tot_steps.text() == self.cal_offset.text():
            self.tot_steps.setStyleSheet("background-color: #90EE90;")
        else:
            self.tot_steps.setStyleSheet("background-color: #F08080;")
            
    def commit_cal(self):
        logging.debug(tc.FCTCALL + "commit calibration", self.card_ID,":",  self.pcs_str, tc.ENDC)
        self.commitFlag = True
        self.cal_offset.setStyleSheet("background-color: #90EE90;")
        self.tot_steps.setStyleSheet("background-color: #90EE90;")
        self.cal_off = int(self.tot_steps.text())
        self.coeffs[self.counter] = self.cal_off
        self.cal_offset.setText(str(self.cal_off))
        self.cal_off_deg.setText(str(self.cal_off*9))
        if self.counter == 0:
            self.phase_trim_spin.setValue(0)
            self.tot_steps.setText(str(self.cal_off))
            self.trim_deg.setText(str(0))
            self.tot_degs.setText(str(self.cal_off*9))
            mask = 0xfffe01f
            wregval = (self.GPI1 & mask) | ((self.cal_off & 0xff) << 5)
            logging.debug("send phase trim coefficient & slot:", self.cal_off,"/", self.slot)
            self.sendReg(wregval)
        self.ptrim[self.counter] = int(self.tot_degs.text())
        self.commitFlag = False
        
        

    def calcounter(self):
        logging.debug(tc.FCTCALL + "calibrate counter", self.card_ID,":", self.pcs_str, tc.ENDC)
        
        if self.counter == 0:
            '''
             This case implements a firmware PLL reset & auto calibration.
             The auto calibration applies the slot & phase trim offsets to ALL counters.
             As a result of the RESET process, the other individual counter offsets are reset as well.
             In this case the code must reset the spin values and total steps of the other counters to 0.
             But changing a spin value calls newvalue which automatically steps the phase.
             So we must use a flag, self.resetFlag, set in resetPhase, to branch out of NEWVALUE when called from there.
            '''
            logging.debug(tc.FCTCALL + "firmware calibrate ALL counters:", tc.ENDC)
            logging.debug("send GPI1: phase trim coefficient & slot:", self.cal_offset.text(),"/", self.slot)
            wregval = (self.GPI1 & self.phtr_mask) | ((int(self.cal_offset.text()) & 0xff) << 5)
            self.sendReg(wregval)
            
            logging.debug("send GPI2: phase adjust counter (pll_counter)")
            self.sendReg(self.GPI2)
            

            self.parent.resetALLphase()         # reset ALL phase
            self.tot_steps.setText(self.cal_offset.text())
            self.tot_degs.setText(str(int(self.tot_steps.text())*9))

            logging.debug(tc.FCTCALL + "firmware autocal (for phase trim calibration coefficient & slot offsets):", tc.ENDC)
            
            self.enableDPR()

            logging.debug("phase calibrate = True")
            self.sendReg(self.GPI3 + 0b1001)    # PC/SMenb => HI

            logging.debug("phase calibrate = False")
            self.sendReg(self.GPI3 + 0b0001)    # PC => LO (SMenb => HI)    

            self.disableDPR()
            

        else:
            steps = self.phase_trim_spin.value() - int(self.cal_offset.text())
            logging.debug("software calibrate (for phase trim):", -steps, "phase steps applied")
            while steps != 0:
                if steps > 0:
                    self.phase_trim_spin.setValue(self.phase_trim_spin.value() - 1)
                    steps = steps -1
                if steps < 0:
                    self.phase_trim_spin.setValue(self.phase_trim_spin.value() + 1)
                    steps = steps +1
            
        
        self.cal_offset.setStyleSheet("background-color: #90EE90;")
        self.tot_steps.setStyleSheet("background-color: #90EE90;")
        self.ptrim[self.counter] = int(self.tot_steps.text())
        
    def loadCal(self, cal_val):
        logging.debug(tc.FCTCALL + "load calibration", self.card_ID,":", self.pcs_str, tc.ENDC)
#         self.cal_off = self.coeffs[self.counter]
#         print self.counter, self.cal_off
        self.cal_off = cal_val
        self.cal_offset.setText(str(self.cal_off))
        self.cal_off_deg.setText(str(self.cal_off*9))
        if self.tot_steps.text() == self.cal_offset.text():
            self.tot_steps.setStyleSheet("background-color: #90EE90;")
        else:
            self.tot_steps.setStyleSheet("background-color: #F08080;")            
        
               
    def enableDPR(self):            # set True before sending GPI 2/3
        logging.debug("enable DPR", tc.ENDC)
        self.sendReg(self.GPI3 + 0b1)

    def disableDPR(self):            # set False after sending GPI 2/3
        logging.debug("disable DPR", tc.ENDC)
        self.sendReg(self.GPI3)

    def resetPhase(self):
        logging.debug(tc.FCTCALL + "reset phase", self.card_ID,":", self.pcs_str, tc.ENDC)
        
        self.resetFlag = True
        if self.counter == 0:                   # firmware PLL reset
            self.enableDPR()
            addr = 3
            wreg = addr << 17
            wregval = wreg + 5                  # PR/SMenb bit HI
            logging.debug("phase reset = True")
            self.sendReg(wregval)
            wregval = wreg + 1                  # PR bit LO (SMenb HI)
            logging.debug("phase reset = False")
            self.sendReg(wregval)
            self.allCountersReset = True
            self.disableDPR()
            
            
        'reset spin & total values to 0'
        
        self.phase_trim_spin.setValue(0)
        self.tot_steps.setText(str(0))
        self.trim_deg.setText(str(0))
        self.tot_degs.setText(str(0))
        
        self.ptrim[self.counter] = int(self.tot_steps.text())
        
        self.lastSpinVal = 0
        self.lastTotVal = 0

        'if total value differ from calibration offsets set total cells RED to indicate out of calibration'
        
        if self.tot_steps.text() == self.cal_offset.text():
            self.tot_steps.setStyleSheet("background-color: #90EE90;")
        else:
            self.tot_steps.setStyleSheet("background-color: #F08080;")
        
        self.resetFlag = False
            
    def sendReg(self, wregval):
        logging.debug(tc.COMMAND + "send to address", self.address, ":", tc.BOLD, wregval, tc.ENDC)
        b0 = (wregval & 0x7f ) << 1            # 1st 7 bits shifted up 1
        b1 = ((wregval >> 7) & 0x7f) <<  1     # 2nd 7 bits shifted up 1
        b2 = ((wregval >> 14) & 0x7f) << 1     # 3rd 7 bits shifted up 1
        b3 = ((wregval >> 21) & 0x7f) << 1     # 4th 7 bits shifted up 1
        b4 = (self.address << 1) + 1           # Address shifted up 1 bit with address bit set

        msg = struct.pack('BBBBB', b0, b1, b2, b3, b4)
#         print bin(b4)[2:].zfill(8),b3,b2,b1,b0
        self.serialport.write(msg)
        time.sleep(0.001)

def main():
     
    app = QApplication(sys.argv)
    ex = dpr_counter()
    sys.exit(app.exec_())
 
if __name__ == '__main__':
    main()
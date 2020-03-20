import sys
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt, SIGNAL
from PyQt4.QtGui import QDoubleSpinBox, QSpinBox, QFrame, QGroupBox

# import named_serial
import struct

class dpr_counter(QtGui.QWidget):
    
    def __init__(self, parent=None, layout=None, pcs=0, idx=0, slot=1, coeffs=[0], appTrim=[0], serialport=None, cardaddr=3):

        super(dpr_counter, self).__init__()
        
        self.COMMAND = '\033[95m'
        self.FCTCALL = '\033[94m'
        self.INIT = '\033[92m'
        self.WARNING = '\033[93m'
        self.FAIL = '\033[91m'
        self.ENDC = '\033[0m'
        self.BOLD = "\033[1m"

        
        self.green = "90EE90"
        self.red ="F08080"
        
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
                
        self.layout_widget = QtGui.QGroupBox(self)
        self.layout_widget.setTitle(self.pcs_str)
        self.layout_widget.setStyleSheet("font-size: 14px")
#         self.layout_widget.setStyleSheet("font-style: italic")
        self.row_layout = QtGui.QGridLayout(self.layout_widget)
        self.row_layout.setSpacing(5)
        
        self.counter_label = QtGui.QLabel(self.ctr_dsc)
        self.counter_label.setFixedWidth(200)
        self.counter_label.setAlignment(QtCore.Qt.AlignLeft)
                
        self.row_layout.addWidget(self.counter_label,0,0,1,1,QtCore.Qt.AlignCenter)
        
        self.counter_label = QtGui.QLabel(self.ctr_fct)
        self.counter_label.setFixedWidth(200)
        self.counter_label.setAlignment(QtCore.Qt.AlignLeft)
                
        self.row_layout.addWidget(self.counter_label,1,0,1,1,QtCore.Qt.AlignCenter)
        
        self.cal_offset = QtGui.QLineEdit()
        self.cal_offset.setReadOnly(True)
        self.cal_offset.setFixedWidth(50)
        self.cal_offset.setAlignment(QtCore.Qt.AlignRight)
        self.cal_offset.setText(str(self.cal_off))
        
        self.row_layout.addWidget(self.cal_offset,0,1,QtCore.Qt.AlignCenter)

        self.seperator_label = QtGui.QLabel(sep_str)
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

        self.result1_label = QtGui.QLabel("=")
        self.result1_label.setFixedWidth(20)
        self.result1_label.setAlignment(QtCore.Qt.AlignCenter)
                
        self.row_layout.addWidget(self.result1_label,0,4,QtCore.Qt.AlignCenter)
         
        self.tot_steps = QtGui.QLineEdit()
        self.tot_steps.setReadOnly(True)
        self.tot_steps.setFixedWidth(50)
        self.tot_steps.setAlignment(QtCore.Qt.AlignRight)
        self.tot_steps.setText(str(self.cal_off+self.phase_trim_spin.value()))
        
        self.row_layout.addWidget(self.tot_steps,0,5,QtCore.Qt.AlignCenter)
        
        self.step_label = QtGui.QLabel("steps")
        self.step_label.setFixedWidth(50)
         
        self.row_layout.addWidget(self.step_label,0,6,QtCore.Qt.AlignLeft)
        
        self.calibrate = QtGui.QPushButton(self, text = "Calibrate "+self.pcs_str)
        self.calibrate.setFixedWidth(180)
        self.calibrate.clicked.connect(self.calcounter)

        self.row_layout.addWidget(self.calibrate,0,7,QtCore.Qt.AlignCenter)


        self.cal_off_deg = QtGui.QLineEdit()
        self.cal_off_deg.setReadOnly(True)
        self.cal_off_deg.setFixedWidth(50)
        self.cal_off_deg.setAlignment(QtCore.Qt.AlignRight)
        self.cal_off_deg.setText(str(self.cal_off*9))
        
        self.row_layout.addWidget(self.cal_off_deg,1,1,QtCore.Qt.AlignCenter)

        self.seperator2_label = QtGui.QLabel(sep_str)
        self.seperator2_label.setFixedWidth(20)
        self.seperator2_label.setAlignment(QtCore.Qt.AlignCenter)
                
        self.row_layout.addWidget(self.seperator2_label,1,2,QtCore.Qt.AlignCenter)

        self.trim_deg = QtGui.QLineEdit()
        self.trim_deg.setReadOnly(True)
        self.trim_deg.setFixedWidth(43)
        self.trim_deg.setAlignment(QtCore.Qt.AlignRight)
        self.trim_deg.setText(str(self.phase_trim_spin.value()*9))
        
        self.row_layout.addWidget(self.trim_deg,1,3,QtCore.Qt.AlignLeft)

        self.result2_label = QtGui.QLabel("=")
        self.result2_label.setFixedWidth(20)
        self.result2_label.setAlignment(QtCore.Qt.AlignCenter)
         
        self.row_layout.addWidget(self.result2_label,1,4,QtCore.Qt.AlignLeft)
         
        self.tot_degs = QtGui.QLineEdit()
        self.tot_degs.setReadOnly(True)
        self.tot_degs.setFixedWidth(50)
        self.tot_degs.setAlignment(QtCore.Qt.AlignRight)
        self.tot_degs.setText(str((self.cal_off+self.phase_trim_spin.value())*9))
        
        self.row_layout.addWidget(self.tot_degs,1,5,QtCore.Qt.AlignCenter)
 
        self.degrees_label = QtGui.QLabel("deg")
        self.degrees_label.setFixedWidth(50)
                
        self.row_layout.addWidget(self.degrees_label,1,6,QtCore.Qt.AlignCenter)

        self.commit = QtGui.QPushButton(self, text = "Commit Calibration")
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
            print self.FCTCALL + "step phase", self.card_ID,":", self.pcs_str, "from", self.lastTotVal,"to", self.tot_steps.text(), self.ENDC
        else:
            self.tot_steps.setText(str(self.phase_trim_spin.value()))
            print self.FCTCALL + "step phase:", self.card_ID,":", self.pcs_str, "from", self.lastSpinVal,"to", val, self.ENDC
        self.tot_degs.setText(str(int(self.tot_steps.text())*9))
        self.ptrim[self.counter] = int(self.tot_degs.text())

        self.enableDPR()
        addr = 2
        wreg = addr << 17
        wregval1 = wreg + (self.counter << 2)
        print "set phase adjust register"
        self.sendReg(wregval1)
        if val > self.lastSpinVal:
            wregval2 = wregval1 + 1
            print "increment "+ self.pcs_str
            self.sendReg(wregval2)
        else:
            wregval2 = wregval1 + 2
            print "decrement "+ self.pcs_str
            self.sendReg(wregval2)
        wregval3 = wregval1 + (0 << 2)
        print "reset phase adjust register"
        self.sendReg(wregval3)
        self.disableDPR()

        self.lastSpinVal = val
        self.lastTotVal = int(self.tot_steps.text())
        if self.tot_steps.text() == self.cal_offset.text():
            self.tot_steps.setStyleSheet("background-color: #90EE90;")
        else:
            self.tot_steps.setStyleSheet("background-color: #F08080;")
        print
            
    def commit_cal(self):
        print self.FCTCALL + "commit calibration", self.card_ID,":",  self.pcs_str, self.ENDC
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
            self.tot_degs.setText(str(self.cal_off*9))
#             addr = 1
#             wreg = addr << 17
            mask = 0xfffe01f
#             trim = (self.cal_off & 0xff) << 5
            wregval = (self.GPI1 & mask) | (self.cal_off << 5)
            print "send phase trim coefficient & slot:", self.cal_off,"/", self.slot
            self.sendReg(wregval)
        print
        self.ptrim[self.counter] = int(self.tot_degs.text())
        self.commitFlag = False
        

    def calcounter(self):
        print self.FCTCALL + "calibrate counter", self.card_ID,":", self.pcs_str, self.ENDC
        if self.counter == 0:
            '''
             This case implements a firmware PLL reset & auto calibration.
             The auto calibration applies the slot & phase trim offsets to ALL counters.
             As a result of the RESET process, the other individual counter offsets are reset as well.
             In this case the code must reset the spin values and total steps of the other counters to 0.
             But changing a spin value calls newvalue which automatically steps the phase.
             So we must use a flag, self.resetFlag, set in resetPhase, to branch out of NEWVALUE when called from there.
            '''
            print "firmware calibrate ALL" 
            print "send phase trim coefficient & slot:", self.cal_off,"/", self.slot
#             print "send slot & phase trim:"
            self.sendReg(self.GPI1)
            self.enableDPR()
            self.parent.resetALLphase()         # reset ALL phase
            self.tot_steps.setText(self.cal_offset.text())
            self.tot_degs.setText(str(int(self.tot_steps.text())*9))
            addr = 3
            wreg = addr << 17
            wregval = wreg + 9                  # PC/SMenb => HI
            print "phase calibrate = True"
            self.sendReg(wregval)
            wregval = wreg + 1                  # PC => LO (SMenb => HI)
            print "phase calibrate = False"
            self.sendReg(wregval)
            self.disableDPR()
            print
        else:
            steps = self.phase_trim_spin.value() - int(self.cal_offset.text())
            print "software calibrate:", -steps, "phase steps applied"
            print
            while steps != 0:
                if steps > 0:
                    self.phase_trim_spin.setValue(self.phase_trim_spin.value() - 1)
                    steps = steps -1
                if steps < 0:
                    self.phase_trim_spin.setValue(self.phase_trim_spin.value() + 1)
                    steps = steps +1
        print
        
        self.cal_offset.setStyleSheet("background-color: #90EE90;")
        self.tot_steps.setStyleSheet("background-color: #90EE90;")
        self.ptrim[self.counter] = int(self.tot_degs.text())
        
    def loadCal(self):
        print self.FCTCALL + "load calibration", self.card_ID,":", self.pcs_str, self.ENDC
        self.cal_off = self.coeffs[self.counter]
        self.cal_offset.setText(str(self.cal_off))
        self.cal_off_deg.setText(str(self.cal_off*9))
        if self.tot_steps.text() == self.cal_offset.text():
            self.tot_steps.setStyleSheet("background-color: #90EE90;")
        else:
            self.tot_steps.setStyleSheet("background-color: #F08080;")            
               
    def enableDPR(self):            # set True before sending wreg 2/3
        print self.FCTCALL + "enable DPR", self.ENDC
        addr = 3
        wreg = addr << 17
        wregval = wreg + 1
#         print "address =", int(bin(wregval)[2:].zfill(28)[3:11])
#         print "SMenb =", bool(bin(wregval)[2:].zfill(28)[27])
        self.sendReg(wregval)

    def disableDPR(self):            # set False after sending wreg 2/3
        print self.FCTCALL + "disable DPR", self.ENDC
        addr = 3
        wreg = addr << 17
        wregval = wreg + 0
        self.sendReg(wregval)

    def resetPhase(self):
        print self.FCTCALL + "reset phase", self.card_ID,":", self.pcs_str, self.ENDC
        self.resetFlag = True
        if self.counter == 0:                   # firmware PLL reset
            addr = 3
            wreg = addr << 17
            wregval = wreg + 5                  # PR/SMenb bit HI
            print "phase reset = True"
            self.sendReg(wregval)
            wregval = wreg + 1                  # PR bit LO (SMenb HI)
            print "reset phase = False"
            self.sendReg(wregval)
            self.allCountersReset = True
            
        'reset spin & total values to 0'
        
        self.phase_trim_spin.setValue(0)
        self.tot_steps.setText(str(0))
        self.trim_deg.setText(str(0))
        self.tot_degs.setText(str(0))
        
        self.ptrim[self.counter] = int(self.tot_degs.text())
        
        self.lastSpinVal = 0
        self.lastTotVal = 0

        'if total value differ from calibration offsets set total cells RED to indicate out of calibration'
        
        if self.tot_steps.text() == self.cal_offset.text():
            self.tot_steps.setStyleSheet("background-color: #90EE90;")
        else:
            self.tot_steps.setStyleSheet("background-color: #F08080;")
        
        self.resetFlag = False
            
    def sendReg(self, wregval):
        print self.COMMAND + "send to address", self.address, ":", self.BOLD, wregval, self.ENDC
        b0 = (wregval & 0x7f ) << 1            # 1st 7 bits shifted up 1
        b1 = ((wregval >> 7) & 0x7f) <<  1     # 2nd 7 bits shifted up 1
        b2 = ((wregval >> 14) & 0x7f) << 1     # 3rd 7 bits shifted up 1
        b3 = ((wregval >> 21) & 0x7f) << 1     # 4th 7 bits shifted up 1
        b4 = (self.address << 1) + 1           # Address shifted up 1 bit with address bit set

        msg = struct.pack('BBBBB', b0, b1, b2, b3, b4)
#         print bin(b4)[2:].zfill(8),b3,b2,b1,b0
        self.serialport.write(msg)

def main():
     
    app = QtGui.QApplication(sys.argv)
    ex = dpr_counter()
    sys.exit(app.exec_())
 
if __name__ == '__main__':
    main()
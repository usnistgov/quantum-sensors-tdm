import sys
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

# import named_serial
import struct

class badChn(QWidget):
    
    def __init__(self, parent=None, layout=None, chn=0, cardaddr=3, serialport=None, master=None):

        super(badChn, self).__init__()

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
        
        self.parent = parent
        self.layout = layout
        
        self.chn = chn
        self.address = cardaddr
        self.serialport = serialport

        self.green = "90EE90"
        self.red ="F08080"
        
        self.unlocked = 1
        
        self.dc = False
        self.lohi = True
        self.tri = False
                
#         self.layout_widget = QHBox(self)
#         self.layout_widget.setStyleSheet("font-size: 14px")
#         self.layout_widget.setStyleSheet("font-style: italic")
        self.row_layout = QGridLayout(self)
#         self.row_layout = QHBoxLayout(self)
        self.row_layout.setContentsMargins(0,0,0,0)
        self.row_layout.setSpacing(5)
         
#         self.counter_label = QLabel(str(self.chn))
#         self.counter_label.setFixedWidth(20)
#         self.counter_label.setAlignment(QtCore.Qt.AlignRight)

        self.counter_label = QLineEdit()
        self.counter_label.setReadOnly(True)
        self.counter_label.setFixedWidth(36)
        self.counter_label.setAlignment(QtCore.Qt.AlignRight)
        self.counter_label.setStyleSheet("background-color: #" + self.yellow + ";")
        self.counter_label.setFocusPolicy(QtCore.Qt.NoFocus)
        self.counter_label.setText(str(self.chn))
        self.row_layout.addWidget(self.counter_label,0,0)
        
        self.row_ht = 25

        self.dc_button = QToolButton(self, text = 'dc')
        self.dc_button.setFixedHeight(self.row_ht)
        self.dc_button.setCheckable(1)
        self.dc_button.setChecked(self.dc)
        self.dc_button.setStyleSheet("background-color: #" + self.red + ";")
        self.row_layout.addWidget(self.dc_button,0,1)

        self.LoHi_button = QToolButton(self, text = 'HI')
        self.LoHi_button.setFixedHeight(self.row_ht)
        self.LoHi_button.setCheckable(1)
        self.LoHi_button.setChecked(self.lohi)
        self.LoHi_button.setStyleSheet("background-color: #" + self.green + ";")
        self.row_layout.addWidget(self.LoHi_button,0,2)

        self.Tri_button = QToolButton(self, text = 'tri')
        self.Tri_button.setFixedHeight(self.row_ht)
        self.Tri_button.setCheckable(1)
        self.Tri_button.setChecked(self.tri)
        self.Tri_button.setStyleSheet("background-color: #" + self.red + ";")
        self.row_layout.addWidget(self.Tri_button,0,3)

        self.d2a_lo_spin = QSpinBox()
        self.d2a_lo_spin.setRange(0, 16383)
        self.d2a_lo_spin.setFixedWidth(80)
        self.d2a_lo_spin.setFixedHeight(self.row_ht)
        self.d2a_lo_spin.setSingleStep(1)
        self.d2a_lo_spin.setKeyboardTracking(0)
        self.d2a_lo_spin.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.row_layout.addWidget(self.d2a_lo_spin,0,4)

        self.d2a_lo_min_button = QPushButton("<")
        self.d2a_lo_min_button.setFixedWidth(25)
        self.d2a_lo_min_button.setFixedHeight(self.row_ht)        
        self.row_layout.addWidget(self.d2a_lo_min_button,0,5)
        
        self.d2a_lo_slider = QSlider(QtCore.Qt.Horizontal)
        self.d2a_lo_slider.setTickPosition(QSlider.TicksBelow)
        self.d2a_lo_slider.setRange(0,16383)
        self.d2a_lo_slider.setFixedWidth(150)
        self.d2a_lo_slider.setFixedHeight(self.row_ht)        
        self.d2a_lo_slider.setTickInterval(1024)
        self.d2a_lo_slider.setSingleStep(32)
        self.d2a_lo_slider.setPageStep(512)
        self.d2a_lo_slider.setValue(0)
        self.d2a_lo_slider.setTracking(1)
        self.d2a_lo_slider.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.row_layout.addWidget(self.d2a_lo_slider,0,6)

        self.d2a_lo_max_button = QPushButton(">")
        self.d2a_lo_max_button.setFixedWidth(25)
        self.d2a_lo_max_button.setFixedHeight(self.row_ht)        
        self.row_layout.addWidget(self.d2a_lo_max_button,0,7)

        self.d2a_hi_spin = QSpinBox()
        self.d2a_hi_spin.setRange(0, 16383)
        self.d2a_hi_spin.setFixedWidth(80)
        self.d2a_hi_spin.setFixedHeight(self.row_ht)        
        self.d2a_hi_spin.setSingleStep(1)
        self.d2a_hi_spin.setKeyboardTracking(0)
        self.d2a_hi_spin.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.row_layout.addWidget(self.d2a_hi_spin,0,8)

        self.d2a_hi_min_button = QPushButton("<")
        self.d2a_hi_min_button.setFixedWidth(25)
        self.d2a_hi_min_button.setFixedHeight(self.row_ht)        
        self.row_layout.addWidget(self.d2a_hi_min_button,0,9)
        
        self.d2a_hi_slider = QSlider(QtCore.Qt.Horizontal)
        self.d2a_hi_slider.setTickPosition(QSlider.TicksBelow)
        self.d2a_hi_slider.setRange(0,16383)
        self.d2a_hi_slider.setFixedWidth(150)
        self.d2a_hi_slider.setFixedHeight(self.row_ht)        
        self.d2a_hi_slider.setTickInterval(1024)
        self.d2a_hi_slider.setSingleStep(32)
        self.d2a_hi_slider.setPageStep(512)
        self.d2a_hi_slider.setValue(0)
        self.d2a_hi_slider.setTracking(1)
        self.d2a_hi_slider.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.row_layout.addWidget(self.d2a_hi_slider,0,10)

        self.d2a_hi_max_button = QPushButton(">")
        self.d2a_hi_max_button.setFixedWidth(25)
        self.d2a_hi_max_button.setFixedHeight(self.row_ht)        
        self.row_layout.addWidget(self.d2a_hi_max_button,0,11)

        self.chn_send = QPushButton(self, text = "send channel")
        self.chn_send.setFixedHeight(self.row_ht)
        self.chn_send.setFixedWidth(160)
        self.row_layout.addWidget(self.chn_send,0,12)
        
#         self.mode_menu = QMenu(self)
#         self.mode_menu.addAction('static')
#         self.mode_menu.addAction('track')
#         self.mode_menu.addAction('lock')

        self.lock_button = QToolButton(self, text = 'dynamic')
#         self.lock_button.setMenu(self.mode_menu)
        self.lock_button.setFixedWidth(50)
        self.lock_button.setFixedHeight(self.row_ht)
        self.lock_button.setCheckable(1)
        self.lock_button.setChecked(1)
        self.lock_button.setStyleSheet("background-color: #" + self.green + ";")
        self.row_layout.addWidget(self.lock_button,0,13)
        
        if master != None:
            self.chn_lbl = QLabel("CHN")
            self.chn_lbl.setToolTip("DAC channel number (corresponds to front panel)")
            self.chn_lbl.setFrameStyle(50)
            self.chn_lbl.setFrameStyle(50)
            self.chn_lbl.setAlignment(QtCore.Qt.AlignCenter)
            self.chn_lbl.setStyleSheet("background-color: #" + self.grey + ";color : #" + self.white)
            self.row_layout.addWidget(self.chn_lbl,1,0)
            
            self.bool_lbl = QLabel("booleans")
            self.bool_lbl.setToolTip("select DC mode, HI/LO level for DC mode, and/or TRI output")
    #         self.bool_lbl.setFixedWidth(85)
            self.bool_lbl.setFrameStyle(50)
            self.bool_lbl.setAlignment(QtCore.Qt.AlignCenter)
            self.bool_lbl.setStyleSheet("background-color: #" + self.grey + ";color : #"+self.white)
            self.row_layout.addWidget(self.bool_lbl,1,1,1,3)

            self.DAClo_lbl = QLabel("DAC low level")
#             self.DAClo_lbl.setFixedWidth(327)
            self.DAClo_lbl.setFrameStyle(50)
            self.DAClo_lbl.setAlignment(QtCore.Qt.AlignCenter)
            self.DAClo_lbl.setStyleSheet("background-color: #" + self.grey + ";color : #"+self.white)
            self.row_layout.addWidget(self.DAClo_lbl,1,4,1,4)
    
            self.DAChi_lbl = QLabel("DAC high level")
#             self.DAChi_lbl.setFixedWidth(327)
            self.DAChi_lbl.setFrameStyle(50)
            self.DAChi_lbl.setAlignment(QtCore.Qt.AlignCenter)
            self.DAChi_lbl.setStyleSheet("background-color: #" + self.grey + ";color : #"+self.white)
            self.row_layout.addWidget(self.DAChi_lbl,1,8,1,4)
    
            self.command_lbl = QLabel("command")
#             self.command_lbl.setFixedWidth(150)
            self.command_lbl.setFrameStyle(50)
            self.command_lbl.setAlignment(QtCore.Qt.AlignCenter)
            self.command_lbl.setStyleSheet("background-color: #" + self.grey + ";color : #"+self.white)
            self.row_layout.addWidget(self.command_lbl,1,12,1,1)
    
            self.mode_lbl = QLabel("mode")
#             self.mode_lbl.setFixedWidth(50)
            self.mode_lbl.setFrameStyle(50)
            self.mode_lbl.setAlignment(QtCore.Qt.AlignCenter)
            self.mode_lbl.setStyleSheet("background-color: #" + self.grey + ";color : #"+self.white)
            self.row_layout.addWidget(self.mode_lbl,1,13,1,1)

        if master == None:
            self.dc_button.toggled.connect(self.dc_changed)
            self.LoHi_button.toggled.connect(self.LoHi_changed)
            self.Tri_button.toggled.connect(self.tri_changed)
            self.d2a_lo_spin.valueChanged.connect(self.d2a_lo_spin_changed)
            self.d2a_lo_min_button.clicked.connect(self.d2a_lo_setMin)
            self.d2a_lo_slider.valueChanged.connect(self.d2a_lo_slider_changed)
            self.d2a_lo_max_button.clicked.connect(self.d2a_lo_setMax)
            self.d2a_hi_spin.valueChanged.connect(self.d2a_hi_spin_changed)
            self.d2a_hi_min_button.clicked.connect(self.d2a_hi_setMin)
            self.d2a_hi_slider.valueChanged.connect(self.d2a_hi_slider_changed)
            self.d2a_hi_max_button.clicked.connect(self.d2a_hi_setMax)        
            self.chn_send.clicked.connect(self.send_channel)
#             self.lock_button.mode_menu.triggered('static').connect(self.lock_channel)
            self.lock_button.toggled.connect(self.lock_channel)
            
        if master != None:
            self.dc_button.toggled.connect(parent.dc_changed, self.dc_button.isChecked())
            self.LoHi_button.toggled.connect(parent.LoHi_changed, self.LoHi_button.isChecked())
            self.Tri_button.toggled.connect(parent.tri_changed, self.Tri_button.isChecked())
            self.d2a_lo_spin.valueChanged.connect(parent.d2a_lo_spin_changed, self.d2a_lo_spin.value())
            self.d2a_lo_min_button.clicked.connect(parent.d2a_lo_setMin)
            self.d2a_lo_slider.valueChanged.connect(parent.d2a_lo_slider_changed, self.d2a_lo_slider.value())
            self.d2a_lo_max_button.clicked.connect(parent.d2a_lo_setMax)
            self.d2a_hi_spin.valueChanged.connect(parent.d2a_hi_spin_changed)
            self.d2a_hi_min_button.clicked.connect(parent.d2a_hi_setMin)
            self.d2a_hi_slider.valueChanged.connect(parent.d2a_hi_slider_changed, self.d2a_hi_slider.value())
            self.d2a_hi_max_button.clicked.connect(parent.d2a_hi_setMax)        
            self.chn_send.clicked.connect(parent.send_channel)
            self.lock_button.toggled.connect(parent.lock_channel, self.lock_button.isChecked())
        
#         self.setStyleSheet("background-color: #" + self.grey + ";")
        
        if self.parent != None:
            self.layout.addWidget(self) 
       
#         self.show()
        
    def dc_changed(self):
        self.dc = self.dc_button.isChecked()
        if self.dc ==1:
            self.dc_button.setStyleSheet("background-color: #" + self.green + ";")
        else:
            self.dc_button.setStyleSheet("background-color: #" + self.red + ";")            
#         if self.unlocked == 1:
        self.send_channel()
        
    def LoHi_changed(self):
        self.lohi = self.LoHi_button.isChecked()
        if self.lohi ==1:
            self.LoHi_button.setStyleSheet("background-color: #" + self.green + ";")
            self.LoHi_button.setText('HI')
        else:
            self.LoHi_button.setStyleSheet("background-color: #" + self.red + ";")            
            self.LoHi_button.setText('LO')
#         if self.unlocked == 1:
        self.send_channel()
        
    def tri_changed(self):
        self.tri = self.Tri_button.isChecked()
        if self.tri ==1:
            self.Tri_button.setStyleSheet("background-color: #" + self.green + ";")
        else:
            self.Tri_button.setStyleSheet("background-color: #" + self.red + ";")  
#         if self.unlocked == 1:
        self.send_channel()
            
    def d2a_lo_spin_changed(self):
        self.d2a_lo_slider.setValue(self.d2a_lo_spin.value())
        if self.unlocked == 1:
            self.send_channel()
        
    def d2a_lo_slider_changed(self):
        self.d2a_lo_spin.setValue(self.d2a_lo_slider.value())
        
    def d2a_lo_setMin(self):
        self.d2a_lo_slider.setValue(0)
        
    def d2a_lo_setMax(self):
        self.d2a_lo_slider.setValue(16383)

    def d2a_hi_spin_changed(self):
        self.d2a_hi_slider.setValue(self.d2a_hi_spin.value())
        if self.unlocked == 1:
            self.send_channel()
        
    def d2a_hi_slider_changed(self):
        self.d2a_hi_spin.setValue(self.d2a_hi_slider.value())
        
    def d2a_hi_setMin(self):
        self.d2a_hi_slider.setValue(0)
        
    def d2a_hi_setMax(self):
        self.d2a_hi_slider.setValue(16383)
        
    def send_channel(self):
        print(self.FCTCALL + "send BAD16 CHN", self.chn, ": index & arrayed register values", self.ENDC)
        self.send_wreg2()
        self.send_wreg4()
        self.send_wreg5()
        print()

    def lock_channel(self):
        self.unlocked = self.lock_button.isChecked()
        if self.unlocked ==1:
            self.lock_button.setStyleSheet("background-color: #" + self.green + ";")
            self.lock_button.setText('dynamic')
        else:
            self.lock_button.setStyleSheet("background-color: #" + self.red + ";")            
            self.lock_button.setText('static')
    
    def send_wreg2(self):
        print("BAD16:WREG2: channel index")
        wreg = 2 << 25
        wregval = wreg | self.chn
        self.sendReg(wregval)
        print()

    def send_wreg4(self):
        print("BAD16:WREG4: channel booleans & DAC high value")
        wreg = 4 << 25
        wreg = wreg | (int(self.dc) << 21)
        wreg = wreg | (int(self.lohi) << 20)
        wreg = wreg | (int(self.tri) << 19)
        wregval = wreg | self.d2a_hi_spin.value()
        self.sendReg(wregval)
        print()

    def send_wreg5(self):
        print("BAD16:WREG5: channel DAC low value")
        wreg = 5 << 25
        wregval = wreg | (self.d2a_lo_slider.value() << 8)
        self.sendReg(wregval)
        print()
        
    def sendReg(self, wregval): 
        print(self.COMMAND + "send to address", self.address, ":", self.BOLD, wregval, self.ENDC)
        b0 = (wregval & 0x7f ) << 1            # 1st 7 bits shifted up 1
        b1 = ((wregval >> 7) & 0x7f) <<  1     # 2nd 7 bits shifted up 1
        b2 = ((wregval >> 14) & 0x7f) << 1     # 3rd 7 bits shifted up 1
        b3 = ((wregval >> 21) & 0x7f) << 1     # 4th 7 bits shifted up 1
        b4 = (self.address << 1) + 1           # Address shifted up 1 bit with address bit set
 
        msg = struct.pack('BBBBB', b0, b1, b2, b3, b4)
        self.serialport.write(msg)
        
    def packChannel(self):
        self.ChannelVector    =    {
            'dc'            :    self.dc_button.isChecked(),
            'LoHi'          :    self.LoHi_button.isChecked(),
            'tri'           :    self.Tri_button.isChecked(),
            'd2a_lo'        :    self.d2a_lo_spin.value(),
            'd2a_hi'        :    self.d2a_hi_spin.value(),
                                }
        
    def unpackChannel(self, loadChannel):
            self.dc_button.setChecked(loadChannel['dc'])
            self.LoHi_button.setChecked(loadChannel['LoHi'])
            self.Tri_button.setChecked(loadChannel['tri'])
            self.d2a_lo_spin.setValue(loadChannel['d2a_lo'])
            self.d2a_hi_spin.setValue(loadChannel['d2a_hi'])

def main():
     
    app = QApplication(sys.argv)
    ex = badChn()
    sys.exit(app.exec_())
 
if __name__ == '__main__':
    main()
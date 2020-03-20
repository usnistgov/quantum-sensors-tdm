import sys
import time
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt, SIGNAL
from PyQt4.QtGui import QWidget, QDoubleSpinBox, QSpinBox, QFrame, QGroupBox,QToolButton, QPushButton, QSlider, QMenu

# import named_serial
import struct

class dfbChn(QtGui.QWidget):
    
    def __init__(self, parent=None, layout=None, state=0, chn=0, cardaddr=3, serialport=None, master=None):

        super(dfbChn, self).__init__()

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
        
        self.address = cardaddr
        self.serialport = serialport

        self.state = state
        self.chn = chn
        self.triA = 0
        self.triB = 0
        self.a2d_lockpt = 0
        self.d2a_A = 0
        self.d2a_B = 0
        self.SM = 0
        self.P = 0
        self.I = 0
        self.FBA = 0
        self.FBB = 0
        self.ARL = 0
        
        self.unlocked = 1
        
        self.dc = False
        self.lohi = True
        self.tri = False
        
        self.saveState = {}
                
#         self.layout_widget = QtGui.QHBox(self)
#         self.layout_widget.setStyleSheet("font-size: 14px")
#         self.layout_widget.setStyleSheet("font-style: italic")
        self.row_layout = QtGui.QGridLayout(self)
#         self.row_layout = QtGui.QHBoxLayout(self)
        self.row_layout.setMargin(0)
        self.row_layout.setSpacing(5)
        
#         self.counter_label = QtGui.QLabel(str(self.chn))
#         self.counter_label.setFixedWidth(20)
#         self.counter_label.setAlignment(QtCore.Qt.AlignRight)

        self.counter_label = QtGui.QLineEdit()
        self.counter_label.setReadOnly(True)
        self.counter_label.setFixedWidth(36)
        self.counter_label.setAlignment(QtCore.Qt.AlignRight)
        self.counter_label.setStyleSheet("background-color: #" + self.yellow + ";")
        self.counter_label.setFocusPolicy(Qt.NoFocus)
        self.counter_label.setText(str(self.state))
        self.row_layout.addWidget(self.counter_label,0,0)

        self.TriA_button = QToolButton(self, text = '^A')
        self.TriA_button.setFixedHeight(25)
        self.TriA_button.setCheckable(1)
        self.TriA_button.setChecked(self.triA)
        self.TriA_button.setStyleSheet("background-color: #" + self.red + ";")
        self.row_layout.addWidget(self.TriA_button,0,1)

        self.TriB_button = QToolButton(self, text = '^B')
        self.TriB_button.setFixedHeight(25)
        self.TriB_button.setCheckable(1)
        self.TriB_button.setChecked(self.triB)
        self.TriB_button.setStyleSheet("background-color: #" + self.red + ";")
        self.row_layout.addWidget(self.TriB_button,0,2)

        self.a2d_lockpt_spin = QSpinBox()
        self.a2d_lockpt_spin.setRange(0, 4095)
        self.a2d_lockpt_spin.setFixedWidth(60)
        self.a2d_lockpt_spin.setFixedHeight(25)
        self.a2d_lockpt_spin.setSingleStep(1)
        self.a2d_lockpt_spin.setKeyboardTracking(0)
        self.a2d_lockpt_spin.setFocusPolicy(Qt.StrongFocus)
        self.row_layout.addWidget(self.a2d_lockpt_spin,0,3)

#         self.d2a_lo_min_button = QPushButton("<")
#         self.row_layout.addWidget(self.d2a_lo_min_button,0,5)
        
        self.a2d_lockpt_slider = QSlider(Qt.Horizontal)
        self.a2d_lockpt_slider.setTickPosition(QSlider.TicksBelow)
        self.a2d_lockpt_slider.setRange(0,4095)
        self.a2d_lockpt_slider.setFixedWidth(120)
        self.a2d_lockpt_slider.setFixedHeight(25)
        self.a2d_lockpt_slider.setTickInterval(256)
        self.a2d_lockpt_slider.setSingleStep(8)
        self.a2d_lockpt_slider.setPageStep(128)
        self.a2d_lockpt_slider.setValue(0)
        self.a2d_lockpt_slider.setTracking(1)
        self.a2d_lockpt_slider.setFocusPolicy(Qt.StrongFocus)
        self.row_layout.addWidget(self.a2d_lockpt_slider,0,4)

#         self.d2a_lo_max_button = QPushButton(">")
#         self.row_layout.addWidget(self.d2a_lo_max_button,0,7)

        self.d2a_A_spin = QSpinBox()
        self.d2a_A_spin.setRange(0, 16383)
        self.d2a_A_spin.setFixedWidth(80)
        self.d2a_A_spin.setFixedHeight(25)
        self.d2a_A_spin.setSingleStep(1)
        self.d2a_A_spin.setKeyboardTracking(0)
        self.d2a_A_spin.setFocusPolicy(Qt.StrongFocus)
        self.row_layout.addWidget(self.d2a_A_spin,0,5)
        
        self.d2a_A_slider = QSlider(Qt.Horizontal)
        self.d2a_A_slider.setTickPosition(QSlider.TicksBelow)
        self.d2a_A_slider.setRange(0,16383)
        self.d2a_A_slider.setFixedWidth(120)
        self.d2a_A_slider.setFixedHeight(25)
        self.d2a_A_slider.setTickInterval(1024)
        self.d2a_A_slider.setSingleStep(32)
        self.d2a_A_slider.setPageStep(512)
        self.d2a_A_slider.setValue(0)
        self.d2a_A_slider.setTracking(1)
        self.d2a_A_slider.setFocusPolicy(Qt.StrongFocus)
        self.row_layout.addWidget(self.d2a_A_slider,0,6)

        self.d2a_B_spin = QSpinBox()
        self.d2a_B_spin.setRange(0, 16383)
        self.d2a_B_spin.setFixedWidth(80)
        self.d2a_B_spin.setFixedHeight(25)
        self.d2a_B_spin.setSingleStep(1)
        self.d2a_B_spin.setKeyboardTracking(0)
        self.d2a_B_spin.setFocusPolicy(Qt.StrongFocus)
        self.row_layout.addWidget(self.d2a_B_spin,0,7)
        
        self.d2a_B_slider = QSlider(Qt.Horizontal)
        self.d2a_B_slider.setTickPosition(QSlider.TicksBelow)
        self.d2a_B_slider.setRange(0,16383)
        self.d2a_B_slider.setFixedWidth(120)
        self.d2a_B_slider.setFixedHeight(25)
        self.d2a_B_slider.setTickInterval(1024)
        self.d2a_B_slider.setSingleStep(32)
        self.d2a_B_slider.setPageStep(512)
        self.d2a_B_slider.setValue(0)
        self.d2a_B_slider.setTracking(1)
        self.d2a_B_slider.setFocusPolicy(Qt.StrongFocus)
        self.row_layout.addWidget(self.d2a_B_slider,0,8)
        
        self.data_packet = QtGui.QComboBox()
        self.data_packet.setFixedHeight(25)
        self.data_packet.addItem('FBA, ERR')
        self.data_packet.addItem('FBB, ERR')
        self.data_packet.addItem('FBB, FBA')            # changed label order to reflect data data packing 8mar16
        self.data_packet.addItem('test pattern')        
        self.row_layout.addWidget(self.data_packet,0,9)

        self.P_spin = QSpinBox()
        self.P_spin.setRange(-511, 511)
        self.P_spin.setFixedWidth(60)
        self.P_spin.setFixedHeight(25)
        self.P_spin.setSingleStep(8)
        self.P_spin.setKeyboardTracking(0)
        self.P_spin.setFocusPolicy(Qt.StrongFocus)
        self.row_layout.addWidget(self.P_spin,0,10)

        self.I_spin = QSpinBox()
        self.I_spin.setRange(-511, 511)
        self.I_spin.setFixedWidth(60)
        self.I_spin.setFixedHeight(25)
        self.I_spin.setSingleStep(8)
        self.I_spin.setKeyboardTracking(0)
        self.I_spin.setFocusPolicy(Qt.StrongFocus)
        self.row_layout.addWidget(self.I_spin,0,11)

        self.FBA_button = QToolButton(self, text = 'FB[A]')
        self.FBA_button.setFixedHeight(25)
#         self.FBA_button.setFixedWidth(25)
        self.FBA_button.setCheckable(1)
        self.FBA_button.setChecked(self.triA)
        self.FBA_button.setStyleSheet("background-color: #" + self.red + ";")
        self.row_layout.addWidget(self.FBA_button,0,12)

        self.FBB_button = QToolButton(self, text = 'FB[B]')
        self.FBB_button.setFixedHeight(25)
#         self.FBB_button.setFixedWidth(25)
        self.FBB_button.setCheckable(1)
        self.FBB_button.setChecked(self.triB)
        self.FBB_button.setStyleSheet("background-color: #" + self.red + ";")
        self.row_layout.addWidget(self.FBB_button,0,13)
        
        self.ARL_button = QToolButton(self, text = 'ARL')
        self.ARL_button.setFixedHeight(25)
#         self.FBB_button.setFixedWidth(25)
        self.ARL_button.setCheckable(1)
        self.ARL_button.setChecked(self.triB)
        self.ARL_button.setStyleSheet("background-color: #" + self.red + ";")
        self.row_layout.addWidget(self.ARL_button,0,14)
        
        self.chn_send = QtGui.QPushButton(self, text = "send channel")
        self.chn_send.setFixedHeight(25)
#         self.chn_send.setFixedWidth(160)
        self.row_layout.addWidget(self.chn_send,0,15)

        self.lock_button = QToolButton(self, text = 'dynamic')
#         self.lock_button.setMenu(self.mode_menu)
        self.lock_button.setFixedWidth(50)
        self.lock_button.setFixedHeight(25)
        self.lock_button.setCheckable(1)
        self.lock_button.setChecked(1)
        self.lock_button.setStyleSheet("background-color: #" + self.green + ";")
        self.row_layout.addWidget(self.lock_button,0,16)
        
        if master != None:
            self.chn_lbl = QtGui.QLabel("state")
            self.chn_lbl.setToolTip("DAC channel number (corresponds to front panel)")
            self.chn_lbl.setFrameStyle(50)
            self.chn_lbl.setFrameStyle(50)
            self.chn_lbl.setAlignment(QtCore.Qt.AlignCenter)
            self.chn_lbl.setStyleSheet("background-color: #6A6A6A; color: #EFEFEF")
            self.row_layout.addWidget(self.chn_lbl,1,0)
            
            self.bool_lbl = QtGui.QLabel("triangle")
            self.bool_lbl.setToolTip("select DC mode, HI/LO level for DC mode, and/or TRI output")
    #         self.bool_lbl.setFixedWidth(85)
            self.bool_lbl.setFrameStyle(50)
            self.bool_lbl.setAlignment(QtCore.Qt.AlignCenter)
            self.bool_lbl.setStyleSheet("background-color: #6A6A6A; color: #EFEFEF")
            self.row_layout.addWidget(self.bool_lbl,1,1,1,2)

            self.DAClo_lbl = QtGui.QLabel("ADC lock point")
            self.DAClo_lbl.setFixedHeight(25)
            self.DAClo_lbl.setFrameStyle(50)
            self.DAClo_lbl.setAlignment(QtCore.Qt.AlignCenter)
            self.DAClo_lbl.setStyleSheet("background-color: #6A6A6A; color: #EFEFEF")
            self.row_layout.addWidget(self.DAClo_lbl,1,3,1,2)
    
            self.DAChi_lbl = QtGui.QLabel("DAC A offset")
            self.DAChi_lbl.setFixedHeight(25)
            self.DAChi_lbl.setFrameStyle(50)
            self.DAChi_lbl.setAlignment(QtCore.Qt.AlignCenter)
            self.DAChi_lbl.setStyleSheet("background-color: #6A6A6A; color: #EFEFEF")
            self.row_layout.addWidget(self.DAChi_lbl,1,5,1,2)
    
            self.command_lbl = QtGui.QLabel("DAC B offset")
            self.command_lbl.setFixedHeight(25)
            self.command_lbl.setFrameStyle(50)
            self.command_lbl.setAlignment(QtCore.Qt.AlignCenter)
            self.command_lbl.setStyleSheet("background-color: #6A6A6A; color: #EFEFEF")
            self.row_layout.addWidget(self.command_lbl,1,7,1,2)
    
            self.mode_lbl = QtGui.QLabel("send mode")
            self.mode_lbl.setFixedHeight(25)
            self.mode_lbl.setFrameStyle(50)
            self.mode_lbl.setAlignment(QtCore.Qt.AlignCenter)
            self.mode_lbl.setStyleSheet("background-color: #6A6A6A; color: #EFEFEF")
            self.row_layout.addWidget(self.mode_lbl,1,9,1,1)

            self.P_lbl = QtGui.QLabel("P")
            self.P_lbl.setFixedHeight(25)
            self.P_lbl.setFrameStyle(50)
            self.P_lbl.setAlignment(QtCore.Qt.AlignCenter)
            self.P_lbl.setStyleSheet("background-color: #6A6A6A; color: #EFEFEF")
            self.row_layout.addWidget(self.P_lbl,1,10,1,1)

            self.I_lbl = QtGui.QLabel("I")
            self.I_lbl.setFixedHeight(25)
            self.I_lbl.setFrameStyle(50)
            self.I_lbl.setAlignment(QtCore.Qt.AlignCenter)
            self.I_lbl.setStyleSheet("background-color: #6A6A6A; color: #EFEFEF")
            self.row_layout.addWidget(self.I_lbl,1,11,1,1)

            self.FB_lbl = QtGui.QLabel("feedback")
            self.FB_lbl.setFixedHeight(25)
            self.FB_lbl.setFrameStyle(50)
            self.FB_lbl.setAlignment(QtCore.Qt.AlignCenter)
            self.FB_lbl.setStyleSheet("background-color: #6A6A6A; color: #EFEFEF")
            self.row_layout.addWidget(self.FB_lbl,1,12,1,3)

            self.command_lbl = QtGui.QLabel("command")
            self.command_lbl.setFixedHeight(25)
            self.command_lbl.setFrameStyle(50)
            self.command_lbl.setAlignment(QtCore.Qt.AlignCenter)
            self.command_lbl.setStyleSheet("background-color: #6A6A6A; color: #EFEFEF")
            self.row_layout.addWidget(self.command_lbl,1,15,1,1)

            self.mode_lbl = QtGui.QLabel("track")
            self.mode_lbl.setFixedHeight(25)
            self.mode_lbl.setFrameStyle(50)
            self.mode_lbl.setAlignment(QtCore.Qt.AlignCenter)
            self.mode_lbl.setStyleSheet("background-color: #6A6A6A; color: #EFEFEF")
            self.row_layout.addWidget(self.mode_lbl,1,16,1,1)

        '''
        call self routines
        '''
        if master == None:
            self.TriA_button.toggled.connect(self.triA_changed)
            self.TriB_button.toggled.connect(self.triB_changed)
            self.a2d_lockpt_spin.valueChanged.connect(self.a2d_lockpt_spin_changed)
            self.a2d_lockpt_slider.valueChanged.connect(self.a2d_lockpt_slider_changed)
            self.d2a_A_spin.valueChanged.connect(self.d2a_A_spin_changed)
            self.d2a_A_slider.valueChanged.connect(self.d2a_A_slider_changed)
            self.d2a_B_spin.valueChanged.connect(self.d2a_B_spin_changed)
            self.d2a_B_slider.valueChanged.connect(self.d2a_B_slider_changed)
            self.data_packet.currentIndexChanged.connect(self.data_packet_changed)
            self.P_spin.valueChanged.connect(self.P_spin_changed)
            self.I_spin.valueChanged.connect(self.I_spin_changed)
            self.FBA_button.toggled.connect(self.FBA_changed)
            self.FBB_button.toggled.connect(self.FBB_changed)
            self.ARL_button.toggled.connect(self.ARL_changed)
            self.chn_send.clicked.connect(self.send_channel)
            self.lock_button.toggled.connect(self.lock_channel)

#             self.a2d_lockpt_slider.mouseDoubleClickEvent()

        
        '''
        call parent routines
        '''    
        if master != None:
            self.TriA_button.toggled.connect(parent.triA_changed, self.TriA_button.isChecked())
            self.TriB_button.toggled.connect(parent.triB_changed, self.TriB_button.isChecked())
            self.a2d_lockpt_spin.valueChanged.connect(parent.a2d_lockpt_spin_changed, self.a2d_lockpt_spin.value())
            self.a2d_lockpt_slider.valueChanged.connect(parent.a2d_lockpt_slider_changed, self.a2d_lockpt_slider.value())
            self.d2a_A_spin.valueChanged.connect(parent.d2a_A_spin_changed, self.d2a_A_spin.value())
            self.d2a_A_slider.valueChanged.connect(parent.d2a_A_slider_changed, self.d2a_A_slider.value())
            self.d2a_B_spin.valueChanged.connect(parent.d2a_B_spin_changed, self.d2a_B_spin.value())
            self.d2a_B_slider.valueChanged.connect(parent.d2a_B_slider_changed, self.d2a_B_slider.value())
            self.data_packet.currentIndexChanged.connect(parent.data_packet_changed, self.data_packet.currentIndex())
            self.P_spin.valueChanged.connect(parent.P_spin_changed, self.P_spin.value())
            self.I_spin.valueChanged.connect(parent.I_spin_changed, self.I_spin.value())
            self.FBA_button.toggled.connect(parent.FBA_changed, self.FBA_button.isChecked())
            self.FBB_button.toggled.connect(parent.FBB_changed, self.FBB_button.isChecked())
            self.ARL_button.toggled.connect(parent.ARL_changed, self.ARL_button.isChecked())
            self.chn_send.clicked.connect(parent.send_channel)
            self.lock_button.toggled.connect(parent.lock_channel, self.lock_button.isChecked())
        
        if self.parent != None:
            self.layout.addWidget(self) 
        
        if parent == None:       
            self.show()
#             print self.width()
        
    def triA_changed(self):
        self.triA = self.TriA_button.isChecked()
        if self.triA ==1:
            self.TriA_button.setStyleSheet("background-color: #" + self.green + ";")
        else:
            self.TriA_button.setStyleSheet("background-color: #" + self.red + ";")  
        self.send_wreg0()
        self.send_wreg2()
            
    def triB_changed(self):
        self.triB = self.TriB_button.isChecked()
        if self.triB ==1:
            self.TriB_button.setStyleSheet("background-color: #" + self.green + ";")
        else:
            self.TriB_button.setStyleSheet("background-color: #" + self.red + ";")  
        self.send_wreg0()
        self.send_wreg2()
            
    def a2d_lockpt_spin_changed(self):
        self.a2d_lockpt = self.a2d_lockpt_spin.value()
        self.a2d_lockpt_slider.setValue(self.a2d_lockpt_spin.value())
        if self.unlocked == 1:
            self.send_wreg0()
            self.send_wreg1()

    def a2d_lockpt_sense(self):
        print("Double Click event")
        
    def a2d_lockpt_slider_changed(self):
        self.a2d_lockpt_spin.setValue(self.a2d_lockpt_slider.value())
        
    def d2a_A_spin_changed(self):
        self.d2a_A = self.d2a_A_spin.value()
        self.d2a_A_slider.setValue(self.d2a_A_spin.value())
        if self.unlocked == 1:
            self.send_wreg0()
            self.send_wreg2()
        
    def d2a_A_slider_changed(self):
        self.d2a_A_spin.setValue(self.d2a_A_slider.value())
        
#     def d2a_lo_setMin(self):
#         self.d2a_lo_slider.setValue(0)
#         
#     def d2a_lo_setMax(self):
#         self.d2a_lo_slider.setValue(16383)

    def d2a_B_spin_changed(self):
        self.d2a_B = self.d2a_B_spin.value()
        self.d2a_B_slider.setValue(self.d2a_B_spin.value())
        if self.unlocked == 1:
            self.send_wreg0()
            self.send_wreg5()
        
    def d2a_B_slider_changed(self):
        self.d2a_B_spin.setValue(self.d2a_B_slider.value())
        
    def data_packet_changed(self):
        self.SM = self.data_packet.currentIndex()
        if self.unlocked ==1:
            self.send_wreg0()
            self.send_wreg5()
            
    def P_spin_changed(self):
        self.P = self.P_spin.value()
        if self.unlocked == 1:
            self.send_wreg0()
            self.send_wreg3()
        
    def I_spin_changed(self):
        self.I = self.I_spin.value()
        if self.unlocked == 1:
            self.send_wreg0()
            self.send_wreg3()
            
    def FBA_changed(self):
        self.FBA = self.FBA_button.isChecked()
        if self.FBA == 1:
            self.FBA_button.setStyleSheet("background-color: #" + self.green + ";")
            self.FBB_button.setChecked(0)
        else:
            self.FBA_button.setStyleSheet("background-color: #" + self.red + ";")  
        self.send_wreg0()
        self.send_wreg3()
            
    def FBB_changed(self):
        self.FBB = self.FBB_button.isChecked()
        if self.FBB == 1:
            self.FBB_button.setStyleSheet("background-color: #" + self.green + ";")
            self.FBA_button.setChecked(0)
        else:
            self.FBB_button.setStyleSheet("background-color: #" + self.red + ";")  
        self.send_wreg0()
        self.send_wreg3()
            
    def ARL_changed(self):
        self.ARL = self.ARL_button.isChecked()
        if self.ARL == 1:
            self.ARL_button.setStyleSheet("background-color: #" + self.green + ";")
        else:
            self.ARL_button.setStyleSheet("background-color: #" + self.red + ";")  
        self.send_wreg0()
        self.send_wreg3()
        
#     def d2a_hi_setMin(self):
#         self.d2a_A_slider.setValue(0)
#         
#     def d2a_hi_setMax(self):
#         self.d2a_A_slider.setValue(16383)
        
    def send_channel(self):
        print(self.FCTCALL + "send DFB STATE parameters", self.state, ": index & arrayed register values", self.ENDC)
        self.send_wreg0()
        self.send_wreg1()
        self.send_wreg2()
        self.send_wreg3()
        self.send_wreg5()
        print()

    def lock_channel(self):
        self.unlocked = self.lock_button.isChecked()
        if self.unlocked == 1:
            self.lock_button.setStyleSheet("background-color: #" + self.green + ";")
            self.lock_button.setText('dynamic')
        else:
            self.lock_button.setStyleSheet("background-color: #" + self.red + ";")            
            self.lock_button.setText('static')
    
    def send_wreg0(self):
        print("DFB:WREG0: page index (GPI/channel/state)")
        wreg = 0 << 25
        wregval = wreg | (self.chn << 6) | self.state
        self.sendReg(wregval)
        print()

    def send_wreg1(self):
        print("DFB:WREG1: arrayed state variables: ADC lock point")
        wreg = 1 << 25
        wregval = wreg | self.a2d_lockpt
        self.sendReg(wregval)
        print()

    def send_wreg2(self):
        print("DFB:WREG2: arrayed state variables: triangle booleans & DAC offset A")
        wreg = 2 << 25
        wregval = wreg | (self.triA << 16) | (self.triB << 17) | self.d2a_A
        self.sendReg(wregval)
        print()

    def send_wreg3(self):
        print("DFB:WREG3: arrayed state feedback parameters: tri, arl, P, I")
        wreg = 3 << 25
        wreg = wreg | (int(self.FBA) << 24)
        wreg = wreg | (int(self.FBB) << 23)
        wreg = wreg | (int(self.ARL) << 21)
        wreg = wreg | ((int(self.P)&0x3ff) << 10)
        wregval = wreg | (int(self.I)&0x3ff)
        self.sendReg(wregval)
        print()

    def send_wreg5(self):
        print()
        print("DFB:WREG5: DAC offset B & send mode")
        wreg = 5 << 25
        wregval = wreg | (self.d2a_B << 11) | self.SM
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
        time.sleep(0.001)
        
    def packState(self):
        self.stateVector    =    {
            'triA'          :    self.TriA_button.isChecked(),
            'triB'          :    self.TriB_button.isChecked(),
            'a2d_lockpt'    :    self.a2d_lockpt_spin.value(),
            'd2a_A'         :    self.d2a_A_spin.value(),
            'd2a_B'         :    self.d2a_B_spin.value(),
            'SM'            :    self.data_packet.currentIndex(),
            'P'             :    self.P_spin.value(),
            'I'             :    self.I_spin.value(),
            'FBA'           :    self.FBA_button.isChecked(),
            'FBB'           :    self.FBB_button.isChecked(),
            'ARL'           :    self.ARL_button.isChecked()
                                }
        
    def unpackState(self, loadState):
            self.TriA_button.setChecked(loadState['triA'])
            self.TriB_button.setChecked(loadState['triB'])
            self.a2d_lockpt_spin.setValue(loadState['a2d_lockpt'])
            self.d2a_A_spin.setValue(loadState['d2a_A'])
            self.d2a_B_spin.setValue(loadState['d2a_B'])
            self.data_packet.setCurrentIndex(loadState['SM'])
            self.P_spin.setValue(loadState['P'])
            self.I_spin.setValue(loadState['I'])
            self.FBA_button.setChecked(loadState['FBA'])
            self.FBB_button.setChecked(loadState['FBB'])
            self.ARL_button.setChecked(loadState['ARL'])

def main():
     
    app = QtGui.QApplication(sys.argv)
    app.setStyle("plastique")
    app.setStyleSheet("""    QPushbutton{font: 10px; padding: 6px}
                            QToolButton{font: 10px; padding: 6px}""")
    ex = dfbChn()
    sys.exit(app.exec_())
 
if __name__ == '__main__':
    main()
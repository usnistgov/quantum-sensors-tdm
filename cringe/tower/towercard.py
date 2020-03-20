import sys
import time
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt, SIGNAL
from PyQt4.QtGui import QWidget, QDoubleSpinBox, QSpinBox, QFrame, QGroupBox,QToolButton, QPushButton, QSlider, QMenu

# import named_serial
import struct
from . import towerchannel

class TowerCard(QtGui.QWidget):
    
    def __init__(self, parent=None, cardaddr=3, serialport="tower", shockvalue=65535,name="default"):

        super(type(self), self).__init__(parent)

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
        self.layout = QtGui.QGridLayout(self)
        
        self.address = cardaddr
        self.serialport = serialport
        self.name=name

        l=QtGui.QLabel(self.name)
        l.setFixedWidth(60)
        self.layout.addWidget(l,0,1,1,1)
        l=QtGui.QLabel("%i"%self.address)
        l.setFixedWidth(60)
        self.layout.addWidget(l,0,0,1,1)
        
        self.shocksleepseconds = 0.1




        self.towerchannels=[]
        for i in range(8):
            tc = towerchannel.TowerChannel(parent=None, chn=i, cardaddr=self.address, serialport=serialport, shockvalue=shockvalue)
            self.layout.addWidget(tc,0,i+2,1,1)
            self.towerchannels.append(tc)

        self.allcontrolchannel = towerchannel.TowerChannel(parent=None, chn=-1, cardaddr=self.address, serialport=serialport,dummy=True)
        self.layout.addWidget(self.allcontrolchannel,0,10,1,1)

        self.shockbutton = QPushButton("shock")
        self.layout.addWidget(self.shockbutton,0,11,1,1)
        self.shockbutton.clicked.connect(self.shock)
        self.shockbutton.pressed.connect(self.gored)
        self.shockbutton.released.connect(self.gowhite)


        self.allcontrolchannel.dacspin.valueChanged.connect(self.allcontrolchannel_dacspin_changed)
        
        if parent == None:       
            self.show()
            print(self.width())
        
            
    def allcontrolchannel_dacspin_changed(self):
        value = self.allcontrolchannel.dacspin.value()
        for tc in self.towerchannels:
            tc.dacspin.setValue(value)

    def shock(self):
        oldvalues = [tc.dacspin.value() for tc in self.towerchannels]
        for tc in self.towerchannels:
            tc.dacspin.setValue(tc.shockvalue)
        print(("sleeping for %0.2f seconds"%self.shocksleepseconds))
        time.sleep(self.shocksleepseconds)
        for (tc, oldvalue) in zip(self.towerchannels, oldvalues):
            tc.dacspin.setValue(oldvalue)


    def gored(self):
        self.shockbutton.setStyleSheet("background-color: #" + self.red + ";")
        for tc in self.towerchannels:
            tc.dacspin.setStyleSheet("background-color: #" + self.red + ";")

    def gowhite(self):
        self.shockbutton.setStyleSheet("background-color: #" + self.white + ";")
        for tc in self.towerchannels:
            tc.dacspin.setStyleSheet("background-color: #" + self.white + ";")

    def packState(self):
        self.stateVector    =    {
            'dacvalue'          :    self.TriA_button.isChecked(),
            'addr'          :    self.TriB_button.isChecked(),
            'chn'    :    self.a2d_lockpt_spin.value(),
            'serialport'         :    self.d2a_A_spin.value()
            }
        
    def unpackState(self, loadState):
            self.TriA_button.setChecked(loadState['triA'])

    

def main():
     
    app = QtGui.QApplication(sys.argv)
    app.setStyle("plastique")
    app.setStyleSheet("""    QPushbutton{font: 10px; padding: 6px}
                            QToolButton{font: 10px; padding: 6px}""")
    ex = TowerCard()
    sys.exit(app.exec_())
 
if __name__ == '__main__':
    main()

import sys
import time
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import struct
from . import towerchannel
from cringe.shared import terminal_colors as tc
import logging


class TowerCard(QWidget):
    
    def __init__(self, parent=None, cardaddr=3, serialport="tower", shockvalue=65535,name="default"):

        super(type(self), self).__init__(parent)
        
        self.parent = parent
        self.layout = QGridLayout(self)
        
        self.address = cardaddr
        self.serialport = serialport
        self.name=name

        l=QLabel(self.name)
        l.setFixedWidth(60)
        self.layout.addWidget(l,0,1,1,1)
        l=QLabel("%i"%self.address)
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
            logging.debug(self.width())
        
            
    def allcontrolchannel_dacspin_changed(self):
        value = self.allcontrolchannel.dacspin.value()
        for tc in self.towerchannels:
            tc.dacspin.setValue(value)

    def shock(self):
        oldvalues = [tc.dacspin.value() for tc in self.towerchannels]
        for tc in self.towerchannels:
            tc.dacspin.setValue(tc.shockvalue)
        logging.debug(("sleeping for %0.2f seconds"%self.shocksleepseconds))
        time.sleep(self.shocksleepseconds)
        for (tc, oldvalue) in zip(self.towerchannels, oldvalues):
            tc.dacspin.setValue(oldvalue)


    def gored(self):
        self.shockbutton.setStyleSheet("background-color: #" + tc.red + ";")
        for tc in self.towerchannels:
            tc.dacspin.setStyleSheet("background-color: #" + tc.red + ";")

    def gowhite(self):
        self.shockbutton.setStyleSheet("background-color: #" + tc.white + ";")
        for tc in self.towerchannels:
            tc.dacspin.setStyleSheet("background-color: #" + tc.white + ";")

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
     
    app = QApplication(sys.argv)
    app.setStyle("plastique")
    app.setStyleSheet("""    QPushbutton{font: 10px; padding: 6px}
                            QToolButton{font: 10px; padding: 6px}""")
    ex = TowerCard()
    sys.exit(app.exec_())
 
if __name__ == '__main__':
    main()

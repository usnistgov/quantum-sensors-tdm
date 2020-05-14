import sys
import time
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

# import named_serial
import struct
from instruments import bluebox
from cringe.shared import terminal_colors as tc
from cringe.shared import log

class TowerChannel(QWidget):
    
    def __init__(self, parent=None, chn=0, cardaddr=3, serialport="tower",dummy=False, shockvalue=65535):

        super(type(self), self).__init__(parent)
        
        self.parent = parent
        self.layout = QHBoxLayout(self)
        
        self.address = cardaddr
        self.chn = chn
        self.serialport = serialport
        self.shockvalue=65535

        self.bluebox = bluebox.BlueBox(port=serialport, version='tower', address=self.address, channel=self.chn, shared=True)

        self.saveState = {}
                
        self.dacspin = QSpinBox()
        self.dacspin.setRange(0, 65535)
        self.dacspin.setFixedWidth(70)
        self.dacspin.setFixedHeight(25)
        self.dacspin.setSingleStep(250)
        self.dacspin.setKeyboardTracking(0)
        self.dacspin.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.layout.addWidget(self.dacspin)
        
        self.dummy=dummy
        if self.dummy:
            self.dacspin.setStyleSheet("background-color: #" + tc.yellow + ";")
        else:
            self.dacspin.valueChanged.connect(self.dacspin_changed)

        
        if parent == None:       
            self.show()
            log.debug(self.width())
        
            
    def dacspin_changed(self):
        value = self.dacspin.value()
        self.sendvalue(value)

    def sendvalue(self,dacvalue):
        log.debug(("towerchannel sending %g to addr %g, chn %g"%(dacvalue,self.address,self.chn)))
        self.bluebox.setVoltDACUnits(dacvalue)


    @property 
    def dacvalue(self):
        self.dacspin.value()

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
    ex = TowerChannel()
    sys.exit(app.exec_())
 
if __name__ == '__main__':
    main()

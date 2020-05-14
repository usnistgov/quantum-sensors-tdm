import sys
import time
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import struct
from . import towercard
from cringe.shared import terminal_colors as tc
from cringe.shared import log


class LabelWidget(QWidget):
    def __init__(self, parent=None):
        super(type(self), self).__init__(parent)
        self.layout=QGridLayout(self)


class TowerWidget(QWidget):

    def __init__(self, parent=None, nameaddrlist=["DB1", "13", "SAb", "4", "SQ1b", "12"], serialport="tower", shockvalue=65535):

        super(type(self), self).__init__()

        self.parent = parent
        self.layout = QVBoxLayout(self)


        label = LabelWidget()
        self.layout.addWidget(label)
        self.towercards = {}
        for i in range(len(nameaddrlist)//2):
            name = nameaddrlist[2*i]
            addr = int(nameaddrlist[2*i+1])
            tc=towercard.TowerCard(parent=self, name=name,cardaddr=addr, serialport=serialport, shockvalue=shockvalue)
            self.towercards[name]=tc
            self.layout.addWidget(tc)


        for i,s in enumerate(["addr","name","CX","CY","DX","DY","AX","AY","BX","BY","all chn", "shock"]):
            l = QLabel(s)
            label.layout.addWidget(l,0,i,1,1)

        sendallbutton = QPushButton("send all tower")
        sendallbutton.clicked.connect(self.sendall)
        self.layout.addWidget(sendallbutton)

        launchtowerpowersupplyguibutton = QPushButton("Power Supply GUI")
        launchtowerpowersupplyguibutton.clicked.connect(self.launchtowerpowersupplygui)
        self.layout.addWidget(launchtowerpowersupplyguibutton)




        if parent == None:
            self.show()
            #print self.width()

    def sendall(self):
        for key,tc in self.towercards.items():
            for tchn in tc.towerchannels:
                value = tchn.dacspin.value()
                tchn.sendvalue(value)

    def packState(self):
        dacvalues = []
        for key,tc in self.towercards.items():
            for tchn in tc.towerchannels:
                dacvalues.append(tchn.dacspin.value())
        self.stateVector    =    {
            'dacvalues'          :dacvalues}
        return self.stateVector

    def unpackState(self, loadState):
        dacvalues = loadState["dacvalues"][:]
        log.debug(dacvalues)
        if len(dacvalues) != len(self.towercards)*8:
            # silentley ignore saved values if there are the wrong number
            # used to allow changiing the tower setup from the command line
            log.debug("wrong number of dacvalues for towerwidget.unpackState")
            return
        for key,tc in self.towercards.items():
            for tchn in tc.towerchannels:
                dacvalues.append(tchn.dacspin.setValue(dacvalues.pop(0)))

    def launchtowerpowersupplygui(self):
        from subprocess import Popen
        Popen(["tower_power_gui"])


def main():

    app = QApplication(sys.argv)
    app.setStyle("plastique")
    app.setStyleSheet("""    QPushbutton{font: 10px; padding: 6px}
                            QToolButton{font: 10px; padding: 6px}""")
    ex = TowerWidget()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

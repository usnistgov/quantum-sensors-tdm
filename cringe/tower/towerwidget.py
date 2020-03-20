import sys
import time
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt, SIGNAL
from PyQt4.QtGui import QWidget, QDoubleSpinBox, QSpinBox, QFrame, QGroupBox,QToolButton, QPushButton, QSlider, QMenu

# import named_serial
import struct
from . import towercard

class LabelWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(type(self), self).__init__(parent)
        self.layout=QtGui.QGridLayout(self)

class TowerWidget(QtGui.QWidget):

    def __init__(self, parent=None, nameaddrlist=["DB1", "13", "SAb", "4", "SQ1b", "12"], serialport="tower", shockvalue=65535):

        super(type(self), self).__init__()

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
        self.layout = QtGui.QVBoxLayout(self)


        label = LabelWidget()
        self.layout.addWidget(label)
        self.towercards = {}
        for i in range(len(nameaddrlist)/2):
            name = nameaddrlist[2*i]
            addr = int(nameaddrlist[2*i+1])
            tc=towercard.TowerCard(parent=self, name=name,cardaddr=addr, serialport=serialport, shockvalue=shockvalue)
            self.towercards[name]=tc
            self.layout.addWidget(tc)


        for i,s in enumerate(["addr","name","CX","CY","DX","DY","AX","AY","BX","BY","all chn", "shock"]):
            l = QtGui.QLabel(s)
            label.layout.addWidget(l,0,i,1,1)

        sendallbutton = QtGui.QPushButton("send all tower")
        sendallbutton.clicked.connect(self.sendall)
        self.layout.addWidget(sendallbutton)

        launchtowerpowersupplyguibutton = QtGui.QPushButton("Power Supply GUI")
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
        print(dacvalues)
        if len(dacvalues) != len(self.towercards)*8:
            # silentley ignore saved values if there are the wrong number
            # used to allow changiing the tower setup from the command line
            print("wrong number of dacvalues for towerwidget.unpackState")
            return
        for key,tc in self.towercards.items():
            for tchn in tc.towerchannels:
                dacvalues.append(tchn.dacspin.setValue(dacvalues.pop(0)))

    def launchtowerpowersupplygui(self):
        from subprocess import Popen
        Popen(["python","/home/pcuser/nist_qsp_readout/instruments/tower_power_supply_gui.py"])


def main():

    app = QtGui.QApplication(sys.argv)
    app.setStyle("plastique")
    app.setStyleSheet("""    QPushbutton{font: 10px; padding: 6px}
                            QToolButton{font: 10px; padding: 6px}""")
    ex = TowerWidget()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

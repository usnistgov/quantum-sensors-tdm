#cd ~/gitrepo/nist_lab_internals/viper/cringe/tower/
import sys
import time
from PyQt5 import QtGui, QtCore
from PyQt4.QtCore import Qt, SIGNAL
from PyQt4.QtGui import QWidget, QDoubleSpinBox, QSpinBox, QFrame, QGroupBox,QToolButton, QPushButton, QSlider, QMenu

# import named_serial
import struct
import towerwidget

app = QApplication(sys.argv)
app.setStyle("plastique")
app.setStyleSheet("""    QPushbutton{font: 10px; padding: 6px}
                        QToolButton{font: 10px; padding: 6px}""")

widget = towerwidget.TowerWidget(nameaddrlist=['SAB1', '1', 'SAFB', '15', 'SQ1FB', '11'])

widget.towercards['SAB1'].towerchannels[6].dacspin.setValue(500)
widget.towercards['SAFB'].towerchannels[6].dacspin.setValue(500)
widget.towercards['SQ1FB'].towerchannels[0].dacspin.setValue(500)

'''
Created on July 20, 2011

@author: schimaf

Versions:

1.0.2   10/16/2012   Check if the power supplies are powered in the GUI. Remove unused imports.

'''

import sys
from PyQt4.QtCore import SIGNAL, Qt
from PyQt4.QtGui import QWidget, QMainWindow,  QVBoxLayout, QPushButton, QHBoxLayout, QApplication, \
                        QLabel, QIcon, QPixmap

import tower_power_supplies

class MainWindow(QMainWindow):
    def __init__(self, app):
        ''' Constructor '''
        super(MainWindow,self).__init__()

        self.app = app
        self.power_on = False
        self.power_state_string = "Power is OFF"
        pixmap = QPixmap("towerpowericon.png")
        self.setWindowIcon(QIcon(pixmap))

        self.version = "1.0.2"

        self.setWindowTitle("Tower Power Supply GUI %s" % self.version)
        self.setGeometry(100, 100, 360, 100)

        self.power_supplies = tower_power_supplies.TowerPowerSupplies()

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout_widget = QWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.top_text_label = QLabel("Tower Power Supply Control", self.central_widget)
        psa_text = "Power Supply A: %s %s (pad=%s)" % \
            (self.power_supplies.power_supply_1.manufacturer,
             self.power_supplies.power_supply_1.model_number,
             str(self.power_supplies.power_supply_1.pad))
        self.psa_label = QLabel(psa_text, self.central_widget)
        psb_text = "Power Supply B: %s %s (pad=%s)" % \
            (self.power_supplies.power_supply_2.manufacturer,
             self.power_supplies.power_supply_2.model_number,
             str(self.power_supplies.power_supply_2.pad))
        self.psb_label = QLabel(psb_text, self.central_widget)
        self.power_state_label = QLabel(self.power_state_string, self.central_widget)
        self.power_on_button = QPushButton("Power ON", self.central_widget)
        self.power_off_button = QPushButton("Power OFF", self.central_widget)
        self.quit_button = QPushButton("Quit", self.central_widget)
        self.readingLabel = QLabel("no readings yet", self.central_widget)

        self.buttons_layout_widget = QWidget(self.central_widget)
        self.buttons_layout = QHBoxLayout(self.buttons_layout_widget)

        self.layout.addWidget(self.top_text_label, 0, Qt.AlignHCenter)
        self.layout.addWidget(self.psa_label)
        self.layout.addWidget(self.psb_label)
        self.layout.addWidget(self.buttons_layout_widget)
        self.layout.addWidget(self.power_state_label, 0, Qt.AlignHCenter)
        self.layout.addWidget(self.readingLabel)
        # self.readingLabel.setPixmap(pixmap) # test that the pixmap was loaded


        self.buttons_layout.addWidget(self.power_on_button)
        self.buttons_layout.addWidget(self.power_off_button)
        self.buttons_layout.addWidget(self.quit_button)

        self.connect(self.power_on_button, SIGNAL("clicked()"), self.power_on_event)
        self.connect(self.power_off_button, SIGNAL("clicked()"), self.power_off_event)
        self.connect(self.quit_button, SIGNAL("clicked()"), self.quit_event)

        # Update the state of the power supplies
        self.updatePowerOnString()

    def power_on_event(self):

        s = self.power_supplies.powerOnSequence()
        self.updatePowerOnString()
	self.readingLabel.setText("most recent readings:\n"+s)

    def power_off_event(self):

        self.power_supplies.powerOffSupplies()
        self.updatePowerOnString()

    def quit_event(self):

        self.app.quit()

    def updatePowerOnString(self):

        if self.power_supplies.powered == True:
            self.power_state_string = "Power is ON"
        else:
            self.power_state_string = "Power is OFF"
        self.power_state_label.setText(self.power_state_string)

def main(args):
    app = QApplication(args)
    win = MainWindow(app)
    win.show()
    win.setWindowIcon(QIcon("towerpowericon.png"))
    app.exec_()

if __name__=="__main__":
    main(sys.argv)

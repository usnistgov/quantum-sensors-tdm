#-*- coding: utf-8 -*-
import sys
import optparse
import struct
import time

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import nasa_client

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import numpy as np
def find(condition):
    res, = np.nonzero(np.ravel(condition))
    return res

class TuneClient(QWidget):
    def __init__(self, parent):
        super(type(self), self).__init__(parent)
        self.layout = QHBoxLayout(self)
        self.statustext = QLabel("not connected to server")
        self.startclientbutton = QPushButton(self, text = "startclient")

        self.layout.addWidget(self.statustext)
        self.layout.addWidget(self.startclientbutton)

        self.startclientbutton.clicked.connect(self.startclient)

        self.client = nasa_client.EasyClient(clockmhz=125, setupOnInit=False)

        # timer = QtCore.QTimer()
        # timer.setSingleShot(False)
        # timer.timeout.connect(self.startclient)
        # timer.start(4000)
        # self.startclient()


    def startclient(self):
        self.statustext.setText("connecting...")
        QtCore.QCoreApplication.processEvents() # allows text change to actually happen before blocking
        try: # blocks for 1 second if server isn't there
            self.client = nasa_client.EasyClient(clockmhz=125)
            self.getNewData = self.client.getNewData
            self.client.setupAndChooseChannels()
            self.statustext.setText("connected to server, lysnc=%g, ncol=%g, nrow=%g, nsamp=%g"%(self.lsync, self.ncol, self.nrow, self.nsamp))
            return True
        except:
            self.statustext.setText("failed to connect to server")
            return False

    def plotpi(self):
        window = Window(self)
        window.show()

    def plotdacLockpoints(self):
        window = Window(self)
        window.show()
        window.plotdata(self.mm.getbaddacHighs())

    @property
    def ncol(self):
        return self.client.ncol

    @property
    def nrow(self):
        return self.client.nrow

    @property
    def lsync(self):
        return self.client.lsync

    @property
    def sample_rate(self):
        return self.client.sample_rate

    @property
    def nsamp(self):
        return self.client.num_of_samples

#-*- coding: utf-8 -*-
import sys
import optparse
import struct
import time

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt, SIGNAL,QTimer
from PyQt4.QtGui import QFileDialog, QPalette, QSpinBox, QToolButton, QVBoxLayout, QLabel

import easyClient

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import numpy as np
from pylab import find

class TuneClient(QtGui.QWidget):
    def __init__(self, parent):
        super(type(self), self).__init__(parent)
        self.layout = QtGui.QHBoxLayout(self)
        self.statustext = QtGui.QLabel("not connected to server")
        self.startclientbutton = QtGui.QPushButton(self, text = "startclient")

        self.layout.addWidget(self.statustext)
        self.layout.addWidget(self.startclientbutton)

        self.startclientbutton.clicked.connect(self.startclient)

        self.client = None

        timer = QTimer()
        timer.setSingleShot(False)
        timer.timeout.connect(self.startclient)
        timer.start(4000)
        self.startclient()


    def startclient(self):
        self.statustext.setText("connecting...")
        QtCore.QCoreApplication.processEvents() # allows text change to actually happen before blocking
        self.client = easyClient.EasyClient(clockmhz=125)
        self.getNewData = self.client.getNewData
        try: # blocks for 1 second if server isn't there
            self.client.setupAndChooseChannels()
            self.statustext.setText("connected to server, lysnc=%g, ncol=%g, nrow=%g, nsamp=%g"%(self.lsync, self.ncol, self.nrow, self.nsamp))
        except:
            self.statustext.setText("failed to connect to server")

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

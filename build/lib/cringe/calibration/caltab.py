#-*- coding: utf-8 -*-
import sys
import optparse
import struct
import time
import pickle
import os.path
import os

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import numpy as np

class CalTab(QWidget):
    def __init__(self, parent):
        super(type(self),self).__init__(parent)
        self.mm = parent.tune_widget.mm
        self.c = parent.tune_widget.c
        self.layout = QVBoxLayout(self)

        self.layout.addWidget(self.c)

        hline = QFrame()
        hline.setFrameStyle(QFrame.HLine)
        self.layout.addWidget(hline)

        self.caldemo =CalDemo(self,self.mm, parent, self.c)
        self.layout.addWidget(self.caldemo)

        hline = QFrame()
        hline.setFrameStyle(QFrame.HLine)


    def packState(self):
        stateVector = {}
        return stateVector


    def unpackState(self, loadState):
        pass


class CalDemo(QWidget):
    def __init__(self, parent, mm, cringe, client):
        super(type(self), self).__init__(parent)
        self.layout = QHBoxLayout(self)
        self.mm=mm
        self.cringe=cringe
        self.c=client

        self.layout.addWidget(QLabel("calibration"))

        layout = QHBoxLayout()
        self.button = QPushButton(self,text="increment counter")
        self.button.clicked.connect(self.increment_counter)
        layout.addWidget(self.button)

        self.button_bittest = QPushButton(self,text="print test pattern")
        self.button_bittest.clicked.connect(self.printtestpattern)
        layout.addWidget(self.button_bittest)

        self.button_correct = QPushButton(self,text="number correct")
        self.button_correct.clicked.connect(self.get_correct_number)
        layout.addWidget(self.button_correct)

        self.button_sweep = QPushButton(self,text="sweep")
        self.button_sweep.clicked.connect(self.sweepcounter)
        layout.addWidget(self.button_sweep)

        self.layout.addLayout(layout)

    def calDemoCard1Channel1(self):
        log.debug("starting demo")

    def printtestpattern(self):
        log.debug(self.parent)
        tp_mode = self.cringe.tp_mode
        log.debug((tp_mode.itemText(tp_mode.currentIndex())))
        TP = self.cringe.TP
        lobytes,hibytes=self.cringe.lohibytes()
        log.debug(hex(lobytes),hex(hibytes))

    def increment_counter(self):
        dfbcard = self.cringe.crate_widgets[1]
        counter = dfbcard.dfbx2_widget3.phase_counters[0]
        counter.phase_trim_spin.setValue(counter.phase_trim_spin.value()+1)
        counter.commit_cal()
        counter.calcounter()

    def get_correct_number(self):
        lobytes,hibytes=self.cringe.lohibytes()
        data = self.c.getNewData(divideNsamp=False, sendMode="raw")
        ncorrect = (data[0,0,:,0]==lobytes).sum()+(data[0,0,:,0]==hibytes).sum()
        log.debug((list(map(hex, data[0,0,:20,0]))))
        log.debug((list(map(hex, data[0,0,:20,1]))))
        log.debug((ncorrect/float(data.size)))
        return ncorrect

    def sweepcounter(self):
        dfbcard = self.cringe.crate_widgets[1]
        counter = dfbcard.dfbx2_widget3.phase_counters[0]
        max = counter.phase_trim_spin.maximum()
        min = counter.phase_trim_spin.minimum()
        values = list(range(min, max+1))
        ncorrect = []
        for value in values:
            counter.phase_trim_spin.setValue(value)
            ncorrect.append(self.get_correct_number())

        plots = ColPlots(self,1,1)
        plots.plot(values,np.array(ncorrect,ndmin=3))
        plots.title("counter sweep")
        plots.xlabel("counter offset")
        plots.ylabel("number number of correct 4-byte units")
        plots.show()


class ColPlots(QDialog):
    def __init__(self, parent,ncol,nrow):
        super(type(self), self).__init__(parent)
        self.ncol = ncol
        self.nrow = nrow
        self.numXSubplots = int((ncol+1)/2.0)
        self.numYSubplots = 1+int(ncol>1)
        self.figure = plt.figure(figsize=(self.numXSubplots*6, self.numYSubplots*4))
        self.canvas = FigureCanvas(self.figure)
        self.titlelabel = QLabel("")
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout = QVBoxLayout()
        layout.addWidget(self.titlelabel)
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)

        self.createaxes()
        self.plottitle()
        self.xlabel("triangle")
        self.ylabel("err")
        self.title("joint title")

    def createaxes(self):
        self.axes = []
        for col in range(self.ncol):
            ax = plt.subplot(self.numYSubplots, self.numXSubplots, col+1)
            self.axes.append(ax)

    def xlabel(self,s):
        for col in range(self.ncol):
            ax=self.axes[col]
            if col>=self.numXSubplots*(self.numYSubplots-1):
                ax.set_xlabel(s)

    def ylabel(self,s):
        for col in range(self.ncol):
            ax=self.axes[col]
            if col%self.numXSubplots==0:
                ax.set_ylabel(s)

    def plottitle(self,s=[""]*32):
        for col in range(self.ncol):
            ax=self.axes[col]
            ax.set_title(("col %g"%col)+s[col])

    def plot(self,x,y):
        for col in range(self.ncol):
            ax=self.axes[col]
            for row in range(self.nrow):
                ax.plot(x,y[col,row,:],".-")
        self.draw()

    def plotbigx(self, x, y):
        for col in range(self.ncol):
            ax=self.axes[col]
            for row in range(self.nrow):
                ax.plot(x[col,row,:],y[col,row,:])
        self.draw()

    def semilogy(self,x,y):
        for col in range(self.ncol):
            ax=self.axes[col]
            for row in range(self.nrow):
                ax.semilogy(x,y[col,row,:],".")
        self.draw()

    def semilogx(self,x,y):
        for col in range(self.ncol):
            ax=self.axes[col]
            for row in range(self.nrow):
                ax.semilogx(x,y[col,row,:])
        self.draw()


    def plotdiffx(self,x,y):
        for col in range(self.ncol):
            ax=self.axes[col]
            for row in range(self.nrow):
                ax.plot(x[col],y[col,row,:],".")
        self.draw()



    def title(self,s=""):
        self.titlelabel.setText(s)

    def xlim(self,xlims):
        for col in range(self.ncol):
            ax=self.axes[col]
            ax.set_xlim(xlims)

    def ylim(self,ylims):
        for col in range(self.ncol):
            ax=self.axes[col]
            ax.set_ylim(ylims)

    def cla(self):
        for col in range(self.ncol):
            ax=self.axes[col]
            ax.clear()

    def draw(self):
        self.canvas.draw()

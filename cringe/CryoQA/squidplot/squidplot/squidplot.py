# -*- coding: utf-8 -*-

import sys
import os.path
from argparse import ArgumentParser
import numpy as np
from PyQt4 import QtCore, QtGui
from .squidplotUI import Ui_MainWindow
from .squid_data_files import squidfile
from .quietset import quiet_set
import pyqtgraph
import signal


DESC_STRING = 'dumb little plotter'

class SquidPlot(QMainWindow):
    '''
    example modulation file gui plotter
    '''


    def __init__(self, app):
        super(SquidPlot, self).__init__()
        self.app = app
        self.ui = Ui_MainWindow()
        self.squidfile = None
        pyqtgraph.setConfigOption('background', 'w')
        pyqtgraph.setConfigOption('foreground', 'k')
        self.ui.setupUi(self)
        self.statusLabel = QLabel('File: None  Channel count: 0')
        self.basewindowtitle = self.windowTitle()
        self.ui.statusbar.insertWidget(0, self.statusLabel)
        self.uiconnect()
        self.update_file()
            
    def process_args(self):
        parser = ArgumentParser(prog='squidplot', description=DESC_STRING)
        parser.add_argument('file', nargs='?', help='Input file - .npy modulation file')
        parser.add_argument('-d', '--debug', help='print extra debugging messages', action='store_true')
#         parser.add_argument('-p', '--permissive', help='allow stupid layouts', action='store_true')
#         parser.add_argument('-n', '--new', nargs=1, help='start new reticleset (use a simple distinct name) ')

        argresult = parser.parse_args()
        self.debug =  argresult.debug
        if argresult.file:
            if not os.path.exists(argresult.file):
                parser.error('Unable to open file \"%s\".\n'%argresult.file)
            self.openfile(argresult.file)
        
    def uiconnect(self):
        self.ui.actionExit.triggered.connect(self.myquit)
        self.ui.actionOpen.triggered.connect(self.openfile)
        self.ui.columnComboBox.currentIndexChanged.connect(self.columnchanged)
        self.ui.pageComboBox.currentIndexChanged.connect(self.pagechanged)
        self.ui.channelSlider.valueChanged.connect(self.sliderchanged)
        
    def closeEvent(self, event):
        self.myquit()
        event.accept()

    def myquit(self):
        self.app.quit()

    def openfile(self, filename):
        if filename == False:
            dlg = QFileDialog.getOpenFileNameAndFilter(parent=self, caption='Open modulation file')
            filename = str(dlg[0])
            if filename == '':
                return
        try:
            self.squidfile = squidfile(filename)
        except IOError as e:
            print(e)
            QMessageBox.warning(self, 'File Error', 'File error - %s'%e)
            self.squidfile = None


        self.update_file()
        self.setWindowTitle('%s %s'%(self.basewindowtitle, self.squidfile.shortfilename))
        
    def update_file(self):
        self.ui.plotwidget.clear()
        self.plotitems = []
        if self.squidfile is None:
            self.currentcolumn = None
            self.currentpage = None
            self.currentchannel = None
            self.ui.columnGroupBox.setEnabled(False)
            self.ui.pageGroupBox.setEnabled(False)
            self.ui.filetypeLineEdit.setText('None')
            self.ui.channelGroupBox.setEnabled(False)
            self.statusLabel.setText('No file loaded')
            return
        
        self.ui.filetypeLineEdit.setText(self.squidfile.filetype)
        self.ui.columnComboBox.blockSignals(True)
        self.ui.pageComboBox.blockSignals(True)
        self.ui.channelSlider.blockSignals(True)
        
        if self.squidfile.columns:
            self.ui.columnGroupBox.setEnabled(True)
            self.ui.columnComboBox.addItems(list(map(str, list(range(self.squidfile.columns)))))
            quiet_set(self.ui.columnComboBox, 0)
            self.currentcolumn = 0
        else:
            self.ui.columnGroupBox.setEnabled(False)
            for i in range(self.ui.columnComboBox.count()):
                self.ui.columnComboBox.removeItem(0)
        
        if self.squidfile.pages:
            self.ui.pageGroupBox.setEnabled(True)
            self.ui.pageComboBox.addItems(list(map(str, list(range(self.squidfile.pages)))))
            quiet_set(self.ui.pageComboBox, 0)
            self.currentpage = 0
        else:
            self.ui.pageGroupBox.setEnabled(False)
            for i in range(self.ui.pageComboBox.count()):
                self.ui.pageComboBox.removeItem(0)
                    
        self.ui.channelGroupBox.setEnabled(True)
        self.ui.channelSlider.setMaximum(self.squidfile.channels-1)
        quiet_set(self.ui.channelSlider, 0)
        self.currentchannel = 0
#        self.sliderchanged(0) 
        self.statusLabel.setText('File: %s Channel count: %d'%(self.squidfile.shortfilename, self.squidfile.channels))
        initialdata = self.squidfile.get_data(column = self.currentcolumn, page=self.currentpage, channel=self.currentchannel)
        for i in range(self.squidfile.numcurves):
            self.plotitems.append(pyqtgraph.PlotDataItem(initialdata[i]))
            self.ui.plotwidget.addItem(self.plotitems[-1])
            
        self.ui.columnComboBox.blockSignals(False)
        self.ui.pageComboBox.blockSignals(False)
        self.ui.channelSlider.blockSignals(False)


        
    def update_plot(self):
#        print 'update_plot'
        newdata = self.squidfile.get_data(column = self.currentcolumn, page=self.currentpage, channel=self.currentchannel)
        for i in range(self.squidfile.numcurves):
            self.plotitems[i].setData(newdata[i])
    

    def columnchanged(self, index):
#        print 'column combo changed'
        self.currentcolumn = int(self.ui.columnComboBox.currentText())
        self.update_plot

    def pagechanged(self, index):
#        print 'page combo changed'
        self.currentpage = int(self.ui.pageComboBox.currentText())
        self.update_plot()
    
    def sliderchanged(self, channel):
#        print 'slider changed'
        self.ui.channelLineEdit.setText('%d'%channel)
        self.currentchannel = channel
        self.update_plot()
        
         
        
def squidplotmain():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QApplication(sys.argv)
    window=SquidPlot(app)
    window.show()
    window.process_args()
    # was
#    sys.exit(app.exec_()) # with nothing after  always got a pixmap before qdeivce error
    app.exec_()
    app.deleteLater()
    sys.exit()

if __name__ == "__main__":
    squidplotmain()
   

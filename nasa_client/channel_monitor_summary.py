#!/usr/bin/env python

"""
channel_monitor.py

An example GUI nasa_client that connects to a server and reports the latest values.
"""

import numpy
import os 
import sys
import time

import nasa_client

from lxml import etree
from PyQt4 import QtGui, QtCore #, QtNetwork

class ChannelMonitor(nasa_client.GUIClient):
    
    VERSION = "1.0.0"
    MAXROWS = 32
    MAXCOLS = 4
    DEFAULTS_DIRECTORY = sys.path[0]
    DEFAULTS_FILENAME  = "channel_monitor_defaults.xml"
    SETTINGS_VERSION = 1

    def __init__(self, host=None, port=None, title="NASA Example Client"):
        nasa_client.GUIClient.__init__(self, host=host, port=port, title=title)
        self.isStreaming = False
        self.loadDefaults()
        self.stream_channels=(self.CHANNEL_CODE_ERR_SUMMARY,
                              self.CHANNEL_CODE_FB_SUMMARY,)

    def quit_app(self):
        self.saveDefaults()
        nasa_client.GUIClient.quit_app(self)
        
    def connectToServer(self):
        self.buildStatsBox()
        
    def saveSettings(self, filename=None):
        '''
        save settings as XML. Optionally add the filename otherwise it will ask for one. 
        '''

        print("Save settings as xml.")
        
        # Pick a file to save to
        if filename == None:
            print("No save file specified.")
            return
        if (filename != []):
            if filename[-4:] == '.xml':
                savename = filename
            else:
                savename = filename + '.xml'
        
        f = open(savename, "w")

        # Create an xml object
        root = etree.Element("channel_monitor")

        root.set("settings_version", "%s" % self.SETTINGS_VERSION)

        text = str(self.hostLineEdit.text())
        child_connection_type = etree.Element("hostname")
        child_connection_type.text = text
        root.append(child_connection_type)

        text = str(self.portLineEdit.text())
        child_serial_port = etree.Element("port")
        child_serial_port.text = text
        root.append(child_serial_port)

        # Save it to a file
        f.write(etree.tostring(root, pretty_print=True))
        f.close()

    def loadSettings(self, filename=""):
        if len(filename) > 0:
            print(("Load XML settings! [%s]" % filename))
            f = None
            root = None
            try:
                f = open(filename)
                root = etree.parse(f)
            except:
                print("No default settings file found!")
                return

            if root is not None:
                # Globals
                child = root.find("hostname")
                if child is not None:
                    value = child.text
                    self.hostLineEdit.setText(value)

                child = root.find("port")
                if child is not None:
                    value = child.text
                    self.portLineEdit.setText(value)


    def loadDefaults(self):
        self.loadSettings(filename=self.DEFAULTS_DIRECTORY + os.sep + self.DEFAULTS_FILENAME)
    

    def saveDefaults(self):
        self.saveSettings(filename=self.DEFAULTS_DIRECTORY + os.sep + self.DEFAULTS_FILENAME)

    
    def buildStatsBox(self):
        self.ndata=numpy.zeros((self.nchan), dtype=numpy.int)
        self.means=numpy.zeros((self.nchan), dtype=numpy.float)
        self.errmeans=numpy.zeros((self.nchan), dtype=numpy.float)
        self.last_packet_time=0.0

        print(('There are %d channels available'%self.nchan))

        self.stats_widget = QtGui.QWidget(self.central_widget)
        print(('There are %d channels available'%self.nchan))
        self.stats_grid = QtGui.QGridLayout()
        print(('There are %d channels available'%self.nchan))

        self.stats_labels=[]
        for r in range(self.nrow):
            if r>=self.MAXROWS: break
            self.stats_labels.append([])
            for c in range(self.ncol):
                if c>=self.MAXCOLS: break
                label = QtGui.QLabel("(%d,%d)"%(c,r))
                label.setAlignment(QtCore.Qt.AlignRight)
                label.setFrameStyle(QtGui.QFrame.StyledPanel)
                self.stats_labels[-1].append(label)
                self.stats_grid.addWidget(self.stats_labels[-1][-1], r+1, c+1)
        self.stats_widget.setLayout(self.stats_grid)
        self.layout.addWidget(self.stats_widget)

        # Label the columns
        for c in range(self.ncol):
            if c>self.MAXCOLS: break
            label = QtGui.QLabel("Col %2d"%c)
            label.setAlignment(QtCore.Qt.AlignCenter)
            self.stats_grid.addWidget(label, 0, c+1)

        # Label the rows
        for r in range(self.nrow):
            if r>=self.MAXROWS: break
            label = QtGui.QLabel("Row %3d"%r)
            label.setAlignment(QtCore.Qt.AlignRight)
            self.stats_grid.addWidget(label, r+1, 0)



    def disconnectFromServer(self):
        self.stats_grid.close()
        self.stats_widget.close()

    def processPacket(self, p, h):
        chan = h['chan']
        if chan not in self.stream_channels: return
        if chan == self.CHANNEL_CODE_ERR_SUMMARY:
            for c,d in enumerate(p):
                self.errmeans[c] = d
            return
   
        assert chan == self.CHANNEL_CODE_FB_SUMMARY

        REPAINT_DELAY=0.5 # sec
        now = time.time()
        dt = now-self.last_packet_time
        if dt>REPAINT_DELAY:
            self.last_packet_time = now

        for c,d in enumerate(p):
            self.means[c] = d
            self.ndata[c] += 1

            col = c/self.nrow
            row = c%self.nrow
            if row<self.MAXROWS and col<self.MAXCOLS and dt>REPAINT_DELAY:
                text = "%6.0f/%5.0f N=%6d"%(self.errmeans[c],self.means[c],self.ndata[c])
                self.stats_labels[row][col].setText(text)

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    client = ChannelMonitor(host="localhost", title="NDFB Channel Monitor")
    client.show()
    returnval = app.exec_()
    sys.exit(returnval)

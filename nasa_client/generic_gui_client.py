#!/usr/bin/env python

"""
generic_gui_client.py

An example GUI nasa_client that connects to a server.  Use it as a building block, I guess.

Usage notes:

The idea is to make a new class that inherits from GUIClient.  If you want something to happen
upon connection, disconnection, or starting or stopping streaming, then you want to add
one of the following methods.  The generic connect/disconnect/start/stop activities will
happen first, then your callback will be called.

connectToServer
disconnectFromServer
startStreaming
stopStreaming

Here's an example:

class AwesomeClient(nasa_client.GUIClient):
    def connectToServer(self):
        print 'Now that we are connected, I have things to do.'
        ....



Second, you need to copy the chunk of code at the bottom, starting with 
if __name__=='__main__': into your own application.

Most applications will want to do something with some or all packets that arrive.  For that,
you must provide a callback processPacket(pkt, hdr).  The <pkt> is a numpy.ndarray object
containing all the samples in the packet.  The <hdr> is a dictionary containing all the
information from the packet header.
"""

import sys
import socket
from . import client as nasa_client
from PyQt4 import QtCore, QtGui

class GUIClient(nasa_client.ZMQClient, QtGui.QMainWindow):
    
    DEFAULT_TIMER_PERIOD=100 # ms
    
    def __init__(self, host=None, port=None, title="NASA Client"):
        QtGui.QDialog.__init__(self, parent=None)
        nasa_client.Client.__init__(self, host, port)
        
        self.setWindowTitle(title)

        self.central_widget = QtGui.QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout_widget = QtGui.QWidget(self.central_widget)
        self.layout = QtGui.QVBoxLayout(self.central_widget)

        self.hostLabel = QtGui.QLabel("&Server name:")
        self.portLabel = QtGui.QLabel("&Server port:")

        self.hostLineEdit = QtGui.QLineEdit(self.host, self.central_widget)
        self.portLineEdit = QtGui.QLineEdit(str(self.port), self.central_widget)
        self.hostLineEdit.textChanged.connect(self.__updateHostPort)
        self.portLineEdit.textChanged.connect(self.__updateHostPort)
        #self.connect(self.hostLineEdit, QtCore.SIGNAL("textChanged(QString)"), self.__updateHostPort)
        #self.connect(self.portLineEdit, QtCore.SIGNAL("textChanged(QString)"), self.__updateHostPort)
        self.portLineEdit.setValidator(QtGui.QIntValidator(1, 65535, self))

        self.network_layout_widget = QtGui.QWidget(self.central_widget)
        self.network_layout = QtGui.QHBoxLayout(self.network_layout_widget)
        self.network_layout.setMargin(2)
        self.network_layout.setSpacing(2)
        
        self.network_layout.addWidget(self.hostLabel)
        self.network_layout.addWidget(self.hostLineEdit)
        self.network_layout.addWidget(self.portLabel)
        self.network_layout.addWidget(self.portLineEdit)

        self.layout.addWidget(self.network_layout_widget)

        self.hostLabel.setBuddy(self.hostLineEdit)
        self.portLabel.setBuddy(self.portLineEdit)

        self.connectButton = QtGui.QPushButton("Connect", self.central_widget)
        #self.streamSwitch = QtGui.QCheckBox("Streaming", self.central_widget)
        self.streamSwitch = QtGui.QToolButton(self.central_widget)
        self.streamSwitch.setText("Streaming")
        self.streamSwitch.setCheckable(True)
        self.disconnectButton = QtGui.QPushButton("Disconnect", self.central_widget)
        self.quitButton = QtGui.QPushButton("Quit", self.central_widget)
        #self.streamSwitch.setEnabled(False)
        #self.disconnectButton.setEnabled(False)
        
        self.button_layout_widget = QtGui.QWidget(self.central_widget)
        self.button_layout = QtGui.QHBoxLayout(self.button_layout_widget)
        self.button_layout.setMargin(2)
        self.button_layout.setSpacing(2)
        
        self.button_layout.addWidget(self.connectButton)
        self.button_layout.addWidget(self.streamSwitch)
        self.button_layout.addWidget(self.disconnectButton)
        self.button_layout.addWidget(self.quitButton)

        self.layout.addWidget(self.button_layout_widget)
        
        self.extra_input_widgets_widget = QtGui.QWidget(self.central_widget)
        self.extra_input_widgets_layout = QtGui.QVBoxLayout(self.extra_input_widgets_widget)
        
        self.layout.addWidget(self.extra_input_widgets_widget)
        
        self.status_label = QtGui.QLabel(self.central_widget)
        self.layout.addWidget(self.status_label)
        self.layout.setSizeConstraint(QtGui.QLayout.SetMaximumSize)

        self._isStreaming = False
        self._isConnected = False
        self.updateStatusLabel()
        self.updateButtons()

#        self.command_popup.currentIndexChanged.connect(self.addInputWidgets)
#        self.send_command_button.clicked.connect(self.execute_command)
        self.connectButton.clicked.connect(self.__connectToServer)
        self.streamSwitch.clicked.connect(self.__startStopStreaming)
        #self.connect(self.streamSwitch, QtCore.SIGNAL('stateChanged(int)'), self.__startStopStreaming)
        self.disconnectButton.clicked.connect(self.__disconnectFromServer)
        self.quitButton.clicked.connect(self.quit_app)
        self.portLineEdit.setFocus() ##  What does this do?

        self.net_timer = QtCore.QTimer()
        self.connect(self.net_timer, QtCore.SIGNAL("timeout()"), self.__checkForData)
        self.net_timer.start(self.DEFAULT_TIMER_PERIOD)

    def updateButtons(self):
        
        if self._isConnected == True:
            # Connected
            self.connectButton.setEnabled(False)
            self.disconnectButton.setEnabled(True)
            self.streamSwitch.setEnabled(True)
        else:
            # Disconnected
            self.connectButton.setEnabled(True)
            self.disconnectButton.setEnabled(False)
            self.streamSwitch.setEnabled(False)
            self.streamSwitch.setChecked(False)

    def updateStatusLabel(self, extra_text=""):
        
        text = ""
        if self._isConnected == True:
            text +=  "Connected to %s:%s (c:%d r:%d)." % (self.host, self.port, self.ncol, self.nrow)
            if self._isStreaming == True:
                text += " Streaming %i channels."  % (len(self.stream_channels))
        else:
            text = "Disconnected."

        text += extra_text

        self.status_label.setText(text)

    def quit_app(self):
        self.__stopStreaming()
        self.__disconnectFromServer()
        QtCore.QCoreApplication.instance().quit()

    def setTimerPeriod(self, msec):
        self.net_timer.stop()
        self.net_timer.start(msec)

    def __updateHostPort(self):
        self.host = self.hostLineEdit.text()
        self.port = int(self.portLineEdit.text())
    
    def __connectToServer(self):
        #self.connectButton.setEnabled(False)
        try:
            print(("generic client connecting to server %s/%i" % (self.host, self.port)))
            self.connect_server()
            self._isConnected = True
        except socket.error as e:
            print(("Could not connect to server: error '%s'" % e))
            #self.connectButton.setEnabled(True)
            self.updateStatusLabel()
            self.updateButtons()
            return
        #self.disconnectButton.setEnabled(True)
        #self.streamSwitch.setEnabled(True)
        self.updateStatusLabel()
        self.updateButtons()
        #self.status_label.setText("Connected to %s:%s (c:%d r:%d)" % (self.host, self.port, self.ncol, self.nrow))
        self.__subclassCallback('connectToServer')

    def __disconnectFromServer(self):
        #self.status_label.setText("Disconnected")
        #self.disconnectButton.setEnabled(False)
        #self.streamSwitch.setChecked(False)
        #self.streamSwitch.setEnabled(False)
        self.disconnect_server()
        #self.connectButton.setEnabled(True)
        self._isConnected = False
        self._isStreaming = False
        self.updateStatusLabel()
        self.updateButtons()
        self.__subclassCallback('disconnectFromServer')

    def __startStopStreaming(self):
        self._isStreaming = self.streamSwitch.isChecked()
        if self._isStreaming:
            self.__startStreaming()
        else:
            self.__stopStreaming()
            
    def __startStreaming(self):
        self.start_streaming()
        #self.status_label.setText("Connected to %s:%s (c:%d r:%d) streaming %i channels" \
        #                          % (self.host, self.port, self.ncol, self.nrow, len(self.stream_channels)))
        self.updateStatusLabel()
        print('Start streaming GUI')
        self.__subclassCallback('startStreaming')
        
    def __stopStreaming(self):
        try:
            self.stop_streaming()
            self.updateStatusLabel()
            #self.status_label.setText("Connected to %s:%s (c:%d r:%d)" % (self.host, self.port, self.ncol, self.nrow))
            print('Stop streaming GUI')
        except:
            pass
        self.__subclassCallback('stopStreaming')

    def __subclassCallback(self, callbackName, *args, **kwargs):
        try:
            exec('self.%s(*args, **kwargs)'%callbackName)
        except AttributeError:
            pass

    def __checkForData(self):
        if not self._isStreaming: return 
        packets, headers = self.get_data_packets()
        if len(packets)==0: return
        for p,h in zip(packets, headers):
            self.__subclassCallback('processPacket',p,h)


        
        
# Append the following to your application *.py file (replacing GUIClient with the
# name of your subclass), and it will run.  Hooray!
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    client = GUIClient(host="localhost", title="NDFB Channel Monitor")
    client.show()
    returnval = app.exec_()
    sys.exit(returnval)

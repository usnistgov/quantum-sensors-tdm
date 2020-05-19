import zmq

from PyQt5 import QtGui, QtCore, QtWidgets, QtNetwork
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import cringe
from cringe.shared import log


class ZmqRep(QWidget):
    gotMessage = QtCore.Signal(str)
    def __init__(self, parent, address_with_port):
        print("YYOYOYOYOYOYOOYOYOYOYOYOYOYOYOYOYOYOYOYOYOYOYOYOYOYOYOYOYOYOYOYOYOYOYO")
        llog = log.child("ZmqRep: __init__:")
        log.debug("__init__")
        super(type(self), self).__init__(parent)
        self._context = zmq.Context()
        self._zmq_sock = self._context.socket(zmq.REP)
        self._zmq_sock.LINGER = 0
        llog.debug(f"bind on: {address_with_port}")
        self._zmq_sock.bind(address_with_port)
        # print(self._zmq_sock.recv())
        self._timer = QtCore.QTimer()
        self._timer.start(100)
        self._timer.timeout.connect(self.handleTimeout)
        self._i=0

    def handleTimeout(self):
        self._i+=1
        llog = log.child("ZmqRep: handleTimeout:")
        llog.debug("start", self._i)
        self.emit_if_has_message()

    def emit_if_has_message(self):
        llog = log.child("ZmqRep: emit_if_has_message:")
        llog.debug("start")
        try:
            message = self._zmq_sock.recv(zmq.NOBLOCK)
            self._zmq_sock.send_string("ok")
            self.gotMessage.emit(message.decode())
            llog.info(f"got: {message}")
        except zmq.Again:
            pass

    
import zmq

from PyQt5 import QtGui, QtCore, QtWidgets, QtNetwork
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import cringe
from cringe.shared import log


class ZmqRep(QWidget):
    _testOnlySignalOnInvalidMessage = QtCore.Signal()
    def __init__(self, parent, address_with_port):
        llog = log.child("ZmqRep: __init__:")
        log.debug("__init__")
        super(type(self), self).__init__(parent)
        self._context = zmq.Context()
        self._zmq_sock = self._context.socket(zmq.REP)
        self._zmq_sock.LINGER = 0
        llog.debug(f"bind on: {address_with_port}")
        self._zmq_sock.bind(address_with_port)
        self._timer = QtCore.QTimer()
        self._timer.start(100)
        self._timer.timeout.connect(self.handle_timeout)
        self._i=0
        self._d = {}

    def handle_timeout(self):
        self._i+=1
        llog = log.child("ZmqRep: handle_timeout:")
        llog.debug("start", self._i)
        message_or_none = self.read_socket()
        if message_or_none is None:
            return
        message = message_or_none
        llog.info(f"got: `{message}`")
        reply = self.resolve_callback(message)
        llog.info(f"send reply: `{reply}`")
        self._zmq_sock.send_string(reply)
        llog.info(f"reply sent")

    def read_socket(self):
        # llog = log.child("ZmqRep: read_socket:")
        # llog.debug("start")
        try:
            message = self._zmq_sock.recv(zmq.NOBLOCK).decode()
            # self.gotMessage.emit(message.decode(), self.complete_message)
            # llog.info(f"got: {message}")
        except zmq.Again:
            message = None
        return message

    def resolve_callback(self, message):
        if not message in self._d.keys():
            return f"fail: `{message}` is not among known messages = {list(self._d.keys())}"
            self._testOnlySignalOnInvalidMessage.emit()
        callback = self._d[message]
        try:
            success, extra_info = callback()
        except Exception as ex:
            success = False
            extra_info = ex
        if success:
            return f"ok: {extra_info}"
        else:
            return f"fail: {extra_info}"

    def register(self, message, callback):
        """when a message is recieved that matches message, call callback
        callback must be a zero argument callable and return (succes, extra_info) 
        where success is a bool 
        and extra_info is something that can be string formatted
        
        I don't use normal QT message here because I want to get a response on the success or failure
        of the request"""
        if message in self._d.keys():
            raise Exception(f"message={message} is already registered with callback {self._d[message]}")
        self._d[message] = callback

    
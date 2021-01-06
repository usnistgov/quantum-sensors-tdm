import zmq

from PyQt5 import  QtCore, QtWidgets

#import cringe
from cringe.shared import log

class ZmqRep(QtWidgets.QWidget):
    ''' ZMQ server (reply socket) for use with cringe
    You can find a nice simple example example of how to talk to 
    this at https://zeromq.org/languages/python/'''
    gotMessage = QtCore.Signal(str)
    def __init__(self, parent, address_with_port):
        llog = log.child("ZmqRep: __init__:")
        log.debug("__init__")
        #super(type(self), self).__init__(parent)
        super().__init__(parent)
        self._context = zmq.Context()
        self._zmq_sock = self._context.socket(zmq.REP)
        self._zmq_sock.LINGER = 0
        llog.debug(f"bind on: {address_with_port}")
        self._zmq_sock.bind(address_with_port)
        self._timer = QtCore.QTimer()
        self._timer.start(100)
        self._timer.timeout.connect(self.handleTimeout)
        self._i=0

    def handleTimeout(self):
        self._i+=1
        llog = log.child("ZmqRep: handleTimeout:")
        # llog.debug("start", self._i)
        self.emit_if_has_message()

    def emit_if_has_message(self):
        llog = log.child("ZmqRep: emit_if_has_message:")
        # llog.debug("start")
        try:
            message = self._zmq_sock.recv(zmq.NOBLOCK).decode()
            llog.info(f"got: {message}")
            self.gotMessage.emit(message)
        except zmq.Again:
            pass

    def resolve_message(self, success, extra_info):
        llog = log.child("ZmqRep: resolve_message:")
        if success:
            reply = f"ok: {extra_info}"
        else:
            reply = f"failure: {extra_info}"
        llog.info(f"reply: `{reply}`")
        self._zmq_sock.send_string(reply)

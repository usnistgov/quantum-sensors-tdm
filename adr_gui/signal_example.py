from PyQt4 import QtCore, QtGui

class Window(QtGui.QWidget):
    customSignal = QtCore.pyqtSignal()

    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.edit = QtGui.QLineEdit(self)
        self.edit.textChanged.connect(self.handleTextChanged)
        self.button = QtGui.QPushButton(self)
        layout = QtGui.QVBoxLayout(self)
        layout.addWidget(self.edit)
        layout.addWidget(self.button)
        self.machine = QtCore.QStateMachine()
        self.off = QtCore.QState()
        self.off.assignProperty(self.button, 'text', 'Off')
        self.on = QtCore.QState()
        self.on.assignProperty(self.button, 'text', 'On')
        self.foo = QtCore.QState()
        self.foo.assignProperty(self.button, 'text', 'Foo')
        self.off.addTransition(self.button.clicked, self.on)
        self.on.addTransition(self.button.clicked, self.foo)
        self.foo.addTransition(self.customSignal, self.off)
        self.machine.addState(self.off)
        self.machine.addState(self.on)
        self.machine.addState(self.foo)
        self.machine.setInitialState(self.off)
        self.machine.start()
        self.show()

    def handleTextChanged(self, text):
        if text == 'foo':
            self.edit.clear()
            self.customSignal.emit()
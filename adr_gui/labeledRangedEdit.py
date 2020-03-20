import numpy

'''
Created on Oct 27, 2010

@author: schimaf
'''

from PyQt5 import QtGui, QtCore, QtWidgets

class labeledRangedEdit(QtWidgets.QWidget):
    def __init__(self, label_text = "", allowed_range = (0,1), startval=0, settings=False, parent=None):
        super(labeledRangedEdit, self).__init__(parent)
        self.allowed_range = (float(min(allowed_range)), float(max(allowed_range)))
        self.label = QtGui.QLabel(label_text)
        self.lineEdit = QtGui.QLineEdit()
        self.lineEdit.setText(str(self.allowed_range[0]))
        self.rangeLabel = QtGui.QLabel("%0.2f-%0.2f"%self.allowed_range)
        
        layout = QtGui.QHBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignRight)
        layout.setMargin(0)
        layout.setSpacing(5)
        layout.setSizeConstraint(QtGui.QLayout.SetFixedSize)
        layout.addWidget(self.label)
        layout.addWidget(self.lineEdit)
        layout.addWidget(self.rangeLabel)
        self.setLayout(layout)
        
        self.lineEdit.editingFinished.connect(self.enforceAlllowedRange)
        self.lineEdit.returnPressed.connect(self.enforceAlllowedRange)
        
        self.settings = settings
        try:
            self.value = self.settings.value(self.label.text(), type=float)
        except: 
            self.value = startval

        
    def enforceAlllowedRange(self):
        v = float(self.lineEdit.text())
        if v < self.allowed_range[0]: v = self.allowed_range[0]
        if v > self.allowed_range[1]: v = self.allowed_range[1]
        self.lineEdit.setText(str(v))
        self._value = v
        if self.settings:
            self.settings.setValue(self.label.text(), self.value)
    
    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self,v):
        self.lineEdit.setText(str(v))
        self.enforceAlllowedRange()


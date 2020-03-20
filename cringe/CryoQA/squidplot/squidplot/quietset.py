from PyQt4 import QtGui
def quiet_set(myobject,value):
    '''
    Sets the value of a widget without having the callback called
    '''
    myobject.blockSignals(True)

    if isinstance(myobject, QtGui.QComboBox):
        myobject.setCurrentIndex(value)
    elif isinstance(myobject, QtGui.QSpinBox):
        if value is None:
            myobject.clear()
        else:
            myobject.setValue(value)
    elif isinstance(myobject, QtGui.QDoubleSpinBox):
        if value is None:
            myobject.clear()
        else:
            myobject.setValue(value)
    elif isinstance(myobject, QtGui.QLineEdit):
        if value is None:
            myobject.clear()
        else:        
            myobject.setText(value)
    elif isinstance(myobject, QtGui.QCheckBox):
        myobject.setChecked(value)
    elif isinstance(myobject, QtGui.QPlainTextEdit):
        if value is None:
            myobject.clear()
        else:
            myobject.appendPlainText(value)
    elif isinstance(myobject, QtGui.QSlider):
        if value is None:
            myobject.setValue(myobject.minimum())
        else:
            myobject.setValue(value)

    myobject.blockSignals(False)

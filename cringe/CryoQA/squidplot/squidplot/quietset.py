from PyQt4 import QtGui
def quiet_set(myobject,value):
    '''
    Sets the value of a widget without having the callback called
    '''
    myobject.blockSignals(True)

    if isinstance(myobject, QComboBox):
        myobject.setCurrentIndex(value)
    elif isinstance(myobject, QSpinBox):
        if value is None:
            myobject.clear()
        else:
            myobject.setValue(value)
    elif isinstance(myobject, QDoubleSpinBox):
        if value is None:
            myobject.clear()
        else:
            myobject.setValue(value)
    elif isinstance(myobject, QLineEdit):
        if value is None:
            myobject.clear()
        else:        
            myobject.setText(value)
    elif isinstance(myobject, QCheckBox):
        myobject.setChecked(value)
    elif isinstance(myobject, QPlainTextEdit):
        if value is None:
            myobject.clear()
        else:
            myobject.appendPlainText(value)
    elif isinstance(myobject, QSlider):
        if value is None:
            myobject.setValue(myobject.minimum())
        else:
            myobject.setValue(value)

    myobject.blockSignals(False)

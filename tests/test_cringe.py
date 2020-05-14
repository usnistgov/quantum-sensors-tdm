from cringe import cringe
from PyQt5 import QtCore, QtWidgets
from named_serial import named_serial

print()
named_serial._setup_for_testing({"rack" : "dummy_rack", "tower": "dummy_tower"})


def test_cringe(qtbot): # see pytest-qt for info qtbot
    widget = cringe.Cringe(None, addr_vector=[0, 1, 2], slot_vector=[0, 1, 2], 
    class_vector=["DFBCLK", "DFBx2", "BAD16"],
    seqln=13, lsync=32, tower_vector=["DB1", 13], calibrationtab=True)
    qtbot.addWidget(widget)

    assert widget.seqln == 13
    assert widget.lsync == 32
    assert widget.tune_widget._test_check_true

    qtbot.mouseClick(widget.sys_glob_send, QtCore.Qt.LeftButton)

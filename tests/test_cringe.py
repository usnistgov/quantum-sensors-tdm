from cringe import cringe


def test_cringe(qtbot):
    widget = cringe.Cringe(None, addr_vector=[0, 1, 2], slot_vector=[0, 1, 2], 
    class_vector=["DFBCLK", "DFBx2", "BAD16"],
    seqln=13, lsync=32, tower_vector=["DB1", 13], calibrationtab=True)
    qtbot.addWidget(widget)

    assert widget.seqln == 13
    assert widget.lsync == 32
    assert widget.tune_widget._test_check_true
import sys
import optparse
import struct

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import named_serial
from . import badrap
from . import sv_array
from cringe.shared import terminal_colors as tc
from cringe.shared import log
import dprcal


class badass(QWidget):

    def __init__(self, parent=None, addr=None, slot=None, seqln=None, lsync=32):

        super(badass, self).__init__()

        self.serialport = named_serial.Serial(port='rack', shared = True)

        self.chns = 16

        self.addr = addr
        self.slot = slot
        self.seqln = seqln


        self.delay = 0
        self.led = 0
        self.status = 0

        self.dwell_val = 0
        self.dwellDACunits = float(0)
        self.range_val = 1
        self.rangeDACunits = float(2)
        self.step_val = 1
        self.stepDACunits = float(1)
        self.tri_idx = 0

        self.mode = 1


        self.chn_vectors = []
# 		self.enb = [0,0,0,0,0,0,0]
# 		self.cal_coeffs = [0,0,0,0,0,0,0]
# 		self.appTrim =[0,0,0,0,0,0,0]

        self.setWindowTitle("BAD16: %d/%d"%(slot, addr))	# Phase Offset Widget
        self.setGeometry(30,30,800,1000)
        self.setContentsMargins(0,0,0,0)

        self.layout_widget = QWidget(self)
        self.layout = QVBoxLayout(self)

        '''
        build widget for file management controls
        '''
        self.file_mgmt_widget = QGroupBox(self)
# 		self.file_mgmt_widget.setFlat(1)
        self.file_mgmt_widget.setFixedWidth(1080)
        self.file_mgmt_widget.setFocusPolicy(QtCore.Qt.NoFocus)
        self.file_mgmt_widget.setTitle("FILE MANAGEMENT INTERFACE")

        self.file_mgmt_layout = QGridLayout(self.file_mgmt_widget)
        self.file_mgmt_layout.setContentsMargins(5,5,5,5)
        self.file_mgmt_layout.setSpacing(5)

        self.loadsetup = QPushButton(self, text = "load setup")
        self.loadsetup.setFixedHeight(25)
        self.file_mgmt_layout.addWidget(self.loadsetup,0,0,1,1,QtCore.Qt.AlignLeft)

        self.savesetup = QPushButton(self, text = "save setup")
        self.savesetup.setFixedHeight(25)
        self.file_mgmt_layout.addWidget(self.savesetup,0,1,1,1,QtCore.Qt.AlignLeft)

        self.sendALLchns = QPushButton(self, text = "send setup")
        self.sendALLchns.setFixedHeight(25)
        self.file_mgmt_layout.addWidget(self.sendALLchns,0,2,1,1,QtCore.Qt.AlignLeft)

        self.filenameEdit = QLineEdit()
        self.filenameEdit.setReadOnly(True)
        self.file_mgmt_layout.addWidget(self.filenameEdit,0,4,1,4)

        self.filename_label = QLabel("file")
        self.file_mgmt_layout.addWidget(self.filename_label,0,3,1,1,QtCore.Qt.AlignRight)

        self.layout.addWidget(self.file_mgmt_widget)

        '''
        build widget for SYSTEM GLOBALS header
        '''
        self.sys_glob_hdr_widget = QGroupBox(self)
        self.sys_glob_hdr_widget.setFixedWidth(1080)
        self.sys_glob_hdr_widget.setFocusPolicy(QtCore.Qt.NoFocus)
        self.sys_glob_hdr_widget.setTitle("SYSTEM GLOBALS")

        self.sys_glob_layout = QGridLayout(self.sys_glob_hdr_widget)
        self.sys_glob_layout.setContentsMargins(5,5,5,5)
        self.sys_glob_layout.setSpacing(5)

        self.seqln_indicator = QLineEdit()
        self.seqln_indicator.setReadOnly(True)
        self.seqln_indicator.setText('%3d'%seqln)
        self.seqln_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.seqln_indicator.setFixedSize(self.seqln_indicator.sizeHint())
        self.seqln_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
        self.sys_glob_layout.addWidget(self.seqln_indicator,0,0,1,1)

        self.seqln_lbl = QLabel("sequence length")
# 		self.seqln_lbl.setAlignment(QtCore.Qt.AlignLeft)
        self.sys_glob_layout.addWidget(self.seqln_lbl,0,1,1,1,QtCore.Qt.AlignLeft)

        self.lsync_indicator = QLineEdit()
        self.lsync_indicator.setReadOnly(True)
        self.lsync_indicator.setText('%4d'%lsync)
        self.lsync_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.lsync_indicator.setFixedSize(self.seqln_indicator.sizeHint())
        self.lsync_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
        self.sys_glob_layout.addWidget(self.lsync_indicator,0,2,1,1)

        self.seqln_lbl = QLabel("line period")
        self.sys_glob_layout.addWidget(self.seqln_lbl,0,3,1,5,QtCore.Qt.AlignLeft)

        self.layout.addWidget(self.sys_glob_hdr_widget)

        '''
        build widget for CLASS GLOBALS header
        '''
        self.class_glob_hdr_widget = QGroupBox(self)
        self.class_glob_hdr_widget.setFixedWidth(1080)
        self.class_glob_hdr_widget.setFocusPolicy(QtCore.Qt.NoFocus)
        self.class_glob_hdr_widget.setTitle("CLASS GLOBALS")

        self.class_glob_layout = QGridLayout(self.class_glob_hdr_widget)
        self.class_glob_layout.setContentsMargins(5,5,5,5)
        self.class_glob_layout.setSpacing(5)

        self.card_delay = QSpinBox()
        self.card_delay.setRange(0, 15)
# 		self.card_delay.setFixedWidth(45)
        self.card_delay.setSingleStep(1)
        self.card_delay.setKeyboardTracking(0)
        self.card_delay.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.class_glob_layout.addWidget(self.card_delay,0,0,1,1)
        self.card_delay.valueChanged.connect(self.card_delay_changed)

        self.card_delay_lbl = QLabel("card delay")
        self.class_glob_layout.addWidget(self.card_delay_lbl,0,1,1,7,QtCore.Qt.AlignLeft)

        self.layout.addWidget(self.class_glob_hdr_widget)

        '''
        build widget for card INTERFACE PARAMETERS header
        '''
        self.class_interface_widget = QGroupBox(self)
        self.class_interface_widget.setFixedWidth(1080)
        self.class_interface_widget.setFocusPolicy(QtCore.Qt.NoFocus)
        self.class_interface_widget.setTitle("CARD INTERFACE PARAMETERS")

        self.controls_layout = QGridLayout(self.class_interface_widget)
        self.controls_layout.setContentsMargins(5,5,5,5)
        self.controls_layout.setSpacing(5)

# 		self.controls_widget = QWidget(self.layout_widget)
# 		self.globals_layout = QGridLayout(self.globals_widget)		


        self.addr_indicator = QLineEdit()
        self.addr_indicator.setReadOnly(True)
        self.addr_indicator.setText(str(addr))
        self.addr_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.addr_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
        self.controls_layout.addWidget(self.addr_indicator,0,0,1,1,QtCore.Qt.AlignRight)

        self.addr_label = QLabel("card address")
        self.controls_layout.addWidget(self.addr_label,0,1,1,1,QtCore.Qt.AlignLeft)

        self.slot_indicator = QLineEdit()
        self.slot_indicator.setReadOnly(True)
# 		self.addr_indicator.setFixedWidth(40)
        self.slot_indicator.setText('%2d'%slot)
        self.slot_indicator.setAlignment(QtCore.Qt.AlignRight)
        self.slot_indicator.setFocusPolicy(QtCore.Qt.NoFocus)
        self.controls_layout.addWidget(self.slot_indicator,0,2,1,1,QtCore.Qt.AlignRight)

        self.slot_label = QLabel("card slot")
        self.controls_layout.addWidget(self.slot_label,0,3,1,5,QtCore.Qt.AlignLeft)

        self.layout.addWidget(self.class_interface_widget)

        '''
        create TAB widget for embedding BAD16 functional widgets
        '''
        self.bad16_widget = QTabWidget(self)

        self.badrap_widget1 = badrap.badrap(parent=self, addr=addr, slot=slot, seqln=seqln, lsync=lsync)
        self.bad16_widget.addTab(self.badrap_widget1, "channels")

        badrap_widget2 = sv_array.SV_array(parent=self, seqln=seqln, addr=addr)
        self.bad16_widget.addTab(badrap_widget2, "states")

        badrap_widget3 = dprcal.dprcal(ctype="BAD16", addr=addr, slot=slot)
        self.bad16_widget.addTab(badrap_widget3, "phase")

        self.layout.addWidget(self.bad16_widget)

        '''
        resize widgets for relative, platform dependent variability
        '''
        rm = 45
        self.file_mgmt_widget.setFixedWidth(self.badrap_widget1.arrayframe.width()+rm)
        self.sys_glob_hdr_widget.setFixedWidth(self.badrap_widget1.arrayframe.width()+rm)
        self.class_glob_hdr_widget.setFixedWidth(self.badrap_widget1.arrayframe.width()+rm)
        self.class_interface_widget.setFixedWidth(self.badrap_widget1.arrayframe.width()+rm)


    def card_delay_changed(self):
        '''
        not sure what the best structure is for class global commanding
        at child level need to change control to indicator (QSpinBOx to QLineEdit)
        1 - can preserve function call triggered from within child as 'QLineEdit.textChanged'
        or
        2 - can pass parent.QSpinBox.value() to child function as parameter
        '''
        self.badrap_widget1.card_delay.setText(str("%2d"%self.card_delay.value()))
        self.badrap_widget1.card_delay_changed(self.card_delay.value())


def main():

    app = QApplication(sys.argv)
    app.setStyle("plastique")
    app.setStyleSheet("""	QPushbutton{font: 10px; padding: 6px}
                            QToolButton{font: 10px; padding: 6px}""")
    win = badass(addr=addr, slot=slot, seqln=seqln)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    p = optparse.OptionParser()
# 	p.add_option('-C','--card_type', action='store', dest='ctype', type='str',
# 				 help='Type of card to calibrate (default=DFBx2).')
    p.add_option('-A','--card_address', action='store', dest='addr', type='int',
                 help='Hardware address of card (default=32).')
    p.add_option('-S','--slot', action='store', dest='slot', type='int',
                 help='Host slot in crate (default=9)')
    p.add_option('-L','--length', action='store', dest='seqln', type='int',
                 help='Number of states in sequence (default=4')
# 	p.set_defaults(ctype="DFBx2")
    p.set_defaults(addr=32)
    p.set_defaults(slot=9)
    p.set_defaults(seqln=4)
    opt, args = p.parse_args()
# 	ctype = opt.ctype
    addr = opt.addr
    slot = opt.slot
    seqln = opt.seqln
    main()


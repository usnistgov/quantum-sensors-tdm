# Author velociraptor Genjix <aphidia@hotmail.com>

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import PyQt5.uic
from . import matplotlibCanvas
import numpy
import time
from . import tempControl
from instruments import pccc_card
from instruments import tower_power_supplies
import argparse
import os
import zmq
import struct
from lxml import etree
from nasa_client import JSONClient  # for stopping dastard
from cringe import zmq_rep
from . import adr_gui_control
import sys

DONT_RAMP_IF_ABOVE_K = 4.45


class MyLogger:
    def __init__(self):
        logDirectory = os.path.join(os.path.dirname(__file__), "logs")
        XML_CONFIG_FILE = "/etc/adr_system_setup.xml"
        if os.path.isfile(XML_CONFIG_FILE):
            f = open(XML_CONFIG_FILE, "r")
            root = etree.parse(f)
            child = root.find("log_folder")
            if child is not None:
                value = child.text
                logDirectory = value
        if not os.path.isdir(logDirectory):
            os.mkdir(logDirectory)

        self.filename = os.path.join(
            logDirectory, time.strftime("ADRLog_%Y%m%d_t%H%M%S.txt")
        )
        self.file = open(self.filename, "w")
        print(f"adr_gui log directory: {logDirectory}")
        print(f"adr_gui log filename: {self.filename}")

    def log(self, s):
        self.file.write(s + "\n")
        self.file.flush()


logger = MyLogger()


def adrMagTick(
    i_now, i_target, i_max=1.0, i_min=0.0, duration_s=60 * 1.0, step_time_s=0.5
):
    # calculate the heater out percent for a single step in a magnet cycle
    # print i_now, i_target, i_max, i_min, duration_s, step_time_s
    step_fraction = step_time_s / float(duration_s)
    # max_step_fraction = 0.005
    # if step_fraction > max_step_fraction:
    #     step_fraction = max_step_fraction
    #     print("slowing magCycleLogic to a max step percent of %f"%max_step_fraction)
    i_step = i_max * step_fraction * numpy.sign(i_target - i_now)
    i_new = i_now + i_step
    if i_new > i_max:
        i_new = i_max
    if i_new < i_min:
        i_new = i_min
    if i_step > 0:
        if i_new >= i_target:
            i_new = i_target
            return i_new, True
    elif i_step < 0:
        if i_new <= i_target:
            i_new = i_target
            return i_new, True
    else:
        print("i_step = 0")
    return i_new, False


class ADR_Gui(PyQt5.QtWidgets.QMainWindow):
    SIG_magUpDone = pyqtSignal()
    SIG_holdAfterMagUpDone = pyqtSignal()
    SIG_magDownDone = pyqtSignal()
    SIG_holdAfterMagDownDone = pyqtSignal()
    SIG_startgoingToMagUp = pyqtSignal()
    SIG_startMagUp = pyqtSignal()
    SIG_panic = pyqtSignal()
    SIG_skipToControl = pyqtSignal()
    SIG_startInitialState_ZeroCurrent = pyqtSignal()

    def __init__(self, min_mag_time=20):
        super(ADR_Gui, self).__init__()
        PyQt5.uic.loadUi(os.path.join(os.path.dirname(__file__), "adr_gui.ui"), self)
        self.setWindowIcon(QIcon("adr_gui.png"))

        # give variabiles initial values
        self.stateStartTime = time.time()
        self.lastTemp_K, self.lastHOut = 0, 0
        self.thresholdTemperatureK = 6

        self.setAttribute(Qt.WA_DeleteOnClose)

        self.settings = QSettings("adr_gui", "adr_gui")

        # Set up labels for settings file, allowed ranges, and start values
        # these really should be spins with all the ranges set in qt designer
        self.setPointmKEdit.label_text = "temperature set point (mK) "
        self.setPointmKEdit.allowed_range = (0.0, 1000.0)
        self.setPointmKEdit.startval = 80.0

        self.startTimeEdit.label_text = "start mag cycle at (time as decimal 24 hour) "
        self.startTimeEdit.allowed_range = (0.0, 23.9)
        self.startTimeEdit.startval = 7.0

        self.startOutEdit.label_text = (
            "dont start mag cycle if heater out is above (%) "
        )
        self.startOutEdit.allowed_range = (0.0, 100.0)
        self.startOutEdit.startval = 10.0

        self.maxHeatOutEdit.label_text = "mag up to this heater out percent (%) "
        self.maxHeatOutEdit.allowed_range = (0.0, 100.0)
        self.maxHeatOutEdit.startval = 35.0

        self.magUpMinsEdit.label_text = "mag up takes this long (mins) "
        self.magUpMinsEdit.allowed_range = (min_mag_time, 100.0)
        self.magUpMinsEdit.startval = 40.0

        self.magUpHoldMinsEdit.label_text = "hold after mag up takes this long (mins) "
        self.magUpHoldMinsEdit.allowed_range = (0.0, 200.0)
        self.magUpHoldMinsEdit.startval = 40.0

        self.magDownMinsEdit.label_text = "mag down takes this long (mins) "
        self.magDownMinsEdit.allowed_range = (min_mag_time, 100.0)
        self.magDownMinsEdit.startval = 40.0

        self.magDownHoldMinsEdit.label_text = (
            "hold after mag down takes this long (mins) "
        )
        self.magDownHoldMinsEdit.allowed_range = (0.0, 200.0)
        self.magDownHoldMinsEdit.startval = 15.0

        # Load line edit starting values from QSettings
        self.edits = [
            self.setPointmKEdit,
            self.startTimeEdit,
            self.startOutEdit,
            self.maxHeatOutEdit,
            self.magUpMinsEdit,
            self.magUpHoldMinsEdit,
            self.magDownMinsEdit,
            self.magDownHoldMinsEdit,
        ]
        for iEdit in self.edits:
            try:
                iEdit.value = self.settings.value(iEdit.label_text, type=float)
            except:
                iEdit.value = iEdit.startval
            iEdit.setText(str(iEdit.value))

        self.setPointmKEdit.editingFinished.connect(
            lambda: self.enforceAllowedRange(self.setPointmKEdit)
        )
        self.startTimeEdit.editingFinished.connect(
            lambda: self.enforceAllowedRange(self.startTimeEdit)
        )
        self.startOutEdit.editingFinished.connect(
            lambda: self.enforceAllowedRange(self.startOutEdit)
        )
        self.maxHeatOutEdit.editingFinished.connect(
            lambda: self.enforceAllowedRange(self.maxHeatOutEdit)
        )
        self.magUpMinsEdit.editingFinished.connect(
            lambda: self.enforceAllowedRange(self.magUpMinsEdit)
        )
        self.magUpHoldMinsEdit.editingFinished.connect(
            lambda: self.enforceAllowedRange(self.magUpHoldMinsEdit)
        )
        self.magDownMinsEdit.editingFinished.connect(
            lambda: self.enforceAllowedRange(self.magDownMinsEdit)
        )
        self.magDownHoldMinsEdit.editingFinished.connect(
            lambda: self.enforceAllowedRange(self.magDownHoldMinsEdit)
        )

        # self.enableTimeBasedMagCheckbox.setCheckState(Qt.Checked)
        # self.heatSwitchIsClosedCheckBox.setCheckState(Qt.Unchecked)

        self.excitationCurrentValues = (
            numpy.sqrt(numpy.logspace(1, 21, 21)) * 10.0 ** -12
        )
        self.excitationCurrentStrings = [
            "3.16 pA",
            "10.0 pA",
            "31.6 pA",
            "100 pA",
            "316 pA",
            "1.00 nA",
            "3.16 nA",
            "10.0 nA",
            "31.6 nA",
            "100 nA",
            "316 nA",
            "1.00 uA",
            "3.16 uA",
            "10.0 uA",
            "31.6 uA",
            "100 uA",
            "316 uA",
            "1.00 mA",
            "3.16 mA",
            "10.0 mA",
            "31.6 mA",
        ]
        self.currentExcitationComboBox.addItems(self.excitationCurrentStrings)
        self.currentExcitationComboBox.label_text = "excitation current"
        self.currentExcitationComboBox.startindex = 7  # 10.0 nA

        # disable some possible currents for now
        for i in range(
            self.excitationCurrentStrings.index("100 uA"),
            len(self.excitationCurrentStrings),
        ):
            self.currentExcitationComboBox.model().item(i).setEnabled(False)

        self.controlChannelValues = numpy.arange(16) + 1
        self.controlChannelStrings = list(map(str, self.controlChannelValues))
        self.controlChannelComboBox.addItems(self.controlChannelStrings)
        self.controlChannelComboBox.label_text = "control channel"
        self.controlChannelComboBox.startindex = 0

        self.comboBox_altChannel.addItems(self.controlChannelStrings)
        self.comboBox_altChannel.label_text = "Alt Channel"
        self.comboBox_altChannel.startindex = 0
        self.pushButton_altChannel.clicked.connect(self.handleCheckAltTemp)

        # Load combo box starting values from QSettings
        self.comboBoxes = [
            self.currentExcitationComboBox,
            self.controlChannelComboBox,
            self.comboBox_altChannel,
        ]
        for iComboBox in self.comboBoxes:
            try:
                iComboBox.indexValue = self.settings.value(
                    iComboBox.label_text, type=int
                )
            except:
                iComboBox.indexValue = iComboBox.startindex
            iComboBox.setCurrentIndex(iComboBox.indexValue)

        self.currentExcitationCurrent = self.excitationCurrentValues[
            self.currentExcitationComboBox.currentIndex()
        ]
        self.controlChannel = self.controlChannelValues[
            self.controlChannelComboBox.currentIndex()
        ]
        # Send current excitation to lakeshore here
        # Set some flag for server/client type here

        self.currentExcitationComboBox.currentIndexChanged.connect(
            self.excitationComboBoxChanges
        )
        self.controlChannelComboBox.currentIndexChanged.connect(
            self.controlComboBoxChanges
        )

        self.tempPlot = matplotlibCanvas.DynamicMplCanvas(
            "time (s)", "temperature (K)", ""
        )
        self.currentPlot = matplotlibCanvas.DynamicMplCanvas(
            "time (s)", "heater out %", ""
        )
        self.tempPlotLayout.addWidget(self.tempPlot)
        self.currentPlotLayout.addWidget(self.currentPlot)

        self.machine = QStateMachine(self)
        self.states = {}
        self.states["magUp"] = QState()
        self.states["holdAfterMagUp"] = QState()
        self.states["magDown"] = QState()
        self.states["holdAfterMagDown"] = QState()
        self.states["control"] = QState()
        self.states["goingToMagUp"] = QState()
        self.states["goingToInitialState_ZeroCurrent"] = QState()
        self.states["initialState_ZeroCurrent"] = QState()
        self.states["panic"] = QState()

        self.lastHOut = 0.0

        self.states["magUp"].addTransition(
            self.SIG_magUpDone, self.states["holdAfterMagUp"]
        )
        self.states["holdAfterMagUp"].addTransition(
            self.SIG_holdAfterMagUpDone, self.states["magDown"]
        )
        self.states["magDown"].addTransition(
            self.SIG_magDownDone, self.states["holdAfterMagDown"]
        )
        self.states["holdAfterMagDown"].addTransition(
            self.SIG_holdAfterMagDownDone, self.states["control"]
        )
        self.states["control"].addTransition(
            self.SIG_startgoingToMagUp, self.states["goingToMagUp"]
        )
        self.states["control"].addTransition(
            self.stopControlButton.clicked,
            self.states["goingToInitialState_ZeroCurrent"],
        )
        self.states["holdAfterMagDown"].addTransition(
            self.SIG_startgoingToMagUp, self.states["goingToMagUp"]
        )
        self.states["initialState_ZeroCurrent"].addTransition(
            self.SIG_startgoingToMagUp, self.states["goingToMagUp"]
        )
        self.states["initialState_ZeroCurrent"].addTransition(
            self.controlNowButton.clicked, self.states["control"]
        )
        self.states["initialState_ZeroCurrent"].addTransition(
            self.SIG_skipToControl, self.states["control"]
        )
        self.states["goingToMagUp"].addTransition(
            self.SIG_startMagUp, self.states["magUp"]
        )
        self.states["goingToInitialState_ZeroCurrent"].addTransition(
            self.SIG_startInitialState_ZeroCurrent,
            self.states["initialState_ZeroCurrent"],
        )
        self.magNowButton.clicked.connect(self.SIG_startgoingToMagUp.emit)
        for sname in self.states:
            self.states[sname].assignProperty(
                self.stateLabel, "text", "Current State: %s" % sname
            )
            self.states[sname].addTransition(self.SIG_panic, self.states["panic"])
            self.machine.addState(self.states[sname])

        # only allow certain changes in initial state
        self.states["initialState_ZeroCurrent"].exited.connect(
            self.disableModifyControlSettings
        )
        self.states["initialState_ZeroCurrent"].entered.connect(
            self.enableModifyControlSettings
        )

        timer = QTimer(self)
        timer.timeout.connect(self.timerHandler)
        self.tickDuration_s = 2
        timer.start(1000 * self.tickDuration_s)
        self.startTime = time.time()
        self.tickTime = time.time()

        self.timeOfLastAutorange = time.time() - 60

        self.tempControl = tempControl.TempControl(
            PyQt5.QtWidgets.QApplication,
            0.001,
            controlThermExcitation=self.currentExcitationCurrent,
            channel=self.controlChannel,
        )

        # these are to turn on and off the crate and tower during and after mags
        self.power_supplies = tower_power_supplies.TowerPowerSupplies()
        self.pccc_card = pccc_card.PCCC_Card()

        self.SIG_startMagUp.connect(self.prepForMagup)
        self.SIG_holdAfterMagDownDone.connect(self.setupTempControl)
        self.controlNowButton.clicked.connect(self.setupTempControl)
        self.SIG_holdAfterMagDownDone.connect(self.powerOnCrateTower)
        self.clearPlotsButton.clicked.connect(self.clearPlots)
        self.SIG_startgoingToMagUp.connect(self.powerOffCrateTower)

        self.machine.setInitialState(self.states["initialState_ZeroCurrent"])
        self.machine.start()

        self.errorIntegral = 0.0

        self.zmq_context = zmq.Context()

        self.control_socket = zmq_rep.ZmqRep(
            self, adr_gui_control.build_zmq_addr(host="*")
        )
        self.control_socket.gotMessage.connect(self.handleMessage)
        # maybe abstracted too far?  Have the data structure point to
        # the method name - here add a element 'func' as unbound method
        self.adr_gui_commands = adr_gui_control.ADR_GUI_COMMANDS.copy()
        for command in self.adr_gui_commands:
            self.adr_gui_commands[command]["func"] = self.__getattribute__(
                self.adr_gui_commands[command]["fname"]
            )

    def handleMessage(self, message):
        logger.log(f"got message: {message}")
        command_words = message.split()
        # original commands were uppercase - still accept those
        command = command_words[0].lower()
        command_args = command_words[1:]
        if command in self.adr_gui_commands:
            f = self.adr_gui_commands[command]["func"]
            logger.log(f"calling: {f}")
            try:
                success, extra_info = f(*command_args)
            except Exception as ex:
                success = False
                import traceback
                import sys

                exc_type, exc_value, exc_traceback = sys.exc_info()
                s = traceback.format_exception(exc_type, exc_value, exc_traceback)
                print("TRACEBACK")
                print("".join(s))
                print("TRACEBACK DONE")
                extra_info = f"Exception: {ex}\n{s}"
        else:
            success = False
            extra_info = f"`{message}` invalid, must be one of {list(self.adr_gui_commands.keys())}"
        self.control_socket.resolve_message(success, extra_info)

    def rpc_get_temp_k(self):
        return True, self.lastTemp_K

    def rpc_get_temp_rms_uk(self):
        assert self.isControlState()
        return True, self._last_stddev_uk

    def rpc_get_slope_hout_per_hour(self):
        assert self.isControlState()
        return True, self._last_slope_hout_per_hour

    def rpc_get_hout(self):
        assert self.isControlState()
        return True, self.lastHOut

    def rpc_set_temp_k(self, requested_setpoint_k):
        assert self.isControlState()
        requested_setpoint_mk = float(requested_setpoint_k) * 1000
        lo, hi = self.setPointmKEdit.allowed_range
        if requested_setpoint_mk > hi:
            return False, f"requested set point > than max, max = {hi}"
        if requested_setpoint_mk < lo:
            return False, f"requested set point < than min, min = {lo}"
        self.setPointmKEdit.setText(str(requested_setpoint_mk))
        self.enforceAllowedRange(self.setPointmKEdit)
        achieved_setpoint_mk = self.setPointmKEdit.value
        if achieved_setpoint_mk == requested_setpoint_mk:
            return True, achieved_setpoint_mk * 1e-3
        else:
            return False, achieved_setpoint_mk * 1e-3

    def rpc_echo(self, x):
        return True, x

    def enforceAllowedRange(self, line_edit):
        v = float(line_edit.text())
        if v < line_edit.allowed_range[0]:
            v = line_edit.allowed_range[0]
        if v > line_edit.allowed_range[1]:
            v = line_edit.allowed_range[1]
        line_edit.setText(str(v))
        line_edit.value = v
        if self.settings:
            self.settings.setValue(line_edit.label_text, line_edit.value)

    def excitationComboBoxChanges(self):
        if self.settings:
            self.settings.setValue(
                self.currentExcitationComboBox.label_text,
                self.currentExcitationComboBox.currentIndex(),
            )
        self.currentExcitationCurrent = self.excitationCurrentValues[
            self.currentExcitationComboBox.currentIndex()
        ]
        self.tempControl.controlThermExcitation = self.currentExcitationCurrent

    def controlComboBoxChanges(self):
        if self.settings:
            self.settings.setValue(
                self.controlChannelComboBox.label_text,
                self.controlChannelComboBox.currentIndex(),
            )
        self.controlChannel = self.controlChannelValues[
            self.controlChannelComboBox.currentIndex()
        ]
        self.tempControl.controlChannel = self.controlChannel

    def ensureHeatSwitchIsClosed(self):
        print("heat switch ensure closed")
        if self.heatSwitchIsClosedCheckBox.isChecked():
            print("heatswitch closed, doing nothing")
            return
        else:
            print("heat switch open, closing")
            self.tempControl.a.heat_switch.CloseHeatSwitch()
            self.heatSwitchIsClosedCheckBox.setCheckState(Qt.Checked)
            print("heat switch now closed")

    def ensureHeatSwitchIsOpen(self):
        print("heat switch ensure open")
        if self.heatSwitchIsClosedCheckBox.isChecked():
            print("heat switch closed, opening")
            self.tempControl.a.heat_switch.OpenHeatSwitch()
            self.heatSwitchIsClosedCheckBox.setCheckState(Qt.Unchecked)
            print("heat switch now open")
        else:
            print("heat switch open, doing nothing")

    def autorange(self):
        if time.time() - self.timeOfLastAutorange > 4:
            didAutorange = self.tempControl.safeAutorange()
            if didAutorange:
                self.timeOfLastAutorange = time.time()

    def timerHandler(self):
        self.pollTempControl()
        self.stateTick()
        self.updateTempPlot()
        self.updateCurrentPlot()
        self.settings.sync()

    def powerOffCrateTower(self):
        ########################################################################
        # Trying to stop the ndfb_server.
        ########################################################################
        print("tell the server to stop")
        req = self.zmq_context.socket(zmq.REQ)
        req.connect("tcp://localhost:2011")
        req.setsockopt(zmq.RCVTIMEO, 1000)
        req.send(struct.pack("!HHHHII", 0, 7, 0, 0, 0, 0))
        try:
            print((req.recv()))
        except zmq.Again:
            pass
        finally:
            print("close zmq socket")
            req.close()
        ########################################################################
        # Trying to stop dastard source.
        ########################################################################
        try:
            rpc = JSONClient(("localhost", 5500))
            rpc.call("SourceControl.Stop", "")
            rpc.close()
        except Exception as ex:
            print(("WARNING: exception {} during stopping dastard".format(ex)))
        print("wait 5 seconds before turning off crate to give server time to stop")
        time.sleep(5)  # give the server time to stop before turning off the crate

        print("turning off the crate power")
        self.pccc_card.powerOff()  # turn off crate when mag up starts
        print("turning off the tower power supplies")
        self.power_supplies.powerOffSupplies()  # turn off all tower outputs when mag up starts

    #         print("turning off outputs from DFB07 cards")
    #         dfb07_cards = self.tempControl.a.crate.dfb07_cards
    #         dfb07_card_channels = []
    #         for i in range(len(dfb07_cards)):
    #                 if not dfb07_cards[i].is_clock_card:
    #                     dfb07_card_channels.append(dfb07_cards[i].channel1)
    #                     dfb07_card_channels.append(dfb07_cards[i].channel2)
    #                     # note that this misses the dfb07 card channel that is part of the clock card
    #
    #         for card_channel in dfb07_card_channels:
    #             card_channel.setOutputsOff()
    #             card_channel.sendAllRows()
    #         ra16_cards = self.tempControl.a.crate.ra16_cards
    #         print("turning off outputs from RA16 cards")
    #         for card in ra16_cards:
    #             card.disable("all")
    #             card.setHigh("all",0)
    #             card.sendChannel("all")

    def powerOnCrateTower(self):
        print("turning on the tower power supplies and crate")
        self.pccc_card.powerOn()
        self.power_supplies.powerOnSequence()

    def prepForMagup(self):
        print("prepping for magup")
        # self.powerOffCrateTower() # now done on SIG_startgoingToMagUp
        self.ensureHeatSwitchIsClosed()
        self.tempControl.setupRamp()

    def updateTempPlot(self):
        self.tempPlot.add_point(time.time() - self.startTime, self.lastTemp_K)

    def updateCurrentPlot(self):
        self.currentPlot.add_point(time.time() - self.startTime, self.lastHOut)

    def clearPlots(self):
        self.tempPlot.clear_points()
        self.currentPlot.clear_points()

    def isControlState(self):
        return self.stateLabel.text().split(": ")[1] == "control"

    def stateTick(self):
        sname = self.stateLabel.text().split(": ")[1]
        self.autorange()

        if sname == "magUp":
            self.magUpStateTick()
        elif sname == "holdAfterMagUp":
            self.holdAfterMagUpStateTick()
        elif sname == "magDown":
            self.magDownStateTick()
        elif sname == "holdAfterMagDown":
            self.holdAfterMagDownStateTick()
        elif sname == "control":
            self.controlStateTick()
        elif sname == "initialState_ZeroCurrent":
            if self.lastHOut != 0:
                # check to see if we are in control state
                now_settings = {}
                tc = self.tempControl.a.temperature_controller
                control_mode = tc.getControlMode()
                ramp_status, ramp_rate = tc.getRamp()
                setpoint = tc.getTemperatureSetPoint()
                temperror = self.tempControl.getTempError()
                if (
                    control_mode == "closed"
                    and ramp_status == "on"
                    and 0 < ramp_rate < 0.2
                    and 0.04 < setpoint < 0.35
                    and numpy.abs(temperror) < 0.005
                ):
                    print(
                        (
                            "started with heater out %0.2f, but determined it is already in control state"
                            % self.lastHOut
                        )
                    )
                    print("STARTING IN CONTROL STATE")
                    self.heatSwitchIsClosedCheckBox.setCheckState(Qt.Unchecked)
                    self.stateStartTime = time.time()
                    self.tempControl.readyToControl = True
                    self.SIG_skipToControl.emit()
                else:
                    warningBox = QMessageBox()
                    warningBox.setText(
                        "Warning heater out is %0.2f, should be zero in initial state (unless its in closed loop control).  Manually correct system and try again.  Will exit after this."
                        % self.lastHOut
                    )
                    warningBox.exec_()
                    sys.exit()
            elif self.adrShouldMagup(
                self.startTimeEdit.value, self.startOutEdit.value, self.lastHOut
            ):
                self.SIG_startgoingToMagUp.emit()
                self.stateStartTime = time.time()
            else:
                self.printStatus("intital state zero current")
        elif sname == "goingToMagUp":
            self.goingToMagUpTick()
        elif sname == "goingToInitialState_ZeroCurrent":
            self.goingToIntitalState_ZeroCurrentTick()
        elif sname == "panic":
            if self.tempControl.readyToRamp:
                self.magDownStateTick()
            elif self.tempControl.readyToControl:
                self.tempControl.setSetTemp(0.0)
            else:
                self.printStatus(
                    "panic, not doing anything because doesn't seem to be in either control or ramp state"
                )

    def magUpStateTick(self):
        i_new, done = adrMagTick(
            self.lastHOut,
            i_target=self.maxHeatOutEdit.value,
            i_max=self.maxHeatOutEdit.value,
            i_min=0.0,
            duration_s=self.magUpMinsEdit.value * 60,
            step_time_s=self.tickDuration_s,
        )
        if self.tempControl.readyToRamp:
            self.setManualHeaterOut(i_new)
            self.printStatus("magging up")
        else:
            self.printStatus(
                "should be magging up, but not readyToRamp, trying to setupRamp, heater out must be 0"
            )
            self.tempControl.setupRamp()
        if done:
            self.SIG_magUpDone.emit()
            self.stateStartTime = time.time()

    def holdAfterMagUpStateTick(self):
        if time.time() - self.stateStartTime > 60 * self.magUpHoldMinsEdit.value:
            self.SIG_holdAfterMagUpDone.emit()
            self.stateStartTime = time.time()
            self.ensureHeatSwitchIsOpen()
        self.printStatus(
            "holding after Mag Up %d s left"
            % (60 * self.magUpHoldMinsEdit.value - time.time() + self.stateStartTime)
        )

    def magDownStateTick(self):
        i_new, done = adrMagTick(
            self.lastHOut,
            i_target=0.0,
            i_max=self.maxHeatOutEdit.value,
            i_min=0.0,
            duration_s=self.magDownMinsEdit.value * 60,
            step_time_s=self.tickDuration_s,
        )
        if self.tempControl.readyToRamp:
            self.setManualHeaterOut(i_new)
        self.printStatus("magging down")
        if done:
            self.SIG_magDownDone.emit()
            self.stateStartTime = time.time()

    def holdAfterMagDownStateTick(self):
        if time.time() - self.stateStartTime > 60 * self.magDownHoldMinsEdit.value:
            self.SIG_holdAfterMagDownDone.emit()
            self.stateStartTime = time.time()
        self.printStatus(
            "hold after mag down %d s left"
            % (60 * self.magDownHoldMinsEdit.value - time.time() + self.stateStartTime)
        )

    def controlStateTick(self):
        currentSetPoint = self.tempControl.getSetTemp()
        #         if currentSetPoint != 1e-3*self.setPointmKEdit.value:
        #             self.tempControl.setSetTemp(1e-3*self.setPointmKEdit.value)
        #             self.printStatus("changed temp set point to %f mK"%self.setPointmKEdit.value)
        #             time.sleep(2)
        error = 1e-3 * self.setPointmKEdit.value - self.lastTemp_K  # kelvin
        self.errorIntegral += error
        newSetPoint = (
            1e-3 * self.setPointmKEdit.value + error + self.errorIntegral * (1 / 12.0)
        )

        if error > 0.0001:
            self.errorIntegral = 0

        #         self.tempControl.setSetTemp(2*1e-3*self.setPointmKEdit.value-self.lastTemp_K)
        if newSetPoint < 1e-3 * (self.setPointmKEdit.value - 0.1):
            newSetPoint = 1e-3 * (self.setPointmKEdit.value - 0.1)
        if newSetPoint > 1e-3 * (self.setPointmKEdit.value + 0.1):
            newSetPoint = 1e-3 * (self.setPointmKEdit.value + 0.1)

        self.tempControl.setSetTemp(newSetPoint)

        #         print("%g %g %g"%(self.setPointmKEdit.value, self.lastTemp_K, 2*self.setPointmKEdit.value*1e-3-self.lastTemp_K))
        stddev, duration = self.controlTempStdDev()
        slope_hour, duration = self.controlHeaterSlope()
        if numpy.isnan(duration):
            self.printStatus(
                "waiting for more data points before calculating stddev and heater slope"
            )
            self._last_stddev_uk = -1e9
            self._last_slope_hout_per_hour = 1e9
        else:
            self._last_stddev_uk = stddev * 1e6
            self._last_slope_hout_per_hour = slope_hour
            self.printStatus(
                "temp stddev = %0.2f uK, heater slope = %0.2f %%/hour over last %g seconds"
                % (self._last_stddev_uk, slope_hour, numpy.round(duration))
            )
        if self.adrShouldMagup(
            self.startTimeEdit.value, self.startOutEdit.value, self.lastHOut
        ):
            self.stateStartTime = time.time()
            self.tempControl.setSetTemp(0.0)
            self.SIG_startgoingToMagUp.emit()

    def adrShouldMagup(
        self, startTime24Hour, thresholdHeaterPercent, currentHeaterPercent
    ):
        t = time.localtime()
        h = numpy.floor(startTime24Hour)
        m = numpy.floor((startTime24Hour - h) * 60)
        if (
            t.tm_hour == h
            and t.tm_min == m
            and self.enableTimeBasedMagCheckbox.isChecked()
        ):
            if self.lastTemp_K > 4:
                print(
                    (
                        "not magging up because temp is %f, not below 4 K"
                        % self.lastTemp_K
                    )
                )
                return False
            return True
        return False

    def goingToMagUpTick(self):
        if self.tempControl.readyToControl and self.lastHOut > 0.0:
            self.printStatus(
                "temp set point should = 0, waiting for heater out = 0.0 before switching to mag up"
            )
            if self.tempControl.getSetTemp() > 0.001:
                self.tempControl.setSetTemp(0.001)
        elif self.lastHOut == 0.0:
            if self.lastTemp_K > DONT_RAMP_IF_ABOVE_K:
                print(
                    f"in GoingToMagUp state, but temp = {self.lastTemp_K} is too high, needs to be below {DONT_RAMP_IF_ABOVE_K}"
                )
            else:
                time.sleep(5.0)
                self.SIG_startMagUp.emit()
                self.stateStartTime = time.time()
        else:
            self.printStatus(
                "something is wrong, in state goingToMagUp with nonzero heater out and not readyToControl"
            )

    def goingToIntitalState_ZeroCurrentTick(self):
        if self.tempControl.readyToControl and self.lastHOut > 0.0:
            self.printStatus(
                "temp set point should = 0, waiting for heater out = 0.0 before switching to initialState_ZeroCurrent"
            )
            if self.tempControl.getSetTemp() > 0.001:
                self.tempControl.setSetTemp(0.001)
        elif self.lastHOut == 0.0:
            if self.lastTemp_K > DONT_RAMP_IF_ABOVE_K:
                self.printStatus(
                    f"in goingToIntitalState_ZeroCurrent state, but temp = {self.lastTemp_K} is too high, needs to be below {DONT_RAMP_IF_ABOVE_K}"
                )
            else:
                time.sleep(5.0)
                self.SIG_startInitialState_ZeroCurrent.emit()
                self.stateStartTime = time.time()
        else:
            self.printStatus(
                "something is wrong, in state goingToIntitalState_ZeroCurrentTicket with nonzero heater out and not readyToControl"
            )

    def controlTempStdDev(self):
        last_n_points = self.tempPlot.last_n_points(61)
        stddev, duration = numpy.NAN, numpy.NAN
        if last_n_points is not None:
            stddev = numpy.std(last_n_points[1])
            duration = last_n_points[0][-1] - last_n_points[0][0]
        return stddev, duration

    def controlHeaterSlope(self):
        last_n_points = self.currentPlot.last_n_points(61)
        slope_hour = numpy.NAN
        duration_s = numpy.NAN
        if last_n_points is not None:
            duration_s = last_n_points[0][-1] - last_n_points[0][0]
            d_heater = last_n_points[1][-1] - last_n_points[1][0]
            slope_hour = 3600 * d_heater / float(duration_s)
        return slope_hour, duration_s

    def pollTempControl(self):
        self.lastTemp_K = self.tempControl.getTemp()
        self.lastHOut = self.tempControl.getHeaterOut()
        logger.log(
            "%s, %f, %f, %f"
            % (time.asctime(), time.time(), self.lastTemp_K, self.lastHOut)
        )
        if self.lastTemp_K > 1000 * self.thresholdTemperatureK:
            self.SIG_panic.emit()

    def setManualHeaterOut(self, v):
        self.tempControl.setHeaterOut(v)

    def printStatus(self, s):
        self.statusLabel.setText("status: %s" % s)

    def setupTempControl(self):
        self.tempControl.setupTempControl()

    def disableModifyControlSettings(self):
        self.controlChannelComboBox.setEnabled(False)
        self.currentExcitationComboBox.setEnabled(False)
        self.controlNowButton.setEnabled(False)
        self.stopControlButton.setEnabled(True)

    def enableModifyControlSettings(self):
        self.controlChannelComboBox.setEnabled(True)
        self.currentExcitationComboBox.setEnabled(True)
        self.controlNowButton.setEnabled(True)
        self.stopControlButton.setEnabled(False)

    def handleCheckAltTemp(self):
        alt_ch = self.controlChannelValues[self.comboBox_altChannel.currentIndex()]
        temp_K = self.tempControl.readAltChannelAndReturnToControlChannel(alt_ch)
        time_read = time.strftime("%m/%d %H:%M:%S")
        self.label_altTempReading.setText(
            f"Channel {alt_ch} was {temp_K:.3f} K at {time_read}"
        )
        if self.settings:
            self.settings.setValue(
                self.comboBox_altChannel.label_text,
                self.comboBox_altChannel.currentIndex(),
            )


def main():
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fastmag",
        help="remove mag up and down times for testing",
        action="store_true",
    )
    args = parser.parse_args()
    if args.fastmag:
        min_mag_time = 0
    else:
        min_mag_time = 20

    app = PyQt5.QtWidgets.QApplication(sys.argv)
    mainWin = ADR_Gui(min_mag_time)
    mainWin.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

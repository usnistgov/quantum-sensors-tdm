#! /usr/bin/env python
''' 
adrTempMonitorGui.py 

monitor all temperatures in the ADR.  
Includes all channels in Lakeshore370 and all channels on cryocon

Based on adr_gui

@author: JH, 12/2020

to do:
1) pass in optional arguments
2) initialize both LS and CC?
3) make it executable from outside qsp/src/nistqsptdm/adr_gui
'''

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import PyQt5.uic
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure

import numpy as np
import time, sys, os
from instruments import Lakeshore370, Cryocon22 
from lxml import etree

DEMAG_TICK_MS = 2000
SENSOR_NAMES = {1:'faa',2:'fp',3:'s1',4:'s2',5:'ggg',9:"s2_calib"}
DEFAULT_SENSORS = [1,2,3,4,5]


class MyLogger():
    def __init__(self,file_pattern="adrFullLog_%Y%m%d_t%H%M%S.txt"):
        logDirectory = os.path.join(os.path.dirname(__file__),"logs")
        XML_CONFIG_FILE = "/etc/adr_system_setup.xml"
        if os.path.isfile(XML_CONFIG_FILE):
            f = open(XML_CONFIG_FILE, 'r')
            root = etree.parse(f)
            child = root.find("log_folder")
            if child is not None:
                value = child.text
                logDirectory = value
        if not os.path.isdir(logDirectory): 
            os.mkdir(logDirectory)

        self.filename = os.path.join(logDirectory,time.strftime(file_pattern))
        self.file = open(self.filename,"w")
        print(f"adr_gui log directory: {logDirectory}")
        print(f"adr_gui log filename: {self.filename}")

    def writeHeader(self,header):
        self.log(header)

    def log(self,s):
	    self.file.write(s+"\n")
	    self.file.flush()
#logger = MyLogger()

class MplCanvas(FigureCanvasQTAgg):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        # We want the axes cleared every time plot() is called

        self.compute_initial_figure()

        #
        FigureCanvasQTAgg.__init__(self, fig)
        self.setParent(parent)

        FigureCanvasQTAgg.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

    def compute_initial_figure(self):
        pass

    def sizeHint(self):
        return QSize(700,500) # this seems to be big enough to make the axes legible without dragging

class DynamicMplCanvas(MplCanvas):
    """A canvas that updates itself every second with a new plot."""
    def __init__(self, xlabel="time (s)", ylabel="data (arb)", title="a plot", max_points = 3000, legend=None, plotstyle='linear', **kwargs):
        MplCanvas.__init__(self, **kwargs)
        self.number_of_lines = 1
        self.x = [[]]
        self.y = [[]]
        self.style = "-"
        self.max_points = max_points
        self.set_axis_labels(xlabel, ylabel, title, legend, plotstyle)
        
    def set_axis_labels(self, xlabel="time (s)", ylabel="data (arb)", title="a plot",legend=(), plotstyle='linear'):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.legend = legend
        self.plotstyle = plotstyle
        self.update_figure()

    def add_point(self, x,y, line_number=0):
        self.x[line_number].append(x)
        self.y[line_number].append(y)
        if len(self.x[line_number]) > self.max_points:
            self.x[line_number] = self.x[line_number][-self.max_points:]
            self.y[line_number] = self.y[line_number][-self.max_points:]
        self.update_figure()

    def add_line(self):
        self.x.append([])
        self.y.append([])
        self.number_of_lines = len(self.x)

    def clear_points(self, line_number=0):
        self.x[line_number] = []
        self.y[line_number] = []
        self.update_figure()

    def last_n_points(self,n, line_number=0):
        if len(self.x[line_number]) < n:
            return None
        else:
            return self.x[line_number][-n:], self.y[line_number][-n:]

    def update_figure(self):
        # Build a list of 4 random integers between 0 and 10 (both inclusive)
        self.axes.cla()
        for line_number in range(self.number_of_lines):
            if self.plotstyle=='linear':
                self.axes.plot(self.x[line_number], self.y[line_number], self.style)
            elif self.plotstyle=='semilogy':
                self.axes.semilogy(self.x[line_number], self.y[line_number], self.style)
            elif self.plotstyle=='semilogx':
                self.axes.semilogx(self.x[line_number], self.y[line_number], self.style)
            elif self.plotstyle=='loglog':
                self.axes.loglog(self.x[line_number], self.y[line_number], self.style)
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.ylabel)
        self.axes.set_title(self.title)
        self.axes.grid('on')
        if self.legend != None:
            self.axes.legend(self.legend,loc='upper left')
        self.draw()


class adrTempMonitorGui(PyQt5.QtWidgets.QMainWindow):

    def __init__(self,ls370_channels={1:'faa',2:'fp',3:'s1',4:'s2',5:'ggg'}, cc_channels={'A':'bb1','B':'bb2'}, plotstyle='semilogy'):
        super(adrTempMonitorGui, self).__init__()
        self.demag=False
        self.printToScreen = True 
        self.plotstyle=plotstyle
        self.do_calibration=False
        self.ls370_channels = list(ls370_channels.keys())
        self.ls370_channel_names = list(ls370_channels.values())  
        self.ls370_allowed_channels = range(1,17)
        self.N_ls370_channels = len(self.ls370_channels)
        self.cc_allowed_channels = ['a','b','A','B']
        self.names = self.ls370_channel_names
        self.channel_dict = ls370_channels

        # handle cases with cryocon channels and without
        if cc_channels != None:
            self.cc_channels = list(cc_channels.keys())
            self.cc_channel_names = list(cc_channels.values())
            self.N_cc_channels = len(self.cc_channels)
            self.names.extend(self.cc_channel_names)
            if self.N_cc_channels == 2:
                self.cc_get_temperature_toggle='both' # need a better name here
            elif self.N_cc_channels == 1:
                self.cc_get_temperature_toggle=self.cc_channels[0]
        else:
            self.cc_channels = cc_channels
            self.N_cc_channels = 0
        
        self.num_therms = self.N_ls370_channels + self.N_cc_channels
        
        # define some globals, initialize
        self.waitBeforeLSsample = 11 # seconds to wait before grabing temperature reading (the LS is slow)
        self.loopPeriod = self.waitBeforeLSsample*self.N_ls370_channels + 2
        self.temp = np.zeros(self.num_therms) 

        # error handling on allowed channels
        if any(np.array(self.ls370_channels)>17) or any(np.array(self.ls370_channels)<1):
            print('invalid ls370_channels.  Must be 1 through 16')
            sys.exit()
        # if self.cc_channels != None and any(self.cc_channels not in self.cc_allowed_channels: 
        #     print('invalid cc_channels: ',self.cc_channels, 'Valid cc_channels: ',self.cc_allowed_channels)
        #     sys.exit()
        

        # classes
        self.ls370 = Lakeshore370()
        self.ls370.setRamp(rampmode = 'off')
        self.ls370.setControlMode('off') 
        for sensor in SENSOR_NAMES.keys():
            if sensor in self.ls370_channels:
                self.ls370.turnChannelOn(sensor)
            else:
                self.ls370.turnChannelOff(sensor)
        self.ls370.setScan(self.ls370_channels[0],'on') # instead of micromanaging the lakeshore, let it do its thing
        if cc_channels != None:
            self.cc = Cryocon22()
            self.cc.getTemperature(self.cc_get_temperature_toggle) # 1st use of this returns nan for channel B, so exercise this command 1st (a cludge)
        self.logger = MyLogger()
        self.logger.writeHeader('#date, epoch'+self.num_therms*', %s'%tuple(self.names))

        # initialize user interface stuff
        #self.initLakeShore370()
        self.initUI()

        print('Launching adrTempMonitorGui with %d thermometers and sample period = %d s'%(self.num_therms,self.loopPeriod))
        

    def initUI(self):
        ''' initialize UI '''
        self.startTime = time.time()
        self.tempPlot = DynamicMplCanvas(xlabel='time (s)', ylabel='temperature (K)', title='', legend=self.names, plotstyle=self.plotstyle)
        self.setCentralWidget(self.tempPlot)
        for ii in range(self.num_therms):
            self.tempPlot.add_line()
        self.tempPlot.update_figure()
        timer = QTimer(self)
        timer.timeout.connect(self.timerHandler)
        timer.start(1000*self.loopPeriod)
        self.startTime = time.time()
        self.tickTime = time.time()

        self.setGeometry(500, 300, 1000, 500)
        self.setWindowTitle('ADR Temperature Monitor')

    # def initLakeShore370():
    #     ''' initial LakeShore, put into a state to scan all LS thermometers '''
    #     ls_init = input('Is the lakeshore setup for monitoring? 1=yes else=no')
    #     if ls_init==1:
    #         pass
    #     else:
    #         print('Setup the lakeshore manually for monitoring and try again')
    #         sys.exit()
        
    def getCCtemps(self):
        ''' return list of cryocon thermometer temperatures '''
        return list(self.cc.getTemperature(self.cc_get_temperature_toggle)) 

    def getLS370temps(self):
        ''' return list of lakeshore thermometer temperatures '''
        temps=[0]*self.N_ls370_channels
        for ii in range(self.N_ls370_channels):
            # self.ls370.setScan(self.ls370_channels[ii],'off')
            # time.sleep(self.waitBeforeLSsample)
            temps[ii] = self.ls370.getTemperature(channel=self.ls370_channels[ii])
        # self.ls370.setScan(self.ls370_channels[ii],'off')
        if self.do_calibration:
            temp_cal = temps[self.ls370_channels.index(self.calibrated_sensor)]
            resistance = self.ls370.getResistance(channel=self.sensor_to_calibrate)
            self.cal_logger.log(f"{time.asctime()}, {time.time():.1f}, {temp_cal}, {resistance}")
        return temps

    def getAllTemps(self):
        ''' return all temperatures in the ADR in a list '''
        if self.cc_channels==None:
            return self.getLS370temps()
        else:
            temps = self.getLS370temps()
            cctemps = self.getCCtemps()
            temps.extend(cctemps)
            return temps
        
    def updateTempPlot(self):
        for ii in range(self.num_therms):
            self.tempPlot.add_point(x=time.time()-self.startTime, y=self.temp[ii],line_number=ii)
        self.tempPlot.update_figure()

    def pollTemp(self):
        self.temp = self.getAllTemps()
        t1 = time.asctime(); t2 = time.time()
        # FP sensor is very reliable*. If we get above 10 K let's disable GGG
        # GGG seems reliable below 10 K but above 10 K it's flaky and can put
        # lakeshore into an infinite loop trying to set proper ranges for it
        #
        # * It's not very *accurate* above 30 K but at least it's reliable.
        try:
            fp_temp = self.temp[self.ls370_channels.index(2)]
            if 5 in self.ls370_channels:
                if fp_temp > 10:
                    self.ls370.turnChannelOff(5)
        #         else:
        #             self.ls370.turnChannelOn(5) # It's not reliable to turn it on automatically.
        except IndexError:
            pass
        if self.printToScreen:
            print("%f"%(t2 - self.startTime), ", %f"*self.num_therms%tuple(self.temp))
        self.logger.log("%s, %f"%(t1,t2)+", %f"*self.num_therms%tuple(self.temp))

    def timerHandler(self):
        self.pollTemp()
        self.updateTempPlot()
        #self.settings.sync()

    def start_demag(self, demag_time_min):
        demag_timer = QTimer(self)
        demag_timer.timeout.connect(self.demagTimerHandler)
        demag_timer.start(DEMAG_TICK_MS)
        self.demag_timer = demag_timer
        heater_now = self.ls370.GetManualHeaterOut()  
        print("Starting Demag! DID YOU OPEN THE HEAT SWITCH.")
        self.heater_dt = heater_now / (demag_time_min * 60000/DEMAG_TICK_MS)

    def demagTimerHandler(self):
        heater = self.ls370.GetManualHeaterOut()
        heater -= self.heater_dt
        heater = max(heater, 0)
        if heater == 0:
            print("demag done")
            self.demag_timer.stop()
        print(f"setting magnet to {heater:.2f}%")
        self.ls370.SetManualHeaterOut(heater)
    
    def add_calibration(self, args):
        self.do_calibration=True
        self.calibrated_sensor = args[0]
        self.sensor_to_calibrate = args[1]
        self.cal_logger = MyLogger(file_pattern="ThermometerCalibration_%Y%m%d_t%H%M%S.txt")
        sensorname_1 = self.channel_dict[self.calibrated_sensor]
        sensorname_2 = self.channel_dict[self.sensor_to_calibrate]
        self.cal_logger.writeHeader(f"#date, epoch, {sensorname_1} [K], {sensorname_2} [Ohms]")

def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(
                    prog='ADR Temperature Monitor',
                    description='Monitors temperatures in Velma cryostat mostly for cooldown and warm up')
    parser.add_argument("--include_coldload",type=bool,help="boolean to include monitoring cold load channels with the cryocon 22.",default=False)
    parser.add_argument('-9','--sensor-nine',help="read out 'sensor #9' on the lakeshore (the calibrated cernox)",action="store_true")
    parser.add_argument('-D','--demag',help="demagnetize the ADR over the course of n minutes while reading temperatures.",type=int)
    parser.add_argument(
        '-c',
        '--calibration-mode',
        help="write an extra file containing temperatures from the 1st sensor and resistances from the 2nd sensor",
        nargs=2,
        type=int,
        metavar=('calibrated','to_calibrate')
    )
    parser.add_argument(
        '-O',
        '--calibration-only',
        help="Requires -c, ignores all sensors other than the 2 you're calibrating.",
        action='store_true'
    )
    args=parser.parse_args()
    sensors = DEFAULT_SENSORS
    if args.sensor_nine:
        sensors += [9]

    if args.calibration_only:
        sensors = args.calibration_mode

    app = PyQt5.QtWidgets.QApplication(sys.argv)
    if args.include_coldload:
        mainWin = adrTempMonitorGui() # excludes use of sensor 9
    else:
        mainWin = adrTempMonitorGui(
            cc_channels=None,
            ls370_channels=dictionary_subset(SENSOR_NAMES, sensors)
        )

    if args.demag:
        mainWin.start_demag(args.demag)

    if args.calibration_mode:
        mainWin.add_calibration(args.calibration_mode)
    mainWin.show()
    sys.exit(app.exec_())

def dictionary_subset(dict, keys_list):
    return {k: dict[k] for k in dict.keys() & set(keys_list)}
    # here & is the set intersection operator

if __name__ == '__main__':
    main()

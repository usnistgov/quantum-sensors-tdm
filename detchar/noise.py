'''
noise.py  - Software to acquire multiplexed noise data with the nistqsptdm package.
Note microscope has psd function that is great for realtime work.  This software package
is suited for storing data and scripting to loop over temperature and bias point.

@author JH, 5/2023
'''

from nasa_client import EasyClient
from cringe.cringe_control import CringeControl
from adr_gui.adr_gui_control import AdrGuiControl
from iv_data import NoiseData, NoiseSweepData

import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.signal

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import PyQt5.uic
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure

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

class NoiseAcquire():
    ''' Acquire multiplexed, averaged noise data.
        No sensor setup is done.
    '''
    def __init__(self, column_str, row_sequence_list, m_ratio, rfb_ohm, f_min_hz=1, num_averages=10,
                 easy_client=None, adr_gui_control=None):
        self.column = column_str
        self.row_sequence = row_sequence_list # mux row order
        self.m_ratio = m_ratio # mutual inductance ratio of SQ1
        self.rfb_ohm = rfb_ohm
        self.num_averages = num_averages
        self.f_min_hz = f_min_hz # requested minimum frequency (affects acquisition time)

        self.ec = self._handle_easy_client_arg(easy_client)
        self.adr_gui_control = self._handle_adr_gui_control_arg(adr_gui_control)
        self.numPoints = self._handle_num_points(f_min_hz)
        self.dfb_bits_to_A = ((2**14-1)*m_ratio*(rfb_ohm+50))**-1 # conversion from dfb counts to amps

        # initialize main class attributes
        self.measured=False
        self.freqs = None # 1D array
        self.Pxx = None # averaged psds of shape (numRows,num_psd_pts)
        self.Pxx_all = None

    def _handle_num_points(self,f_min_hz,force_power_of_two=True):
        npts = int(self.ec.sample_rate/f_min_hz)
        if force_power_of_two:
            npts=2**(int(np.log2(npts)))
        return int(npts)

    def _handle_easy_client_arg(self, easy_client):
        if easy_client is not None:
            return easy_client
        easy_client = EasyClient()
        easy_client.setupAndChooseChannels()
        return easy_client

    def _handle_adr_gui_control_arg(self, adr_gui_control):
        if adr_gui_control is not None:
            return adr_gui_control
        return AdrGuiControl()

    def take(self,extra_info={},force_power_of_two=True):
        ''' get psds.  Store the averaged PSD as class variable Pxx. 
            return the fft bins (freqs) and the individual psds for each measurement 
            in the data structure ret_arr as a numpy array with the data structure 
            (rows,sample,measurement index)
        '''
        numPoints = self._handle_num_points(self.f_min_hz,force_power_of_two)
        Pxx_all = np.zeros(((self.ec.numRows,int(numPoints/2+1),self.num_averages))) # [row,sample,measurement #]
        pre_time = time.time()
        pre_temp_k = self.adr_gui_control.get_temp_k()

        for ii in range(self.num_averages):
            print('Noise PSD, average number = %d'%ii)
            y_ii = self.ec.getNewData(delaySeconds = 0.001, minimumNumPoints = numPoints, exactNumPoints = True, retries = 1)
            (freqs, Pxx_ii) = scipy.signal.periodogram(y_ii[0,:,:,1], fs=self.ec.sample_rate, window='boxcar',
                                                       nfft=None,detrend='constant',scaling='density',axis=-1)
            Pxx_all[:,:,ii]=Pxx_ii
            
        self.measured=True
        self.freqs=freqs
        self.Pxx = np.mean(Pxx_all,axis=-1)
        self.Pxx_all = Pxx_all

        return NoiseData(freq_hz=freqs, Pxx=self.Pxx, 
                         column=self.column, row_sequence=self.row_sequence,
                         num_averages=self.num_averages, pre_temp_k=pre_temp_k, pre_time_epoch_s=pre_time,
                         dfb_bits_to_A=self.dfb_bits_to_A, rfb_ohm=self.rfb_ohm, m_ratio=self.m_ratio,
                         extra_info=extra_info)
        
    def plot_avg_psds(self,physical_units=True):
        assert self.measured, 'You have not taken a measurement yet.  Use the take() method.'
        fig,ax = plt.subplots(1,1)
        fig.suptitle('Column %s averaged noise'%(self.column))
        if physical_units:
            y=self.Pxx.transpose()*self.dfb_bits_to_A**2
            ax.set_ylabel('PSD (A$^2$/Hz)')
        else:
            y=self.Pxx.transpose()
            ax.set_ylabel('PSD (arb$^2$/Hz)')
        ax.loglog(self.freqs,y)
        ax.set_xlabel('Frequency (Hz)')
        ax.legend(range(self.ec.numRows))


    def plot_psds_for_row(self,row_index,physical_units=True):
        assert self.measured, 'You have not taken a measurement yet.  Use the take() method.'
        fig,ax = plt.subplots(1,1)
        fig.suptitle('Column %s Row %02d noise'%(self.column,self.row_sequence[row_index]))
        if physical_units:
            m = self.dfb_bits_to_A
            ax.set_ylabel('PSD (A$^2$/Hz)')
        else:
            m=1
            ax.set_ylabel('PSD (arb$^2$/Hz)')
        for ii in range(self.num_averages):
            ax.loglog(self.freqs,self.Pxx_all[row_index,:,ii])
        ax.set_xlabel('Frequency (Hz)')
        ax.legend(range(self.num_averages))


    # def _init_ui(self):
    #     ''' initialize UI '''
    #     self.MainPlot = DynamicMplCanvas(xlabel='sample #', ylabel='Signal', title='', legend=range(self.ec.numRows), plotstyle=self.plotstyle)
    #     self.setCentralWidget(self.MainPlot)
    #     for ii in range(self.ec.numRows):
    #         self.MainPLot.add_line()
    #     self.MainPlot.update_figure()
    #     timer = QTimer(self)
    #     timer.timeout.connect(self.timerHandler)
    #     timer.start(1000*self.loopPeriod)
    #     self.startTime = time.time()
    #     self.tickTime = time.time()

    #     self.setGeometry(500, 300, 1000, 500)
    #     self.setWindowTitle('Noise Plot')

    # def update_plot(self):
    #     for ii in range(self.ec.numRows):
    #         self.MainPlot.add_point(x=time.time()-self.startTime, y=self.temp[ii],line_number=ii)
    #     self.MainPlot.update_figure()

class NoiseSweep(NoiseAcquire):
    ''' class to collect noise data as a function of temperature and
        detector bias.
    '''
    def __init__(self, column_str, row_sequence_list, m_ratio, rfb_ohm,
                 f_min_hz=1, num_averages=10,
                 easy_client=None, adr_gui_control=None,
                 detector_bias_list = [10000,20000],
                 temperature_list_k = [0.1],
                 signal_column_index=0,
                 voltage_source='tower',
                 db_cardname = 'DB',
                 db_tower_channel='0',
                 cringe_control=None):

        super().__init__(column_str, row_sequence_list, m_ratio, rfb_ohm, f_min_hz, num_averages,
                         easy_client, adr_gui_control)

        self.temp_list_k = temperature_list_k
        self.db_list = self._handle_db_input(detector_bias_list)
        self.db_cardname = db_cardname
        self.db_tower_channel = db_tower_channel
        self.signal_column_index=signal_column_index

        # globals hidden from class initialization
        self.temp_settle_delay_s = 30 # wait time after commanding an ADR set point

        self.cc = self._handle_cringe_control_arg(cringe_control)
        self.set_volt = self._handle_voltage_source_arg(voltage_source)

    def _handle_db_input(self,db_list):
        if type(db_list[0]) == list:
            output = db_list
        else:
            print('First element of detector_bias_list is not a list.  Using the same detector bias settings for all temperatures.')
            output = [db_list]*len(self.temp_list_k)

        # ensure descending order (useful so that only one autobias is needed)
        output_sorted = []
        for db in output:
            db.sort(reverse=True)
            output_sorted.append(db)
        return output_sorted

    def _handle_cringe_control_arg(self, cringe_control):
        if cringe_control is not None:
             return cringe_control
        return CringeControl()

    def _handle_voltage_source_arg(self,voltage_source):
        # set "set_volt" to either tower or bluebox
        if voltage_source == None or voltage_source == 'tower':
            set_volt = self.set_tower # 0-2.5V in 2**16 steps
        elif voltage_source == 'bluebox':
            self.bb = BlueBox(port='vbox', version='mrk2')
            set_volt = self.set_bluebox # 0 to 6.5535V in 2**16 steps
        return set_volt

    def set_tower(self, dacvalue):
        self.cc.set_tower_channel(self.db_cardname, self.db_tower_channel, int(dacvalue))

    def set_bluebox(self, dacvalue):
        self.bb.setVoltDACUnits(int(dacvalue))

    def _is_temp_stable(self, setpoint_k, tol=.005, time_out_s=180):
        ''' determine if the servo has reached the desired temperature '''
        assert time_out_s > 10, "time_out_s must be greater than 10 seconds"
        cur_temp=self.adr_gui_control.get_temp_k()
        it_num=0
        while abs(cur_temp-setpoint_k)>tol:
            time.sleep(10)
            cur_temp=self.adr_gui_control.get_temp_k()
            print('Current Temp: ' + str(cur_temp))
            it_num=it_num+1
            if it_num>round(int(time_out_s/10)):
                print('exceeded the time required for temperature stability: %d seconds'%(round(int(10*it_num))))
                return False
        return True

    def set_temp(self,temp_k):
        self.adr_gui_control.set_temp_k(float(temp_k))
        stable = self._is_temp_stable(temp_k)
        print('Temperature has been reached, waiting %d s to stabilize'%self.temp_settle_delay_s)
        time.sleep(self.temp_settle_delay_s)
        return stable

    def run(self, skip_wait_on_first_temp=False, force_power_of_two=True, extra_info={}):
        temp_output = []
        for ii,temp in enumerate(self.temp_list_k): #loop over temperature list
            print('Setting to temperature %.1f mK'%(temp*1000))
            if np.logical_and(ii==0,skip_wait_on_first_temp):
                self.adr_gui_control.set_temp_k(float(temp))
            else:
                self.set_temp(temp)
            print('Detector bias list: ',self.db_list[ii])
            if self.db_list[ii][0] != 0: # if detector bias non-zero, autobias device onto transition
                print('overbiasing detector, dropping bias down, then waiting 30s')
                self.set_volt(65535)
                time.sleep(0.3)
                self.set_volt(self.db_list[ii][0])
                time.sleep(30)
            det_bias_output = []
            for jj,db in enumerate(self.db_list[ii]): # loop over detector bias
                print('Setting detector bias to %d, then relocking'%db)
                self.set_volt(db)
                self.cc.relock_all_locked_fba(self.signal_column_index)
                time.sleep(0.1)
                print('Collecting Noise PSD')
                result = self.take(extra_info={},force_power_of_two=force_power_of_two)
                det_bias_output.append(result) # return data structure indexes as [temp_index,det_bias_index,result]
            temp_output.append(det_bias_output)

        return NoiseSweepData(data=temp_output, column=self.column, row_sequence=self.row_sequence,
                              temp_list_k=self.temp_list_k, db_list=self.db_list,
                              signal_column_index=self.signal_column_index,
                              db_cardname=self.db_cardname, db_tower_channel_str=self.db_tower_channel,
                              temp_settle_delay_s=self.temp_settle_delay_s,
                              extra_info=extra_info)

if __name__ == "__main__":
    path='/data/uber_omt/20230517/'
    filename = 'colB_noise_20230606_5.json'
    skip_wait = False
    row_sequence_list=[8,9,11,16,22]
    temp_list_k = [0.1,0.15]
    db_list = [ [21992, 18203, 16691, 15205, 14042, 12931, 11795,10565, 9904] , 
                [18511, 14773, 13255, 11898, 10889,  9979,  9049, 8065, 6995, 6406]
                ]
    #temp_list_k=[.15]
    #db_list=[0]
    comment = 'lsync=256, sett=110,nsamp=144'
    
    ns = NoiseSweep(column_str='B',
                      row_sequence_list=row_sequence_list, 
                      m_ratio = 15.08,
                      rfb_ohm = 1700,
                      f_min_hz = 1, 
                      num_averages=100,
                      detector_bias_list = db_list,
                      temperature_list_k = temp_list_k,
                      signal_column_index=0,
                      voltage_source='tower',
                      db_cardname = 'DB',
                      db_tower_channel='1',
                      cringe_control=None)
    data = ns.run(skip_wait_on_first_temp=skip_wait,extra_info={'comment':comment})
    data.to_file(path+filename,overwrite=True)
    print('wrote file %s to disk'%(path+filename))
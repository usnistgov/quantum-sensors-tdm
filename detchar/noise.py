''' 
noise.py  - Software to acquire multiplexed noise data with the nistqsptdm package.
Note microscope has psd function that is great for realtime work.  This software package 
is suited for storing data and scripting to loop over temperature and bias point. 

@author JH, 5/2023
'''

from nasa_client import EasyClient
from cringe.cringe_control import CringeControl
from adr_gui.adr_gui_control import AdrGuiControl
from .iv_data import NoiseData

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

    

if __name__ == "__main__":
    na = NoiseAcquire('A',[6,7,8,10,12,14,16,18,19,21,22,23], 15.08, 1700,f_min_hz=10, num_averages=10)
    data = na.take()
    na.plot_avg_psds(physical_units=True)
    na.plot_psds_for_row(0,physical_units=True)
    data.to_file('noise_data_example.json')
    plt.show()



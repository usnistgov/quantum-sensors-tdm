''' 
noise.py  - Software to acquire multiplexed noise data with the nistqsptdm package.
Note microscope has psd function that is great for realtime work.  This software package 
is suited for storing data and scripting to loop over temperature and bias point. 

@author JH, 5/2023
'''

import numpy as np
from nasa_client import EasyClient
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
    def __init__(self, f_min_hz=1, num_averages=10, easy_client=None, m_ratio=15.08,rfb_ohm=1750):
        self.ec = self._handle_easy_client_arg(easy_client)
        self.num_averages = num_averages 
        self.numPoints = self._handle_num_points(f_min_hz)
        self.to_I = ((2**14-1)*m_ratio*(rfb_ohm+50))**-1
        
        self.measured=False
        self.freqs = None # 1D array
        self.Pxx = None # averaged psds of shape (numRows,num_psd_pts)

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

    def take(self,showplot=False):
        ''' get psds.  Store the averaged PSD as class variable Pxx. 
            return the fft bins (freqs) and the individual psds for each measurement 
            in the data structure ret_arr as a numpy array with the data structure 
            (rows,sample,measurement index)
        '''
        if showplot:
            fig,ax = plt.subplots(1,1)
        ret_arr = np.zeros(((self.ec.numRows,int(self.numPoints/2+1),self.num_averages))) # [row,sample,measurement #]
        for ii in range(self.num_averages):
            print('average number = %d'%ii)
            y_ii = self.ec.getNewData(delaySeconds = 0.001, minimumNumPoints = self.numPoints, exactNumPoints = True, retries = 1)
            (freqs, Pxx_ii) = scipy.signal.periodogram(y_ii[0,:,:,1], fs=self.ec.sample_rate, window='boxcar',
                                                       nfft=None,detrend='constant',scaling='density',axis=-1)
            ret_arr[:,:,ii]=Pxx_ii
            
        if showplot:
            ax.loglog(freqs,np.mean(ret_arr[:,:,:],axis=-1).transpose())
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('PSD')
            ax.legend(range(self.ec.numRows))
            plt.show()

        self.freqs=freqs
        self.Pxx = np.mean(ret_arr,axis=-1)
        self.measured=True
        return freqs,ret_arr

    def plot_avg_psds(self,physical_units=True):
        assert self.measured, 'You have not taken a measurement yet.  Use the take() method.'
        fig,ax = plt.subplots(1,1)
        if physical_units:
            y=self.Pxx.transpose()*self.to_I**2
            ax.set_ylabel('PSD (A$^2$/Hz)')
        else:
            y=self.Pxx.transpose() 
            ax.set_ylabel('PSD (arb$^2$/Hz)')
        ax.loglog(self.freqs,y)
        ax.set_xlabel('Frequency (Hz)')
        ax.legend(range(self.ec.numRows))
        plt.show()

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
    na = NoiseAcquire(f_min_hz=.5)
    na.take()
    na.plot_avg_psds()



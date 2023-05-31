'''noiseAnalysis_utils.py

utility functions for analyzing noise data
@author JH 5/2023
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
sys.path.append('/Users/hubmayr/nistgit/nistqsptdm/detchar')
from iv_data import NoiseSweepData

def notchFilter(x,y,notch_x,x_widths):
    if type(x_widths)==int:
        x_widths = [x_widths]*len(notch_x)
    for ii,f in enumerate(notch_x):
        dexs = np.where(np.logical_or(x<f-x_widths[ii]/2,x>f+x_widths[ii]/2))[0]
        x=x[dexs]; y=y[dexs]
    return x,y

class FitNoise():
    def __init__(self,xdata,ydata,x_fit_limits=[1,1e4]):
        ''' xdata must be in ascending frequency order '''
        self.xdata = np.array(xdata); self.ydata = np.array(ydata) # raw, untouched data
        self.x_lim = x_fit_limits
        self.x, self.y = self.get_fit_data() # data that will be fit

        # initialize main data products
        self.popt=None # order is low frequency white noise, 2\pi*\tau, noise floor, 1/f amp, 1/f index
        self.pcov=None

    def get_fit_data(self):
        # explicitly remove DC term, which otherwise makes curve_fit fail
        if self.xdata[0]==0: x=self.xdata[1:]; y=self.ydata[1:]
        else: x=self.xdata ; y=self.ydata
        # normalize, I find curve_fit fails with DC value of 1e-18
        self.y_norm = y[0]
        y=y/self.y_norm
        # cull fit range
        dexes = np.where(np.logical_and(x>=self.x_lim[0],x<=self.x_lim[1]))[0]
        x = x[dexes] ; y = y[dexes]
        return x,y

    def notch_frequencies(self,f_notch=None,notch_width=None):
        if f_notch is None:
            f_notch = np.array([1,2,3,4,5,6,7,8,9,10,11])*60
        if notch_width is None:
            notch_width = 10
        self.x, self.y = notchFilter(self.x,self.y,f_notch,notch_width)

    def fit_func(self,x,a,b,c,d,e):
        return a*(1+(b*x)**2)**-1+c+d*x**e

    def fit(self, p0 = (1,1,1,1,-1), printresult=True,debug=False):
        try:
            popt,pcov = fit_result = curve_fit(self.fit_func, self.x, self.y, p0=p0, sigma=None, absolute_sigma=False,
                                               check_finite=True, bounds=([0,0,0,0,-np.inf],[np.inf,1,np.inf,np.inf,0]), method=None)
                           #jac=None, full_output=False)
            self.popt = popt; self.pcov = pcov
            fit_success=True
        except:
            print('Fit failed')
            self.popt=self.pcov=None
            fit_success=False
        if debug:
            fig,ax = plt.subplots(1,1)
            ax.loglog(self.xdata,self.ydata,'bo')
            ax.loglog(self.x,self.y*self.y_norm,'r.')
            #x_fit = np.linspace(np.min(xdata),np.max(xdata),1000)
            y_init = self.fit_func(self.x,p0[0],p0[1],p0[2],p0[3],p0[4])
            ax.loglog(self.x,y_init*self.y_norm,'k--')
            if self.popt is not None:
                y_fit = self.fit_func(self.x,self.popt[0],self.popt[1],self.popt[2],self.popt[3],self.popt[4])
                ax.loglog(self.x,y_fit*self.y_norm,'k-')
                ax.legend(('data','fit region','initial guess','fit'))
            else:
                ax.legend(('data','fit region','initial guess'))
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('PSD')
            plt.show()

        if np.logical_and(printresult,fit_success): self.print_results()
        return popt,pcov

    def plot(self,fig=None,ax=None):
        ''' plot data, fit, and residuals'''
        if fig is None:
            fig,ax = plt.subplots(2,1)
        ax[0].loglog(self.xdata,self.ydata,'r.')
        #x_fit = np.linspace(np.min(xdata),np.max(xdata),1000)
        y_fit = self.fit_func(np.array(self.xdata),self.popt[0],self.popt[1],self.popt[2],self.popt[3],self.popt[4])
        ax[0].loglog(self.xdata,y_fit,'k-')
        ax[1].plot(self.xdata,self.ydata-y_fit)
        ax[1].set_xlabel('Frequency (Hz)')
        ax[0].set_ylabel('Response')
        ax[0].set_ylabel('Data - Fit')
        return fig,ax

    def print_results(self):
        assert self.popt is not None, 'You must fit the data first.  Use fit() method.'
        print('Fit results (error estimate assumes no correlation b/w parameters)\n'+'-'*80)
        labels=['Low frequency white noise','2*pi*time constant','Noise floor','amplitude of 1/f','1/f frequency']
        for ii in range(len(self.popt)):
            print(labels[ii]+': ',self.popt[ii],' +/- ',self.pcov[ii][ii]**0.5)

class DetermineRsh():
    ''' From a NoiseSweepData instance or filename to that instance, determine the
        shunt resistance values
    '''
    def __init__(self,noise_sweep,dfb_bits_to_A=None):
        self.ns = self._handle_data_in(noise_sweep)
        self.nrows,self.nsamples = np.shape(self.ns.data[0][0].Pxx)
        self.row_sequence = self.ns.data[0][0].row_sequence
        self.f = self.ns.data[0][0].freq_hz
        self.dfb_bits_to_A = self._handle_conversion(dfb_bits_to_A)
        self.temp_list_k = self.ns.temp_list_k
        self.num_temps = len(self.temp_list_k)
        self.temp, self.temp_m, self.Pxx_list = self.get_singlebias_data(bias=0)

    def _handle_data_in(self,noise_sweep):
        dtype = type(noise_sweep)
        if dtype == str:
            ns = NoiseSweepData.from_file(noise_sweep)
        elif dtype == NoiseSweepData:
            ns = noise_sweep
        return ns

    def _handle_conversion(self,dfb_bits_to_A):
        if dfb_bits_to_A is None:
            toI = self.ns.data[0][0].dfb_bits_to_A
        else:
            toI = dfb_bits_to_A
        return toI

    def get_singlebias_data(self,bias=0):
        ''' return temperature list, measured temperature list, list of NoiseData instances
            which correspond to the selected bias
        '''
        df = []; temp_list=[]; temp_meas_list = []
        for ii,temp in enumerate(self.temp_list_k):
            db_list = self.ns.db_list[ii]
            if bias in db_list:
                temp_list.append(temp)
                dex = db_list.index(bias)
                nd_ii = self.ns.data[ii][dex] # individual noise data
                temp_meas_list.append(nd_ii.pre_temp_k)
                df.append(nd_ii.Pxx)
            else:
                continue
        return temp_list, temp_meas_list, df

    def plot_tempsweep_for_rows(self,row_index_list=None):
        if row_index_list is None:
            row_index_list = list(range(self.nrows))
        elif type(row_index_list) ==  int:
            row_index_list = [row_index_list]
        for ii, row_index in enumerate(row_index_list):
            row = self.row_sequence[row_index]
            fig,ax = plt.subplots(1,1,num=ii)
            fig.suptitle('Row %02d (index = %d)'%(row,row_index))
            for jj,temp in enumerate(self.temp):
                y = np.array(self.Pxx_list[jj])[row_index,:]*self.dfb_bits_to_A**2
                ax.loglog(self.f,np.array(self.Pxx_list[jj])[row_index,:]*self.dfb_bits_to_A**2)
            #ax.set_ylim((1e-21,2e-19))
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('PSD ')
            ax.legend((self.temp))

    def fit_tempsweeps(self):
        for ii in range(self.nrows): # loop over rows / signal channels
            for jj,temp in enumerate(self.temp): # loop over temperatures
                y = np.array(self.Pxx_list[jj])[ii,:]*self.dfb_bits_to_A**2
                fn = FitNoise(self.f,y)
                fn.notch_frequencies()
                popt,pcov = fn.fit((1,.01,.01,1,-0.5),debug=False)


def testFitNoise():
    npts=10000
    p=(100,1/100,10,100,-1)
    x=np.linspace(1,1e4,npts)
    fn = FitNoise(None,None)
    y_sig=fn.fit_func(x,p[0],p[1],p[2],p[3],p[4])
    y = y_sig+(np.random.rand(npts)-0.5)*10
    fn = FitNoise(x,y)
    popt,pcov = fn.fit()
    fn.plot()
    plt.show()

def testDetermineRsh():
    file = '/Users/hubmayr/tmp/colA_zerobias_noise_tsweep_230528_3.json'
    rsh = DetermineRsh(file)
    #rsh.plot_tempsweep_for_rows()
    rsh.fit_tempsweeps()
    plt.show()

if __name__ == "__main__":
    #testFitNoise()
    testDetermineRsh()

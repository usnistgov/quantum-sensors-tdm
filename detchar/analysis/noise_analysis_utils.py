''' noise_analysis_utils.py 

JH 04/2024 
'''

import matplotlib.pyplot as plt
import numpy as np 
from scipy.optimize import curve_fit
from scipy.constants import k as kb
from detchar.iv_data import NoiseSweepData

class FitNoiseSingle():
    ''' Fit a measurement of current noise density to a single pole roll-off model.  
        Typical protocol would be to initialize the class, use plot() to determine what 
        noise spikes to remove and what frequency range to fit.  
        Remove noise spikes with remove_freqs.  Trim the fit range with trim_fit_range().  
        Then fit the data with fit().
    '''
    def __init__(self, f_hz,Ixx):
        ''' f_hz: frequency vector in hz
            Ixx: current noise density in A^2/Hz
        '''
        self.f = f_hz # frequency in hz
        self.fs = self.f[1]-self.f[0]
        self.Ixx = Ixx # current noise density A^2/Hz
        self.data_cleaned=False

    def plot(self):
        plt.loglog(self.f,self.Ixx,'.')
        if self.data_notched:
            plt.loglog(self.f_notched,self.Ixx_notched,'.')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Current Noise Density (A$^2$/Hz)')
        plt.grid('on')

    def func(self,x,a,b,c):
        r""" white noise term rolled off to background noise 
            input 
            x: x-axis variable, (frequency)
            a: white noise amplitude at low frequency 
            b: determines roll-off frequency.  If x = frequency in Hz, then b=(2\pi\tau)**2
            c: white noise amplitude at high frequency
        """
        return a*(1+b*x**2)**-1+c

    def remove_freqs(self,toremove_hz=[60,180,300],bw=[5,0,1]):
        ''' remove pesky noise spikes that throw off the fit.  
            input:
            toremove_hz: list of center frequencies to remove (typically 60Hz and harmonics)
            bw: list of bandwidths (in number of single sided frequency bins) to remove

        '''
        indices = []
        for ii,f in enumerate(toremove_hz):
            dex = np.argmin(abs(f-self.f))
            indices.extend(list(range(dex-bw[ii],dex+bw[ii]+1)))  
        self.data_cleaned = True
        self.Ixx_to_fit = np.delete(self.Ixx,indices)
        self.f_to_fit = np.delete(self.f,indices)

    def trim_fit_range(self,fmin=10,fmax=10e3):
        if self.data_cleaned:
            x = self.f_to_fit 
        else:
            x = self.f 
        indices = np.where((x<fmax) & (x>fmin))[0]
        self.f_to_fit = self.f_to_fit[indices]
        self.Ixx_to_fit = self.Ixx_to_fit[indices]

    def fit(self,showplot=False,verbose=False):
        p0 = self.get_init_guess()
        if self.data_cleaned:
            x=self.f_to_fit ;  y = self.Ixx_to_fit 
        else:
            x=self.f; y=Ixx
        popt, pcov = curve_fit(self.func, x,y,p0=p0)#,bounds=(0, [3., 1., 0.5]))
        if verbose: print('fit: a=%5.3e, b=%5.3e, c=%5.3e' % tuple(popt))

        if showplot:
            fig,ax = plt.subplots(2,1)
            ax[0].loglog(self.f,self.Ixx,'.',label='raw data')
            ax[0].loglog(x,y,'.',label='fit data')
            ax[0].loglog(x,self.func(x, *popt), 'r--', label='fit: a=%5.1e, b=%5.1e, c=%5.1e' % tuple(popt))
            ax[0].set_ylabel('Current noise density (A$^2$/Hz)')
            ax[0].legend(loc='lower left')
            ax[1].semilogx(x,y-self.func(x, *popt),'.')
            ax[1].set_xlabel('Frequency (Hz)')
            ax[1].set_xlim(ax[0].get_xlim())
            ax[1].set_ylabel('Residuals')
            plt.tight_layout()

        return popt, pcov 
    
    def get_init_guess(self):
        a = self.Ixx[np.where((self.f>10) & (self.f<45))[0]].mean() 
        b = self.f[np.argmin(abs(self.Ixx - a/2))]**-2
        c = self.Ixx[np.where((self.f>5000) & (self.f<6000))[0]].mean() 
        return a,b,c

class FitNoiseTemperatureSweep():
    ''' Fit measurements of current noise density as a function of temperature for a single channel to 
        a single pole roll-off model.  The main purpose is to extract the shunt resistance value. 
    '''  
        
    def __init__(self, f_hz,Ixx_arr,temp_list_k):
        ''' f_hz: frequency vector in hz
            Ixx: current noise density array [N frequency pts x M temperatures], units in A^2/Hz
            temp_list_k: list of temperatures in k at which the measurement was taken. temp_list_mK[ii] corresponds to Ixx_arr[:,ii] 
        '''
        self.f = f_hz 
        self.Ixx_arr = Ixx_arr 
        self.temp_list = temp_list_k 
        self.npts,self.ntemps = np.shape(Ixx_arr)
        self.fns = []
        for ii in range(self.ntemps):
            self.fns.append(FitNoiseSingle(f_hz,Ixx_arr[:,ii]))
        self.is_fit = False

    def plot(self):
        for ii in range(self.ntemps):
            plt.loglog(self.f,self.Ixx_arr[:,ii])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Current noise density (A$^2$/Hz)')
        plt.legend(self.temp_list)

    def fit(self,to_remove=None,bw=None,showplot=False):
        popt_list = []; cov_list =[]; fit_data_list=[]
        for ii,fns in enumerate(self.fns):
            if to_remove is None:
                fns.remove_freqs()
            else:
                fns.remove_freqs(to_remove,bw)
            fns.trim_fit_range()
            popt,cov = fns.fit()
            popt_list.append(popt); cov_list.append(cov)
            fit_data_list.append([fns.f_to_fit,fns.Ixx_to_fit])
        if showplot:
            for ii,fit in enumerate(fit_data_list):
                plt.loglog(fit[0],fit[1])
            plt.legend(self.temp_list)
            for ii, fit in enumerate(fit_data_list):
                plt.loglog(fit[0],self.fns[0].func(fit[0], *popt_list[ii]), 'k--')    
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Current noise density (A$^2$/Hz)')
        self.popt_list = popt_list
        self.cov_list = cov_list
        self.fit_data_list = fit_data_list 
        self.is_fit = True

        self.fit_parameters=np.zeros((self.ntemps,3))
        for ii, popt in enumerate(self.popt_list):
            self.fit_parameters[ii,:] = popt
    
    def plot_fit_v_temp(self):
        assert self.is_fit, 'Hey man, you gotta fit the data first before you can plot the fit parameters versus temperature'
        fig,ax = plt.subplots(3,1)
        ax[0].plot(self.temp_list,self.fit_parameters[:,0],'o')
        ax[1].plot(self.temp_list,self.fit_parameters[:,1]**0.5/2/np.pi,'o')
        ax[2].plot(self.temp_list,self.fit_parameters[:,2],'o')
        ylabels = ['white noise (A$^2$/Hz) ','time constant (s)','readout noise (A$^2$/Hz)']
        for ii in range(3):
            ax[ii].set_ylabel(ylabels[ii])
        ax[2].set_xlabel('Temperature (K)')
        plt.tight_layout()

    def get_R(self,showplot=False):
        ''' determine the resistance of the Johnson noise emitting resistor from a fit of white noise vs temperature '''
        p = np.polyfit(np.array(self.temp_list),self.fit_parameters[:,0],1)
        if showplot:
            fig,ax = plt.subplots(2,1)
            ax[0].plot(self.temp_list,self.fit_parameters[:,0],'.',label='data')
            ax[0].plot(self.temp_list,np.polyval(p,self.temp_list), 'k--', label='fit: m=%5.1e, b=%5.1e' % tuple(p))
            ax[0].set_ylabel('Current noise density (A$^2$/Hz)')
            ax[0].legend(loc='upper left')
            ax[1].plot(self.temp_list,self.fit_parameters[:,0]-np.polyval(p,self.temp_list),'.')
            ax[1].set_xlabel('Frequency (Hz)')
            ax[1].set_ylabel('Residuals')
            plt.tight_layout()  
        R_ohms = 4*kb/p[0]
        print('Resistance is %.3f microOhms'%(R_ohms*1e6))
        return R_ohms  

def noise_sweep_data_parser(noisesweepdata_filename,detbias_index=0,row_sequence_index=0,temp_list=None):
    ''' helps parse the raw NoiseSweepData json file into the structure that FitNoiseTemperatureSweep expects '''
    ns = NoiseSweepData.from_file(noisesweepdata_filename)
    print("Loading noise sweep:")
    print(f"    temperatures: {ns.temp_list_k}")
    print(f"    biases: {ns.db_list}")
    m, y_label_str = ns._phys_units(True)
    if temp_list is None:
        ntemps = len(ns.temp_list_k)
        temp_list = ns.temp_list_k
    else:
        ntemps = len(temp_list)

    npts= len(ns.data[0][detbias_index].freq_hz)
    Ixx = np.zeros((npts,ntemps))
    meas_temp=[]
    for ii,temp in enumerate(temp_list):
        dex = ns.temp_list_k.index(temp)
        Ixx[:,ii] = np.array(ns.data[dex][0].Pxx[row_sequence_index])*m**2
        meas_temp.append(ns.data[dex][0].pre_temp_k)
    return FitNoiseTemperatureSweep(np.array(ns.data[0][0].freq_hz),Ixx,meas_temp)

if __name__ == "__main__":
    # from detchar.iv_data import NoiseSweepData
    # ns=NoiseSweepData.from_file('/data/uber_omt/20240329/rsh_noise4.json')
    # df = ns.data[4][0]
    # m, y_label_str = ns._phys_units(True)
    # fns=FitNoiseSingle(np.array(df.freq_hz),np.array(df.Pxx)[0,:]*m**2)
    # fns.remove_freqs()
    # fns.trim_fit_range()
    # fns.fit(showplot=True,verbose=True)
    # plt.show()

    fnts = noise_sweep_data_parser('/data/uber_omt/20240329/rsh_noise2.json')
    fnts.plot()
    plt.show() 
    fnts.fit(to_remove=[60,120,180,300,420,540,660,780],bw=[30,1,2,4,1,1,2,1],showplot=True)
    plt.show()
    fnts.plot_fit_v_temp()
    plt.show()
    fnts.get_R(True)
    plt.show()
    

    



    
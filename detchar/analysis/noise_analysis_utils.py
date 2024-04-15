''' noise_analysis_utils.py 

JH 04/2024 
'''

import matplotlib.pyplot as plt
import numpy as np 
from scipy.optimize import curve_fit

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
        ''' white noise term rolled off to background noise 
            input 
            x: x-axis variable, (frequency)
            a: white noise amplitude at low frequency 
            b: determines roll-off frequency.  If x = frequency in Hz, then b=(2\pi\tau)**2
            c: white noise amplitude at high frequency
        '''
        return a*(1+b*x**2)**-1+c

    def remove_freqs(self,toremove_hz=[60,180,300],bw=[5,0,0]):
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

    def trim_fit_range(self,fmax=40e3):
        if self.data_cleaned:
            indices = np.where(self.f_to_fit<fmax)[0]
            self.f_to_fit = self.f_to_fit[indices]
            self.Ixx_to_fit = self.Ixx_to_fit[indices]
        else:
            indices = np.where(self.f<fmax)[0]
            self.f_to_fit = self.f[indices]
            self.Ixx_to_fit = self.Ixx[indices]    

    def fit(self,p0=None,showplot=False,verbose=False):
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
            ax[0].loglog(x,self.func(x, *popt), 'r--', label='fit: a=%5.3e, b=%5.3e, c=%5.3e' % tuple(popt))
            ax[0].set_ylabel('Current noise density (A$^2$/Hz)')
            ax[0].legend()
            ax[1].semilogx(x,y-self.func(x, *popt),'.')
            ax[1].set_xlabel('Frequency (Hz)')
            ax[1].set_ylabel('Residuals')
            
        return popt, pcov 
    
    def get_init_guess(self):
        a = self.Ixx[np.where((self.f>1) & (self.f<45))[0]].mean() 
        b = self.f[np.argmin(abs(self.Ixx - a/2))]**-2
        c = self.Ixx[np.where((self.f>5000) & (self.f<6000))[0]].mean() 
        return a,b,c

if __name__ == "__main__":
    from detchar.iv_data import NoiseSweepData
    ns=NoiseSweepData.from_file('/data/uber_omt/20240329/rsh_noise2.json')
    df = ns.data[4][0]
    fns=FitNoiseSingle(np.array(df.freq_hz),np.array(df.Pxx)[0,:])
    fns.remove_freqs()
    fns.trim_fit_range()
    fns.fit(showplot=True,verbose=True)
    plt.show()
    

    



    
'''noiseAnalysis_utils.py

utility functions for analyzing noise data
@author JH 5/2023
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class FitNoise():
    def __init__(self,xdata,ydata):
        self.xdata = xdata
        self.ydata=ydata
        self.popt=None
        self.pcov=None

    def fit_func(self,x,a,b,c,d,e):
        return a*(1+(b*x)**2)**-1+c+d*x**e

    def fit(self, printresult=True):
        popt,pcov = fit_result = curve_fit(self.fit_func, self.xdata, self.ydata, p0=(1,1,1,1,-0.5), sigma=None, absolute_sigma=False,
                                           check_finite=True, bounds=([0,0,0,0,-np.inf],[np.inf,1,np.inf,np.inf,0]), method=None)
                           #jac=None, full_output=False)
        self.popt = popt; self.pcov = pcov
        if printresult: self.print_results()
        return popt,pcov

    def plot(self,fig=None,ax=None):
        ''' plot data, fit, and residuals'''
        if fig is None:
            fig,ax = plt.subplots(2,1)
        ax[0].loglog(self.xdata,self.ydata,'r.')
        #x_fit = np.linspace(np.min(xdata),np.max(xdata),1000)
        y_fit = self.fit_func(self.xdata,self.popt[0],self.popt[1],self.popt[2],self.popt[3],self.popt[4])
        ax[0].loglog(self.xdata,y_fit,'k-')
        ax[1].plot(self.xdata,self.ydata-y_fit)
        ax[1].set_xlabel('Frequency (Hz)')
        ax[0].set_ylabel('Response')
        ax[0].set_ylabel('Data - Fit')

    def print_results(self):
        assert self.popt is not None, 'You must fit the data first.  Use fit() method.'
        print('Fit results (error estimate assumes no correlation b/w parameters)\n'+'-'*80)
        labels=['Low frequency white noise','2*pi*time constant','Noise floor','amplitude of 1/f','1/f frequency']
        for ii in range(len(self.popt)):
            print(labels[ii]+': ',self.popt[ii],' +/- ',self.pcov[ii][ii]**0.5)

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

if __name__ == "__main__":
    testFitNoise()

''' passband_utils.py

    utilities for simulated passbands and calculation of metrics
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k,h,c
from scipy.integrate import quad, simps,cumtrapz


class LoadingCalculation():
    def Pnu_thermal(self,nu,T):
        ''' power spectral density (W/Hz) of single mode from thermal source at
            temperature T (in K) and frequency nu (Hz).
        '''
        x = h*nu/(k*T)
        B = h*nu * (np.exp(x)-1)**-1
        return B

    def thermalPower(self,nu1,nu2,T,passband_arr=None,debug=False):
        ''' Calculate the single mode thermal power emitted from a blackbody
            at temperature T (in Kelvin) from frequency nu1 to nu2 (in Hz).
            passband_arr is an N x 2 np.array with the 1st column the frequency in Hz (monotonically increasing),
            and the second column the absolute response. The default is None, in which case a
            top hat band is assumed.  If passband_arr provided, nu1 and nu2 are unused.
        '''
        try:
            if passband_arr==None: # case for tophat
                P = quad(self.Pnu_thermal,nu1,nu2,args=(T))[0] # toss the error
        except: # case for arbitrary passband shape F
            N,M = np.shape(passband_arr)
            nu = np.linspace(passband_arr[0,0],passband_arr[-1,0],N)
            integrand = self.Pnu_thermal(nu,T)*passband_arr[:,1]
            P = simps(integrand,nu)
            if debug:
                print('calculating power from %.1f GHz to %.1f GHz'%(passband_arr[0,0]/1e9,passband_arr[-1,0]/1e9))
                yy = self.Pnu_thermal(nu,T)
                plt.plot(passband_arr[:,0],passband_arr[:,1]*np.max(yy),'k-')
                plt.plot(nu,yy,'b-')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Power (W)')
                plt.legend(('passband','thermal spectrum'))
                plt.title('Debug plot for thermalPower calculation')
                plt.show()
        return P


class PassbandMetrics():
    def __cull_range(self,x,y,x_range):
        indices = np.where((x >= x_range[0]) & (x <= x_range[1]))
        return x[indices], y[indices]

    def calc_bandwidth(self,f_ghz,B,f_range_ghz=None):
        ''' calculate bandwidth of passband from equation:
            bw = [ \int_f1^f2 B(f_ghz) df ] ^2 / [ \int_f1^f2 B(f_ghz)^2 df ]
        '''
        if f_range_ghz != None:
            f_ghz, B = self.__cull_range(f_ghz,B,f_range_ghz)

        integral_numerator = simps(y=B, x=f_ghz) #simps(y, x=None, dx=1, axis=-1, even='avg')
        integral_denom = simps(y=B**2,x=f_ghz)
        return integral_numerator**2 / integral_denom

    def calc_center_frequency(self,f_ghz,B,f_range_ghz=None,source_index=0):
        ''' calculate the center frequency of the passband as

            fc = \int_f1^f2 f B(f) S(f) f / \int_f1^f2 B(f) S(f) df

            where B(f) is the spectrum, f is the frequency, S(f) is the the frequency
            dependance of the source = f^-0.7, 3.6, and 1 for synchrotron, dust, and thermal BB
        '''
        if f_range_ghz != None:
            f_ghz, B = self.__cull_range(f_ghz,B,f_range_ghz)

        fc = simps(y=B*f_ghz*f_ghz**source_index,x=f_ghz) / simps(y=B*f_ghz**source_index,x=f_ghz)
        return fc

    def get_fwhm_freqs(self,f_ghz,B,normalize=True,fig_num=1,showplot=False):
        ''' assumes B is linear (not in dB) '''
        if normalize:
            B = B/np.max(B)
        #interpolate up to 1 MHz resolution
        f_interp = np.arange(np.min(f_ghz),np.max(f_ghz),0.001)
        B_interp = np.interp(f_interp,f_ghz,B)
        N = len(B_interp)
        peak_index = np.argmax(B_interp)
        f_low_index = np.argmin(abs(B_interp[0:peak_index]-0.5))
        f_high_index = np.argmin(abs(B_interp[peak_index:]-0.5))
        f_low = f_interp[f_low_index]
        f_high = f_interp[peak_index+f_high_index]
        fwhm = f_high - f_low
        if showplot:
            plt.plot(f_ghz,B,'ko-')
            plt.plot([f_low],B_interp[f_low_index],'ro')
            plt.plot([f_high],B_interp[peak_index+f_high_index],'ro')
            plt.xlabel('Frequency (GHz)')
            plt.ylabel('Response')
        return f_low, f_high, fwhm

class PassbandModel(PassbandMetrics,LoadingCalculation):
    def __init__(self,txtfilename=None,f_ghz=None,B=None,f_range_ghz=None):
        self.txtfilename = txtfilename
        self.f_ghz = f_ghz
        self.B = B
        self.__handle_txtfile(txtfilename)
        self.f_low_ghz, self.f_high_ghz = self.__handle_f_range(f_range_ghz)
        self.bandwidth_ghz = self.calc_bandwidth(self.f_ghz,self.B,f_range_ghz)
        self.f_center_ghz = self.calc_center_frequency(self.f_ghz,self.B,f_range_ghz)

    def __handle_txtfile(self,txtfilename):
        self.txtfile_loaded=False
        if txtfilename != None:
            self.txtfile_loaded = True
            self.n_freqs, self.n_cols, self.header, self.model = self.from_file(txtfilename)
            self.plot_model_from_file(1)
            self.f_ghz = self.model[:,0]
            self.response_index = int(input('select index of desired passband: '))
            self.B = self.model[:,self.response_index]

            # how the hell do you do below without 3 lines of code?
            self.passband_arr = np.zeros((len(self.f_ghz),2))
            self.passband_arr[:,0] = self.f_ghz
            self.passband_arr[:,1] = self.B

        else:
            self.passband_arr=None

    def from_file(self, txtfilename):
        model = np.loadtxt(txtfilename,skiprows=1)
        f=open(txtfilename,'r')
        header_raw = f.readline()
        f.close()
        header = header_raw[0:-1].split('\t')
        n_freqs, n_cols = np.shape(model)
        return n_freqs, n_cols, header, model

    def plot_model_from_file(self,fig_num=1):
        ''' plot all columns in a model file to explore what is in the file '''
        assert self.txtfile_loaded == True, 'This method only plots models loaded from a file'
        plt.figure(fig_num)
        for ii in range(1,self.n_cols):
            plt.plot(self.model[:,0],self.model[:,ii],label=self.header[ii]+' index=%d'%ii)
        plt.xlabel(self.header[0])
        plt.ylabel('Response')
        plt.legend(loc='best')
        plt.show()

    def __handle_f_range(self,f_range_ghz):
        if f_range_ghz == None:
            f_low_ghz = np.min(self.f_ghz)
            f_high_ghz = np.max(self.f_ghz)
        else:
            f_low_ghz = f_range_ghz[0]
            f__ghz = f_range_ghz[1]
        return f_low_ghz, f_high_ghz

    def plot_response(self):
        plt.plot(self.f_ghz,self.B)
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Response (arb)')
        plt.grid()

    def get_PvT(self,T_list_k = [4,5,6,7,8,9,10],showplot=False):
        P = []
        for ii, T_k in enumerate(T_list_k):
            pii = self.thermalPower(self.f_low_ghz*1e9,self.f_high_ghz*1e9,T=T_k,passband_arr=self.passband_arr)
            print(T_k,pii)
            P.append(pii)
        if showplot:
            plt.plot(T_list_k,P,'bo-')
            plt.xlabel('Temperature (K)')
            plt.ylabel('Power (W)')
        return P

    def get_dPvdT(self,T_list_k=[4,5,6,7,8,9,10],showplot=False):
        ''' T_o assumed to be the 0th index (lowest T) '''
        P = self.get_PvT(T_list_k,False)
        dT = np.array(T_list_k) - T_list_k[0]
        dP = np.array(P) - P[0]
        if showplot:
            plt.plot(dT,dP,'bo-')
            plt.xlabel('T - T=%.2fK (K)'%T_list_k[0])
            plt.ylabel('dP (W)')
        return dT,dP

if __name__ == "__main__":
    #path = '/home/pcuser/data/lbird/20210320/fts_raw/modeled_response/' # on velma
    path = '/Users/hubmayr/projects/lbird/HFTdesign/hft_v0/modeled_response/' # on Hannes' machine
    filename='hftv0_hft2_diplexer_model.txt'
    pb = PassbandModel(path+filename)
    #plt.plot(pb.passband_arr[:,0],pb.passband_arr[:,1])
    pb.get_PvT(showplot=True)
    plt.show()
    #pb.get_fwhm_freqs(pb.f_ghz,pb.B,normalize=True,fig_num=1,showplot=True)
    #pb.plot_response()
    #plt.show()
    # plt.plot(pb.model[:,0],pb.model[:,2],'k--')
    # plt.plot(pb.model[:,0],pb.model[:,3],'k--')
    # plt.show()

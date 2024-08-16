'''

tools.py

Useful/general software for detector characterization.

DO TO:
0): check num_pts_per_period for square wave input
1) DETERMINE IF TIME DOMAIN APPROACH OR FREQUENCY DOMAIN APPROACH TO LOCK-IN IS "BETTER"
2) rolling average of lock-in signal
3) ~1-2% bias in lock-in detection.  Probably due to integer # of periods?

'''

import numpy as np
from nasa_client import EasyClient
import matplotlib.pyplot as plt
from scipy.signal import hilbert,square
from scipy.optimize import curve_fit
import time

class SignalAnalysis():
    def get_amplitude_of_sinusoid_high_s2n(self,a):
        ''' return amplitude of a wave.
            Algorithm only good for high signal to noise
        '''
        N=len(a)
        fa = np.fft.fft(a)
        aa = fa.conjugate()*fa
        amp = np.sqrt(2*np.mean(np.real(aa))/N)
        #amp = np.sqrt(2*np.sum(np.real(aa))/N**2)
        return amp

    def get_amplitude_of_sinusoid(self,a,nbins=0,debug=False):
        ''' return amplitude of a real-valued wave.
            Expects "a" is a pure harmonic.  If multiple tones, will return
            the amplitude of the largest tone.

            input:
            a: 1D array-like
            nbins: <int>, number of bins to (left and right) to integrate around max
            debug: <bool>, if True, plot some stuff

            output: amplitude (float)
        '''
        N=len(a)
        fa = np.fft.fft(a)
        aa = fa.conjugate()*fa
        aa=aa.real
        dex = np.where(aa[1:] == np.max(aa[1:]))[0][0]+1 # excluding the zero frequency "peak", which is related to offset
        #dex = np.where(aa==np.max(aa))[0][0]
        aa_slice = aa[dex-nbins:dex+nbins+1]
        amp = np.sqrt(4*np.sum(aa_slice))/N

        if debug:
            print('recovered amplitude = ',amp)
            indices=range(N)
            plt.plot(aa,'bo')
            plt.plot(dex,aa[dex],'r*')
            plt.plot(indices[dex-nbins:dex+nbins+1],aa_slice,'go')
            plt.show()
        return amp

    def get_fundamental_frequency(self,a,samp_int):
        ''' return the strongest harmonic component of input array a given the sampling interval ''' 
        N = len(a)
        ref_fft = np.fft.fft(ref_sq)
        ref_psd = ref_sq_fft*ref_sq_fft.conj()
        dex = np.argmax(ref_psd[1:N//2])+1 # avoid DC term
        fft_freqs = np.fft.fftfreq(N,samp_int)
        return fft_freqs[dex]

    def get_amplitude_of_squarewave(self,a,debug=False):
        ''' return amplitude of a real-valued square wave.

            input:
            a: 1D array-like
            nbins: <int>, number of bins to (left and right) to integrate around max
            debug: <bool>, if True, plot some stuff

            output: amplitude (float)
        '''
        N=len(a)
        fa = np.fft.fft(a)
        aa = fa.conjugate()*fa
        aa=aa.real
        amp = np.sqrt(np.sum(aa[1:]))/N

        if debug:
            print('recovered amplitude = ',amp)
            print('max - min / 2 = ',((np.max(a)-np.min(a))*.5))
            fig,ax = plt.subplots(2)
            ax[0].plot(a,'bo-')
            ax[0].set_ylabel('time domain')
            ax[1].plot(aa,'bo-')
            ax[1].set_ylabel('frequency domain')
            plt.show()
        return amp

    def fit_sin(self,yy):
        '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
        tt = np.array(range(len(yy)))
        yy = np.array(yy)
        ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
        Fyy = abs(np.fft.fft(yy))
        guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
        guess_amp = np.std(yy) * 2.**0.5
        guess_offset = np.mean(yy)
        guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

        def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
        popt, pcov = curve_fit(sinfunc, tt, yy, p0=guess)
        A, w, p, c = popt
        f = w/(2*np.pi)
        fitfunc = lambda t: A * np.sin(w*t + p) + c
        return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

    def get_num_points_per_period(self,arr,fit_wave=False,debug=False):
        ''' return the number of points in "arr" which corresponds to one
            period of the wave.  Note that fit_wave=True is required if
            a contains <~10-20 periods

            input::
            arr: 1d array
            fit_wave: <bool> if true "arr" will be fit to a sinusoid
            debug: <bool>, if true, plot some stuff

            return N <int> the number of points which correspond to one period of the wave "arr"
        '''
        if fit_wave:
            fit_out = self.fit_sin(arr)
            period = fit_out['period']
        else:
            ff = np.fft.fftfreq(len(arr))
            Fyy = abs(np.fft.fft(arr))
            freq = abs(ff[np.argmax(Fyy[1:])+1]) # excluding the zero frequency "peak", which is related to offset
            period = 1/freq
        N = int(round(period))
        if debug:
            fig = plt.figure()
            plt.plot(arr,'o-')
            plt.plot(range(0,N),arr[0:N],'o-')
            plt.title('Period of wave')
            plt.show()
        return N

    def get_num_points_per_period_squarewave(self,arr,threshold=0.5,debug=False):
        arr_diff = np.diff(arr)
        pp = np.max(arr_diff)
        pos_dex = np.where(arr_diff > 0)
        #indices = np.where(arr_diff > arr_diff.mean() * threshold)[0]+1 # +1 since diff makes one less pt
        indices = np.where(arr_diff > arr_diff.max()*threshold)[0]+1 #np.max(arr_diff)*threshold)[0]+1 # +1 since diff makes one less pt
        indices = indices[1::2]
        num_pts_per_period_arr = np.diff(indices)
        num_pts_per_period = int(round(num_pts_per_period_arr.mean()))

        if debug:
            print('Number of points per period list: ',num_pts_per_period_arr)
            fix,ax = plt.subplots(2)
            ax[0].plot(arr,'bo-')
            ax[0].plot(indices,arr[indices],'r*')
            ax[1].plot(arr_diff,'bo-')
            ax[1].plot(indices-1, arr_diff[indices-1],'r*')
            ax[1].axhline(arr_diff.max()*threshold)
            plt.show()
        return num_pts_per_period

    def lowpass(self, x, alpha=0.001):
        data = [x[0]]
        for a in x[1:]:
            data.append(data[-1] + (alpha*(a-data[-1])))
        return np.array(data)

class SoftwareLockIn(SignalAnalysis):
    def lockin_func(self,sig,ref,threshold=0.5,debug=False):
        ''' 
            DSP quadrature detection using software generated I and Q sine waves based on sampled square wave.
            Algorithm forces an integer number of modulation periods and as such windowing is not required.
            new by JH on 5/17/2024
            
            sig: signal, 1D array
            ref: sampled square wave 1D array
            threshold: <float> with range between 0 and 1.  Detection threshold to define a square wave edge.  
                    A square wave edge is defined if the difference between successively sampled points 
                    is > threshold x peak_to_peak of square wave
            debug: <bool>, show plots for debugging

            return I,Q (in-phase and quadrature detection in amplitude units (not rms or peak!!))

            
        '''
        assert threshold>0 and threshold<1,'threshold out of range. 0 < threshold < 1'
        N = len(ref)
        a = ref - ref.mean()
        pp = a[a>0].mean() - a[a<0].mean() # get estimate of peak to peak square wave reference 
        ref_diff = np.diff(ref) 
        pos_dex = np.where(ref_diff>pp*threshold)[0] # find indices where signal goes from negative to positive
        neg_dex = np.where(ref_diff<-1*pp*threshold)[0] # find indices where signal goes from positive to negative

        assert len(pos_dex)>1 or len(neg_dex)>1,'Less than one modulation period in timestream.  Increase data acquisition!'
    
        # determine if 1st edge is positive or negative
        if pos_dex.min()<neg_dex.min():
            ref_sign = 1
        else:
            ref_sign = -1
    
        dex_min = np.hstack((pos_dex,neg_dex)).min() + 1 # find sample *right after* 1st edge
        n = np.hstack((np.diff(pos_dex),np.diff(neg_dex))).mean() # average samples per period
        n_per = (N-dex_min)//n # number of whole periods from dex_min to N
    
        sig_cut = sig[dex_min:dex_min+round(n_per*n)]
        ref_cut = ref[dex_min:dex_min+round(n_per*n)] # keep only integer periods
    
        N_p = len(sig_cut)
        ref_amp = np.sin(2*np.pi*n_per*np.linspace(0,1,N_p))*ref_sign
        rh = hilbert(ref_amp)
    
        sig_cut_mean_rm = sig_cut - sig_cut.mean()
        It = rh.real*sig_cut_mean_rm
        Qt = rh.imag*sig_cut_mean_rm
    
        I = 2*It.mean() # factor of 2 for proper normalization
        Q = 2*Qt.mean()
    
        if debug:
            amp=np.sqrt(I**2+Q**2)
            fig,ax = plt.subplots()
            ax.plot(sig,'b.')
            ax.plot(ref,'g.')
            ax.plot(range(N_p)+dex_min,sig_cut,'k--')
            ax.plot(range(N_p)+dex_min,ref_cut,'r-')
            ax.plot(pos_dex,ref[pos_dex],'*')
            ax.plot(neg_dex,ref[neg_dex],'*')
            ax.legend(('sig','ref','sig lock-in','ref lock-in','pos','neg'))
            ax.set_title('raw data')
        
            fig2,ax2 = plt.subplots()
            ax2.plot(sig_cut,'.-')
            ax2.plot(ref_cut,'.-')
            ax2.plot(rh.real*sig_cut_mean_rm.max(),'-')
            ax2.legend(('sig','ref','Imix'))
            ax2.set_title('lock-in data')
        
            fig3,ax3 = plt.subplots()
            ax3.plot(sig_cut_mean_rm,'.')
            ax3.plot(amp*rh.real,'-')
            ax3.plot(amp*rh.imag,'-')
            ax3.legend(('sig','Imix','Qmix'))
            ax3.set_title('lock-in data')
        
        return I,Q

    def _handle_integer_periods(self,integer_periods,num_pts_per_period,sig,ref,debug):

        if integer_periods:
            if num_pts_per_period is not None:
                pass
            else:
                #num_pts_per_period = self.get_num_points_per_period(ref,fit_wave=True,debug=debug)
                num_pts_per_period = self.get_num_points_per_period_squarewave(ref,threshold=0.5,debug=False)
            N = len(sig)//num_pts_per_period * num_pts_per_period
            sig=sig[0:N]
            ref=ref[0:N]
        else:
            N = len(sig)
        return sig, ref, N

class SoftwareLockinAcquire(SoftwareLockIn):
    ''' One column data acquisition to lock-into a small signal.
        The signal comes form one dfb card input.
        The references comes from another dfb card input
    '''
    def __init__(self, easy_client=None, signal_column_index=0,reference_column_index=1,
                 signal_feedback_or_error='feedback'):
        '''
        input:
        easy_client: instance of easyClient
        signal_column_index: column index in EasyClient.getNewData return which corresponds to the signal.
        reference_column_index: column index in EasyClient.getNewData return which corresponds to the reference
        signal_feedback_or_error: 'feedback' or 'error', defines which vector describes the signal.
                                  Typical use case is 'feedback', which is default.
        '''
        # constant globals
        self.ref_pp_min = 100

        self.ec = self._handle_easy_client_arg(easy_client)
        self._check_num_cols_()
        self.sig_col_index = signal_column_index
        self.ref_col_index = reference_column_index
        #self.num_pts_per_period = self._handle_pts_per_period_arg(num_pts_per_period)
        self.signal_feedback_or_error = signal_feedback_or_error
        self.sig_index = self._handle_feedback_or_error_arg(signal_feedback_or_error)

    def _check_num_cols_(self):
        assert self.ec.numColumns>1, 'SoftwareLockinAcquire needs a minimum of two 2 columns.  Have you selected fibers correctly in Dastard Commander?'

    def _handle_feedback_or_error_arg(self, signal_feedback_or_error):
        if signal_feedback_or_error == 'feedback':
            dex = 1
        elif signal_feedback_or_error == 'error':
            dex = 0
        else:
            print('unknown signal_feedback_or_error: ',signal_feedback_or_error)
        return dex

    def _handle_easy_client_arg(self, easy_client):
        if easy_client is not None:
            return easy_client
        easy_client = EasyClient()
        easy_client.setupAndChooseChannels()
        return easy_client

    # def _handle_pts_per_period_arg(self,num_pts_per_period):
    #     if num_pts_per_period is not None:
    #         num_pts = num_pts_per_period
    #     else:
    #         ii = 0
    #         retries=10
    #         while ii < retries:
    #             data = self.ec.getNewData(minimumNumPoints=10000)
    #             if isinstance(data, (np.ndarray, np.generic)):
    #                 break
    #             else: 
    #                 print('easy_client getNewData failed on attempt number %d'%ii)
    #                 time.sleep(.1)
    #                 ii+=1 
    #                 continue 
    #         if ii==retries:
    #             raise Exception('getData failed, likely due to dropped packets')
    #         arr = data[self.ref_col_index,0,:,0]
    #         arr_pp = np.max(arr) - np.min(arr)
    #         assert arr_pp > self.ref_pp_min, print('Reference signal amplitude (%.1f )is too low.  Increase and try again.'%(arr_pp/2))
    #         #num_pts = self.get_num_points_per_period(arr,fit_wave=False,debug=False)
    #         num_pts = self.get_num_points_per_period_squarewave(arr,debug=False)
    #     return num_pts

    def getData(self, minimumNumPoints=390000, threshold=0.5, debug=False):
        '''
        Acquire data and return the locked in signal for each row in one column of data.

        input:
        minimumNumPoints: mimimum number of points to grab in raw buffer.  The exact number will be taken.  
        threshold: <float> with range between 0 and 1.  Detection threshold to define a square wave edge.  
                    A square wave edge is defined if the difference between successively sampled points 
                    is > threshold x peak_to_peak of square wave.  
        debug: boolean, if true plot some stuff
        
        output: iq_arr, n_row x 2 numpy.array of I and Q
        '''
        
        dataOut = self.ec.getNewData(delaySeconds = 0.001, minimumNumPoints = minimumNumPoints, exactNumPoints = False) # dataOut[col,row,frame,error=0/fb=1]
        iq_arr = np.empty((self.ec.numRows,2))
        for ii in range(self.ec.numRows):
            iq_arr[ii,:] = np.array(self.lockin_func(dataOut[self.sig_col_index,ii,:,self.sig_index],dataOut[self.ref_col_index,ii,:,0],threshold=threshold,debug=False))

        if debug:
            for ii in range(self.ec.numRows):
                ref_amp = self.get_amplitude_of_squarewave(dataOut[self.ref_col_index,ii,:,0],debug=False)
                #ref_amp = self.get_amplitude_of_sinusoid(dataOut[reference_column_index,ii,:,0],nbins=2,debug=True)
                sig_amp = self.get_amplitude_of_sinusoid(dataOut[self.sig_col_index,ii,:,self.sig_index],nbins=2,debug=False)
                lock_in_amp = np.sqrt(iq_arr[ii,0]**2+iq_arr[ii,1]**2)
                print('Row index %02d: (I, Q) = (%.3e,%.3e)'%(ii,iq_arr[ii,0],iq_arr[ii,1]))
                print('lock-in amplitude: ',lock_in_amp,'\nsignal amplitude: ',sig_amp, '\nref_amp: ',ref_amp)

                ref = dataOut[self.ref_col_index,ii,:,0] 
                ref_ac = ref - np.mean(ref)
                sig = dataOut[self.sig_col_index,ii,:,self.sig_index]
                sig_ac = sig - np.mean(sig) 

                fig,ax = plt.subplots(2,1)
                fig.suptitle('Row index = %02d'%ii)
                ax[0].plot(sig,label='signal')
                ax[0].plot(ref,label='reference (0-4095,1V)')
                ax[0].set_ylabel('raw counts')
                ax[0].legend()
                ax[1].plot(sig_ac,label='signal')
                ax[1].plot(ref_ac/np.max(ref_ac)*np.max(sig_ac),label='reference')
                ax[1].set_xlabel('sample #')
                plt.show()
        return iq_arr

# functions used in debugging -----------------------------
def make_simulated_lock_in_data(sig_params=[3,5,0,0],ref_params=[1,5,0,0],N=1024,noise_to_signal=0,ref_type='sine',plotfig=False):
    ''' sig/ref_params[amplitude, number of periods, phase, DC offset] '''
    t = np.linspace(0,1,N,endpoint=False)
    # reference square wave
    if ref_type == 'square':
        ref = ref_params[0]*square(2*np.pi*ref_params[1]*t+ref_params[2])+ref_params[3]
    elif ref_type == 'sine':
        ref = ref_params[0]*np.sin(2*np.pi*ref_params[1]*t+ref_params[2])+ref_params[3]

    noise = np.random.normal(0,noise_to_signal*sig_params[0],(N))
    sig = sig_params[0]*np.sin(2*np.pi*sig_params[1]*t+sig_params[2]) + sig_params[3] + noise

    if plotfig:
        plt.plot(sig)
        plt.plot(ref)
        plt.show()
    return sig, ref

def test_lockin_func(f=7,phase=0,s2n=10,s2n_ref=10,threshold=0.5):
    sl = SoftwareLockIn()

    # simulate a f Hz signal and reference for lock-in 
    N=1000
    A=10    
    offset=50
    t=np.linspace(0,1,N)
    sig=A*(np.sin(2*np.pi*f*t-phase)+2/s2n*(np.random.random(N)-0.5))+offset

    # make a fake sampled square wave
    pp_actual = 10
    offset_actual = 0
    ref = (square(2*np.pi*f*np.linspace(0,1,N))+(np.random.random(N)-0.5)/s2n_ref + 1)/2*pp_actual+offset_actual
    I,Q=sl.lockin_func(sig=sig,ref=ref,threshold=threshold,debug=True)
    print('Input signal amp ',A)
    print('Input signal phase',phase)
    print('I,Q,amp,phase: ',I,Q,np.sqrt(I**2+Q**2),np.arctan(Q/I))


def test_lockin_acq():
    sla = SoftwareLockinAcquire(signal_feedback_or_error='error')
    sla.getData(minimumNumPoints=1000, window=False,debug=True)

def test_get_num_points_per_period_squarewave():
    sig,ref = make_simulated_lock_in_data(sig_params=[3,5,0,0],ref_params=[1,5,0,0],N=1024,noise_to_signal=0,ref_type='square',plotfig=False)
    SignalAnalysis().get_num_points_per_period_squarewave(ref,debug=True)

if __name__ == "__main__":
    sla = SoftwareLockinAcquire()
    sampling_rate = int(125e6/142/4) # number of samples per second
    f = 20 # Hz 
    samples_per_period = int(sampling_rate/f)
    print(samples_per_period*20)
    sla.getData(minimumNumPoints=20*samples_per_period, debug=True)

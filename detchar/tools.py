'''

tools.py

Useful/general software for detector characterization.

DO TO:
1) DETERMINE IF TIME DOMAIN APPROACH OR FREQUENCY DOMAIN APPROACH TO LOCK-IN IS "BETTER"
2) WRITE FUNCTION TO USE SQUARE WAVE REFERENCE
3) MAKE FUNCTION THAT ACCEPTS easy_client return data structure as input to lock-in
4) IMPROVE LOCK-IN DATA ACQUISITION FUNCTION
5) CLEAN UP UN-NEEDED FUNCTIONS IN THIS FILE
6) rolling average of lock-in signal 

'''

import numpy as np
#from nasa_client import EasyClient
import matplotlib.pyplot as plt
from scipy.signal import hilbert,square
from scipy.optimize import curve_fit

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

    def get_amplitude_of_sinusoid(self,a,nbins=0,debug=True):
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
        #dex = np.where(aa==np.max(aa[1:]))[0][0]+1 # excluding the zero frequency "peak", which is related to offset
        dex = np.where(aa==np.max(aa))[0][0]
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

class SoftwareLockIn(SignalAnalysis):
    def lockin_reference_is_sine(self,sig,ref,window=True,integer_periods=True,num_pts_per_period=None,debug=False):
        ''' software lock in using sine-wave mixing in time domain

            sig: 1d array, raw signal
            ref: 1d array, reference signal
            window: <bool>, if True apply hanning window to data before lock-in
            integer_periods: <bool>, if True truncate the data to an integer number of periods before lock-in.
                             A bias results from locking into a non-integer number of periods.
                             Thus only set to false if you are passing an integer number of periods in sig and ref.
            num_pts_per_period: only used if integer_periods=True.  If None, determine this from the reference signal.
            debug: <bool>, if True, plot some stuff

            return: I, Q (single points for *all* of sig)

        '''
        if integer_periods:
            if num_pts_per_period is not None:
                pass
            else:
                num_pts_per_period = self.get_num_points_per_period(ref,fit_wave=True,debug=debug)
            N = len(sig)//num_pts_per_period * num_pts_per_period
            sig=sig[0:N]
            ref=ref[0:N]
        else:
            N = len(sig)

        sig = sig - sig.mean()
        ref = ref - ref.mean()
        ref = ref/self.get_amplitude_of_sinusoid(ref,nbins=1,debug=False) # make reference wave from +/- 1

        rh = hilbert(ref)
        if window:
            x=sig*rh.real*np.hanning(N)
            y=sig*rh.imag*np.hanning(N)
            I = np.mean(x)*4
            Q = np.mean(y)*4
        else:
            x=sig*rh.real
            y=sig*rh.imag
            I = np.mean(x)*2
            Q = np.mean(y)*2

        if debug:
            print('I=',I,'\nQ=',Q)
            fig, ax = plt.subplots(3)
            ax[0].plot(sig,'o-')
            ax[0].set_title('Signal')
            ax[1].plot(rh.real)
            ax[1].plot(rh.imag)
            ax[1].set_title('Analytic reference')
            ax[1].legend(('real','imag'))
            ax[2].set_title('mix (prefilter). Window = %s'%(window))
            ax[2].plot(x)
            ax[2].plot(y)
            ax[2].legend(('I','Q'))
            plt.show()
        return I,Q

    def lockin_reference_is_square(self,sig,ref,window=True,integer_periods=True,num_pts_per_period=None,debug=False):
        print('To be written!!!')

    # def lock_in_1d(self,sig, mix, mix_type='square',sig_type='sine',debug=True):
    #     ''' inspired by J. McMahon
    #
    #         input:
    #             sig: raw input signal wave
    #             mix: reference wave
    #             mix_type: 'square' or 'sine', this affects the normalization
    #             sig_type: 'square' or 'sine', this affects the normalization
    #             data_pts_factor2: <bool> if true, truncate data to highest factor of 2 for speed
    #             debug:<bool> print some stuff if True
    #
    #         output: tuple
    #             I: real part of transfer function
    #             Q: imaginary part of transfer function
    #     '''
    #
    #     if mix_type=='square':
    #         mix_norm = 0.5
    #     elif mix_type=='sine':
    #         mix_norm = 2.0
    #     if sig_type=='square':
    #         sig_norm = 0.5
    #     elif sig_type=='sine':
    #         sig_norm = 2.0
    #
    #     N = len(mix)
    #
    #     # remove DC offset
    #     sig = sig - np.mean(sig)
    #     mix = mix - np.mean(mix)
    #
    #     # 90 degree phase shift of reference.
    #     rh = hilbert(mix) # make the `analytic signal' signal of the reference.  x+iy, now a complex array
    #     fr = np.fft.fft(rh,N)  # mix frequency spectrum
    #     fs = np.fft.fft(sig,N) # signal frequency spectrum
    #
    #     # auto and cross correlations
    #     ss = fs.conjugate()*fs # autocorrelation of signal
    #     rr = fr.conjugate()*fr # autocorrelation of reference
    #     qs = fr.conjugate()*fs # cross correlation of 'analytic reference' and signal
    #
    #     mix_amp = np.sqrt(mix_norm*np.mean(np.real(rr))/N)
    #     sig_amp = np.sqrt(sig_norm*np.mean(np.real(ss))/N)
    #
    #     I = np.mean(qs.real) # in-phase
    #     Q = np.mean(qs.imag) # quad
    #
    #     if debug:
    #         print('mix_amp: ',mix_amp)
    #         print('sig_amp:',sig_amp)
    #         print('I: ',I)
    #         print('Q: ',Q)
    #         plt.plot(qs.real,label='cross-real')
    #         plt.plot(qs.imag,label='cross-imag')
    #         plt.plot(ss.real,label='auto-real')
    #         plt.legend()
    #     return I, Q

    def software_lock_in(self, v_signal, v_reference, reference_type='square',response_type='sine'):
        ''' lockin algorithm written by J. McMahon

            input:
                v_signal: the output signal
                v_reference: the reference
                mix_type: 'square' or 'sine', this affects the normalization

            output: tuple

                I: real part of transfer function
                Q: imaginary part of transfer function
                v_reference_amp = amplitude of reference signal

            NOTES:
            To get the the amplitude of the locked in signal compute: sqrt(I**2+Q**2)/v_reference_amp
            algorithm checked for normalization on 5/12/2012

            BUG KNOWN: 'SQUARE' 'SQUARE' CONFIGURATION GIVES THE WRONG AMPLITUDE NORMALIZATION IF
                       THE PHASE IS DIFFERENT THAN ZERO

        '''

        if reference_type=='square':
            v_ref_norm = 1
            IQ_norm = np.pi/2
        elif reference_type=='sine':
            v_ref_norm = 2
            IQ_norm = 2

        if response_type == 'sine':
            res_norm = 1
        elif response_type == 'square':
            res_norm = 2/np.pi

        # perhaps crappy way to make a 1D array compatible with the rest of the function (need 1 x Nsamp array)
        theshape=np.shape(v_reference)
        if len(theshape)==1:
            N = len(v_reference)  # number of samples
            dummy = np.empty((1,N))
            dummy[0,:]=v_signal
            v_signal = dummy
            #v_signal = dummy[:N/2] # take half
            dummy = np.empty((1,N))
            dummy[0,:]=v_reference
            v_reference = dummy
            #v_reference = dummy[::2] # double rate

        dex = 1
        Nd,N=np.shape(v_reference)

        # only take highest power of 2 of the data (faster!)
        N = 2**(int(np.log2(N)))
        v_signal = v_signal[:,0:N]
        v_reference = v_reference[:,0:N]

        # remove the DC offset from signal and reference as well as create hanning window
        ref_mean = np.mean(v_reference,dex)
        dat_mean = np.mean(v_signal,dex)
        ref=np.empty((Nd,N))
        dat=np.empty((Nd,N))
        hanning_window = np.hanning(N)
        hw = np.empty((Nd,N))
        for i in range(Nd):
            ref[i] = v_reference[i]-ref_mean[i]
            dat[i] = v_signal[i]-dat_mean[i]
            hw[i] = hanning_window


        # 90 degree phase shift of reference.  Due to limitation of python2.6.5 scipy.signal.hilbert, I cannot
        # take the hilbert transform of a matrix, just a 1D array.  I need to construct the matrix.  Technically,
        # I should take the reference of each column instead of using one for all.  However, doing this would
        # be slow, and the data is sampled so quickly, it probably doesn't matter

        rh = hilbert(ref[0]) # now a complex array
        ref_hil = np.empty((Nd,N),complex)
        for ii in range(Nd):
            ref_hil[ii] = rh

        # take FFTs
        # fr = np.fft.fft(hw*ref_hil,N)
        # fs = np.fft.fft(hw*dat,N)

        fr = np.fft.fft(ref_hil,N)
        fs = np.fft.fft(dat,N)

        # auto and cross correlations (8/3 accounts for power removed due to hanning window)
        # rr = fr.conjugate()*fr*8/3. # autocorrelation of reference
        # qs = fr.conjugate()*fs*8/3. # cross correlation of 'analytic reference' and signal
        rr = fr.conjugate()*fr # autocorrelation of reference
        qs = fr.conjugate()*fs # cross correlation of 'analytic reference' and signal


        v_reference_amp = np.sqrt(np.mean(np.real(rr),dex)/N*v_ref_norm) # somewhat mislabeled, only true if one harmonic present

        #v_signal_amp = np.sqrt(ss.real.mean()/N*2) # ditto here
        I = np.mean(qs.real,dex)/N*np.sqrt(2)*IQ_norm*res_norm # transfer function in phase component
        Q = np.mean(qs.imag,dex)/N*np.sqrt(2)*IQ_norm*res_norm # transfer function out of phase component

        return I, Q, v_reference_amp

    def getNewData_lockin(self, easy_client, signal_column_index=0,reference_column_index=1,reference_type='square',response_type='sine',
                                     signal_feedback_or_error='feedback', debug=False):
        '''
        Acquire data and return the locked in signal for each row in one column of data.

        input:
        ec: instance of easyClient
        signal_index: column index in EasyClient.getNewData return which corresponds to the signal.
        reference_index: column index in EasyClient.getNewData return which corresponds to the reference
        reference_type: 'square' or 'sine'
        response_type: 'square' or 'sine'
        debug: boolean

        output:
        '''

        #dataOut[col,row,frame,error=0/fb=1]
        if signal_feedback_or_error == 'feedback':
            dex = 1
        elif signal_feedback_or_error == 'error':
            dex = 0
        else:
            print('unknown signal_feedback_or_error: ',signal_feedback_or_error)
        dataOut = easy_client.getNewData(delaySeconds = 0.001, minimumNumPoints = 4000, exactNumPoints = False, sendMode = 0, toVolts=False, divideNsamp=True, retries = 3)
        I,Q,v_ref_amp = self.software_lock_in(dataOut[signal_column_index,:,:,dex],dataOut[reference_column_index,:,:,0],
                                         reference_type=reference_type, response_type = response_type)
        if debug:
            for ii in range(easy_client.numRows):
                plt.figure(ii)
                plt.title('Row index = %02d'%ii)
                plt.plot(dataOut[signal_column_index,ii,:,dex],label='signal')
                plt.plot(dataOut[reference_column_index,ii,:,0],label='reference')

                ref_pp = np.max(dataOut[reference_column_index,ii,:,0]) - np.min(dataOut[reference_column_index,ii,:,0])
                sig_pp = np.max(dataOut[signal_column_index,ii,:,dex]) - np.min(dataOut[signal_column_index,ii,:,dex])
                lock_in_amp = (I**2+Q**2)/v_ref_amp
                print('signal amplitude: ',sig_pp/2, '\nlock-in amplitude: ',lock_in_amp/2,'\nref_pp: ',ref_pp)
                print('Row index %02d: (I, Q, v_ref_amp) = (%.3f,%.3f,.%.3f)'%(ii,I[ii],Q[ii],v_ref_amp[ii]))
        return I,Q,v_ref_amp


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

# def test_software_lock_in():
#     sl=SoftwareLockIn()
#     N=1000
#     t = np.linspace(0,1,N,endpoint=False)
#     # reference square wave
#     ref_freq_hz = 5
#     ref_offset = 0
#     ref_phase = 0
#     ref_amp = 3.0
#     ref = ref_amp*square(2*np.pi*ref_freq_hz*t+ref_phase)+ref_offset
#
#     # signal is a sine wave
#     sig_freq_hz = 5
#     sig_offset = 0
#     sig_phase = 0#np.pi/2.0
#     sig_amp = 3.0
#
#     noise_to_signal = 0
#     noise = (np.random.random(N)-0.5)*noise_to_signal*sig_amp
#     sig = sig_amp*np.sin(2*np.pi*sig_freq_hz*t+sig_phase) + sig_offset + noise
#
#     sh=hilbert(sig)
#     #plt.plot(ref)
#     plt.plot(sh.real)
#     plt.plot(sh.imag)
#     plt.legend(('hil real','hil imag'))
#     plt.show()
#
#     #I,Q,v_ref_amp = software_lock_in(sig, ref, reference_type='square',response_type='sine')
#     I,Q = sl.software_lock_in_1d(sig, mix=ref, mix_type='square',sig_type='sine',data_pts_factor2=False,debug=True)
#     plt.show()
#
#     #lockin_amp = (I**2+Q**2)/v_ref_amp/np.sqrt(2)
#
#     #print('Input signal amplitude = ',sig_amp)
#     #print('lock in signal amplitude = ',lockin_amp)
#
#     plt.plot(t,ref,'o-',label='ref')
#     plt.plot(t,sig,'o-',label='sig')
#     #plt.plot(t,noise)
#
#     plt.show()

def test_lockin_reference_is_sine():
    ''' Test the function lockin_reference_is_sine()

        lessons learned:
        1) In limit of fine sampling and a low number of periods: ~8-10% bias when locking in on a non-integer number of
           periods.  This is likely due to not subtracting the mid-point of the sine wave, as the mean of sig is used to
           remove DC term.  This is therefore an issue with a high S/N wave.
        2) In limit of course sampling and many periods: bias occurs for when using an integer number of periods.
           In practice with "PGE", will never be in this limit since the clock is 125 MHz, and the signals of interest
           are < 100 kHz (or more like <10kHz)
        3) For typical polcal settings, integer_periods = True is demonstrably better than all other methods.
           Applying a hanning window on top of this doesn't provide added benefit
        4) For S/N = 1/10 all methods provide <10% error
        5) Number of periods: likely just a single one is fine!
           For 100 S/N=1 realizations, the percent error is: -0.076 +/- 1.766
           For 100 S/N=0.5 realizations, the percent error is: 0.15 +/- 3.03
           For 100 S/N=0.1 realizations, the percent error is: -0.263 +/- 15.507
           See the pattern here?  For an uncertainty of ~2%, you need the equivalent of one lock-in period at S/N=1
    '''

    sl = SoftwareLockIn()

    signal_amp = 1.0
    samp_int = 25e-6 # 25 micro seconds is typical
    f = 5 # typical chop frequency for polcal
    num_periods = 1.1
    N = int(num_periods/f/samp_int) # number of samples
    print(N)

    #n_to_s = np.linspace(0,10,21) # noise to signal vector
    n_to_s = np.ones(100)*10
    cases = [[False,False],[True,False],[False,True],[True,True]]
    #cases = [[False,False]]

    data_list=[]
    for ii,n2s in enumerate(list(n_to_s)):
        sig,ref = make_simulated_lock_in_data(sig_params=[signal_amp,num_periods,np.pi/3,0],ref_params=[1,num_periods,0,0],N=N,noise_to_signal=n2s,ref_type='sine',plotfig=False)
        data_list.append([sig,ref])

    results = []
    for case in cases:
        print(case)
        tmp_arr = np.empty((len(n_to_s),4))
        for ii,n2s in enumerate(data_list):
            I,Q=sl.lockin_reference_is_sine(n2s[0],n2s[1],window=case[0],integer_periods=case[1],num_pts_per_period=None,debug=False)
            amp = np.sqrt(I**2+Q**2)
            err_pct = ((amp - signal_amp)/amp)*100
            print('amplitude =',amp, '% diff = ',err_pct)
            tmp_arr[ii,:] = np.array([I,Q,amp,err_pct])
        results.append(tmp_arr)

    for ii,result in enumerate(results):
        plt.plot(n_to_s,result[:,3])
        print(result[:,3].mean(),'+/-',result[:,3].std())
    plt.xlabel('Noise/Signal')
    plt.ylabel('% error in amplitude measurement')
    plt.legend(('basic','window','int periods','window and int periods'))
    plt.show()

def test_get_num_points_per_period():
    N = 1001
    y = np.empty((N,2))
    num_periods = np.linspace(1,100,N)
    #t = time()
    for ii in range(N):
        #print(time()-t)
        sig,ref = make_simulated_lock_in_data(sig_params=[1,num_periods[ii],0,0],ref_params=[1,5,0,0],N=1000,noise_to_signal=0,ref_type='sine',plotfig=False)
        N1 = get_num_points_per_period(sig,fit_wave=False,debug=False)
        N2 = get_num_points_per_period(sig,fit_wave=True,debug=False)
        y[ii,:] = np.array([N1,N2])
    plt.plot(num_periods,y[:,0])
    plt.plot(num_periods,y[:,1])
    plt.show()

if __name__ == "__main__":

    test_lockin_reference_is_sine()

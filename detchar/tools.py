'''

tools.py

Useful/general software for detector characterization

'''

import numpy as np
from nasa_client import EasyClient
import matplotlib.pyplot as plt
from scipy.signal import hilbert

def software_lock_in(v_signal, v_reference, reference_type='square',response_type='sine'):
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
    fr = np.fft.fft(hw*ref_hil,N)
    fs = np.fft.fft(hw*dat,N)

    # auto and cross correlations (8/3 accounts for power removed due to hanning window)
    rr = fr.conjugate()*fr*8/3. # autocorrelation of reference
    qs = fr.conjugate()*fs*8/3. # cross correlation of 'analytic reference' and signal

    v_reference_amp = np.sqrt(np.mean(np.real(rr),dex)/N*v_ref_norm) # somewhat mislabeled, only true if one harmonic present

    #v_signal_amp = np.sqrt(ss.real.mean()/N*2) # ditto here
    I = np.mean(qs.real,dex)/N*np.sqrt(2)*IQ_norm*res_norm # transfer function in phase component
    Q = np.mean(qs.imag,dex)/N*np.sqrt(2)*IQ_norm*res_norm # transfer function out of phase component
    #pylab.plot(v_reference[:100],'.-')
    #pylab.plot(v_signal[:100], '.-')
    #pylab.show()

    return I, Q, v_reference_amp

def software_lock_in_acquisition(easy_client, signal_column_index=0,reference_column_index=1,reference_type='square',response_type='sine',
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
    I,Q,v_ref_amp = software_lock_in(dataOut[signal_column_index,:,:,dex],dataOut[reference_column_index,:,:,0],
                                     reference_type=reference_type, response_type = response_type)
    if debug:
        
        for ii in range(easy_client.numRows):
            plt.figure(ii)
            plt.title('Row index = %02d'%ii)
            plt.plot(dataOut[signal_column_index,ii,:,dex],label='signal')
            plt.plot(dataOut[reference_column_index,ii,:,0],label='reference')
            print('Row index %02d: (I, Q, v_ref_amp) = (%.3f,%.3f,.%.3f)'%(ii,I[ii],Q[ii],v_ref_amp[ii]))
    return I,Q,v_ref_amp

if __name__ == "__main__":
    easy_client = EasyClient()
    easy_client.setupAndChooseChannels()
    #attrs = vars(easy_client)
    #print(', '.join("%s: %s" % item for item in attrs.items()))

    I,Q,v_ref_amp = software_lock_in_acquisition(easy_client,signal_feedback_or_error='error',debug=True)
    plt.show()


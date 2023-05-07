''' czAcquire.py

    @authors GCJ and JH 5/2023, based polcal.py.

    Typical hardware setup: 
    1) Agilent33220A output through resistor box (10K) into detector bias line
    2) Agilent33220A sync into DFB card CH2 input.  This corresponds to reference_column_index=1.
    3) Detector column signal corresponds to DFB card CH1 input/output.
'''

from nasa_client import EasyClient
from instruments import BlueBox, Agilent33220A 
from adr_gui.adr_gui_control import AdrGuiControl
#from iv_data import ComplexImpedanceSweepData

import time
import numpy as np
import pylab as plt
import progress.bar
import os
from tools import SoftwareLockinAcquire
import matplotlib.pyplot as plt

def test_for_signal(N_lockins=5):
    cz = ComplexImpedanceSweep(source_amp_volt=3.0, 
                                source_offset_volt=1.5, 
                                source_frequency_hz=np.logspace(0,5,50),
                                num_lockin_periods = 10,
                                row_order=None,
                                )
    
    N=N_lockins
    iq_arr = []
    for ii in range(N):
        iq_arr.append(cz.get_point()) # nrow x 2 array
    
    cz.source.SetOutput(outputstate='off')

    iq_arr2 = []
    for ii in range(N):
        iq_arr2.append(cz.get_point())

    nrow,col = np.shape(iq_arr[0])
    
    for ii in range(nrow):
        plt.figure(ii)
        for jj in range(N):
            amp_on = np.sqrt(iq_arr[jj][ii,0]**2 + iq_arr[jj][ii,1]**2)  
            amp_off = np.sqrt(iq_arr2[jj][ii,0]**2 + iq_arr[jj][ii,1]**2)  
            plt.plot(jj,amp_on,'bo')
            plt.plot(jj,amp_off,'ro')
        plt.legend(('source on','source off'))
        plt.xlabel('measurement index')
        plt.ylabel('Amplitude response')
        plt.title('Row index %d'%(ii))

    plt.show()
    
class SineSweep():
    ''' Sweep constant amplitude sine wave as a function of frequency.  
        Used for example in a single detector bias position for a complex impedance or complex responsivity 
        measurement.
    '''
    def __init__(self,
                 source_amp_volt=0.04, source_offset_volt=0, source_frequency_hz=np.logspace(1,3,50),
                 num_lockin_periods = 10,
                 row_order=None):

        self.source = Agilent33220A()
        self.adr_gui_control = AdrGuiControl() #self._handle_adr_gui_control_arg(adr_gui_control)
        self.init_source(source_amp_volt, source_offset_volt, source_frequency_hz[0])

        # hardware and class initialization
        self.sla = SoftwareLockinAcquire(easy_client=None, signal_column_index=0,reference_column_index=1,
                                         signal_feedback_or_error='feedback',num_pts_per_period=None)

        # input parameters
        self.source_amp_v = source_amp_volt
        self.source_offset_v = source_offset_volt
        self.source_freq_hz = source_frequency_hz
        self.num_lockin_periods = num_lockin_periods
        self.waittime_s = 0.1 # time to wait between setting new frequency and acquiring data
        self.num_freq = len(self.source_freq_hz)
        self.row_order = self._handle_row_to_state(row_order)
        self.freq_hz_measured = None  
        self.iq_v_freq = None

    def _handle_row_to_state(self,row_order):
        if row_order is not None:
            return row_order
        return list(range(self.sla.ec.nrow))

    def init_source(self, amp, offset, frequency):
        self.source.SetFunction(function = 'sine')
        self.source.SetLoad('INF')
        self.source.SetFrequency(frequency)
        self.source.SetAmplitude(amp)
        self.source.SetOffset(offset)
        self.source.SetOutput(outputstate='on')

    def get_point(self,window=False):
        return self.sla.getData(num_periods=self.num_lockin_periods, window=window,debug=False)

    def take(self, extra_info = {}, move_to_zero_at_end = True, turn_off_source_on_end = True):
        pre_time = time.time()
        pre_temp_k = self.adr_gui_control.get_temp_k()

        freq_hz_measured = []
        iq_v_freq = np.empty((self.num_freq,self.sla.ec.nrow,2))
        
        for ii, freq in enumerate(self.source_freq_hz):
            print('Setting function generator to frequency = %.2f Hz'%freq)
            self.source.SetFrequency(freq)
            freq_hz_measured.append(self.source.GetFrequency())
            iq_arr = self.get_point() # nrow x 2 array
            iq_v_freq[ii,:,:] = iq_arr
        post_temp_k = self.adr_gui_control.get_temp_k()
        post_time = time.time()

        print('Acquisition complete.')
        if turn_off_source_on_end: self.source.SetOutput('off')

        self.freq_hz_measured = freq_hz_measured 
        self.iq_v_freq = iq_v_freq 

        # return ComplexImpedanceSweepData(iq_v_freq = iq_v_freq,
        #                               row_order=self.row_order,
        #                               #bayname=self.bayname,
        #                               #db_cardname=self.db_cardname,
        #                               column_number=0,
        #                               source_amp_volt=self.source_amp_v,
        #                               source_offset_volt=self.source_offset_v,
        #                               source_frequency_hz=self.source_freq_hz,
        #                               pre_temp_k=pre_temp_k,
        #                               post_temp_k=post_temp_k,
        #                               pre_time_epoch_s=pre_time,
        #                               post_time_epoch_s=post_time,
        #                               extra_info=extra_info)

    def plot(self,fignum=1):
        fig, ax = plt.subplots(nrows=2,ncols=2,sharex=False,figsize=(12,8),num=fignum)
        for ii in range(self.sla.ec.nrow):
            ax[0][0].plot(self.freq_hz_measured,self.iq_v_freq[:,ii,0],'o-')
            ax[0][1].plot(self.freq_hz_measured,self.iq_v_freq[:,ii,1],'o-')
            ax[1][0].plot(self.freq_hz_measured,self.iq_v_freq[:,ii,0]**2+self.iq_v_freq[:,ii,1]**2,'o-')
            ax[1][1].plot(self.freq_hz_measured,np.arctan(self.iq_v_freq[:,ii,1]/self.iq_v_freq[:,ii,0]),'o-')

        # axes labels
        ax[0][0].set_ylabel('I')
        ax[0][1].set_ylabel('Q')
        ax[1][0].set_ylabel('I^2+Q^2')
        ax[1][1].set_ylabel('Phase')
        ax[1][0].set_xlabel('Freq (Hz)')
        ax[1][1].set_xlabel('Freq (Hz)')    

        ax[1][1].legend(list(range(self.sla.ec.nrow)))

if __name__ == "__main__":
    ss = SineSweep()
    ss.take()
    ss.plot()
    plt.show()


''' complexImpedanceAcquire.py

    Acquire data for complex impedance measurements
    Work in progress!

    @author GCJ 1/2022, based Hannes's polcal.py, based on Jay's software for legacy electronics.
'''

from nasa_client import EasyClient
from instruments import BlueBox, Velmex, Agilent33220A, AerotechXY
from adr_gui.adr_gui_control import AdrGuiControl
from iv_data import ComplexImpedanceSweepData

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
    
        
    

class ComplexImpedanceSweep():
    ''' Acquire  '''
    def __init__(self,
                 source_amp_volt=3.0, source_offset_volt=1.5, source_frequency_hz=np.logspace(0,5,50),
                 num_lockin_periods = 10,
                 row_order=None):

        # hardware and class initialization
        self.sla = SoftwareLockinAcquire(easy_client=None, signal_column_index=0,reference_column_index=1,
                                         signal_feedback_or_error='feedback',num_pts_per_period=None)
        self.source = Agilent33220A()
        self.adr_gui_control = AdrGuiControl() #self._handle_adr_gui_control_arg(adr_gui_control)
        self.init_source(source_amp_volt, source_offset_volt, source_frequency_hz)

        # input parameters
        self.source_amp_v = source_amp_volt
        self.source_offset_v = source_offset_volt
        self.source_freq_hz = source_frequency_hz
        self.num_lockin_periods = num_lockin_periods
        self.waittime_s = 0.1 # time to wait between setting new frequency and acquiring data
        self.num_freq = len(self.source_freq_hz)
        self.row_order = self._handle_row_to_state(row_order)


    def _handle_row_to_state(self,row_order):
        if row_order is not None:
            return row_order
        return list(range(self.sla.ec.nrow))

    def init_source(self, amp, offset, frequency):
        self.source.SetFunction(function = 'sine')
        #self.source.SetFunction(function = 'square')
        self.source.SetLoad('INF')
        self.source.SetFrequency(frequency)
        self.source.SetAmplitude(amp)
        self.source.SetOffset(offset)
        self.source.SetOutput(outputstate='on')

    def get_point(self,window=False):
        return self.sla.getData(num_periods=self.num_lockin_periods, window=window,debug=False)

    def get_complexImpedance(self, extra_info = {}, move_to_zero_at_end = True, turn_off_source_on_end = True):
        pre_time = time.time()
        pre_temp_k = self.adr_gui_control.get_temp_k()

        iq_v_freq = np.empty((self.num_freqs,self.sla.ec.nrow,2))
        measured_freqs = []
        for ii, freq in enumerate(self.source_freq_hz):
            print('Setting function generator to frequency = %.2f Hz'%freq)
            self.source.SetFrequency(freq)
            m_freq = self.source.GetFrequency()
            measured_freqs.append(m_freq)
            iq_arr = self.get_point() # nrow x 2 array
            iq_v_freq[ii,:,:] = iq_arr
        post_temp_k = self.adr_gui_control.get_temp_k()
        post_time = time.time()

        print('Acquisition complete.')
        if turn_off_source_on_end: self.source.SetOutput('off')

        return ComplexImpedanceSweepData(iq_v_freq = iq_v_freq,
                                      row_order=self.row_order,
                                      #bayname=self.bayname,
                                      #db_cardname=self.db_cardname,
                                      column_number=0,
                                      source_amp_volt=self.source_amp_v,
                                      source_offset_volt=self.source_offset_v,
                                      source_frequency_hz=self.source_freq_hz,
                                      pre_temp_k=pre_temp_k,
                                      post_temp_k=post_temp_k,
                                      pre_time_epoch_s=pre_time,
                                      post_time_epoch_s=post_time,
                                      extra_info=extra_info)


if __name__ == "__main__":
    freqs = np.logspace(0,5,50) # Hz
    power_line = 60. # Hz
    delta = 3 # Hz trim freqs within +-delta of power_line
    source_frequency_hz = list(freqs[(freqs<power_line-delta)]) + list(freqs[(freqs>power_line+delta)])
    czss = ComplexImpedanceSweep(source_frequency_hz=source_frequency_hz,num_lockin_periods = 10)



    cz_data = czss.get_complexImpedance()
    cz_data.to_file('test_cz_data',overwrite=True)
    cz_data.plot()

    #test_for_signal()


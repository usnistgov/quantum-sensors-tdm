''' czAcquire.py

    @author JH and GCJ 5/2023, based polcal.py.

    Typical hardware setup:
    1) Agilent33220A output through resistor box (10K) into detector bias line
    2) Agilent33220A sync into DFB card CH2 input.  This corresponds to reference_column_index=1.
    3) Detector column signal corresponds to DFB card CH1 input/output.  This corresponse to signal_column_index=0

    to do:
    1) self.waittime_s currently not used.  Is latency taken care of in instrument or not?
    2) debug the number of periods used.  And is the calculated every getData?  If so, that is inefficient.
    3) write class to loop over detector bias and bath temperature
    4) temperature regulation stuff copied from iv_utils.py duplicate code is bad.
       Make separate class which all can inherit?
'''

from nasa_client import EasyClient
from instruments import BlueBox, Agilent33220A
from adr_gui.adr_gui_control import AdrGuiControl
from cringe.cringe_control import CringeControl
from iv_data import SineSweepData
from tools import SoftwareLockinAcquire

import numpy as np
import matplotlib.pyplot as plt
import time, os
import progress.bar

class SineSweep():
    ''' Sweep constant amplitude sine wave as a function of frequency.
        Used for example in a single detector bias position for a complex impedance or complex responsivity
        measurement.
    '''
    def __init__(self,
                 amp_volt=0.04, offset_volt=0, frequency_hz=np.logspace(1,3,50),
                 num_lockin_periods = 10,
                 row_order=None,
                 signal_column_index=0,
                 reference_column_index=1,
                 column_str='A'):

        self.fg = Agilent33220A() #function generator
        self.adr_gui_control = AdrGuiControl() #self._handle_adr_gui_control_arg(adr_gui_control)
        self.sla = SoftwareLockinAcquire(easy_client=None, signal_column_index=signal_column_index,
                                         reference_column_index=reference_column_index,
                                         signal_feedback_or_error='feedback',num_pts_per_period=None)
        # input parameters
        self.freq_hz = frequency_hz
        self.amp_v = amp_volt
        self.offset_v = offset_volt
        self.signal_column_index = signal_column_index
        self.reference_column_index = reference_column_index
        self.num_lockin_periods = num_lockin_periods
        self.waittime_s = 0 # time to wait between setting new frequency and acquiring data.
        self.row_order = self._handle_row_to_state(row_order)
        self.column_str = column_str # purely for recording purposes
        self.iq_v_freq = None

        self.init_fg(amp_volt, offset_volt, frequency_hz[0])

    def _handle_row_to_state(self,row_order):
        if row_order is not None:
            return row_order
        return list(range(self.sla.ec.nrow))

    def init_fg(self, amp, offset, frequency):
        self.fg.SetFunction(function = 'sine')
        self.fg.SetLoad('INF')
        self.fg.SetFrequency(frequency)
        self.fg.SetAmplitude(amp)
        self.fg.SetOffset(offset)
        self.fg.SetOutput(outputstate='on')

    def get_iq(self,window=False):
        ''' Gets the IQ from the software lock in.
            Return array structure is n_rows x 2 (one for I and one for Q)
        '''
        return self.sla.getData(num_periods=self.num_lockin_periods, window=window,debug=False)

    def set_frequency(self,freq,diff_percent_tol = .1):
        ''' set the frequency and check that the frequency set is within diff_percent_tol % of requested. 
            Why isn't it exact?  Finite # of bits 
        '''
        print('Setting function generator to frequency = %.2f Hz'%freq)
        self.fg.SetFrequency(freq)
        freq_measured = self.fg.GetFrequency()
        diff_percent = ((freq_measured-freq) / freq)*100  
        assert abs(diff_percent) < diff_percent_tol, print('Output frequency differs from requested by > diff_percent_tol. Requested: %.5f, output: %.5f'%(freq,freq_measured))
        return freq_measured

    def take_sweep(self, extra_info = {}, turn_off_source_on_end = True):
        pre_time = time.time()
        pre_temp_k = self.adr_gui_control.get_temp_k()

        iq_v_freq = np.empty((len(self.freq_hz),self.sla.ec.nrow,2))
        freq_m = []
        for ii, freq in enumerate(self.freq_hz):
            freq_m.append(self.set_frequency(freq))
            time.sleep(self.waittime_s)
            iq_v_freq[ii,:,:] = self.get_iq()

        post_temp_k = self.adr_gui_control.get_temp_k()
        post_time = time.time()
        print('Acquisition complete.')
        if turn_off_source_on_end: self.fg.SetOutput('off')
        self.iq_v_freq = iq_v_freq

        return SineSweepData(frequency_hz = self.freq_hz, iq_data = iq_v_freq,
                                         amp_volt=self.amp_v, offset_volt=self.offset_v,
                                         row_order=self.row_order,
                                         column_str = self.column_str,
                                         signal_column_index = self.signal_column_index,
                                         reference_column_index = self.reference_column_index,
                                         number_of_lockin_periods = self.num_lockin_periods,
                                         pre_temp_k=pre_temp_k, post_temp_k=post_temp_k,
                                         pre_time_epoch_s=pre_time, post_time_epoch_s=post_time,
                                         extra_info=extra_info)

    def plot(self,fignum=1,semilogx=True):
        fig, ax = plt.subplots(nrows=2,ncols=2,sharex=False,figsize=(12,8),num=fignum)
        for ii in range(self.sla.ec.nrow):
            if semilogx:
                ax[0][0].semilogx(self.freq_hz,self.iq_v_freq[:,ii,0],'o-')
                ax[0][1].semilogx(self.freq_hz,self.iq_v_freq[:,ii,1],'o-')
                ax[1][0].semilogx(self.freq_hz,self.iq_v_freq[:,ii,0]**2+self.iq_v_freq[:,ii,1]**2,'o-')
                ax[1][1].semilogx(self.freq_hz,np.arctan(self.iq_v_freq[:,ii,1]/self.iq_v_freq[:,ii,0]),'o-')
            else:
                ax[0][0].plot(self.freq_hz,self.iq_v_freq[:,ii,0],'o-')
                ax[0][1].plot(self.freq_hz,self.iq_v_freq[:,ii,1],'o-')
                ax[1][0].plot(self.freq_hz,self.iq_v_freq[:,ii,0]**2+self.iq_v_freq[:,ii,1]**2,'o-')
                ax[1][1].plot(self.freq_hz,np.arctan(self.iq_v_freq[:,ii,1]/self.iq_v_freq[:,ii,0]),'o-')

        # axes labels
        ax[0][0].set_ylabel('I')
        ax[0][1].set_ylabel('Q')
        ax[1][0].set_ylabel('I^2+Q^2')
        ax[1][1].set_ylabel('Phase')
        ax[1][0].set_xlabel('Freq (Hz)')
        ax[1][1].set_xlabel('Freq (Hz)')

        ax[1][1].legend(list(range(self.sla.ec.nrow)))

class ComplexZ(SineSweep):
    '''
    '''
    def __init__(self,
                 amp_volt=0.04, offset_volt=0, frequency_hz=np.logspace(1,3,50),
                 num_lockin_periods = 10,
                 row_order=None,
                 signal_column_index=0,
                 reference_column_index=1,
                 column_str='A',
                 detector_bias_list = [10000,20000],
                 temperature_list_k = [0.1],
                 voltage_source='tower',
                 db_cardname = 'DB',
                 db_tower_channel='0',
                 cringe_control=None):

        super().__init__(amp_volt, offset_volt, frequency_hz, num_lockin_periods,
                         row_order,signal_column_index,reference_column_index,
                         column_str)

        self.detector_bias = detector_bias_list
        self.temp_list_k = temperature_list_k
        self.db_cardname = db_cardname
        self.db_tower_channel = tower_channel

        # globals hidden from class initialization
        self.temp_settle_delay_s = 30 # wait time after commanding an ADR set point

        self.cc = self._handle_cringe_control_arg(cringe_control)
        self.set_volt = self._handle_voltage_source_arg(voltage_source)

    def _handle_cringe_control_arg(self, cringe_control):
        if cringe_control is not None:
             return cringe_control
        return CringeControl()

    def _handle_voltage_source_arg(self,voltage_source):
        # set "set_volt" to either tower or bluebox
        if voltage_source == None or voltage_source == 'tower':
            set_volt = self.set_tower # 0-2.5V in 2**16 steps
        elif voltage_source == 'bluebox':
            self.bb = BlueBox(port='vbox', version='mrk2')
            set_volt = self.set_bluebox # 0 to 6.5535V in 2**16 steps
        return set_volt

    def set_tower(self, dacvalue):
        self.cc.set_tower_channel(self.db_cardname, self.db_tower_channel, int(dacvalue))

    def set_bluebox(self, dacvalue):
        self.bb.setVoltDACUnits(int(dacvalue))

    def _is_temp_stable(self, setpoint_k, tol=.005, time_out_s=180):
        ''' determine if the servo has reached the desired temperature '''
        assert time_out_s > 10, "time_out_s must be greater than 10 seconds"
        cur_temp=self.adr_gui_control.get_temp_k()
        it_num=0
        while abs(cur_temp-setpoint_k)>tol:
            time.sleep(10)
            cur_temp=self.adr_gui_control.get_temp_k()
            print('Current Temp: ' + str(cur_temp))
            it_num=it_num+1
            if it_num>round(int(time_out_s/10)):
                print('exceeded the time required for temperature stability: %d seconds'%(round(int(10*it_num))))
                return False
        return True

    # def _set_temp_and_settle(self, setpoint_k):
    #     self.adr_gui_control.set_temp_k(float(setpoint_k))
    #     self._last_setpoint_k = setpoint_k
    #     print(f"set setpoint to {setpoint_k} K and now sleeping for {self.temp_settle_delay_s} s")
    #     time.sleep(self.temp_settle_delay_s)
    #     print(f"done sleeping")

    def set_temp(self,temp_k):
        self.adr_gui_control.set_temp_k(float(temp_k))
        stable = self._is_temp_stable()
        time.sleep(self.temp_settle_delay_s)
        return stable

    def run(self, extra_info = {}, skip_wait_on_first_temp=False):
        temp_output = []
        for ii,temp in enumerate(self.temp_list_k):
            if np.logical_and(ii==0,skip_wait_on_first_temp):
                self.adr_gui_control.set_temp_k(float(temp_k))
            else:
                self.set_temp(temp)
            det_bias_output = []
            for jj,db in enumerate(self.detector_bias_list):
                self.set_volt(db)
                time.sleep(0.1)
                self.cc.relock_all_locked_fba(self.signal_column_index)
                time.sleep(0.1)
                result = self.take_sweep(extra_info = {}, turn_off_source_on_end = False)
                det_bias_output.append(result) # want a return data structure that indexes like so: [temp_index,det_bias_index,result]
            temp_output.append(det_bias_output)

        return CzData(data = temp_output,
                      detector_bias_list = self.detector_bias,
                      temp_list_k = self.temp_list_k,
                      db_cardname = self.db_cardname,
                      db_tower_channel_str = self.tower_channel,
                      temp_settle_delay_s = self.temp_settle_delay_s,
                      extra_info = extra_info)

if __name__ == "__main__":
    ss = SineSweep(frequency_hz=[10,20,30])
    ss.take_sweep()
    ss.plot()
    plt.show()

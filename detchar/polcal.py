''' polcal.py

    Polarization calibration software using aerotech XY stage, velmex stepper motor controlled wire grid, 
    function generator biased IR source, and NIST `previous generation (TDM) electronics' 

    JH 6/2021, based on Jay's software for legacy electronics.
'''

from dataclasses import dataclass
import dataclasses
from dataclasses_json import dataclass_json
from typing import Any, List

from nasa_client import EasyClient
from instruments import BlueBox, Velmex, Agilent33220A
from adr_gui.adr_gui_control import AdrGuiControl

import time
import numpy as np
import pylab as plt
import progress.bar
import os
from tools import * 

# to do:
# 1) write XY stage class
# 2) ability to handle arbitrary row to state orders
# 3) include bayname and db_cardname?
# 4) limits to input angles so that wires of IR source don't rip.


# questions: is an instance of easy client the right thing for every point?

# general algorithm:
# 1) move XY stage to a given position.
# 2) step through grid positions, for each grid position get lock-in signal from each bolometer on the column.
#
# Data I would like returned:
# - locked in signal for each detector for each grid angular position 
#     - locked in signal data format: I,Q and rms of each?  
#     - grid position for this measurement
# - meta-data:
#         - row select -> state map
#         - temperature of measurement
#         - IR source bias settings
#         - settings for function generator
#         - detector bias settings

@dataclass_json
@dataclass
class PolCalSteppedSweepData():
    angle_deg: List[float] # see if this works
    iq_values: List[Any] = dataclasses.field(repr=False) #actually a list of np arrays
    iq_rms_values: List[Any] = dataclasses.field(repr=False) #actually a list of np arrays
    row_order: List[int]
    #bayname: str
    #db_cardname: str
    column_number: int    
    source_amp_volt: float 
    source_offset_volt: float
    source_frequency_hz: float
    #nominal_temp_k: float
    pre_temp_k: float
    post_temp_k: float
    pre_time_epoch_s: float
    post_time_epoch_s: float
    extra_info: dict

    def to_file(self, filename, overwrite = False):
        if not overwrite:
            assert not os.path.isfile(filename)
        with open(filename, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_file(cls, filename):
        with open(filename, "r") as f:
            return cls.from_json(f.read())

    def plot(self):
        plt.figure()
        plt.plot(self.angle_deg, self.iq_values)
        plt.xlabel("Angle (deg)")
        plt.ylabel("Response (arb)")
        plt.title(f"bay {self.bayname}, db_card {self.db_cardname}, nominal_temp_mk {self.nominal_temp_k*1000}")

# class PolcalPointTaker():
#     ''' Acquire a single polcal data point for all bolometers on a column '''
#     def __init__(self,easy_client):
#         self.ec = self._handle_easy_client_arg(easy_client) # is this the right thing to do for every data point?

#     def _handle_easy_client_arg(self, easy_client):
#         if easy_client is not None:
#             return easy_client
#         easy_client = EasyClient()
#         easy_client.setupAndChooseChannels()
#         return easy_client

#     def get_point():
#         return software_lock_in_acquisition(ec = self.ec, signal_index,reference_index,reference_type='square',response_type='sine',
#                                  debug=False)
    

class PolcalSteppedSweep():
    ''' Aquire polcal at stepped, fixed angles '''
    def __init__(self, angle_list_deg, 
                 source_amp_volt=3.0, source_offset_volt=1.5, source_frequency_hz=5.0, 
                 signal_column_index = 0, 
                 reference_column_index = 1,
                 easy_client=None
                 row_order=None):
        
        # input parameters
        self.angle_deg, self._angle_100th_deg = self._handle_angle_arg(angle_list_arg)  
        self.source_amp_v = source_amp_volt
        self.source_offset_v = source_offset_volt
        self.source_freq_hz = source_frequency_hz 
        self.sig_col_index = signal_column_index 
        self.ref_col_index = reference_column_index 
        self.row_order = self._handle_row_order(row_order)
        self.waittime_s = 1.0 # time to wait between setting grid angle and acquiring data

        # hardware and class initialization
        self.source = Agilent33220A() 
        self.ec = self._handle_easy_client_arg(easy_client)
        self.adr_gui_control = self._handle_adr_gui_control_arg(adr_gui_control)
        self.init_source(source_amp_volt, source_offest_volt, source_frequency_hz)
        self.grid_motor = Velmex(doInit=True)

    def _handle_row_to_state(self,row_order):
        if row_order is not None:
            return row_order 
        return list(range(self.ec.n_rows))

    def _handle_adr_gui_control_arg(self, adr_gui_control):
        if adr_gui_control is not None:
            return adr_gui_control
        return AdrGuiControl()

    def _handle_easy_client_arg(self, easy_client):
        if easy_client is not None:
            return easy_client
        easy_client = EasyClient()
        easy_client.setupAndChooseChannels()
        return easy_client

    def _handle_angle_arg(self,angle):
        ''' velmex motor only has 100th degree step resolution.  So round to nearest '''
        angle_100th_deg = []
        for a in angle:
            angle_100th_deg.append(int(round(a*100)))
        angle_deg = list(np.array(angle_100th_deg)/100))
        return angle_deg, angle_100th_deg

    def init_source(self, amp, offset, frequency):
        self.source.SetFunction(function = 'sine')
        self.source.SetLoad('INF')
        self.source.SetFrequency(frequency)
        self.source.SetAmplitude(amp)
        self.source.SetOffset(offset)
        self.source.SetOutput(outputstate='on')

    def get_point(self):
        I,Q,vref = software_lock_in_acquisition(ec = self.ec, signal_index=self.sig_col_index,reference_index = self.ref_col_index,reference_type='sine',response_type='sine',
                                 debug=False)
        return I, Q 

    def get_polcal(self, extra_info = {}):
        pre_time = time.time()
        pre_temp_k = self.adr_gui_control.get_temp_k()
        for ii, angle in enumerate(self._angle_100th_deg):
            self.grid_motor.move_relative(phi_100th_deg=angle,wait=True,verbose=False)
            time.sleep(self.waittime_s)
            I,Q = self.get_point() 
        post_temp_k = self.adr_gui_control.get_temp_k()
        post_time = time.time()

        return PolCalSteppedSweepData(angle_deg=self.angle_deg, iq_values=iq_values, iq_rms_values=iq_rms_values,
                                      row_select_to_state_dict=self.row_select_to_state_dict,
                                      #bayname=self.bayname,
                                      #db_cardname=self.db_cardname,
                                      column_number=self.sig_col_index,
                                      source_amp_volt=self.source_amp_v,
                                      source_offset_volt=self.source_offset_v,
                                      source_frequency_hz=self.source_freq_hz,
                                      pre_temp_k=pre_temp_k,
                                      post_temp_k=post_temp_k,
                                      pre_time_epoch_s=pre_time,
                                      post_time_epoch_s=post_time,
                                      extra_info=extra_info)

class PolCalSteppedBeamMap():
    ''' Aquire PolcalSteppedSweep for x,y positions ''' 
    def __init__(self):
        print('To be written')


if __name__ == "__main__":
    print('Entered main')

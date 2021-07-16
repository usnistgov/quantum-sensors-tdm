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
from instruments import BlueBox, Velmex, Agilent33220A, AerotechXY
from adr_gui.adr_gui_control import AdrGuiControl

import time
import numpy as np
import pylab as plt
import progress.bar
import os
from tools import SoftwareLockinAcquire

# to do:
# 1) write XY stage class
# 2) ability to handle arbitrary row to state orders
# 3) include bayname and db_cardname?
# 4) limits to input angles so that wires of IR source don't rip.

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
    iq_v_angle: List[Any] = dataclasses.field(repr=False) #actually a list of np arrays
    #iq_rms_values: List[Any] = dataclasses.field(repr=False) #actually a list of np arrays
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

    def plot(self, rows_per_figure=None):
        ''' rows_per_figure is a list of lists to group detector responses
            to be plotted together.  If None will plot in groups of 8.
        '''
        if rows_per_figure is not None:
            pass
        else:
            num_in_group = 8
            n_angles,n_rows,n_iq = np.shape(self.iq_v_angle)
            n_groups = n_rows//num_in_group + 1
            rows_per_figure=[]
            for jj in range(n_groups):
                tmp_list = []
                for kk in range(num_in_group):
                    row_index = jj*num_in_group+kk
                    if row_index>=n_rows: break
                    tmp_list.append(row_index)
                rows_per_figure.append(tmp_list)
        for ii,row_list in enumerate(rows_per_figure):
            fig,ax = plt.subplots(3,num=ii)
            for row in row_list:
                ax[0].plot(self.angle_deg,self.iq_v_angle[:,row,0],'o-',label=row)
                ax[1].plot(self.angle_deg,self.iq_v_angle[:,row,1],'o-',label=row)
                ax[2].plot(self.angle_deg,np.sqrt(self.iq_v_angle[:,row,0]**2+self.iq_v_angle[:,ii,1]**2),'o-',label=row)
            ax[0].set_ylabel('I (DAC)')
            ax[1].set_ylabel('Q (DAC)')
            ax[2].set_ylabel('Amplitude (DAC)')
            ax[2].set_xlabel('Angle (deg)')
            ax[1].legend()
            ax[0].set_title('Column %d, Group %d'%(self.column_number,ii))
        plt.show()

@dataclass_json
@dataclass
class PolCalSteppedBeamMapData():
    xy_position_list: List[Any]
    data: List[PolCalSteppedSweepData]


class PolcalSteppedSweep():
    ''' Aquire polcal at stepped, fixed angles '''
    def __init__(self, angle_throw_amp_deg=360, angle_step_deg=10,
                 source_amp_volt=3.0, source_offset_volt=1.5, source_frequency_hz=5.0,
                 num_lockin_periods = 10,
                 row_order=None):

        # hardware and class initialization
        self.sla = SoftwareLockinAcquire(easy_client=None, signal_column_index=0,reference_column_index=1,
                                         signal_feedback_or_error='feedback',num_pts_per_period=None)
        self.source = Agilent33220A()
        self.adr_gui_control = AdrGuiControl() #self._handle_adr_gui_control_arg(adr_gui_control)
        self.init_source(source_amp_volt, source_offset_volt, source_frequency_hz)
        self.grid_motor = Velmex(doInit=False)
        # print('Sleeping for 15 seconds after homing.')
        # time.sleep(15)
        # print('end sleep.')

        # input parameters
        self.angle_deg, self._angle_100th_deg = self._handle_angle_args(angle_throw_amp_deg, angle_step_deg)
        self.num_angles = len(self.angle_deg)
        self.source_amp_v = source_amp_volt
        self.source_offset_v = source_offset_volt
        self.source_freq_hz = source_frequency_hz
        self.num_lockin_periods = 10
        self.row_order = self._handle_row_to_state(row_order)
        self.waittime_s = 5 # time to wait between setting grid angle and acquiring data

    def _handle_row_to_state(self,row_order):
        if row_order is not None:
            return row_order
        return list(range(self.sla.ec.nrow))

    def _handle_angle_args(self,angle_throw_amp_deg, angle_step_deg):
        ''' velmex motor hase 100th degree step resolution.  So round to nearest.    
            Also, wires underneath motor cannot be continually twisted or risk breaking.  
        '''
        self.grid_motor.ensure_safe_angle(angle_throw_amp_deg*200) # x100 due to expected units of velmex class 
        angle_deg = np.arange(-angle_throw_amp_deg,angle_throw_amp_deg+angle_step_deg,angle_step_deg)
        angle_100th_deg = []
        for ii, angle in enumerate(angle_deg):
            a = int(round(angle*100))
            angle_100th_deg.append(a)
            angle_deg[ii] = a/100
            
        return angle_deg, np.array(angle_100th_deg)

    def init_source(self, amp, offset, frequency):
        #self.source.SetFunction(function = 'sine')
        self.source.SetFunction(function = 'square')
        self.source.SetLoad('INF')
        self.source.SetFrequency(frequency)
        self.source.SetAmplitude(amp)
        self.source.SetOffset(offset)
        self.source.SetOutput(outputstate='on')

    def get_point(self,window=False):
        return self.sla.getData(num_periods=self.num_lockin_periods, window=window,debug=False)

    def get_polcal(self, extra_info = {}):
        pre_time = time.time()
        pre_temp_k = self.adr_gui_control.get_temp_k()
        print('Moving to initial angle and waiting 10s')
        self.grid_motor.move_absolute(self._angle_100th_deg[0])
        time.sleep(10)

        iq_v_angle = np.empty((self.num_angles,self.sla.ec.nrow,2))
        angle_100th_diff = np.diff(self._angle_100th_deg)
        angle_100th_diff = np.insert(angle_100th_diff,0,0)
        for ii, angle in enumerate(angle_100th_diff):
            print('Moving to angle = %.1f deg'%(self.angle_deg[ii]))
            self.grid_motor.move_relative(angle_100th_deg=angle,wait=False,verbose=False)
            print('motor absolute angle is = ',self.grid_motor.current_angle_100th_deg)
            time.sleep(self.waittime_s)
            # iq_arr = self.get_point() # nrow x 2 array
            # iq_v_angle[ii,:,:] = iq_arr
        post_temp_k = self.adr_gui_control.get_temp_k()
        post_time = time.time()

        print('Acquisition done.  Unwinding wires.')
        self.grid_motor.move_relative(-1*self._angle_100th_deg[-1])

        return PolCalSteppedSweepData(angle_deg=self.angle_deg, iq_v_angle = iq_v_angle,
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

class PolCalSteppedBeamMap():
    ''' Acquire PolcalSteppedSweep for x,y positions '''
    def __init__(self,xy_position_list, polcal_stepped_sweep, doXYinit=True):
        self.pcss = polcal_stepped_sweep
        self.xy_pos_list = xy_position_list
        self.x_velocity_mmps = self.y_velocity_mmps = 25 # velocity of xy motion in mm per s
        self.xy = AerotechXY() #
        if doXYinit:
            self.xy.initialize()

    def acquire(self, extra_info = {}):
        data_list = []
        for ii, xy_pos in enumerate(self.xy_pos_list):
            self.xy.move_absolute(xy_pos[0],xy_pos[1],self.x_velocity_mmps,self.y_velocity_mmps)
            data_list.append(self.pcss.get_polcal(extra_info = extra_info))
        return PolCalSteppedBeamMapData(xy_position_list = self.xy_pos_list, data=data_list)

if __name__ == "__main__":
    pcss = PolcalSteppedSweep(angle_throw_amp_deg=40, angle_step_deg=10,
                       source_amp_volt=3.0, source_offset_volt=1.5, source_frequency_hz=5.0,
                       num_lockin_periods = 1,
                       row_order=None)
    pc_data = pcss.get_polcal()
    pc_data.to_file('test_polcal_data',overwrite=True)
    pc_data.plot()

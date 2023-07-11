''' polcal.py

    Polarization calibration software using aerotech XY stage, velmex stepper motor controlled wire grid,
    function generator biased IR source, and NIST `previous generation (TDM) electronics'

    @author JH 7/2021, based on Jay's software for legacy electronics.
'''

from iv_data import PolCalSteppedSweepData, PolCalSteppedBeamMapData, BeamMapSingleGridAngleData

from nasa_client import EasyClient
from instruments import BlueBox, Velmex, Agilent33220A, AerotechXY
from adr_gui.adr_gui_control import AdrGuiControl

import time
import numpy as np
import pylab as plt
import progress.bar
import os
from tools import SoftwareLockinAcquire
import matplotlib.pyplot as plt

class SourceInit():
    ''' class to initialize the function generator, tailored to driving the HawkEye IR source.  
        Hardware is to use blue bias box that combines a DC and AC signal from a function 
        generator, and sends this to the IR source
    ''' 
    def __init__(self,source_amp_volt=3.0, source_offset_volt=1.5, source_frequency_hz=5.0):
        self.source_amp_v = source_amp_volt
        self.source_offset_v = source_offset_volt
        self.source_freq_hz = source_frequency_hz
        self.source = Agilent33220A()
        self.init_source(source_amp_volt, source_offset_volt, source_frequency_hz)

    def init_source(self, amp, offset, frequency):
        #self.source.SetFunction(function = 'sine')
        self.source.SetFunction(function = 'square')
        self.source.SetLoad('INF')
        self.source.SetFrequency(frequency)
        self.source.SetAmplitude(amp)
        self.source.SetOffset(offset)
        self.source.SetOutput(outputstate='on')

    
class PolcalSteppedSweep(SourceInit):
    ''' Acquire polcal at stepped, fixed absolute angles '''
    def __init__(self, angle_deg_list,
                 source_amp_volt=3.0, source_offset_volt=1.5, source_frequency_hz=5.0,
                 num_lockin_periods = 5,
                 row_order=None,
                 grid_motor=None,
                 initialize_grid_motor=True):

        super().__init__(source_amp_volt, source_offset_volt, source_frequency_hz) # initialize the function generator

        # hardware and class initialization
        self.sla = SoftwareLockinAcquire(easy_client=None, signal_column_index=0,reference_column_index=1,
                                         signal_feedback_or_error='feedback')
        
        self.adr_gui_control = AdrGuiControl() #self._handle_adr_gui_control_arg(adr_gui_control)
        self.grid_motor = self._handle_grid_motor_arg(grid_motor,initialize_grid_motor)

        # input parameters
        self.angles = self._handle_angle_arg(angle_deg_list)
        self.num_angles = len(self.angles)
        assert num_lockin_periods >=2,'More than 2 periods needed.'
        self.num_lockin_periods = num_lockin_periods
        self._handle_num_points(num_lockin_periods)
        self.waittime_s = 0.1 # time to wait between setting grid angle and acquiring data
        self.row_order = self._handle_row_to_state(row_order)

    def _handle_grid_motor_arg(self,grid_motor,initialize_grid_motor):
        if grid_motor is not None:
            return grid_motor
        else:
            return Velmex(doInit=initialize_grid_motor)

    def _handle_row_to_state(self,row_order):
        if row_order is not None:
            return row_order
        return list(range(self.sla.ec.nrow))

    def _handle_angle_arg(self,angle_deg_list):
        ''' velmex motor has finite resolution, so map angle_deg_list to this
            resolution.
        '''
        for angle in angle_deg_list:
            self.grid_motor.check_angle_safe(angle) # ensure IR source wires not over-twisted
        return self.grid_motor.angle_list_to_stepper_resolution(angle_deg_list)

    def _handle_num_points(self,num_lockin_periods,verbose=False):
        self.sampling_rate = self.sla.ec.clockMhz*1e6/self.sla.ec.linePeriod/self.sla.ec.nrow # num samples/sec 
        self.samples_per_period = int(self.sampling_rate/self.source_freq_hz)  
        self.num_points = self.samples_per_period*self.num_lockin_periods
        if verbose:
            print('sampling rate (1/s) :', self.sampling_rate)
            print('Source frequency: ',self.source_freq_hz)
            print('samples_per_period: ',self.samples_per_period)
            print('number of lockin periods: ',self.num_lockin_periods)
            print('number of points',self.num_points)

    def get_point(self,window=False):
        return self.sla.getData(minimumNumPoints=self.num_points, window=window,debug=False,num_pts_per_period=self.samples_per_period)

    def get_polcal(self, extra_info = {}, move_to_zero_at_end = True, turn_off_source_on_end = True, source_on=True):
        if source_on:
            self.source.SetOutput(outputstate='on')
        else:
            self.source.SetOutput(outputstate='off')
        pre_time = time.time()
        pre_temp_k = self.adr_gui_control.get_temp_k()

        iq_v_angle = np.empty((self.num_angles,self.sla.ec.nrow,2))
        measured_angles = []
        for ii, angle in enumerate(self.angles):
            print('Moving grid to angle = %.2f deg'%angle)
            self.grid_motor.move_absolute(angle,wait=True)
            m_angle = self.grid_motor.get_current_position()
            measured_angles.append(m_angle)
            print('Motor at angle = %.2f.  Grabbing data'%m_angle)
            iq_arr = self.get_point() # nrow x 2 array
            iq_v_angle[ii,:,:] = iq_arr
        post_temp_k = self.adr_gui_control.get_temp_k()
        post_time = time.time()

        print('Acquisition complete.')
        if move_to_zero_at_end:
            print('Unwinding wires; moving to angle=0')
            self.grid_motor.move_to_zero_index(wait=True)
        if turn_off_source_on_end: self.source.SetOutput('off')

        return PolCalSteppedSweepData(angle_deg_req=self.angles,
                                      angle_deg_meas=measured_angles,
                                      iq_v_angle = iq_v_angle,
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

class PolCalSteppedBeamMap(PolcalSteppedSweep):
    ''' Acquire PolcalSteppedSweep for x,y positions '''
    def __init__(self,xy_position_list, angle_deg_list,
                 source_amp_volt=3.0, source_offset_volt=1.5, source_frequency_hz=5,
                 num_lockin_periods=5,
                 row_order=None,
                 doXYinit=True):

        super().__init__(angle_deg_list,source_amp_volt, source_offset_volt, source_frequency_hz,
                     num_lockin_periods,row_order,None, False)
        self.grid_motor.initialize()

        self.xy_pos_list = xy_position_list
        self.x_velocity_mmps = self.y_velocity_mmps = 25 # velocity of xy motion in mm per s
        self.xy = AerotechXY() #
        if doXYinit:
            self.xy.initialize()
            self.xy.set_wait_mode('MOVEDONE')

    def acquire(self, extra_info = {}):
        data_list = []
        for ii, xy_pos in enumerate(self.xy_pos_list):
            print('Moving XY stage to position: ',xy_pos)
            self.xy.move_absolute(xy_pos[0],xy_pos[1],self.x_velocity_mmps,self.y_velocity_mmps)
            print('Acquiring response versus polarization angle')
            data = self.get_polcal(extra_info = extra_info, move_to_zero_at_end = False, turn_off_source_on_end = False, source_on=True)
            data_list.append(data)
        return PolCalSteppedBeamMapData(xy_position_list = self.xy_pos_list, data=data_list,extra_info=extra_info)

class BeamMapSingleGridAngle(SourceInit):
    ''' 1D cut for a polarized beam map chopping using the IR source. '''
    def __init__(self,xy_position_list, grid_angle,
                 source_amp_volt=3.0, source_offset_volt=1.5, source_frequency_hz=5,
                 num_lockin_periods=5,
                 row_order=None,
                 home_xy=True,
                 grid_motor=None):
        
        self.xy_pos_list = xy_position_list
        self.grid_angle = grid_angle 
        self.num_lockin_periods= num_lockin_periods
        self.x_velocity_mmps = self.y_velocity_mmps = 25 # velocity of xy motion in mm per s

        # initialize instruments: function generator, TDM lock-in data acq, XY stage, wire grid motor, adr_control
        super().__init__(source_amp_volt, source_offset_volt, source_frequency_hz) # initialize the function generator
        self.sla = SoftwareLockinAcquire(easy_client=None, signal_column_index=0,reference_column_index=1,
                                         signal_feedback_or_error='feedback')
        self._handle_num_points(num_lockin_periods)
        self.xy = AerotechXY() # initialize XY stage
        self.xy.initialize(home=home_xy)
        self.grid_motor = self._handle_grid_motor_arg(grid_motor) #initialize the wire grid polarizer 
        self.adr_gui_control = AdrGuiControl()

    def _handle_grid_motor_arg(self,grid_motor):
        if grid_motor is not None:
            return grid_motor
        else:
            return Velmex(doInit=True)

    def _handle_num_points(self,num_lockin_periods,verbose=False):
        self.sampling_rate = self.sla.ec.clockMhz*1e6/self.sla.ec.linePeriod/self.sla.ec.nrow # num samples/sec 
        self.samples_per_period = int(self.sampling_rate/self.source_freq_hz)  
        self.num_points = self.samples_per_period*self.num_lockin_periods
        if verbose:
            print('sampling rate (1/s) :', self.sampling_rate)
            print('Source frequency: ',self.source_freq_hz)
            print('samples_per_period: ',self.samples_per_period)
            print('number of lockin periods: ',self.num_lockin_periods)
            print('number of points',self.num_points)

    def get_point(self,window=False):
        return self.sla.getData(minimumNumPoints=self.num_points, window=window,debug=False,num_pts_per_period=self.samples_per_period)
    
    def acquire(self, extra_info = {}, window=False, xy_shutdown_on_complete=False):
        pre_time = time.time()
        pre_temp_k = self.adr_gui_control.get_temp_k()
        self.grid_motor.move_absolute(self.grid_angle,wait=True) # move grid into position.

        iq_v_pos = np.empty((len(self.xy_pos_list),self.sla.ec.nrow,2))
        for ii, xy_pos in enumerate(self.xy_pos_list): # loop over XY positions
            print('Moving XY stage to position: ',xy_pos)
            self.xy.move_absolute(xy_pos[0],xy_pos[1],self.x_velocity_mmps,self.y_velocity_mmps)
            print('Doing lock-in measurement')
            iq_arr = self.get_point(window) 
            iq_v_pos[ii,:,:] = iq_arr
        
        post_temp_k = self.adr_gui_control.get_temp_k()
        post_time = time.time()
        res = BeamMapSingleGridAngleData(xy_position_list=self.xy_pos_list,
                                        iq_v_pos=iq_v_pos,
                                        grid_angle_deg=self.grid_angle,
                                        source_amp_volt=self.source_amp_v,
                                        source_offset_volt=self.source_offset_v,
                                        source_frequency_hz=self.source_freq_hz,
                                        pre_temp_k=pre_temp_k,
                                        post_temp_k=post_temp_k,
                                        pre_time_epoch_s=pre_time,
                                        post_time_epoch_s=post_time,
                                        num_lockin_periods=self.num_lockin_periods,
                                        extra_info=extra_info)
        if xy_shutdown_on_complete:
            xy.shutdown()

        
        return res 

def pol_to_xy_coordinates(xp,yp,theta_deg,pixel_center):
    ''' xp,yp: 1D array like of same length listing coordinates in detector polarization frame 
        theta_deg: the angle in degrees between the detector polarization axis and the x-axis of the XY stage 
        pixel_center = (Xo,Yo), location of pixel center in XY stage coordiante system
    '''
    assert len(xp)==len(yp), 'xp and yp must be the same length'
    c,s = np.cos(theta_deg*np.pi/180), np.sin(theta_deg*np.pi/180)
    R = np.array(((c,s),(-s,c)))
    x=[]; y=[]
    for ii in range(len(xp)): # this for loop isn't needed, but I'm too stupid to figure it out right now.
        xi,yi = np.matmul(R,np.array([[xp[ii]],[yp[ii]]]))
        x.append(xi[0]+pixel_center[0])
        y.append(yi[0]+pixel_center[1])
    return x,y

def xy_to_list(x,y):
    assert len(x)==len(y), 'x and y must be of same length'
    xy_list = []
    for ii in range(len(x)):
        xy_list.append([x[ii],y[ii]])
    return xy_list

def test_make_xy_list():
    xp=[-5,-4,-3,-2,-1,0,1,2,3,4,5]
    yp=[0]*11
    theta_deg=160
    pixel_center = [326.4663, 306.12]
    x,y = pol_to_xy_coordinates(xp,yp,theta_deg,pixel_center)
    xy_list = xy_to_list(x,y)
    for xy in xy_list:
        print(xy)

if __name__ == "__main__":
    # filename='test.json'
    # extra_info = {'exp_setup':'cmbs4 pixel, db=20000, tb=400mK, on-axis after peak-up on signal of 90A'}
    # angles = list(range(0,360,10))
    # pcss = PolcalSteppedSweep(angle_deg_list=angles,num_lockin_periods = 5,row_order=[0,2,3,4,5,6],source_frequency_hz=5.0)
    # pc_data = pcss.get_polcal(extra_info=extra_info,source_on=True)
    # pc_data.to_file(filename,overwrite=False)
    # pc_data.plot()

    pixel_center = [325.3, 310.64] # center of pixel 
    xp = np.arange(-50,51,10)
    grid_angle = 160
    x,y = pol_to_xy_coordinates(xp,[0]*len(xp),grid_angle,pixel_center)
    xy_position_list=xy_to_list(x,y)
    print(xy_position_list)

    bm = BeamMapSingleGridAngle(xy_position_list, grid_angle,
                                source_amp_volt=3.0, source_offset_volt=1.5, source_frequency_hz=5,
                                num_lockin_periods=5,
                                row_order=[0,2,3,4,5,6],
                                home_xy=False,
                                grid_motor=None)
   
    res = bm.acquire()
    res.to_file('test_beammap_write.json',overwrite=True)


    

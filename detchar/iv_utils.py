''' iv_util.py
    @author Galen O'Neil and Hannes Hubmayr

    was previously "galen_iv.py"
'''
import time
import numpy as np
import pylab as plt
import progress.bar
from detchar.iv_data import IVCurveColumnData, IVTempSweepData, IVColdloadSweepData
from detchar.acquire import Acquire

class IVPointTaker(Acquire):
    """ The IVPointTaker commands the bias value then calls easyClientDastard, which waits `delay_s` and then
    takes data. 

    Note that when you initialize IVPointTaker, the member `self.set_volt` is set to either point to
    the function `self.set_bluebox` or `self.set_tower` depending on the value passed to the initializer
    in the `voltage_source` argument.

    """
    def __init__(self, db_cardname=None, bayname=None, delay_s=0.05, 
                 relock_threshold_lo_hi = (2000, 14000),
    easy_client=None, cringe_control=None, voltage_source = 'tower', 
    column_number = 0, **kwargs):
        """ IVPointTaker class constructor.
        :param db_cardname: which card in the Tower to use for detector biasing. Note this is ignored if `voltage_source` is `bluebox`
        :param bayname: which output on the Tower Card specified in `db_cardname` to set. This is either a string containing a single digit number 
            or a list of strings containing single digit numbers, e.g. `bayname = "1"` or `bayname=["0","1","2"]`
        :param delay_s: how long to wait between setting the bias and taking data
        :param relock_threshold_lo_hi: tuple containing two ints. If a feedback value goes outside this range, attempt to relock.
        :param easy_client: optional. if None, create a new Easy Client.
        :param cringe_control: optional. if None, create a new Cringe Control.
        :param voltage_source: optional. Defaults to Tower. Possible values: 'tower', 'bluebox'
        :param column_number: optional. Defaults to 0. Which column of data should be returned. 
            This param has caveats. Right now it can only be zero, I think. It controls not only which column is selected
            from the data returned by EasyClient in ec.getNewData(...)[self.col,:,:,1].mean(axis=-1), but also which column
            is set up by Cringe with ARL=0. Theoretically should it == bayname? Depends on how the system is wired.
            CTR has three DFB cards wired with dfbx2[0]->"A" dfbx2[1]->"B" dfb_clk->"C". Historically,
            dfbx2[0] was manually physically swapped between cols, which required bayname to change and column_number to remain at 0.


        """
        super().__init__(
            voltage_source,
            db_cardname,
            bayname,
            delay_s,
            relock_threshold_lo_hi,
            easy_client,
            cringe_control,
            column_number, # gets put into self.col. Used for converting Dastard channels to Cringe column indices
            **kwargs
            )
        self.relock_lo_threshold = relock_threshold_lo_hi[0]
        self.relock_hi_threshold = relock_threshold_lo_hi[1]
        assert self.relock_hi_threshold - self.relock_lo_threshold > 2000
        self._relock_offset = np.zeros(self.ec.nrow)

    def get_iv_pt(self, dacvalue):
        self.set_volt(dacvalue)
        #data = self.ec.getNewData(delaySeconds=self.delay_s)
        # GCJ debugging this with try/except/finally
        try:
            data = self.ec.getNewData(delaySeconds=self.acq_delay,minimumNumPoints=16,exactNumPoints=True)
        except:
            print('\ngetNewData failed, trying once more')
            try:
                data = self.ec.getNewData(delaySeconds=self.acq_delay,minimumNumPoints=16,exactNumPoints=True)
            except:
                print('\ngetNewData failed again, trying one last time')
                data = self.ec.getNewData(delaySeconds=self.acq_delay,minimumNumPoints=16,exactNumPoints=True)             
        avg_col = data[self.col,:,:,1].mean(axis=-1)
        rows_relocked_hi = []
        rows_relocked_lo = []
        for row, fb in enumerate(avg_col):
            if fb < self.relock_lo_threshold:
                self.cc.relock_fba(self.col, row)
                rows_relocked_lo.append(row)
            if fb > self.relock_hi_threshold:
                self.cc.relock_fba(self.col, row)
                rows_relocked_hi.append(row)
        avg_col_out = avg_col[:]
        if len(rows_relocked_lo)+len(rows_relocked_hi) > 0:
            # at least one relock occured
            print(f"\nrelocked rows: too low {rows_relocked_lo}, too high {rows_relocked_hi}")
            data_after = self.ec.getNewData(delaySeconds=self.acq_delay,minimumNumPoints=16,exactNumPoints=True)
            avg_col_after = data_after[self.col,:,:,1].mean(axis=-1)
            for row in rows_relocked_lo+rows_relocked_hi:
                self._relock_offset[row] += avg_col_after[row]-avg_col[row]
                avg_col_out[row] = avg_col_after[row]
        return avg_col_out-self._relock_offset

    def prep_fb_settings(self, ARLoff=True, I=None, fba_offset = None):
        if ARLoff:
            print("setting ARL (autorelock) off")
            self.cc.set_arl_off(self.col)
        if I is not None:
            print(f"setting I to {I}")
            self.cc.set_fb_i(self.col, I)
        if fba_offset is not None:
            print(f"setting fba offset to {fba_offset}")
            self.cc.set_fba_offset(self.col, fba_offset)

    def relock_all_locked_rows(self):
        print("relock all locked rows")
        self.cc.relock_all_locked_fba(self.col)


class IVPointTakerMulti(IVPointTaker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._relock_offset = np.zeros(self.ec.nrow * self.ec.ncol)
        self.times_relocked = {}

    def get_iv_pt(self, dacvalue):
        self.set_volt(dacvalue)
        time.sleep(self.acq_delay)
        data = self.ec.getNewData2(16)
        numChans = self.ec.numRows*self.ec.numColumns
        processed_data = np.zeros(numChans)
        relocked_chans = []
        for i in range(numChans):
            try:
                relocks = self.times_relocked[i]
            except KeyError:
                relocks = 0
            chan_ts = data[f"chan{i}"] >> 2  # Following easyClient getNewData, throw out two lsb
            chan_avg = np.mean(chan_ts)
            processed_data[i] = chan_avg
            if not self.relock_lo_threshold < chan_avg < self.relock_hi_threshold:
                if relocks < 5: # Give up on a channel if it relocks a bunch of times really quickly
                    relocked_chans.append(i)
                    cringe_col_for_relock = self.dfb_chans[i // self.ec.numRows]
                    cringe_row_for_relock = i % self.ec.numRows
                    self.cc.relock_fba(cringe_col_for_relock, cringe_row_for_relock)
                    relocks += 1
            else:
                relocks = max(0, relocks-0.2)
                    
                # print(f"Relocked channel {i} (col={cringe_col_for_relock} row={cringe_row_for_relock})")
            self.times_relocked[i] = relocks
                # self.col is the indices for input channels as listed by cringecontrol
        
        if len(relocked_chans) > 0:
            time.sleep(self.acq_delay)
            post_relock_data = self.ec.getNewData2(16)
            print(f"Relocked: {relocked_chans}")
            for i in relocked_chans:
                avg_after = np.mean(post_relock_data[f"chan{i}"] >> 2)
                self._relock_offset[i] += avg_after - processed_data[i]
                processed_data[i] = avg_after

        return processed_data - self._relock_offset

    def prep_fb_settings(self, ARLoff=True, I=None, fba_offset = None):
        for c in self.dfb_chans:
            if ARLoff:
                print("setting ARL (autorelock) off")
                self.cc.set_arl_off(c)
            if I is not None:
                print(f"setting I to {I}")
                self.cc.set_fb_i(c, I)
            if fba_offset is not None:
                print(f"setting fba offset to {fba_offset}")
                self.cc.set_fba_offset(c, fba_offset)

    def relock_all_locked_rows(self):
        print("relock all locked rows")
        for c in self.dfb_chans:
            self.cc.relock_all_locked_fba(c)

class IVCurveTaker():
    def __init__(self, point_taker, temp_settle_delay_s=60, shock_normal_dac_value=2**16-1, zero_tower_at_end=True):
        self.pt = point_taker
        self.temp_settle_delay_s = temp_settle_delay_s
        self._last_setpoint_k = -1e9
        self.shock_normal_dac_value = shock_normal_dac_value
        self.zero_tower_at_end = zero_tower_at_end
        self._was_prepped = False
        self.post_overbias_settle_s = 30.0

    def set_temp_and_settle(self, setpoint_k):
        self.pt.adr.set_temp_k(float(setpoint_k))
        self._last_setpoint_k = setpoint_k
        print(f"set setpoint to {setpoint_k} K and now sleeping for {self.temp_settle_delay_s} s")
        time.sleep(self.temp_settle_delay_s)
        print("done sleeping")

    def overbias(self, overbias_temp_k, setpoint_k, dac_value, verbose=True):
        ''' raise ADR temperature above Tc, overbias bolometer, cool back to base temperature to keep bolometer in the normal state
        '''
        if verbose:
            print('Overbiasing detectors.  Raise temperature to %.1f mK, apply dac_value = %d, then cool to %.1f mK.'%(overbias_temp_k*1000,dac_value,setpoint_k*1000))
        self.pt.adr.set_temp_k(float(overbias_temp_k))
        ThighStable = self.is_temp_stable(overbias_temp_k,tol=0.005,time_out_s=180) # determine that it got to Thigh

        if verbose:
            if ThighStable:
                print('Successfully raised Tb > %.3f K.  Appling detector voltage bias and cooling back down.'%(overbias_temp_k))
            else:
                print('Could not get to the desired temperature above Tc.  Current temperature = ', self.pt.adr.get_temp_k())
        self.pt.set_volt(dac_value) # voltage bias to stay above Tc
        self.pt.adr.set_temp_k(float(setpoint_k)) # set back down to Tbath, base temperature
        TlowStable = self.is_temp_stable(setpoint_k, tol=0.001, time_out_s=180) # determine that it got to Tbath target
        if verbose:
            bar = progress.bar.Bar("Wait for temp to settle after over bias, %d seconds"%self.post_overbias_settle_s,max=100)
            if TlowStable:
                print('Successfully cooled back to base temperature '+str(setpoint_k)+'K')
                # settle after overbias
                for ii in range(100):
                    time.sleep(self.post_overbias_settle_s/100)
                    bar.next()
                bar.finish()
            else:
                print('Could not cool back to base temperature'+str(setpoint_k)+'. Current temperature = ', self.pt.adr.get_temp_k())

    def get_curve(self, dac_values, extra_info = {}, ignore_prep_requirement=False):
        assert ignore_prep_requirement or self._was_prepped, "call prep_fb_settings before get_curve, or pass ignore_prep_requirement=True"
        dac_values = self._handle_dac_values_int(dac_values)
        pre_temp_k = self.pt.adr.get_temp_k()
        pre_time = time.time()
        pre_hout = self.pt.adr.get_hout()
        # temp_rms and slope will not be very useful if you just changed temp, so get them at end only
        self.pt.set_volt(self.shock_normal_dac_value)
        time.sleep(0.05) # inserted because immediately commanding the lower dac value didn't take.
                         # was stuck at shock_normal_dac_value and this affected the first few points
        self.pt.set_volt(dac_values[0]) # go to the first dac value and relock all
        print(f"shocked detectors normal with dacvalue {self.shock_normal_dac_value}")
        self.pt.relock_all_locked_rows()
        time.sleep(5) # wait after shock to settle
        
        fb_values = []
        bar = progress.bar.Bar("getting IV points",max=len(dac_values))
        for dac_value in dac_values:
            fb_values.append(self.pt.get_iv_pt(dac_value))
            bar.next()
        bar.finish()
        post_temp_k = self.pt.adr.get_temp_k()
        post_time = time.time()
        post_hout = self.pt.adr.get_hout()
        #post_temp_rms_uk = self.pt.adr.get_temp_rms_uk()
        post_slope_hout_per_hour = self.pt.adr.get_slope_hout_per_hour()
        if self.zero_tower_at_end:
            print("zero detector bias")
            self.pt.set_volt(0)

        return IVCurveColumnData(nominal_temp_k = self._last_setpoint_k, pre_temp_k=pre_temp_k, post_temp_k=post_temp_k,
        pre_time_epoch_s = pre_time, post_time_epoch_s = post_time, pre_hout = pre_hout, post_hout = post_hout,
        post_slope_hout_per_hour = post_slope_hout_per_hour, dac_values = dac_values, bayname = self.pt.bayname,
        db_cardname = self.pt.db_cardname, column_number = self.pt.col, extra_info = extra_info, fb_values = fb_values,
        pre_shock_dac_value=self.shock_normal_dac_value)

    def _handle_dac_values_int(self,dac_values):
        '''ensure values passed to set_volt are integers, and recorded as such '''
        dac_values_int = []
        for dac_value in dac_values:
            dac_values_int.append(int(round(dac_value)))
        #dv = list(np.round(np.array(dac_values)).astype(int))
        #dac_values_int = getattr(dv, "tolist", lambda: dv)() # tried to be fancy, but didn't work.  Reverted to good old for loop.
        return dac_values_int

    def prep_fb_settings(self, ARLoff=True, I=None, fba_offset = 8000):
        self._was_prepped = True
        return self.pt.prep_fb_settings(ARLoff, I, fba_offset)

class IVTempSweeper():
    def __init__(self, curve_taker, to_normal_method=None, overbias_temp_k=None, overbias_dac_value = None):
        self.curve_taker = curve_taker
        self.to_normal_method = to_normal_method
        self.overbias_temp_k = overbias_temp_k
        self.overbias_dac_value = overbias_dac_value

    def initialize_bath_temp(self,set_temp_k):
        if self.to_normal_method == None:
            self.curve_taker.set_temp_and_settle(set_temp_k)
            self.curve_taker.pt.wait_temp_stable(set_temp_k, tol=.001, time_out_s=180)
        elif self.to_normal_method == "overbias":
            self.curve_taker.overbias(self.overbias_temp_k, setpoint_k = set_temp_k, dac_value=self.overbias_dac_value, verbose=True)
            self.curve_taker.set_temp_and_settle(set_temp_k)

    def get_sweep(self, dac_values, set_temps_k, extra_info={}):
        datas = []
        for set_temp_k in set_temps_k:
            self.initialize_bath_temp(set_temp_k)
            data = self.curve_taker.get_curve(dac_values, extra_info)
            datas.append(data)
        return IVTempSweepData(set_temps_k, datas)

class IVColdloadSweeper():
    def __init__(self, ivsweeper, loop_channel=1):
        self.ivsweeper = ivsweeper # instance of IVTempSweeper
        from instruments import Cryocon22
        self.ccon = Cryocon22()
        self.loop_channel = loop_channel

    def initialize_bath_temp(self,set_temp_k):
        if self.to_normal_method == None:
            self.curve_taker.set_temp_and_settle(set_temp_k)
        elif self.to_normal_method == "overbias":
            self.curve_taker.overbias(self.overbias_temp_k, setpoint_k = set_temp_k, dac_value=self.overbias_dac_value, verbose=True)

    def set_coldload_temp_and_settle(self, set_coldload_temp_k, tolerance_k=0.001,
                                   setpoint_timeout_m=5, post_setpoint_waittime_m=20,
                                   verbose=True):
        ''' servo coldload to temperature T and wait for temperature to stabilize '''
        assert set_coldload_temp_k<50.0, "Coldload temperature setpoint may not exceed 50K"
        if verbose: print('Setting BB temperature to '+str(set_coldload_temp_k)+'K')
        self.ccon.setControlTemperature(temp=set_coldload_temp_k,loop_channel=self.loop_channel)

        # wait for thermometer on coldload to reach set_coldload_temp_k --------------------------------------------
        is_stable = self.ccon.isTemperatureStable(self.loop_channel,tolerance_k)
        stable_num=0
        while not is_stable:
            time.sleep(5)
            is_stable = self.ccon.isTemperatureStable(self.loop_channel,tolerance_k)
            stable_num += 1
            if stable_num*5/60. > setpoint_timeout_m:
                break

        # settle at set_coldload_temp_k
        bar = progress.bar.Bar("Cold load thermalizing over %d minutes"%post_setpoint_waittime_m,max=100)
        for ii in range(100):
            time.sleep(60*post_setpoint_waittime_m/100)
            bar.next()
        bar.finish()

    def _prepareColdload(self,set_cl_temp_k):
        self.ccon.controlLoopSetup(loop_channel=self.loop_channel, control_temp=set_cl_temp_k,
                                t_channel='a',PID=[1,5,0], heater_range='low') # setup BB control
        #control_state = self.ccon.getControlLoopState()
        #if control_state == 'OFF': self.ccon.setControlTemperature(3.0,self.loop_channel) # set to temperature below achievable
        self.ccon.setControlState(state='on')

    def get_sweep(self, dac_values, set_cl_temps_k, set_temps_k, cl_temp_tolerance_k=0.001,
                  cl_settemp_timeout_m=5, cl_post_setpoint_waittime_m=20,
                  skip_first_settle = True,
                  cool_upon_finish = True, extra_info={},
                  write_while_acquire = False, filename=None):

        self._prepareColdload(set_cl_temps_k[0]) # control enabled after this point
        datas = []; pre_cl_temps_k = []; post_cl_temps_k =[]
        for ii, set_cl_temp_k in enumerate(set_cl_temps_k):
            if ii==0 and skip_first_settle: pass
            else:
                self.set_coldload_temp_and_settle(set_cl_temp_k,
                                              tolerance_k=cl_temp_tolerance_k,
                                              setpoint_timeout_m=cl_settemp_timeout_m,
                                              post_setpoint_waittime_m=cl_post_setpoint_waittime_m,
                                              verbose=True)
            pre_cl_temp_k = self.ccon.getTemperature()
            data = self.ivsweeper.get_sweep(dac_values, set_temps_k,
                                            extra_info={'coldload_temp_setpoint':set_cl_temp_k,'pre_coldload_temp':pre_cl_temp_k})
            post_cl_temp_k = self.ccon.getTemperature()
            datas.append(data); pre_cl_temps_k.append(pre_cl_temp_k); post_cl_temps_k.append(post_cl_temp_k)
            extra_info['pre_cl_temps_k']=pre_cl_temps_k
            extra_info['post_cl_temps_k']=post_cl_temps_k
            if write_while_acquire:
                temp_filename = _handle_file_extension(filename)
                temp_df = IVColdloadSweepData(set_cl_temps_k, datas, extra_info)
                temp_df.to_file(temp_filename,overwrite=True)

        if cool_upon_finish:
            print('Setting coldload to base temperature')
            self.ccon.setControlTemperature(3.0)
            self.ccon.setControlState('off')
        return IVColdloadSweepData(set_cl_temps_k, datas, extra_info)

def _handle_file_extension(filename, suffix='json'):
    if filename==None:
        new_filename='tempfile.'+suffix
    else:
        fname_split = filename.split('.')
        if len(fname_split)==1:
            new_filename = filename+'.'+suffix
        elif len(fname_split)==2:
            fext = fname_split[1]
            if fext != suffix:
                new_filename = filename + '.' + suffix
            else:
                new_filename = filename
    return new_filename

def sparse_then_fine_dacs(a, b, c, n_ab, n_bc):
    return np.hstack([np.linspace(a, b, n_ab), np.linspace(b, c, n_bc+1)[1:]])

def tc_tickle():
    plt.ion()
    plt.close("all")
    curve_taker = IVCurveTaker(IVPointTaker("DB1", "BX"), temp_settle_delay_s=0, shock_normal_dac_value=40000)
    curve_taker.set_temp_and_settle(0.05)
    time.sleep(0)
    curve_taker.set_temp_and_settle(0.01)
    t = []
    fbs = []
    while True:
        try:
            t.append(curve_taker.pt.adr.get_temp_k())
            fb0 = curve_taker.pt.get_iv_pt(0)
            fb1 = curve_taker.pt.get_iv_pt(50)
            fbs.append(fb1-fb0)
            plt.clf()
            plt.plot(np.array(t)*1e3, fbs)
            plt.xlabel("temp (mK)")
            plt.pause(0.1)
        except KeyboardInterrupt:
            break
    return t, np.vstack(fbs)


if __name__ == "__main__":
    # DEMONSTRATE IV POINT TAKER WORKS
    # ivpt = IVPointTaker('dfb_card','A',voltage_source='bluebox')
    # #dacvalue = int(0.7/ivpt.max_voltage*(2**16-1))
    # dacvalue = 10000
    # fb_values = ivpt.get_iv_pt(dacvalue)
    # plt.plot(fb_values,'o')
    # plt.show()

    # DEMONSTRATE IVCurveTaker works
    ivpt = IVPointTaker('dfb_card','A',voltage_source='bluebox') # instance of point taker class
    curve_taker = IVCurveTaker(ivpt, temp_settle_delay_s=0, shock_normal_dac_value=65000)
    #curve_taker.overbias(overbias_temp_k=0.2, setpoint_k=0.19, dac_value=10000, verbose=True)
    curve_taker.set_temp_and_settle(setpoint_k=0.1)
    curve_taker.prep_fb_settings(I=16, fba_offset=8192)
    dacs = np.linspace(10000,0,100)
    data = curve_taker.get_curve(dacs, extra_info = {})
    data.plot()
    plt.show()
    #data.to_file('lbird_iv_100mk_20210202_2.json',True)

    # # DEMONSTRATE IVTempSweeper
    # ivpt = IVPointTaker('dfb_card','A',voltage_source='bluebox') # instance of point taker class
    # curve_taker = IVCurveTaker(ivpt, temp_settle_delay_s=0, shock_normal_dac_value=2**16-1)
    # curve_taker.prep_fb_settings(I=16, fba_offset=8192)
    # ivsweeper = IVTempSweeper(curve_taker, to_normal_method=None, overbias_temp_k=0.2, overbias_dac_value = 7000)
    # dacs = np.linspace(10000,0,100)
    # temps = np.linspace(0.1,0.2,11)
    # data = ivsweeper.get_sweep(dacs, temps, extra_info={"state": "data used to develop IV versus temp sweep analysis class"})
    # data.to_file("lbird_hftv0_ivsweep_test.json", overwrite=True)

    # # DEMONSTRATE IVColdloadSweeper
    # filename = 'lbird_hftv0_coldload_sweep_20210203.json'
    # pt_taker = IVPointTaker(db_cardname='dfb_card', bayname='A', voltage_source = 'bluebox')
    # curve_taker = IVCurveTaker(pt_taker, temp_settle_delay_s=60, shock_normal_dac_value=65000, zero_tower_at_end=True, adr_gui_control=None)
    # curve_taker.prep_fb_settings(I=16, fba_offset=8192)
    # btemp_sweep_taker = IVTempSweeper(curve_taker, to_normal_method=None, overbias_temp_k=.21, overbias_dac_value = 7000)
    # clsweep_taker = IVColdloadSweeper(btemp_sweep_taker)
    # dacs = np.linspace(10000,0,100)
    # cl_temps = [4,5,6,7,8,9,10,11,12,11,10,9,8,7,6,5,4]
    # bath_temps = [0.1,.13,.17]
    # data = clsweep_taker.get_sweep(dacs, cl_temps, bath_temps, cl_temp_tolerance_k=0.01,
    #               cl_settemp_timeout_m=10.0, cl_post_setpoint_waittime_m=20.0,
    #               skip_first_settle = True,
    #               cool_upon_finish = True, extra_info={'message':'this is a test'},
    #               write_while_acquire = True, filename=filename)
    # data.to_file(filename,overwrite=True)
    # plt.clf()
    # plt.ion()
    # data.plot_row(1)
    # plt.show()

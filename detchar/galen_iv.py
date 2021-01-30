from cringe.cringe_control import CringeControl
from adr_gui.adr_gui_control import AdrGuiControl
from nasa_client import EasyClient
import time
import numpy as np
import pylab as plt
import progress.bar
import os
#from . iv_data import IVCurveColumnData, IVTempSweepData, IVColdLoadTempSweepData, IVCircuit
from iv_data import IVCurveColumnData, IVTempSweepData, IVColdLoadTempSweepData, IVCircuit
from instruments import BlueBox 

class IVPointTaker():
    def __init__(self, db_cardname, bayname, delay_s=0.05, relock_threshold_lo_hi  = (2000, 14000), 
    easy_client=None, cringe_control=None, voltage_source = None, column_number = 0):
        self.ec = self._handle_easy_client_arg(easy_client)
        self.cc = self._handle_cringe_control_arg(cringe_control)
        # these should be args
        self.delay_s = delay_s
        self.relock_lo_threshold = relock_threshold_lo_hi[0]
        self.relock_hi_threshold = relock_threshold_lo_hi[1]
        assert self.relock_hi_threshold - self.relock_lo_threshold > 2000
        self.db_cardname = db_cardname
        self.bayname = bayname
        self.col = column_number
        self._relock_offset = np.zeros(self.ec.nrow)
        self.set_volt = self._handle_voltage_source_arg(voltage_source)
    
    def _handle_voltage_source_arg(self,voltage_source):
        # set "set_volt" to either tower or bluebox
        if voltage_source == None:
            set_volt = self.set_tower # 0-2.5V in 2**16 steps
            self.max_voltage = 2.5
        elif voltage_source == 'bluebox':
            self.bb = BlueBox(port='vbox', version='mrk2')
            set_volt = self.set_bluebox # 0 to 6.5535V in 2**16 steps
            self.max_voltage = self.bb.max_voltage
        return set_volt

    def _handle_easy_client_arg(self, easy_client):
        if easy_client is not None:
            return easy_client
        easy_client = EasyClient()
        easy_client.setupAndChooseChannels()
        return easy_client           

    def _handle_cringe_control_arg(self, cringe_control):
        if cringe_control is not None:
             return cringe_control
        return CringeControl()

    def get_iv_pt(self, dacvalue):
        self.set_volt(dacvalue)
        data = self.ec.getNewData(delaySeconds=self.delay_s)
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
            data_after = self.ec.getNewData(delaySeconds=self.delay_s)
            avg_col_after = data_after[self.col,:,:,1].mean(axis=-1)
            for row in rows_relocked_lo+rows_relocked_hi:
                self._relock_offset[row] += avg_col_after[row]-avg_col[row]
                avg_col_out[row] = avg_col_after[row]
        return avg_col_out-self._relock_offset

    def set_tower(self, dacvalue):
        self.cc.set_tower_channel(self.db_cardname, self.bayname, int(dacvalue))

    def set_bluebox(self, dacvalue):
        self.bb.setVoltDACUnits(dacvalue)

    def prep_fb_settings(self, ARLoff=True, I=None, fba_offset = 8000):
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

class IVCurveTaker():
    def __init__(self, point_taker, temp_settle_delay_s, shock_normal_dac_value, zero_tower_at_end=True, adr_gui_control=None):
        self.pt = point_taker
        self.adr_gui_control = self._handle_adr_gui_control_arg(adr_gui_control)
        self.temp_settle_delay_s = temp_settle_delay_s
        self._last_setpoint_k = -1e9
        self.shock_normal_dac_value = shock_normal_dac_value
        self.zero_tower_at_end = zero_tower_at_end
        self._was_prepped = False
        self.post_overbias_wait_time_s = 30.0

    def _handle_adr_gui_control_arg(self, adr_gui_control):
        if adr_gui_control is not None:
            return adr_gui_control
        return AdrGuiControl()

    def set_temp_and_settle(self, setpoint_k):
        self.adr_gui_control.set_temp_k(float(setpoint_k))
        self._last_setpoint_k = setpoint_k
        print(f"set setpoint to {setpoint_k} K and now sleeping for {self.temp_settle_delay_s} s")
        time.sleep(self.temp_settle_delay_s)
        print(f"done sleeping")

    def overbias(self, overbias_temp_k, setpoint_k, dac_value, verbose=True):
        ''' raise ADR temperature above Tc, overbias bolometer, cool back to base temperature to keep bolometer in the normal state
        '''
        if verbose:
            print('Overbiasing detectors.  Raise temperature to %.1f mK, apply dac_value = %d, then cool to %.1f mK.'%(overbias_temp_k*1000,dac_value,setpoint_k*1000))
        self.adr_gui_control.set_temp_k(float(overbias_temp_k))
        ThighStable = self.IsTemperatureStable(overbias_temp_k,tol=0.005,time_out_s=180) # determine that it got to Thigh
        
        if verbose:
            if ThighStable:
                print('Successfully raised Tb > %.3f K.  Appling detector voltage bias and cooling back down.'%(overbias_temp_k))
            else:
                print('Could not get to the desired temperature above Tc.  Current temperature = ', self.adr_gui_control.get_temp_k())
        self.pt.set_volt(dac_value) # voltage bias to stay above Tc
        self.adr_gui_control.set_temp_k(float(setpoint_k)) # set back down to Tbath, base temperature
        TlowStable = self.IsTemperatureStable(setpoint_k,tol=0.001,time_out_s=180) # determine that it got to Tbath target
        if verbose:     
            if TlowStable:
                print('Successfully cooled back to base temperature '+str(setpoint_k)+'K')
                time.sleep(self.post_overbias_wait_time_s)
            else:
                print('Could not cool back to base temperature'+str(setpoint_k)+'. Current temperature = ', self.adr_gui_control.get_temp_k())
        

    def IsTemperatureStable(self, setpoint_k, tol=.005, time_out_s=180):
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

    def get_curve(self, dac_values, extra_info = {}, ignore_prep_requirement=False):
        assert ignore_prep_requirement or self._was_prepped, "call prep_fb_settings before get_curve, or pass ignore_prep_requirement=True"
        pre_temp_k = self.adr_gui_control.get_temp_k()
        pre_time = time.time()
        pre_hout = self.adr_gui_control.get_hout()
        # temp_rms and slope will not be very useful if you just changed temp, so get them at end only
        self.pt.set_volt(self.shock_normal_dac_value)
        time.sleep(0.05) # inserted because immediately commanding the lower dac value didn't take.  
                         # was stuck at shock_normal_dac_value and this affected the first few points
        self.pt.set_volt(dac_values[0]) # go to the first dac value and relock all
        self.pt.relock_all_locked_rows()
        print(f"shock detectors normal with dacvalue {self.shock_normal_dac_value}")
        time.sleep(3) # wait after shock to settle
        fb_values = []
        bar = progress.bar.Bar("getting IV points",max=len(dac_values))
        for dac_value in dac_values:
            fb_values.append(self.pt.get_iv_pt(dac_value))
            bar.next()
        bar.finish()
        post_temp_k = self.adr_gui_control.get_temp_k()
        post_time = time.time()
        post_hout = self.adr_gui_control.get_hout()
        post_temp_rms_uk = self.adr_gui_control.get_temp_rms_uk()
        post_slope_hout_per_hour = self.adr_gui_control.get_slope_hout_per_hour()
        if self.zero_tower_at_end:
            print(f"zero detector bias")
            self.pt.set_volt(0)

        return IVCurveColumnData(nominal_temp_k = self._last_setpoint_k, pre_temp_k=pre_temp_k, post_temp_k=post_temp_k,
        pre_time_epoch_s = pre_time, post_time_epoch_s = post_time, pre_hout = pre_hout, post_hout = post_hout,
        post_slope_hout_per_hour = post_slope_hout_per_hour, dac_values = dac_values, bayname = self.pt.bayname,
        db_cardname = self.pt.db_cardname, column_number = self.pt.col, extra_info = extra_info, fb_values = fb_values, 
        pre_shock_dac_value=self.shock_normal_dac_value)

    def prep_fb_settings(self, ARLoff=True, I=None, fba_offset = 8000):
        self._was_prepped = True
        return self.pt.prep_fb_settings(ARLoff, I, fba_offset)

class IVTempSweeper():
    def __init__(self, curve_taker, to_normal_method=None, overbias_temp_k=None, overbias_dac_value = None):
        self.curve_taker = curve_taker
        self.to_normal_method = to_normal_method
        self.overbias_temp_k = overbias_temp_k
        self.overbias_dac_value = overbias_dac_value

    def initializeTemperature(self,set_temp_k):
        if self.to_normal_method == None:
            self.curve_taker.set_temp_and_settle(set_temp_k)
        elif self.to_normal_method == "overbias":
            self.curve_taker.overbias(self.overbias_temp_k, setpoint_k = set_temp_k, dac_value=self.overbias_dac_value, verbose=True)

    def get_sweep(self, dac_values, set_temps_k, extra_info={}):
        datas = []
        for set_temp_k in set_temps_k:
            self.initializeTemperature(set_temp_k) 
            data = self.curve_taker.get_curve(dac_values, extra_info)
            datas.append(data)
        return IVTempSweepData(set_temps_k, datas)

class IVColdLoadTempSweeper():
    def __init__(self, curve_taker):
        self.curve_taker = curve_taker

    def get_sweep(self, dac_values, set_temps_k, extra_info={}):
        datas = []
        for set_temp_k in set_temps_k:
            self.curve_taker.set_temp_and_settle(set_temp_k)
            data = self.curve_taker.get_curve(dac_values, extra_info)
            datas.append(data)
        return IVTempSweepData(set_temps_k, datas)

def iv_to_frac_rn_array(x, y, superconducting_below_x, normal_above_x):
    rs = [iv_to_frac_rn_single(x, y[:, i], superconducting_below_x, normal_above_x) for i in range(y.shape[1])]
    return np.vstack(rs).T

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
            t.append(curve_taker.adr_gui_control.get_temp_k())
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
    # plt.ion()
    # plt.close("all")
    # curve_taker = IVCurveTaker(IVPointTaker("DB1", "BX"), temp_settle_delay_s=0, shock_normal_dac_value=40000)
    # curve_taker.set_temp_and_settle(setpoint_k=0.075)
    # curve_taker.pt.prep_fb_settings(I=10, fba_offset=3000)
    # dacs = sparse_then_fine_dacs(a=40000, b = 4000, c=0, n_ab=40, n_bc=250)
    # data = curve_taker.get_curve(dacs, extra_info = {"magnetic field current (amps)": 1e-6})
    # data.plot()
    # data.to_file("ivtest.json", overwrite=True)
    # data2 = IVCurveColumnData.from_json(data.to_json())
    # assert data2.pre_time_epoch_s == data.pre_time_epoch_s
    # data = IVCurveColumnData.from_file("ivtest.json")
    # x, y = data.xy_arrays_zero_subtracted()
    # r = iv_to_frac_rn_array(x, y, superconducting_below_x=2000, normal_above_x=24000)
    # plt.figure()
    # plt.plot(x, r)
    # plt.xlabel("dac value")
    # plt.ylabel("fraction R_n")
    # plt.legend([f"row{i}" for i in range(r.shape[1])])
    # plt.vlines(2750, 0, 1)
    # plt.grid(True)



    # t, y = tc_tickle()
    # y = np.vstack(fbs)
    # plt.clf()
    # plt.plot(np.array(t)*1e3, y)
    # plt.xlabel("temp (mK)")
    # plt.ylabel("delta fb from 50 dac unit tickle")
    # plt.pause(0.1)

    # plt.ion()
    # plt.close("all")
    # curve_taker = IVCurveTaker(IVPointTaker("DB1", "BX"), temp_settle_delay_s=180, shock_normal_dac_value=40000)
    # curve_taker.prep_fb_settings()
    # temp_sweeper = IVTempSweeper(curve_taker)
    # dacs = sparse_then_fine_dacs(a=40000, b = 10000, c=0, n_ab=20, n_bc=100)
    # temps_mk = np.linspace(60,100,16)
    # print(f"{temps_mk} mK")
    # sweep = temp_sweeper.get_sweep(dacs, 
    #     set_temps_k=temps_mk*1e-3, 
    #     extra_info={"field coil current (Amps)":0})
    # sweep.to_file("iv_sweep_test2.json", overwrite=True)
    # sweep2 = IVTempSweepData.from_file("iv_sweep_test2.json")
    # sweep2.plot_row(row=0)

    # DEMONSTRATE IV POINT TAKER WORKS
    # ivpt = IVPointTaker('dfb_card','A',voltage_source='bluebox')
    # dacvalue = int(0.7/ivpt.max_voltage*(2**16-1))
    # fb_values = ivpt.get_iv_pt(dacvalue)
    # plt.plot(fb_values,'o')
    # plt.show()

    # # DEMONSTRATE IVCurveTaker works 
    # ivpt = IVPointTaker('dfb_card','A',voltage_source='bluebox') # instance of point taker class 
    # curve_taker = IVCurveTaker(ivpt, temp_settle_delay_s=0, shock_normal_dac_value=65535)
    # curve_taker.overbias(overbias_temp_k=0.2, setpoint_k=0.19, dac_value=10000, verbose=True)
    # #curve_taker.set_temp_and_settle(setpoint_k=0.13)
    # curve_taker.prep_fb_settings(I=16, fba_offset=8192)
    # v_bias = np.linspace(0.7,0.0,100)
    # dacs = v_bias/ivpt.max_voltage*(2**16-1); dacs = dacs.astype(int)
    # data = curve_taker.get_curve(dacs, extra_info = {"hannes is rad": "yes"})
    # data.plot()
    # plt.show()

    # DEMONSTRATE IVTempSweeper
    ivpt = IVPointTaker('dfb_card','A',voltage_source='bluebox') # instance of point taker class 
    curve_taker = IVCurveTaker(ivpt, temp_settle_delay_s=0, shock_normal_dac_value=65535)
    curve_taker.prep_fb_settings(I=16, fba_offset=8192)
    ivsweeper = IVTempSweeper(curve_taker, to_normal_method="overbias", overbias_temp_k=0.2, overbias_dac_value = 10000)
    v_bias = np.linspace(0.7,0.0,100)
    dacs = v_bias/ivpt.max_voltage*(2**16-1); dacs = dacs.astype(int)
    #temps = [.13,.14,.15,.16,.17,.18,.19,.2]
    temps = [.19,.2]
    data = ivsweeper.get_sweep(dacs, temps, extra_info={})
    data.to_file("lbird_hftv0_ivsweep_test.json", overwrite=True)
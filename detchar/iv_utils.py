""" iv_util.py
    @author Galen O'Neil and Hannes Hubmayr

    was previously "galen_iv.py"
"""
from cringe.cringe_control import CringeControl
from adr_gui.adr_gui_control import AdrGuiControl
from nasa_client import EasyClient
import time
import numpy as np
import pylab as plt
import progress.bar
import os
from detchar.iv_data import (
    IVCurveColumnData,
    IVTempSweepData,
    IVColdloadSweepData,
    IVCircuit,
)
from instruments import BlueBox


class AdrGuiControlDummy():
    def get_temp_k(self):
        return -1

    def set_temp_k(self, setpoint_k):
        return True

    def get_temp_rms_uk(self):
        return -1

    def get_hout(self):
        return -1

    def get_slope_hout_per_hour(self):
        return 1e6

class IVPointTaker:
    def __init__(
        self,
        db_cardname,
        bayname,
        delay_s=0.05,
        relock_threshold_lo_hi=(2000, 14000),
        easy_client=None,
        cringe_control=None,
        voltage_source=None,
        column_number=0,
    ):
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

    def _handle_voltage_source_arg(self, voltage_source):
        # set "set_volt" to either tower or bluebox
        if voltage_source == None:
            set_volt = self.set_tower  # 0-2.5V in 2**16 steps
        elif voltage_source == "bluebox":
            self.bb = BlueBox(port="tower", version="mrk2")
            set_volt = self.set_bluebox  # 0 to 6.5535V in 2**16 steps
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

    def get_fb_raw(self):
        data = self.ec.getNewData(delaySeconds=self.delay_s)
        avg_col = data[self.col, :, :, 1].mean(axis=-1)
        return np.array(avg_col,dtype="float64") # we can't json serialize np.float32, which is the element type of avg_col

    def get_iv_pt(self, dacvalue):
        self.set_volt(dacvalue)
        avg_col = self.get_fb_raw()
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
        if len(rows_relocked_lo) + len(rows_relocked_hi) > 0:
            # at least one relock occured
            print(
                f"\nrelocked rows: too low {rows_relocked_lo}, too high {rows_relocked_hi}"
            )
            data_after = self.ec.getNewData(delaySeconds=self.delay_s)
            avg_col_after = data_after[self.col, :, :, 1].mean(axis=-1)
            for row in rows_relocked_lo + rows_relocked_hi:
                self._relock_offset[row] += avg_col_after[row] - avg_col[row]
                avg_col_out[row] = avg_col_after[row]
        return avg_col_out - self._relock_offset


    def set_tower(self, dacvalue):
        self.cc.set_tower_channel(self.db_cardname, self.bayname, int(dacvalue))

    def set_bluebox(self, dacvalue):
        self.bb.setVoltDACUnits(int(dacvalue))

    def prep_fb_settings(self, ARLoff=True, I=None, fba_offset=8000):
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

    def reset_for_new_curve(self):
        self._relock_offset = np.zeros(self.ec.nrow)


class IVCurveTaker:
    def __init__(
        self,
        point_taker,
        temp_settle_time_out_s=180,
        temp_settle_tolerance_k=0.0002,
        shock_normal_dac_value=2 ** 16 - 1,
        zero_bias_and_relock_and_record_fb_at_end=True,
        adr_gui_control=None,
    ):
        self.pt = point_taker
        self.adr_gui_control = self._handle_adr_gui_control_arg(adr_gui_control)
        self._last_setpoint_k = -1e9
        self.temp_settle_time_out_s = temp_settle_time_out_s
        self.temp_settle_tolerance_k = temp_settle_tolerance_k
        self.shock_normal_dac_value = shock_normal_dac_value
        self.zero_bias_and_relock_and_record_fb_at_end = zero_bias_and_relock_and_record_fb_at_end
        self._was_prepped = False
        self.post_overbias_settle_s = 30.0

    def _handle_adr_gui_control_arg(self, adr_gui_control):
        if adr_gui_control is not None:
            return adr_gui_control
        return None

    def set_temp_and_settle(
        self, setpoint_k, temp_settle_time_out_s=None, temp_settle_tolerance_k=None
    ):
        if temp_settle_time_out_s is None:
            temp_settle_time_out_s = self.temp_settle_time_out_s
        if temp_settle_tolerance_k is None:
            temp_settle_tolerance_k = self.temp_settle_tolerance_k
        self.adr_gui_control.set_temp_k(float(setpoint_k))
        self._last_setpoint_k = setpoint_k
        print(
            f"set setpoint to {setpoint_k} K and now wait for stable with time out {temp_settle_time_out_s} s and tolerance {temp_settle_tolerance_k} K"
        )
        self.wait_for_temp_stable(
            setpoint_k, temp_settle_tolerance_k, temp_settle_time_out_s
        )
        print(f"done settling")

    def overbias(self, overbias_temp_k, setpoint_k, dac_value, verbose=True):
        """raise ADR temperature above Tc, overbias bolometer, cool back to base temperature to keep bolometer in the normal state"""
        if verbose:
            print(
                "Overbiasing detectors.  Raise temperature to %.1f mK, apply dac_value = %d, then cool to %.1f mK."
                % (overbias_temp_k * 1000, dac_value, setpoint_k * 1000)
            )
        self.adr_gui_control.set_temp_k(float(overbias_temp_k))
        ThighStable = self.is_temp_stable(
            overbias_temp_k, tol=0.005, time_out_s=180
        )  # determine that it got to Thigh

        if verbose:
            if ThighStable:
                print(
                    "Successfully raised Tb > %.3f K.  Appling detector voltage bias and cooling back down."
                    % (overbias_temp_k)
                )
            else:
                print(
                    "Could not get to the desired temperature above Tc.  Current temperature = ",
                    self.adr_gui_control.get_temp_k(),
                )
        self.pt.set_volt(dac_value)  # voltage bias to stay above Tc
        self.adr_gui_control.set_temp_k(
            float(setpoint_k)
        )  # set back down to Tbath, base temperature
        TlowStable = self.is_temp_stable(
            setpoint_k, tol=0.001, time_out_s=180
        )  # determine that it got to Tbath target
        if verbose:
            if TlowStable:
                print(
                    "Successfully cooled back to base temperature "
                    + str(setpoint_k)
                    + "K"
                )
                # settle after overbias
                bar = progress.bar.Bar(
                    "Wait for temp to settle after over bias, %d seconds"
                    % self.post_overbias_settle_s,
                    max=100,
                )
                for ii in range(100):
                    time.sleep(self.post_overbias_settle_s / 100)
                    bar.next()
                bar.finish()
            else:
                print(
                    "Could not cool back to base temperature"
                    + str(setpoint_k)
                    + ". Current temperature = ",
                    self.adr_gui_control.get_temp_k(),
                )

    def wait_for_temp_stable(self, setpoint_k, tolerance_k, time_out_s, sleep_size_s=5):
        """ determine if the servo has reached the desired temperature """
        assert sleep_size_s > 0.001
        latest_temp_k = self.adr_gui_control.get_temp_k()
        tstart = time.time()
        while abs(latest_temp_k - setpoint_k) > tolerance_k:
            time.sleep(sleep_size_s)
            latest_temp_k = self.adr_gui_control.get_temp_k()
            print(f"Current Temp: {latest_temp_k*1e3:.02f} mK")
            elapsed = time.time() - tstart
            if elapsed > time_out_s:
                print(
                    f"wait_for_temp_stable TIMED OUT after {time_out_s} s, setpoint_k={setpoint_k}, with tolerance_k={tolerance_k}"
                )
                return False
        print(
            f"wait_for_temp_stable REACHED TEMP after {time_out_s} s, setpoint_k={setpoint_k}, with tolerance_k={tolerance_k}"
        )
        return True

    def get_curve(self, dac_values, extra_info={}, ignore_prep_requirement=False):
        assert (
            ignore_prep_requirement or self._was_prepped
        ), "call prep_fb_settings before get_curve, or pass ignore_prep_requirement=True"
        dac_values = self._handle_dac_values_int(dac_values)
        pre_temp_k = self.adr_gui_control.get_temp_k()
        pre_time = time.time()
        pre_hout = self.adr_gui_control.get_hout()
        # temp_rms and slope will not be very useful if you just changed temp, so get them at end only
        self.pt.reset_for_new_curve()
        self.pt.set_volt(self.shock_normal_dac_value)
        time.sleep(
            0.05
        )  # inserted because immediately commanding the lower dac value didn't take.
        # was stuck at shock_normal_dac_value and this affected the first few points
        self.pt.set_volt(dac_values[0])  # go to the first dac value and relock all
        time.sleep(0.05)
        self.pt.relock_all_locked_rows()
        fb_values = []
        bar = progress.bar.Bar("getting IV points", max=len(dac_values))
        for dac_value in dac_values:
            fb_values.append(self.pt.get_iv_pt(dac_value))
            bar.next()
        bar.finish()
        post_temp_k = self.adr_gui_control.get_temp_k()
        post_time = time.time()
        post_hout = self.adr_gui_control.get_hout()
        post_temp_rms_uk = self.adr_gui_control.get_temp_rms_uk()
        post_slope_hout_per_hour = self.adr_gui_control.get_slope_hout_per_hour()
        if self.zero_bias_and_relock_and_record_fb_at_end:
            print(f"zero detector bias, relock, and record fb")
            self.pt.set_volt(0)
            time.sleep(0.05)
            self.pt.relock_all_locked_rows()
            zero_bias_fb = self.pt.get_fb_raw()
        else:
            zero_bias_fb = None

        return IVCurveColumnData(
            nominal_temp_k=self._last_setpoint_k,
            pre_temp_k=pre_temp_k,
            post_temp_k=post_temp_k,
            pre_time_epoch_s=pre_time,
            post_time_epoch_s=post_time,
            pre_hout=pre_hout,
            post_hout=post_hout,
            post_slope_hout_per_hour=post_slope_hout_per_hour,
            dac_values=dac_values,
            bayname=self.pt.bayname,
            db_cardname=self.pt.db_cardname,
            column_number=self.pt.col,
            extra_info=extra_info,
            fb_values=fb_values,
            pre_shock_dac_value=self.shock_normal_dac_value,
            zero_bias_fb=zero_bias_fb
        )

    def _handle_dac_values_int(self, dac_values):
        """ensure values passed to set_volt are integers, and recorded as such """
        dac_values_int = []
        for dac_value in dac_values:
            dac_values_int.append(int(round(dac_value)))
        # dv = list(np.round(np.array(dac_values)).astype(int))
        # dac_values_int = getattr(dv, "tolist", lambda: dv)() # tried to be fancy, but didn't work.  Reverted to good old for loop.
        return dac_values_int

    def prep_fb_settings(self, ARLoff=True, I=None, fba_offset=8000):
        self._was_prepped = True
        return self.pt.prep_fb_settings(ARLoff, I, fba_offset)


class IVTempSweeper:
    def __init__(
        self,
        curve_taker,
        to_normal_method=None,
        overbias_temp_k=None,
        overbias_dac_value=None,
    ):
        self.curve_taker = curve_taker
        self.to_normal_method = to_normal_method
        self.overbias_temp_k = overbias_temp_k
        self.overbias_dac_value = overbias_dac_value

    def initialize_bath_temp(self, set_temp_k):
        if self.to_normal_method == None:
            self.curve_taker.set_temp_and_settle(set_temp_k)
        elif self.to_normal_method == "overbias":
            self.curve_taker.overbias(
                self.overbias_temp_k,
                setpoint_k=set_temp_k,
                dac_value=self.overbias_dac_value,
                verbose=True,
            )
            self.curve_taker.set_temp_and_settle(set_temp_k)

    def get_sweep(self, dac_values, set_temps_k, extra_info={}):
        datas = []
        for set_temp_k in set_temps_k:
            self.initialize_bath_temp(set_temp_k)
            data = self.curve_taker.get_curve(dac_values, extra_info)
            datas.append(data)
        return IVTempSweepData(set_temps_k, datas)


class IVColdloadSweeper:
    def __init__(self, ivsweeper, loop_channel=1):
        self.ivsweeper = ivsweeper  # instance of IVTempSweeper
        from instruments import Cryocon22

        self.ccon = Cryocon22()
        self.loop_channel = loop_channel

    def initialize_bath_temp(self, set_temp_k):
        if self.to_normal_method == None:
            self.curve_taker.set_temp_and_settle(set_temp_k)
        elif self.to_normal_method == "overbias":
            self.curve_taker.overbias(
                self.overbias_temp_k,
                setpoint_k=set_temp_k,
                dac_value=self.overbias_dac_value,
                verbose=True,
            )

    def set_coldload_temp_and_settle(
        self,
        set_coldload_temp_k,
        tolerance_k=0.001,
        setpoint_timeout_m=5,
        post_setpoint_waittime_m=20,
        verbose=True,
    ):
        """ servo coldload to temperature T and wait for temperature to stabilize """
        assert (
            set_coldload_temp_k < 50.0
        ), "Coldload temperature setpoint may not exceed 50K"
        if verbose:
            print("Setting BB temperature to " + str(set_coldload_temp_k) + "K")
        self.ccon.setControlTemperature(
            temp=set_coldload_temp_k, loop_channel=self.loop_channel
        )

        # wait for thermometer on coldload to reach set_coldload_temp_k --------------------------------------------
        is_stable = self.ccon.isTemperatureStable(self.loop_channel, tolerance_k)
        stable_num = 0
        while not is_stable:
            time.sleep(5)
            is_stable = self.ccon.isTemperatureStable(self.loop_channel, tolerance_k)
            stable_num += 1
            if stable_num * 5 / 60.0 > setpoint_timeout_m:
                break

        # settle at set_coldload_temp_k
        bar = progress.bar.Bar(
            "Cold load thermalizing over %d minutes" % post_setpoint_waittime_m, max=100
        )
        for ii in range(100):
            time.sleep(60 * post_setpoint_waittime_m / 100)
            bar.next()
        bar.finish()

    def _prepareColdload(self, set_cl_temp_k):
        self.ccon.controlLoopSetup(
            loop_channel=self.loop_channel,
            control_temp=set_cl_temp_k,
            t_channel="a",
            PID=[1, 5, 0],
            heater_range="low",
        )  # setup BB control
        # control_state = self.ccon.getControlLoopState()
        # if control_state == 'OFF': self.ccon.setControlTemperature(3.0,self.loop_channel) # set to temperature below achievable
        self.ccon.setControlState(state="on")

    def get_sweep(
        self,
        dac_values,
        set_cl_temps_k,
        set_temps_k,
        cl_temp_tolerance_k=0.001,
        cl_settemp_timeout_m=5,
        cl_post_setpoint_waittime_m=20,
        skip_first_settle=True,
        cool_upon_finish=True,
        extra_info={},
        write_while_acquire=False,
        filename=None,
    ):

        self._prepareColdload(set_cl_temps_k[0])  # control enabled after this point
        datas = []
        pre_cl_temps_k = []
        post_cl_temps_k = []
        for ii, set_cl_temp_k in enumerate(set_cl_temps_k):
            if ii == 0 and skip_first_settle:
                pass
            else:
                self.set_coldload_temp_and_settle(
                    set_cl_temp_k,
                    tolerance_k=cl_temp_tolerance_k,
                    setpoint_timeout_m=cl_settemp_timeout_m,
                    post_setpoint_waittime_m=cl_post_setpoint_waittime_m,
                    verbose=True,
                )
            pre_cl_temp_k = self.ccon.getTemperature()
            data = self.ivsweeper.get_sweep(
                dac_values,
                set_temps_k,
                extra_info={
                    "coldload_temp_setpoint": set_cl_temp_k,
                    "pre_coldload_temp": pre_cl_temp_k,
                },
            )
            post_cl_temp_k = self.ccon.getTemperature()
            datas.append(data)
            pre_cl_temps_k.append(pre_cl_temp_k)
            post_cl_temps_k.append(post_cl_temp_k)
            extra_info["pre_cl_temps_k"] = pre_cl_temps_k
            extra_info["post_cl_temps_k"] = post_cl_temps_k
            if write_while_acquire:
                temp_filename = _handle_file_extension(filename)
                temp_df = IVColdloadSweepData(set_cl_temps_k, datas, extra_info)
                temp_df.to_file(temp_filename, overwrite=True)

        if cool_upon_finish:
            print("Setting coldload to base temperature")
            self.ccon.setControlTemperature(3.0)
            self.ccon.setControlState("off")
        return IVColdloadSweepData(set_cl_temps_k, datas, extra_info)


def _handle_file_extension(filename, suffix=".json"):
    if filename == None:
        new_filename = "tempfile" + suffix
    else:
        fname_split = filename.split(".")
        if len(fname_split) == 1:
            new_filename = filename + suffix
        elif len(fname_split) == 2:
            fext = fname_split[1]
            if fext != suffix:
                new_filename = filename + suffix
            else:
                new_filename = filename
    return new_filename


def sparse_then_fine_dacs(a, b, c, n_ab, n_bc):
    return np.hstack([np.linspace(a, b, n_ab), np.linspace(b, c, n_bc + 1)[1:]])

from cringe.cringe_control import CringeControl
from adr_gui.adr_gui_control import AdrGuiControl
from nasa_client import EasyClient
import time
from dataclasses import dataclass
import dataclasses
from dataclasses_json import dataclass_json
from typing import Any, List
import numpy as np
import pylab as plt
import progress.bar
import collections
import os
from instruments import bluebox

class IVPointTaker():
    def __init__(self, db_cardname, bayname, delay_s=0.05, relock_threshold_lo_hi  = (2000, 14000), 
    easy_client=None, cringe_control=None, column_number = 0):
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
        self.bb = bluebox.BlueBox(version="mrk2")

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
        self.set_tower(dacvalue)
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
        self.bb.setVoltDACUnits(int(dacvalue))

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


    def get_curve(self, dac_values, extra_info = {}, ignore_prep_requirement=False):
        assert ignore_prep_requirement or self._was_prepped, "call prep_fb_settings before get_curve, or pass ignore_prep_requirement=True"
        pre_temp_k = self.adr_gui_control.get_temp_k()
        pre_time = time.time()
        pre_hout = self.adr_gui_control.get_hout()
        # temp_rms and slope will not be very useful if you just changed temp, so get them at end only
        self.pt.set_tower(self.shock_normal_dac_value)
        self.pt.set_tower(dac_values[0]) # go to the first dac value and relock all
        self.pt.relock_all_locked_rows()
        print(f"shock detectors normal with dacvalue {self.shock_normal_dac_value}")
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
            self.pt.set_tower(0)

        return IVCurveColumnData(nominal_temp_k = self._last_setpoint_k, pre_temp_k=pre_temp_k, post_temp_k=post_temp_k,
        pre_time_epoch_s = pre_time, post_time_epoch_s = post_time, pre_hout = pre_hout, post_hout = post_hout,
        post_slope_hout_per_hour = post_slope_hout_per_hour, dac_values = dac_values, bayname = self.pt.bayname,
        db_cardname = self.pt.db_cardname, column_number = self.pt.col, extra_info = extra_info, fb_values = fb_values, 
        pre_shock_dac_value=self.shock_normal_dac_value)

    def prep_fb_settings(self, ARLoff=True, I=None, fba_offset = 8000):
        self._was_prepped = True
        return self.pt.prep_fb_settings(ARLoff, I, fba_offset)
@dataclass_json
@dataclass
class IVCurveColumnData():
    nominal_temp_k: float
    pre_temp_k: float
    post_temp_k: float
    pre_time_epoch_s: float
    post_time_epoch_s: float
    pre_hout: float
    post_hout: float
    post_slope_hout_per_hour: float
    dac_values: List[float]
    fb_values: List[Any] = dataclasses.field(repr=False) #actually a list of np arrays
    bayname: str
    db_cardname: str
    column_number: int
    extra_info: dict
    pre_shock_dac_value: float

    def plot(self):
        plt.figure()
        plt.plot(self.dac_values, self.fb_values)
        plt.xlabel("dac values (arb)")
        plt.ylabel("fb values (arb)")
        plt.title(f"bay {self.bayname}, db_card {self.db_cardname}, nominal_temp_mk {self.nominal_temp_k*1000}")

    def to_file(self, filename, overwrite = False):
        if not overwrite:
            assert not os.path.isfile(filename)
        with open(filename, "w") as f:
            f.write(self.to_json())   

    @classmethod
    def from_file(cls, filename):
        with open(filename, "r") as f:
            return cls.from_json(f.read())

    def fb_values_array(self):
        return np.vstack(self.fb_values)

    def xy_arrays_zero_subtracted(self):
        dac_values = np.array(self.dac_values)
        dac_zero_ind = np.where(dac_values==0)[0][0]
        fb = self.fb_values_array()
        fb -= fb[dac_zero_ind, :]

        return dac_values, fb
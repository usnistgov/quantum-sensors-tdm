import pylab as plt
import numpy as np
import detchar
from instruments import bluebox
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import List
from detchar import IVCurveColumnData
import os


@dataclass_json
@dataclass
class ICVboxSweepData:
    set_volts_v: List[float]
    data: List[IVCurveColumnData]

    def to_file(self, filename, overwrite=False):
        if not overwrite:
            assert not os.path.isfile(filename)
        with open(filename, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_file(cls, filename):
        with open(filename, "r") as f:
            return cls.from_json(f.read())

    def get_ics_at_set_volt_ind(self, i):
        data = self.data[i]
        fb_values = data.fb_values_array()
        diffs = np.diff(fb_values, axis=0)
        sign = np.median(np.sign(diffs[0, :]))
        inds, rows = np.nonzero(np.diff(np.sign(diffs), axis=0) == -2 * sign)
        inds += 1
        ic_dacs = np.zeros(fb_values.shape[1]) * np.nan
        for ind, row in zip(inds, rows):
            ic_dacs[row] = data.dac_values[ind]
        return ic_dacs

    def get_ics_vs_set_volt_v(self):
        ics = [self.get_ics_at_set_volt_ind(i) for i in range(len(self.data))]
        return np.vstack(ics)


class ICVboxSweeper:
    def __init__(self, curve_taker, vbox):
        self.curve_taker = curve_taker
        self.vbox = vbox

    def get_sweep(self, dac_values, set_volts_v, extra_info={}):
        datas = []
        for set_volt_v in set_volts_v:
            print(f"taking data for set_volt_v={set_volt_v:.2f} V")
            self.vbox.setvolt(set_volt_v)
            data = self.curve_taker.get_curve(dac_values, extra_info)
            datas.append(data)
            temp = ICVboxSweepData(set_volts_v, datas)
            temp.to_file("ic_sweeper_temp.json", overwrite=True)
        return ICVboxSweepData(set_volts_v, datas)
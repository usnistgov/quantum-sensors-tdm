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


plt.ion()
# plt.close("all")
vbox = bluebox.BlueBox(port="extra1")
# vbox.setvolt(1)
curve_taker = detchar.IVCurveTaker(
    detchar.IVPointTaker("DB1", "BX", column_number=0, voltage_source="bluebox"),
    shock_normal_dac_value=0,
    temp_settle_time_out_s=180,
    temp_settle_tolerance_k=0.05 * 1e-3,
)
curve_taker.set_temp_and_settle(setpoint_k=70.2 * 1e-3)
curve_taker.prep_fb_settings(I=10, fba_offset=3500, ARLoff=True)

dacs = (
    detchar.sparse_then_fine_dacs(a=0, b=900, c=4000, n_ab=50, n_bc=150) * 2.5 / 6.5535
)

# dacs = np.linspace(0, 6000, 200)
# data = curve_taker.get_curve(dacs, extra_info={"field coil (amps)": 0})
# data.to_file("latest_single_iv.json", overwrite=True)
# data.plot()
# fb_values = data.fb_values_array()
# flip_fb  = np.max(fb_values) - fb_values
# diffs = np.diff(fb_values, axis=0)
# sign = np.median(np.sign(diffs[0, :]))
# inds, rows = np.nonzero(np.diff(np.sign(diffs), axis=0) == -2 * sign)
# inds += 1

# plt.plot(dacs[inds], data.fb_values_array()[inds, rows], "o")
# ic_dacs = np.zeros(fb_values.shape[1])
# for ind, row in zip(inds, rows):
#     ic_dacs[row] = dacs[ind]


ic_sweeper = ICVboxSweeper(curve_taker, vbox)
data = ic_sweeper.get_sweep(
    dacs,
    set_volts_v=np.linspace(0, 2, 9),
    # extra_info={"field coil polarity": "positive"},
)
data.to_file("ic_sweep_positive10.json", overwrite=True)

data = ICVboxSweepData.from_file("ic_sweep_positive10.json")


def get_ics_at_set_volt_ind(self, i):
    set_volt = self.set_volts_v[i]
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


ICVboxSweepData.get_ics_at_set_volt_ind = get_ics_at_set_volt_ind
ICVboxSweepData.get_ics_vs_set_volt_v = get_ics_vs_set_volt_v
ic_dacs = data.get_ics_vs_set_volt_v()
print(ic_dacs)

plt.figure()
plt.plot(data.set_volts_v, ic_dacs)
plt.xlabel("field coil volts")
plt.ylabel("ic dac")
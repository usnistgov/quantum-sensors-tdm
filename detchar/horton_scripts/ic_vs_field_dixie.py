import pylab as plt
import numpy as np
import detchar
from instruments import bluebox
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import List
from detchar import IVCurveColumnData
import os
from ic_vs_field_util import ICVboxSweepData, ICVboxSweeper


# make a "vbox" out of a crigne controller tower card channel
vbox = detchar.IVPointTaker("FB2", "CX", column_number=6)
vbox.setvolt = lambda v: vbox.set_tower(int((2**16-1)*v/2.5))
vbox.setvolt(.1)

plt.ion()
curve_taker = detchar.IVCurveTaker(
    detchar.IVPointTaker("DB1", "BX", column_number=6),
    shock_normal_dac_value=0,
    temp_settle_time_out_s=180,
    temp_settle_tolerance_k=0.05 * 1e-3,
)
curve_taker.set_temp_and_settle(setpoint_k=63 * 1e-3)
curve_taker.prep_fb_settings(I=10, fba_offset=3500, ARLoff=True)

dacs = (
    detchar.sparse_then_fine_dacs(a=0, b=900, c=4000, n_ab=50, n_bc=150)# * 2.5 / 6.5535
)


ic_sweeper = ICVboxSweeper(curve_taker, vbox)
data = ic_sweeper.get_sweep(
    dacs,
    set_volts_v=np.linspace(0, .2, 10),
    # extra_info={"field coil polarity": "positive"},
)
# data.to_file("ic_sweep_dixie_test.json")

ic_dacs = data.get_ics_vs_set_volt_v_2()
print(ic_dacs)

plt.figure()
plt.plot(data.set_volts_v, ic_dacs)
plt.xlabel("field coil volts")
plt.ylabel("ic dac")
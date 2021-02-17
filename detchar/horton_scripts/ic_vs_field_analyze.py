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

plt.ion()


icdata = ICVboxSweepData.from_file("ic_sweep_positive7.json")


icdata.data[0].plot()
ic_dacs = icdata.get_ics_vs_set_volt_v()
print(ic_dacs)

plt.figure()
plt.plot(icdata.set_volts_v, ic_dacs)
plt.xlabel("field coil volts")
plt.ylabel("ic dac")


icdata = ICVboxSweepData.from_file("ic_sweep_positive10.json")


icdata.data[0].plot()
ic_dacs = icdata.get_ics_vs_set_volt_v()
print(ic_dacs)

plt.figure()
plt.plot(icdata.set_volts_v, ic_dacs)
plt.xlabel("field coil volts")
plt.ylabel("ic dac")
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

icdata1 = ICVboxSweepData.from_file("ic_sweep_both_negative1.json")
icdata2 = ICVboxSweepData.from_file("ic_sweep_negativedb_positivefc.json")
icdata3 = ICVboxSweepData.from_file("ic_sweep_positivedb_negativefc1.json")
icdata4 = ICVboxSweepData.from_file("ic_sweep_positivedb_positivefc1.json")


plt.figure()
plt.title("ic Positve bias")
plt.subplot(1, 2, 1)
ic_dacs4 = icdata4.get_ics_vs_set_volt_v_2()*0.1
plt.plot(icdata4.set_volts_v, ic_dacs4)
plt.xlabel("field coil volts")
plt.ylabel("ic mV")
plt.ylim(0, 200)
plt.title("positive field coil")

plt.subplot(1, 2, 2)
ic_dacs_3 = icdata3.get_ics_vs_set_volt_v_2()*.1
plt.plot(icdata3.set_volts_v, ic_dacs_3)
plt.xlabel("field coil volts")
plt.ylabel("ic dac (mV)")
plt.ylim(0, 200)
plt.title("negative field coil")

plt.savefig("ic_positive_bias.png")

plt.figure()
plt.title("ic negative bias")
plt.subplot(1, 2, 1)
ic_dacs1 = icdata1.get_ics_vs_set_volt_v_2()*.1
plt.plot(icdata1.set_volts_v, ic_dacs1)
plt.xlabel("field coil volts")
plt.ylabel("ic dac (mV)")
plt.ylim(0, 200)
plt.title("positive field coil")

plt.subplot(1, 2, 2)
ic_dacs_2 = icdata2.get_ics_vs_set_volt_v_2()*.1
plt.plot(icdata3.set_volts_v, ic_dacs_2)
plt.xlabel("field coil volts")
plt.ylabel("ic dac (mV)")
plt.ylim(0, 200)
plt.title("negative field coil")


plt.savefig("ic_negative_bias")

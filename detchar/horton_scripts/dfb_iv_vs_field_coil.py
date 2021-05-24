import nasa_client
import pylab as plt
import numpy as np
from cringe.cringe_control import CringeControl
import time
import h5py

plt.ion()
tstart = time.time()
# plt.close("all")

c = nasa_client.EasyClient()
N=2**20
# N = 10000
data_ = c.getNewData()

cc = CringeControl()
fc_cardname = "FB2"
bayname = "CX"
field_coil_dacs = np.array(np.round(np.linspace(0,2**16-1,100)), dtype="int64")

alldata_shape = (len(field_coil_dacs), data_.shape[0], data_.shape[1], N, data_.shape[3])
with h5py.File("dfb_iv.hdf5", "w") as h5:
    # alldata = np.zeros((len(field_coil_dacs), data_.shape[0], data_.shape[1], N, data_.shape[3]),
    # dtype="float32")
    h5["field_coil_dacs"] = field_coil_dacs
    h5["fc_cardname"] = fc_cardname
    h5["bayname"] = bayname
    alldata = h5.create_dataset("alldata", alldata_shape, dtype="float32")
    for i, fc_dac in enumerate(field_coil_dacs):
        print(f"starting dac = {fc_dac}, {i+1}/{len(field_coil_dacs)}, elapsed={time.time()-tstart:.2f}")
        cc.set_tower_channel(fc_cardname, bayname, int(fc_dac))
        time.sleep(0.01)
        for i in range(5):
            try:
                data = c.getNewData(minimumNumPoints=N, exactNumPoints=True)
                alldata[i, :,:,:,:] = data
                break
            except Exception as ex:
                print("failed data aquisition")




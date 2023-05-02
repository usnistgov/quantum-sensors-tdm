import qsghw.instruments.lakeshore370_ser as lakeshore370_ser
import time
import numpy as np

ls = lakeshore370_ser.Lakeshore370_ser(port='/dev/ttyUSB0')
timestr = time.strftime("%Y%m%d-%H%M%S")
dt = np.dtype([('time', np.float64), ('t40K', np.float64), ('tmxc', np.float64),
('tstill', np.float64), ('t3k', np.float64), ('hout', np.float64)])

v = np.zeros(200000, dtype=dt)
sleep_s = 10
for i in range(len(v)):

    t1 = ls.getTemperature(1)
    t2 = ls.getTemperature(2)
    t5 = ls.getTemperature(5)
    t6 = ls.getTemperature(6)
    hout = ls.getHeaterOut()
    t = time.time()


    v[i]=(t, t1, t5, t6, t2, hout)

    time.sleep(sleep_s)
    if i%10 == 0:
        hoursleft = (len(v)-i)*sleep_s/3600
        print(f"i={i}, {hoursleft:.2f} hours left")
        # we just write over the whole file each time for simplicity
        np.save(f"lakeshorelog{timestr}.npz", v[:i])
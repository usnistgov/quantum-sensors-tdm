from cringe.cringe_control import CringeControl
from nasa_client import EasyClient
import time


class IVPointTaker():
    def __init__(self):
        self.ec = EasyClient()
        self.ec.setupAndChooseChannels()
        self.cc = CringeControl()
        # these should be args
        self.delay_s = 0.05
        self.relock_lo_threshold = 3000
        self.relock_hi_threshold = 13000
        self.db_cardname = "DB1"
        self.bayname = "BX"

    def get_iv_pt(self, dacvalue):
        self.cc.set_tower_channel(self.db_cardname, self.bayname, int(dacvalue))
        data = self.ec.getNewData(delaySeconds=self.delay_s)
        avg_col0 = data[0,:,:,1].mean(axis=-1)
        for row, fb in enumerate(avg_col0):
            if fb < self.relock_lo_threshold:
                self.cc.relock_fba(0, row)
                print(f"relock row {row} was too low")
            if fb > self.relock_hi_threshold:
                self.cc.relock_fba(0, row)
                print(f"relock row {row} was too high")
        return avg_col0


# demo with a tc tickle measurement
taker = IVPointTaker()
while True:
    va = taker.get_iv_pt(0)
    time.sleep(1)
    vb = taker.get_iv_pt(250)
    print(vb-va)




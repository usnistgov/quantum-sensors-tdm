from cringe.cringe_control import CringeControl
from adr_gui.adr_gui_control import AdrGuiControl
from instruments.srs980 import SRS980
from nasa_client import EasyClient
from instruments import BlueBox
import time
import numpy as np

class Acquire():
    def __init__(
        self,
        db_source,
        db_card="DB",
        db_bay=None,
        delay_between_acqs=0.05, # wip
        relock_bounds=(2000, 14000), # wip
        easy_client=None,
        cringe_control=None,
        dfb_channels=None, 
        dastard_columns=[0],
        adr_control=None
    ):
        """
        A base class with logic for acquiring
        various forms of data including IV curves
        and noise, as functions of bath temperature
        and detector bias etc.
        :param db_source: 'tower' or 'bluebox', i.e. what
            is providing the detector bias
        :param db_card: required if db_source=='tower', 
            which card in the tower provides bias. 
            Usually this is 'DB'
        :param db_bay: required if db_source=='tower',
            which bay(s) on the db_card to use for bias
            (string or list of strings)
        :param dfb_channels: List of channels to relock with cringecontrol.
            In Velma the mapping is 0: DFBx2 CH1, 1: DFBx2 CH2, 2: DFBx1CLK CH1.
        """
        
        # neat trick: the or operator returns the first non-falseish argument
        # so 0 or 1 -> 1 and false or "hi" -> "hi". None is falseish
        self.ec = easy_client or EasyClient()  # should setupandchoose by itself
        self.cc = cringe_control or CringeControl()
        self.adr = adr_control or AdrGuiControl()
        self.acq_delay = delay_between_acqs
        self.relock_bounds = relock_bounds
        self.dfb_chans = dfb_channels
        self.col=dastard_columns
        if relock_bounds[1] - relock_bounds[0] < 2000:
            raise ValueError("Insufficient range for relock!")

        if db_source == 'tower' or db_source is None:
            self.is_tower = True  # 0-2.5V in 2**16 steps
        elif db_source == 'bluebox':
            self.is_tower = False
            self.bb = BlueBox(port='vbox', version='mrk2')
            # 0 to 6.5535V in 2**16 steps
        elif db_source == 'srs':
            self.is_tower = False
            if type(db_bay) is list:
                self.srs = [SRS980(i) for i in db_bay]
            else:
                self.srs = [SRS980(db_bay)]
            for srs in self.srs:
                srs.output_on()
        else:
            raise ValueError("voltage_source must be None, 'tower', or 'bluebox'.")
        self.db_source = db_source
        self.db_bay = db_bay
        self.bayname = db_bay 
        self.db_cardname = db_card #required that these member variables always exist for backward compatibility 
        # with noise.py at least

    def __del__(self):
        if self.db_source == "srs":
            for srs in self.srs:
                srs.output_off()

    def set_volt(self, voltage):
        voltage = int(voltage)
        if self.is_tower:
            self._set_tower(voltage)
        elif self.db_source == 'bluebox':
            self._set_blue_box(voltage)
        else:
            self._set_srs(voltage)

    def _set_blue_box(self, dac_value):
        # We'll convert dac values here so that blue box acts the same as the tower
        tower_v = dac_value * 2.5 / 2**16
        self.bb.setvolt(tower_v)

    def _set_tower(self, voltage):
        if type(self.db_bay) is list:
            for bay in self.db_bay:
                self.cc.set_tower_channel(self.db_cardname, bay, voltage)
        else:
            self.cc.set_tower_channel(self.db_cardname, self.db_bay, voltage)

    def _set_srs(self, dac_value):
        for srs in self.srs:
            srs.setvolt(dac_value * 2.5 / 2**16)
    # def get_data(self, npts):
    #     data = self.ec.getNewData(minimumNumPoints=npts)
    #     

    def set_temp(self, temp):
        self.adr.set_temp_k(float(temp))

    def set_temp_and_settle(self, temp, tolerance=0.0005, timeout=180):
        self.set_temp(temp)
        return self.wait_temp_stable(temp, tol=tolerance, time_out_s=timeout)

    def wait_temp_stable(self, setpoint_k, tol=.0005, time_out_s=180):
        ''' determine if the servo has reached the desired temperature '''
        poll_time = 10
        assert time_out_s > poll_time, "time_out_s must be greater than 10 seconds"
    
        for i in range(time_out_s//poll_time+1):
            cur_temp = self.adr.get_temp_k()
            at_temp = np.abs(cur_temp-setpoint_k) < tol
            rms = self.adr.get_temp_rms_uk_npts(10)
            stable = rms < tol * 1e6
            print(f'Current Temp: {cur_temp:.4f} K +/- {rms:.1f} uK')
            if at_temp and stable:
                return True

            time.sleep(poll_time)

        return False


def column_name_to_num(col):
    ''' based on the column, select the appropriate tower card detector bias channel.  
        This "channel" is called "bayname" in the iv_utils software and BAY_NAMES 
        in the towerwidget.py.  So I follow this poor naming convention 
    '''
    col = col.upper()
    if len(col) == 1:
        assert col in ['A','B','C','D'], 'unknown column.  Column must be A,B,C, or D'
        if col == 'A':
            bayname = "0"
        elif col == 'B':
            bayname = "1"
        elif col == 'C':
            bayname = "2"
        elif col == 'D':
            bayname = "3"
    elif len(col) > 1:
        bayname = []
        for c in col:
            bayname.append(column_name_to_num(c))
    return bayname

if __name__ == "__main__":
    print("self test")
    aq = Acquire('tower')
    print("set temp and settle")
    print(aq.set_temp_and_settle(0.130,0.0005))
    
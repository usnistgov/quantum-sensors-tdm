from dataclasses import dataclass
import dataclasses
from dataclasses_json import dataclass_json
from typing import Any, List
import numpy as np
import pylab as plt
import progress.bar

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

@dataclass_json
@dataclass
class IVTempSweepData():
    set_temps_k: List[float]
    data: List[IVCurveColumnData]
    
    def to_file(self, filename, overwrite = False):
        if not overwrite:
            assert not os.path.isfile(filename)
        with open(filename, "w") as f:
            f.write(self.to_json())   

    @classmethod
    def from_file(cls, filename):
        with open(filename, "r") as f:
            return cls.from_json(f.read())

    def plot_row(self, row):
        plt.figure()
        for curve in self.data:
            x, y = curve.xy_arrays_zero_subtracted()
            t_mK = curve.nominal_temp_k*1e3
            dt_mK = (curve.post_temp_k-curve.pre_temp_k)*1e3
            plt.plot(x, y[:,row], label=f"{t_mK:0.2f} mK, dt {dt_mK:0.2f} mK")
        plt.xlabel("dac value (arb)")
        plt.ylabel("feedback (arb)")
        plt.title(f"row={row} bayname {curve.bayname}, db_card {curve.db_cardname}")
        plt.legend()


if __name__ == "__main__":
    # plt.ion()
    # plt.close("all")
    # curve_taker = IVCurveTaker(IVPointTaker("DB1", "BX"), temp_settle_delay_s=0, shock_normal_dac_value=40000)
    # curve_taker.set_temp_and_settle(setpoint_k=0.075)
    # curve_taker.pt.prep_fb_settings(I=10, fba_offset=3000)
    # dacs = sparse_then_fine_dacs(a=40000, b = 4000, c=0, n_ab=40, n_bc=250)
    # data = curve_taker.get_curve(dacs, extra_info = {"magnetic field current (amps)": 1e-6})
    # data.plot()
    # data.to_file("ivtest.json", overwrite=True)
    # data2 = IVCurveColumnData.from_json(data.to_json())
    # assert data2.pre_time_epoch_s == data.pre_time_epoch_s
    # data = IVCurveColumnData.from_file("ivtest.json")
    # x, y = data.xy_arrays_zero_subtracted()
    # r = iv_to_frac_rn_array(x, y, superconducting_below_x=2000, normal_above_x=24000)
    # plt.figure()
    # plt.plot(x, r)
    # plt.xlabel("dac value")
    # plt.ylabel("fraction R_n")
    # plt.legend([f"row{i}" for i in range(r.shape[1])])
    # plt.vlines(2750, 0, 1)
    # plt.grid(True)



    # t, y = tc_tickle()
    # y = np.vstack(fbs)
    # plt.clf()
    # plt.plot(np.array(t)*1e3, y)
    # plt.xlabel("temp (mK)")
    # plt.ylabel("delta fb from 50 dac unit tickle")
    # plt.pause(0.1)

    plt.ion()
    plt.close("all")
    curve_taker = IVCurveTaker(IVPointTaker("DB1", "BX"), temp_settle_delay_s=180, shock_normal_dac_value=40000)
    curve_taker.prep_fb_settings()
    temp_sweeper = IVTempSweeper(curve_taker)
    dacs = sparse_then_fine_dacs(a=40000, b = 10000, c=0, n_ab=20, n_bc=100)
    temps_mk = np.linspace(60,100,16)
    print(f"{temps_mk} mK")
    sweep = temp_sweeper.get_sweep(dacs, 
        set_temps_k=temps_mk*1e-3, 
        extra_info={"field coil current (Amps)":0})
    sweep.to_file("iv_sweep_test2.json", overwrite=True)
    sweep2 = IVTempSweepData.from_file("iv_sweep_test2.json")
    sweep2.plot_row(row=0)
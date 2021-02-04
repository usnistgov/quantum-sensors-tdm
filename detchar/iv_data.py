from dataclasses import dataclass
import dataclasses
from dataclasses_json import dataclass_json
from typing import Any, List
import numpy as np
import pylab as plt
import collections
import os
from numpy.polynomial.polynomial import Polynomial

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
    dac_values: List[int]
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

    def xy_arrays_zero_subtracted_at_origin(self):
        dac_values = np.array(self.dac_values)
        dac_zero_ind = np.where(dac_values==0)[0][0]
        fb = self.fb_values_array()
        fb -= fb[dac_zero_ind, :]

        return dac_values, fb

    def xy_arrays_zero_subtracted_at_normal_y_intercept(self, normal_above_fb):
        dac_values = np.array(self.dac_values)
        fb = self.fb_values_array()
        for i in range(fb.shape[1]):
            fb[:,i] = fit_normal_zero_subtract(dac_values, fb[:, i], normal_above_fb)
        return dac_values, fb

    def xy_arrays(self):
        dac_values = np.array(self.dac_values)
        fb = self.fb_values_array()
        for i in range(fb.shape[1]):
            fb[:,i] = fb[:, i]
        return dac_values, fb

def fit_normal_zero_subtract(x, y, normal_above_x):
    normal_inds = np.where(x>normal_above_x)[0]
    pfit_normal = Polynomial.fit(x[normal_inds], y[normal_inds], deg=1)
    normal_y_intersect = pfit_normal(0)
    return y-normal_y_intersect

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

    def plot_row(self, row, zero="origin"):
        plt.figure()
        for curve in self.data:
            if zero == "origin":
                x, y = curve.xy_arrays_zero_subtracted_at_origin()
            elif zero == "fit normal":
                x, y = curve.xy_arrays_zero_subtracted_at_normal_y_intersect(normal_above_fb=25000)
            t_mK = curve.nominal_temp_k*1e3
            dt_mK = (curve.post_temp_k-curve.pre_temp_k)*1e3
            plt.plot(x, y[:,row], label=f"{t_mK:0.2f} mK, dt {dt_mK:0.2f} mK")
        plt.xlabel("dac value (arb)")
        plt.ylabel("feedback (arb)")
        plt.title(f"row={row} bayname {curve.bayname}, db_card {curve.db_cardname}, zero={zero}")
        plt.legend()

@dataclass_json
@dataclass
class IVColdloadSweepData(): #set_cl_temps_k, pre_cl_temps_k, post_cl_temps_k, data
    set_cl_temps_k: List[float]
    data: List[IVTempSweepData]
    extra_info: dict

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
        #n=len(set_cl_temps_k)
        plt.figure()
        for ii, tempSweep in enumerate(self.data): # loop over IVTempSweepData instances (ie coldload temperature settings)
            for jj, set_temp_k in enumerate(tempSweep.set_temps_k): # loop over bath temperatures
                data = tempSweep.data[jj]
                x = data.dac_values ; y=data.fb_values_array()
                plt.plot(x,y[:,row],label='T_cl = %.1fK; T_b = %.1f'%(self.set_cl_temps_k[ii],data.nominal_temp_k))
        plt.xlabel("dac value (arb)")
        plt.ylabel("feedback (arb)")
        plt.title(f"row={row} bayname {data.bayname}, db_card {data.db_cardname}")
        plt.legend()

@dataclass_json
@dataclass
class IVCircuit():
    rfb_ohm: float # feedback resistor
    rbias_ohm: float # bias resistor
    rsh_ohm: float # shunt resistor
    rx_ohm: float # parasitic resistance in series with TES
    m_ratio: float # ratio of feedback mutual inductance to input mutual inductance
    vfb_gain: float # volts/arbs of feeback
    vbias_gain: float # volts/arbs of bias

    def iv_raw_to_physical_fit_rpar(self, vbias_arbs, vfb_arbs, sc_below_vbias_arbs):
        ites0, vtes0 = self.iv_raw_to_physical(vbias_arbs, vfb_arbs, rpar_ohm=0)
        sc_inds = np.where(vbias_arbs<sc_below_vbias_arbs)[0]
        pfit_sc = Polynomial.fit(ites0[sc_inds], vtes0[sc_inds], deg=1)
        rpar_ohm = pfit_sc.deriv(m=1)(0)
        return ites0, vtes0-ites0*rpar_ohm, rpar_ohm

    def iv_raw_to_physical_simple(self, vbias_arbs, vfb_arbs, rpar_ohm):
        #assume rbias >> rshunt
        ifb = vfb_arbs*self.vfb_gain / self.rfb_ohm # feedback current
        ites = ifb / self.m_ratio # tes current
        ibias = vbias_arbs*self.vbias_gain/self.rbias_ohm # bias current
        # rtes = rsh_ohm + rpar_ohm - ibias*rsh_ohm/ites
        vtes = (ibias-ites)*self.rsh_ohm-ites*rpar_ohm
        return ites, vtes

    def to_physical_units(self,dac_values,fb_array):
        y = fb_array*self.vfb_gain *(self.rfb_ohm*self.m_ratio)**-1
        I = dac_values*self.vbias_gain/self.rbias_ohm # current sent to TES bias network
        n,m = np.shape(y)
        x = np.zeros((n,m))
        for ii in range(m):
            #x[:,ii] = I*self.rsh_ohm - y[:,ii]*(self.rsh_ohm+self.rx_ohm[ii]) # for future for unique rx per sensor
            x[:,ii] = I*self.rsh_ohm - y[:,ii]*(self.rsh_ohm+self.rx_ohm)
        return x,y

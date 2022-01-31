from dataclasses import dataclass
import dataclasses
from dataclasses_json import dataclass_json
from typing import Any, List
import numpy as np
import pylab as plt
import collections
import os
from numpy.polynomial.polynomial import Polynomial
import lmfit

@dataclass_json
@dataclass
class IVCurveColumnData:
    nominal_temp_k: float
    pre_temp_k: float
    post_temp_k: float
    pre_time_epoch_s: float
    post_time_epoch_s: float
    pre_hout: float
    post_hout: float
    post_slope_hout_per_hour: float
    dac_values: List[int]
    fb_values: List[Any] = dataclasses.field(repr=False)  # actually a list of np arrays
    bayname: str
    db_cardname: str
    column_number: int
    extra_info: dict
    pre_shock_dac_value: float
    zero_bias_fb: List[int] # after iv, db=0, then relock, then this value is recorded

    def plot(self):
        plt.figure()
        plt.plot(self.dac_values, self.fb_values)
        plt.xlabel("dac values (arb)")
        plt.ylabel("fb values (arb)")
        plt.title(
            f"bay {self.bayname}, db_card {self.db_cardname}, col {self.column_number}, nominal_temp_mk {self.nominal_temp_k*1000}"
        )

    def plot2(self):
        plt.figure()
        x, y = self.xy_arrays

    def to_file(self, filename, overwrite=False):
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
        dac_zero_ind = np.where(dac_values == 0)[0][0]
        fb = self.fb_values_array()
        fb -= fb[dac_zero_ind, :]

        return dac_values, fb

    def xy_arrays_zero_subtracted_at_normal_y_intercept(self, normal_above_fb):
        dac_values = np.array(self.dac_values)
        fb = self.fb_values_array()
        for i in range(fb.shape[1]):
            fb[:, i] = fit_normal_zero_subtract(dac_values, fb[:, i], normal_above_fb)
        return dac_values, fb

    def xyarrays_zero_subtracted_with_post_iv_zero_fb_value(self):
        dac_values = np.array(self.dac_values)
        fb = self.fb_values_array()
        fb -= np.array(self.zero_bias_fb)
        return dac_values, fb       

    def xyarrays_zero_subtracted_and_sc_branch_fixed(self):
        x, y = self.xyarrays_zero_subtracted_with_post_iv_zero_fb_value()
        out = np.zeros_like(y)
        for row in range(self.get_nrows()):
            z = y[:, row]
            # assume we started a high db, and moved down, the sc branch should be the last few points
            zz = np.diff(z)/np.diff(x)
            zzz = (zz/zz[-1])-1 # slope relative to potential sc zone, where sc zone = 0
            # and deviations are in units of the sc zone slope
            inds = np.where(np.abs(zzz)>2)[0]
            if len(inds) > 1:
                ind = inds[-1]
                z[ind:]-=z[-1]
            out[:, row] = z
        # can't actually do this here, because I don't know where the origin is!!
        return x, out


    def xy_arrays(self):
        dac_values = np.array(self.dac_values)
        fb = self.fb_values_array()
        for i in range(fb.shape[1]):
            fb[:, i] = fb[:, i]
        return dac_values, fb

    def get_nrows(self):
        return len(self.fb_values[0])

    def fit_for_rpar(self, circuit, sc_below_vbias_arb):
        """
        loop over rows and fit for r_par (aka parasitic resistance)
        return a list of fitted values
        if a fit fails return np.nan

        circuit: an IVCircuit object, all values except r_par must be accurate
        sc_below_vbias_arb: a value for detector bias in arb units below which the device is superconducting
        """
        nrows = self.get_nrows()
        vbias_arbs, fb_arbs = self.xy_arrays()
        rpar_ohm_by_row = np.zeros(nrows)        
        for row in range(nrows):
            fb_for_this_row = fb_arbs[:, row]
            try:
                _ites, _vtes, rpar_ohm = circuit.iv_raw_to_physical_fit_rpar(vbias_arbs, 
                        fb_for_this_row, sc_below_vbias_arb)
            except np.linalg.LinAlgError:
                rpar_ohm = np.nan
            rpar_ohm_by_row[row] = rpar_ohm
        return rpar_ohm_by_row
        

def fix_sc_branch(x, y):
    """
    locate the superconducting branch based on the fact that it's slope is very different from all other slopes
    force it to end at the origin
    x - detector bias dac units
    y - feedback dac units that has already been set to have 0 correspond to zero current"""
    # assume we started a high db, and moved down, the sc branch should be the last few points
    dfb_dbias = np.diff(y)/np.diff(x)
    z = (dfb_dbias/dfb_dbias[-1])-1 # slope relative to potential sc zone, where sc zone = 0
    # and deviations are in units of the sc zone slope
    inds = np.where(np.abs(z)>2)[0]
    yout = y[:]
    if len(inds) > 1:
        ind = inds[-1]
        yout[ind:] -= yout[-1]
    return yout

def fix_sc_branch_array(x, y):
    """
    locate the superconducting branch based on the fact that it's slope is very different from all other slopes
    force it to end at the origin
    do it for an array of y[i,j] values where i indexes detector bias and j indexes row or temperature
    """
    yout = np.zeros_like(y)
    for i in range(y.shape[1]):
        yout[:, i] = fix_sc_branch(x, y[:,i])
    return yout

def fit_normal_zero_subtract(x, y, normal_above_x):
    normal_inds = np.where(x > normal_above_x)[0]
    pfit_normal = Polynomial.fit(x[normal_inds], y[normal_inds], deg=1)
    normal_y_intersect = pfit_normal(0)
    return y - normal_y_intersect


@dataclass_json
@dataclass
class IVTempSweepData:
    set_temps_k: List[float]
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

    def xyarrays_zero_subtracted_temp_fb_row(self, fix_sc=True):
        last_curves = self.data[-1]
        xlast, ylast = last_curves.xyarrays_zero_subtracted_with_post_iv_zero_fb_value()        
        out = np.zeros( (len(self.data), len(ylast), self.get_nrows()) )        
        for temp_index, curves in enumerate(self.data):
                x, y = curves.xyarrays_zero_subtracted_with_post_iv_zero_fb_value()
                y -= ylast[-1, :] 
                # assume the last (highest temp) iv curve kept lock 
                # and it's last point has zero detector bias, so it can define current zero
                if fix_sc:
                    y_sc_fixed = fix_sc_branch_array(x, y)
                    out[temp_index, :, :] = y_sc_fixed[:, :]
                else:
                    out[temp_index, :, :] = y[:, :]
        return xlast, out

    def xyarrays_zero_subtracted_all_temps_for_one_row(self, row):
        x, y_temp_fb_row = self.xyarrays_zero_subtracted_temp_fb_row()
        return x, y_temp_fb_row[:, :, row]

    def xyarrays_zero_subtracted_all_row_for_one_temp(self, temp_index):
        x, y_temp_fb_row = self.xyarrays_zero_subtracted_temp_fb_row()
        return x, y_temp_fb_row[temp_index, :, :] 
    
    def plot_row(self, row):
        x, y = self.xyarrays_zero_subtracted_all_temps_for_one_row(row)
        plt.figure()
        for temp_index, curves in enumerate(self.data):
            t_mK = curves.nominal_temp_k * 1e3
            dt_mK = (curves.post_temp_k - curves.pre_temp_k) * 1e3
            plt.plot(x, y[temp_index, :], label=f"{t_mK:0.2f} mK, dt {dt_mK:0.2f} mK")
        plt.xlabel("dac value (arb)")
        plt.ylabel("feedback (arb)")
        plt.title(
            f"row={row} bayname {curves.bayname}, db_card {curves.db_cardname}"
        )
        plt.legend()     

    def plot_temp(self, temp_index):
        curves = self.data[0] # grab the first IV to get bayname and cardname
        x, y = self.xyarrays_zero_subtracted_all_row_for_one_temp(temp_index)
        plt.figure()
        for row in range(self.get_nrows()):
            plt.plot(x, y[:, row], label=f"row{row}")
        plt.xlabel("dac value (arb)")
        plt.ylabel("feedback (arb)")
        set_temp_mk = self.set_temps_k[temp_index]*1e3
        plt.title(
            f"set_temp={set_temp_mk:.1f} mk, bayname {curves.bayname}, db_card {curves.db_cardname}"
        )
        plt.legend()          


    def plot_row_old(self, row, zero="origin"):
        "kept around for testing zero subtraction, but basically don't use it"
        plt.figure()
        last_curves = self.data[-1]
        xlast, ylast = last_curves.xyarrays_zero_subtracted_with_post_iv_zero_fb_value()
        for curves in self.data:
            if zero == "origin":
                x, y = curves.xy_arrays_zero_subtracted_at_origin()
            elif zero == "fit normal":
                x, y = curves.xy_arrays_zero_subtracted_at_normal_y_intersect(
                    normal_above_fb=25000
                )
            elif zero == "zero_bias":
                x, y = curves.xyarrays_zero_subtracted_with_post_iv_zero_fb_value()
                y -= ylast[-1, :] # assume the last (highest temp) iv curve kept lock so it can define zero
            elif zero == "none":
                x, y = curves.xy_arrays()
            else:
                raise Exception(f"zero choice {zero} is not valid")
            t_mK = curves.nominal_temp_k * 1e3
            dt_mK = (curves.post_temp_k - curves.pre_temp_k) * 1e3
            plt.plot(x, y[:, row], label=f"{t_mK:0.2f} mK, dt {dt_mK:0.2f} mK")
        plt.xlabel("dac value (arb)")
        plt.ylabel("feedback (arb)")
        plt.title(
            f"row={row} bayname {curves.bayname}, db_card {curves.db_cardname}, zero={zero}"
        )
        plt.legend()
    
    def get_nrows(self):
        return self.data[0].get_nrows()

    def fit_for_rpar(self, circuit, sc_below_vbias_arb, temp_index):
        """
        circuit: an IVCircuit object, all values except r_par must be accurate
        sc_below_vbias_arb: a value for detector bias in arb units below which the device is superconducting
        temp_index: an integer index into self.set_temps_k for which set of IVs to use, typically use the lowest temp"""
        data = self.data[temp_index]
        rpar_ohm_by_row = data.fit_for_rpar(circuit, sc_below_vbias_arb)
        return rpar_ohm_by_row

    def iv_temp_val_row(self, circuit, rpar_ohm_by_row=None, sc_below_vbias_arb=None):
        if rpar_ohm_by_row is None:
            rpar_ohm_by_row = self.fit_for_rpar(circuit, sc_below_vbias_arb, 0)
            #assume lowest temp was first
        x, y_temp_fb_row = self.xyarrays_zero_subtracted_temp_fb_row()
        i_temp_fb_row = np.zeros_like(y_temp_fb_row)        
        v_temp_fb_row = np.zeros_like(y_temp_fb_row)          
        for temp_index in range(len(self.data)):
            for row in range(self.get_nrows()):
                i, v = circuit.iv_raw_to_physical(x, 
                y_temp_fb_row[temp_index, :, row], rpar_ohm=rpar_ohm_by_row[row])
                i_temp_fb_row[temp_index, :, row] = i
                v_temp_fb_row[temp_index, :, row] = v
        return i_temp_fb_row, v_temp_fb_row

    def plot_row_iv(self, row, circuit, rpar_ohm_by_row=None, sc_below_vbias_arb=None, 
    y_quantity="current", x_quantity="voltage"):
        i, v = self.iv_temp_val_row(circuit, rpar_ohm_by_row, sc_below_vbias_arb)
        if y_quantity == "current":
            y = i
            ylabel = "tes current (A)"
        elif y_quantity == "resistance":
            y = v/i
            ylabel = "tes resistance (Ohm)"
        elif y_quantity == "power":
            y = i*v
            ylabel = "tes power (W)"
        else: 
            raise Exception(f"y_quantity = {y_quantity} is invalid")
        if x_quantity == "voltage":
            x = v
            xlabel = "tes voltage (V)"
        elif x_quantity == "arbs":
            x = np.zeros(( len(self.data), len(self.data[0].dac_values),self.get_nrows()))
            for temp_index in range(len(self.data)):
                for row in range(self.get_nrows()):
                    x[temp_index, :, row] = self.data[0].dac_values[:]
            xlabel = "dac units (arb)"
        else:
            raise Exception("fix it")
        print(f"i.shape={i.shape} v.shape={v.shape}")
        plt.figure()
        for temp_index in range(len(self.data)):
            curves = self.data[temp_index]
            t_mK = curves.nominal_temp_k * 1e3
            dt_mK = (curves.post_temp_k - curves.pre_temp_k) * 1e3
            plt.plot(x[temp_index, :, row], y[temp_index, :, row], label=f"{t_mK:0.2f} mK, dt {dt_mK:0.2f} mK")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(
            f"row={row} bayname {curves.bayname}, db_card {curves.db_cardname}"
        )
        plt.legend()        

    def get_power_at_r(self, r_ohm, row, circuit, rpar_ohm_by_row=None, sc_below_vbias_arb=None,
    plot=False):
        i, v = self.iv_temp_val_row(circuit, rpar_ohm_by_row, sc_below_vbias_arb)
        r = v/i
        p = i*v
        p_out = np.zeros(len(self.data))
        for temp_index in range(len(self.data)):
            # not quite sure why the ::-1 helps get better interp results, but without it
            # I get the same power for all temp_index values
            p_out[temp_index] = np.interp(r_ohm, r[temp_index, ::-1, row], p[temp_index, ::-1, row])
        if plot:
            plt.figure()
            for temp_index in range(len(self.data)):
                curves = self.data[temp_index]
                t_mK = curves.nominal_temp_k * 1e3
                dt_mK = (curves.post_temp_k - curves.pre_temp_k) * 1e3
                plt.plot(p[temp_index, :, row], r[temp_index, :, row], label=f"{t_mK:0.2f} mK, dt {dt_mK:0.2f} mK")
            plt.plot(p_out, r_ohm*np.ones(len(p_out)), "o", label="interpolated powers")
            plt.xlabel("tes power (W)")
            plt.ylabel("tes resistance (ohm)")
            plt.title(
                f"row={row} bayname {curves.bayname}, db_card {curves.db_cardname}"
            )
            plt.legend()       
        
        return p_out



@dataclass_json
@dataclass
class IVColdloadSweepData:  # set_cl_temps_k, pre_cl_temps_k, post_cl_temps_k, data
    set_cl_temps_k: List[float]
    data: List[IVTempSweepData]
    extra_info: dict

    def to_file(self, filename, overwrite=False):
        if not overwrite:
            assert not os.path.isfile(filename)
        with open(filename, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_file(cls, filename):
        with open(filename, "r") as f:
            return cls.from_json(f.read())

    def plot_row(self, row):
        # n=len(set_cl_temps_k)
        plt.figure()
        for ii, tempSweep in enumerate(
            self.data
        ):  # loop over IVTempSweepData instances (ie coldload temperature settings)
            for jj, set_temp_k in enumerate(
                tempSweep.set_temps_k
            ):  # loop over bath temperatures
                data = tempSweep.data[jj]
                x = data.dac_values
                y = data.fb_values_array()
                plt.plot(
                    x,
                    y[:, row],
                    label="T_cl = %.1fK; T_b = %.1f"
                    % (self.set_cl_temps_k[ii], data.nominal_temp_k),
                )
        plt.xlabel("dac value (arb)")
        plt.ylabel("feedback (arb)")
        plt.title(f"row={row} bayname {data.bayname}, db_card {data.db_cardname}")
        plt.legend()


@dataclass_json
@dataclass
class IVCircuit:
    rfb_ohm: float  # feedback resistor
    rbias_ohm: float  # bias resistor
    rsh_ohm: float  # shunt resistor
    m_ratio: float  # ratio of feedback mutual inductance to input mutual inductance
    vfb_gain: float  # volts/arbs of feeback
    vbias_gain: float  # volts/arbs of bias
    rpar_ohm: float = 0 # parasitic resistance ohms, is often set to, then fitted for, then set to value

    def iv_raw_to_physical_fit_rpar(self, vbias_arbs, vfb_arbs, sc_below_vbias_arbs):
        ites0, vtes0 = self.iv_raw_to_physical(vbias_arbs, vfb_arbs, rpar_ohm=0)
        sc_inds = np.where(vbias_arbs < sc_below_vbias_arbs)[0]
        x = ites0[sc_inds]
        domain_lo = np.amin(x)
        # this is to fix a bug where some versions of numpy error if x is all the same value
        domain_hi = np.amax(x)
        if domain_hi == domain_lo:
            domain_hi += 1
        pfit_sc = Polynomial.fit(
            ites0[sc_inds], vtes0[sc_inds], deg=1, domain=[domain_lo, domain_hi]
        )
        rpar_ohm = pfit_sc.deriv(m=1)(0)
        return ites0, vtes0 - ites0 * rpar_ohm, rpar_ohm

    def iv_raw_to_physical(self, vbias_arbs, vfb_arbs, rpar_ohm=None):
        """rpar_ohm is None by default, when None it uses the circuit value. otherwise it uses the passed value"""
        if rpar_ohm is None:
            rpar_ohm = self.rpar_ohm
        # assume rbias >> rshunt
        ifb = vfb_arbs * self.vfb_gain / self.rfb_ohm  # feedback current
        ites = ifb / self.m_ratio  # tes current
        ibias = vbias_arbs * self.vbias_gain / self.rbias_ohm  # bias current
        # rtes = rsh_ohm + rpar_ohm - ibias*rsh_ohm/ites
        vtes = (ibias - ites) * self.rsh_ohm - ites * rpar_ohm
        return ites, vtes

def g_fit(tb_k, p_w, k_guess_w_per_t_to_n=1e-9, tc_guess_k=0.01, n_guess=3, prune_nan=True, plot=False):
    if prune_nan:
        inds = ~np.isnan(p_w)
        tb_k = np.array(tb_k)[inds] # swith to array to allow indexing with bool vector
        p_w = p_w[inds]
    def p_model(tb_k, tc_k, gigak, n):
            return 1e-9*gigak*(tc_k**n-tb_k**n)*np.greater(tc_k, tb_k)
    model = lmfit.model.Model(p_model)
    params = model.make_params()
    params["tc_k"].set(70e-3, vary=True)
    params["gigak"].set(1, min=1e-18)
    params["n"].set(3, min=0.5)
    result = model.fit(data=p_w, tb_k=tb_k, params=params)
    p_model_out = result.eval()
    # print(result.fit_report())
    # result.params.pretty_print()
    k = result.params["gigak"].value*1e-9
    n = result.params["n"].value
    tc_k = result.params["tc_k"].value
    G_W_per_K = n*k*tc_k**(n-1)
    if plot:
        plt.figure()
        plt.plot(tb_k, p_w, "o")
        plt.plot(tb_k, p_model_out, label=f"tc={tc_k:.4f} K, k={k:.2g} W/K^n, n={n:.2f}, G={G_W_per_K*1e12:.2f} pW/K")
        plt.xlabel("temp (K)")
        plt.ylabel("power (W)")
        plt.legend()

    return result, k, tc_k, n, G_W_per_K
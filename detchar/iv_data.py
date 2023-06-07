from dataclasses import dataclass
import dataclasses
from dataclasses_json import dataclass_json
from typing import Any, List
import numpy as np
import pylab as plt
import collections
import os
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import leastsq
from IPython import embed

def iv_to_frac_rn_array(x, y, normal_above_x):
    rs = [iv_to_frac_rn_single(x, y[:, i], normal_above_x) for i in range(y.shape[1])]
    return np.vstack(rs).T

def iv_to_frac_rn_single(x,y,normal_above_x):
    if normal_above_x is None:
        normal_above_x = x[10]
    normal_indices = np.where(np.array(x)>normal_above_x)[0]
    x_norm = np.array(x)[normal_indices]
    y_norm = np.array(y)[normal_indices]
    [m,b] = np.polyfit(x_norm[::-1],y_norm[::-1],deg=1)
    r = np.array(x) / (np.array(y)-b)
    return r/r[0]

def get_dac_for_frac_rn_single(x,y,normal_above_x=None,superconducting_below_x=0,
                               rn_frac_list=[0.5],plot=False,fig=None,ax=None,fulloutput=False):
    dex = np.where(np.array(x)>superconducting_below_x)
    x=np.array(x)[dex]; y=np.array(y)[dex]
    r = iv_to_frac_rn_single(x,y,normal_above_x)
    dac_list = list(np.interp(rn_frac_list,r[::-1],x[::-1]))
    for ii,dac in enumerate(dac_list): dac_list[ii] = int(dac)
    if plot:
        if fig is None:
            fig,ax = plt.subplots(1,1)
        plt.plot(x,r)
        plt.plot(dac_list,rn_frac_list,'ro')
        plt.xlabel('dac value')
        plt.ylabel('Rn fraction')
    if fulloutput:
        return dac_list, x, r
    else:
        return dac_list

def fit_normal_zero_subtract(x, y, normal_above_x):
    normal_inds = np.where(x>normal_above_x)[0]
    pfit_normal = Polynomial.fit(x[normal_inds], y[normal_inds], deg=1)
    normal_y_intersect = pfit_normal(0)
    return y-normal_y_intersect

class DataIO():
    def to_file(self, filename, overwrite = False):
        if not overwrite:
            assert not os.path.isfile(filename), print('File %s already exists.  Use overwrite=True to overwrite.'%filename)
        with open(filename, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_file(cls, filename):
        with open(filename, "r") as f:
            return cls.from_json(f.read())

# iv data classes -----------------------------------------------------------------------------
@dataclass_json
@dataclass
class IVCurveColumnData(DataIO):
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

    def xy_arrays_zero_subtracted_at_dac_high(self):
        dac_values = np.array(self.dac_values)
        fb = self.fb_values_array()
        fb = fb - fb[0,:]
        return dac_values, fb

    def xy_arrays(self):
        dac_values = np.array(self.dac_values)
        fb = self.fb_values_array()
        for i in range(fb.shape[1]):
            fb[:,i] = fb[:, i]
        return dac_values, fb

    def get_dac_at_rfrac_for_rows(self,rows=None,rn_frac_list=[0.5],normal_above_x=None,superconducting_below_x=0,plot=False):
        ''' return list of list of dac values which correspond to fraction of Rn for each row in rows '''
        fb = self.fb_values_array()
        npts,nrows = np.shape(fb)
        if rows is None:
            rows = list(range(nrows))
        elif type(rows)==int:
            rows=[rows]
        dac_at_rn_frac = []
        if plot:
            fig,ax=plt.subplots(1,1)
        else:
            fig=ax=None
        for ii, row in enumerate(rows):
            dac_at_rn_frac_ii = get_dac_for_frac_rn_single(self.dac_values,fb[:,row],
                                                           normal_above_x,superconducting_below_x,
                                                           rn_frac_list=rn_frac_list,
                                                           plot=plot,fig=fig,ax=ax)
            dac_at_rn_frac.append(dac_at_rn_frac_ii)
        if plot:
            plt.legend((rows))

        return dac_at_rn_frac

    def get_rfrac_val_for_dac(self,dac):
        ''' return List of r_frac(dac) for all rows '''
        fb = self.fb_values_array()
        npts,nrows = np.shape(fb)
        dac_list = []
        for row in range(nrows):
            foo,x,r = get_dac_for_frac_rn_single(np.array(self.dac_values),fb[:,row],normal_above_x=None,superconducting_below_x=0,
                               rn_frac_list=[0.5],fulloutput=True)
            dac_list.append(np.interp(dac,x[::-1],r[::-1]))
        return dac_list

@dataclass_json
@dataclass
class IVTempSweepData(DataIO):
    set_temps_k: List[float]
    data: List[IVCurveColumnData]

    def plot_row(self, row, zero="dac high"):
        plt.figure()
        for curve in self.data:
            if zero == "origin":
                x, y = curve.xy_arrays_zero_subtracted_at_origin()
            elif zero == "fit normal":
                x, y = curve.xy_arrays_zero_subtracted_at_normal_y_intercept(normal_above_fb=25000)
            elif zero == "dac high":
                x, y = curve.xy_arrays_zero_subtracted_at_dac_high()
            t_mK = curve.nominal_temp_k*1e3
            dt_mK = (curve.post_temp_k-curve.pre_temp_k)*1e3
            plt.plot(x, y[:,row], label=f"{t_mK:0.2f} mK, dt {dt_mK:0.2f} mK")
        plt.xlabel("dac value (arb)")
        plt.ylabel("feedback (arb)")
        plt.title(f"row={row} bayname {curve.bayname}, db_card {curve.db_cardname}, zero={zero}")
        plt.legend()

    def get_dac_at_rfrac(self, rows, rn_frac_list=[0.3,0.5,0.7],normal_above_x=None,superconducting_below_x=0,plot=False):
        ''' returns List of List of List, i.e. dac_list[temp_index][row_index][rfrac_index] '''
        dac_list = []
        for ii,temp in enumerate(self.set_temps_k):
            iv = self.data[ii]
            dac_list.append(iv.get_dac_at_rfrac_for_rows(rows=rows,rn_frac_list=rn_frac_list,
                                         normal_above_x=normal_above_x,
                                         superconducting_below_x=superconducting_below_x,plot=plot))
        return dac_list

    def get_dac_at_rfrac_row_ordered(self, rows, rn_frac_list=[0.3,0.5,0.7],normal_above_x=None,superconducting_below_x=0,plot=False):
        ''' returns List of List of List, i.e. dac_list[row_index][temp_index][rfrac_index] '''
        dac_list=[]
        for jj,row in enumerate(rows):
            dac_list_ii = []
            if plot:
                fig,ax=plt.subplots(1,1,num=jj)
                fig.suptitle('Row%02d'%row)
                ax.set_xlabel('dac values')
                ax.set_ylabel('Rn Frac')

            for ii,temp in enumerate(self.set_temps_k):
                iv = self.data[ii]
                dacs,x,r = get_dac_for_frac_rn_single(iv.dac_values,iv.fb_values_array()[:,row],normal_above_x=normal_above_x,
                                                      superconducting_below_x=superconducting_below_x,
                                                      rn_frac_list=rn_frac_list,plot=False,fig=None,ax=None,fulloutput=True)
                dac_list_ii.append(dacs)
                if plot:
                    ax.plot(x,r)
                    ax.plot(dacs,rn_frac_list,'ro')
            dac_list.append(dac_list_ii)
            if plot:
                ax.set_ylim((-0.10,1.5))
                #ax.legend((self.set_temps_k))

        return dac_list

@dataclass_json
@dataclass
class IVColdloadSweepData(DataIO): #set_cl_temps_k, pre_cl_temps_k, post_cl_temps_k, data
    set_cl_temps_k: List[float]
    data: List[IVTempSweepData]
    extra_info: dict

    def plot_row(self, row):
        #n=len(set_cl_temps_k)
        #plt.figure()
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
class IVCircuit(DataIO):
    rfb_ohm: float # feedback resistor
    rbias_ohm: float # bias resistor
    rsh_ohm: float # shunt resistor
    rx_ohm: float # parasitic resistance in series with TES
    m_ratio: float # ratio of feedback mutual inductance to input mutual inductance
    vfb_gain: float # volts/arbs of feeback (14 bit dac)
    vbias_gain: float # volts/arbs of bias (16 bit dac for blue boxes)

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

### polcal data classes ---------------------------------------------------------------------

@dataclass_json
@dataclass
class PolCalSteppedSweepData(DataIO):
    angle_deg_req: List[float]
    angle_deg_meas: List[float]
    iq_v_angle: List[Any] = dataclasses.field(repr=False) #actually a list of np arrays
    #iq_rms_values: List[Any] = dataclasses.field(repr=False) #actually a list of np arrays
    row_order: List[int]
    #bayname: str
    #db_cardname: str
    column_number: int
    source_amp_volt: float
    source_offset_volt: float
    source_frequency_hz: float
    #nominal_temp_k: float
    pre_temp_k: float
    post_temp_k: float
    pre_time_epoch_s: float
    post_time_epoch_s: float
    extra_info: dict

    def plot(self, rows_per_figure=None):
        ''' rows_per_figure is a list of lists to group detector responses
            to be plotted together.  If None will plot in groups of 8.
        '''
        if rows_per_figure is not None:
            pass
        else:
            num_in_group = 8
            n_angles,n_rows,n_iq = np.shape(self.iq_v_angle)
            n_groups = n_rows//num_in_group + 1
            rows_per_figure=[]
            for jj in range(n_groups):
                tmp_list = []
                for kk in range(num_in_group):
                    row_index = jj*num_in_group+kk
                    if row_index>=n_rows: break
                    tmp_list.append(row_index)
                rows_per_figure.append(tmp_list)
        for ii,row_list in enumerate(rows_per_figure):
            fig,ax = plt.subplots(3,num=ii)
            for row in row_list:
                ax[0].plot(self.angle_deg_meas,self.iq_v_angle[:,row,0],'o-',label=row)
                ax[1].plot(self.angle_deg_meas,self.iq_v_angle[:,row,1],'o-',label=row)
                ax[2].plot(self.angle_deg_meas,np.sqrt(self.iq_v_angle[:,row,0]**2+self.iq_v_angle[:,ii,1]**2),'o-',label=row)
            ax[0].set_ylabel('I (DAC)')
            ax[1].set_ylabel('Q (DAC)')
            ax[2].set_ylabel('Amplitude (DAC)')
            ax[2].set_xlabel('Angle (deg)')
            ax[1].legend()
            ax[0].set_title('Column %d, Group %d'%(self.column_number,ii))
        plt.show()

@dataclass_json
@dataclass
class PolCalSteppedBeamMapData(DataIO):
    xy_position_list: List[Any]
    data: List[PolCalSteppedSweepData]

### complex impedance / responsivity data classes ------------------------------------------

@dataclass_json
@dataclass
class SineSweepData(DataIO):
    frequency_hz: List[Any]
    iq_data: List[Any] = dataclasses.field(repr=False)
    amp_volt: float
    offset_volt: float
    row_order: List[int]
    column_str: str
    rfg_ohm: float
    signal_column_index: int
    reference_column_index: int
    number_of_lockin_periods: int
    pre_temp_k: float
    post_temp_k: float
    pre_time_epoch_s: int
    post_time_epoch_s: int
    extra_info: dict

    def plot(self,fignum=1,semilogx=True):
        fig, ax = plt.subplots(nrows=2,ncols=2,sharex=False,figsize=(12,8),num=fignum)
        n_freq,n_row,foo = np.shape(self.iq_data)
        iq_data = np.array(self.iq_data)
        for ii in range(n_row):
            if semilogx:
                ax[0][0].semilogx(self.frequency_hz,iq_data[:,ii,0],'o-')
                ax[0][1].semilogx(self.frequency_hz,iq_data[:,ii,1],'o-')
                ax[1][0].semilogx(self.frequency_hz,iq_data[:,ii,0]**2+iq_data[:,ii,1]**2,'o-')
                ax[1][1].semilogx(self.frequency_hz,np.arctan(iq_data[:,ii,1]/iq_data[:,ii,0]),'o-')
            else:
                ax[0][0].plot(self.frequency_hz,iq_data[:,ii,0],'o-')
                ax[0][1].plot(self.frequency_hz,iq_data[:,ii,1],'o-')
                ax[1][0].plot(self.frequency_hz,iq_data[:,ii,0]**2+iq_data[:,ii,1]**2,'o-')
                ax[1][1].plot(self.frequency_hz,np.arctan(iq_data[:,ii,1]/iq_data[:,ii,0]),'o-')

        # axes labels
        ax[0][0].set_ylabel('I')
        ax[0][1].set_ylabel('Q')
        ax[1][0].set_ylabel('I^2+Q^2')
        ax[1][1].set_ylabel('Phase')
        ax[1][0].set_xlabel('Freq (Hz)')
        ax[1][1].set_xlabel('Freq (Hz)')

        ax[1][1].legend(self.row_order)

@dataclass_json
@dataclass
class CzData(DataIO):
    data: List[List[SineSweepData]]
    db_list: List[int]
    temp_list_k: List[float]
    db_cardname: str
    db_tower_channel_str: str
    temp_settle_delay_s: float
    extra_info: dict
    normal_amp_volt: float = 0
    superconducting_amp_volt: float = 0
    normal_above_temp_k: float = 0

    def plot(self,row_index = 0, semilogx=True):
        ''' plot a 2x2 of results for data at each temperature '''
        for ii,temp in enumerate(self.temp_list_k): # loop over temp
            fig, ax = plt.subplots(nrows=2,ncols=2,sharex=False,figsize=(12,8),num=2*ii)
            fig2, ax2 = plt.subplots(1,1,num=2*ii+1)

            fig.suptitle('Temperature = %.1f mK'%(temp*1000))
            fig2.suptitle('I-Q, Temperature = %.1f mK'%(temp*1000))
            for jj, db in enumerate(self.db_list[ii]): # loop over detector bias
                ss = self.data[ii][jj]
                n_freq,n_row,foo = np.shape(ss.iq_data)
                iq_data = np.array(ss.iq_data)

                if semilogx:
                    ax[0][0].semilogx(ss.frequency_hz,iq_data[:,row_index,0],'o-')
                    ax[0][1].semilogx(ss.frequency_hz,iq_data[:,row_index,1],'o-')
                    ax[1][0].semilogx(ss.frequency_hz,iq_data[:,row_index,0]**2+iq_data[:,row_index,1]**2,'o-')
                    ax[1][1].semilogx(ss.frequency_hz,np.unwrap(np.arctan2(iq_data[:,row_index,1],iq_data[:,row_index,0])),'o-')
                else:
                    ax[0][0].plot(ss.frequency_hz,iq_data[:,row_index,0],'o-')
                    ax[0][1].plot(ss.frequency_hz,iq_data[:,row_index,1],'o-')
                    ax[1][0].plot(ss.frequency_hz,iq_data[:,row_index,0]**2+iq_data[:,row_index,1]**2,'o-')
                    ax[1][1].plot(ss.frequency_hz,np.unwrap(np.arctan2(iq_data[:,row_index,1],iq_data[:,row_index,0])),'o-')

                # plot I vs Q as second plot
                ax2.plot(iq_data[:,row_index,0],iq_data[:,row_index,1],'o-')

            # axes labels
            ax[0][0].set_ylabel('I')
            ax[0][1].set_ylabel('Q')
            ax[1][0].set_ylabel('I^2+Q^2')
            ax[1][1].set_ylabel('Phase')
            ax[1][0].set_xlabel('Freq (Hz)')
            ax[1][1].set_xlabel('Freq (Hz)')
            ax[1][1].legend(tuple(self.db_list[ii]))

            ax2.set_xlabel('I')
            ax2.set_ylabel('Q')
            ax2.set_aspect('equal','box')
            ax2.legend(tuple(self.db_list[ii]))

    def get_sc_dataset(self,Tc_k=.16):
        sc_indices = []
        for dex in np.where(np.array(self.temp_list_k) < Tc_k)[0]:
            zb_indices = np.where(np.array(self.db_list[dex])==0)[0]
            if len(zb_indices) >0 :
                for zb_dex in zb_indices:
                    sc_indices.append([dex,zb_dex])
        N = len(sc_indices)
        if N>1:
            print('More than one measurement in the superconducting state has been taken.  The indices are: ',sc_indices)
            print('Using the first measurement in the list for calibration.  Index = ',sc_indices[0])
            sc_index = sc_indices[0]
        elif N==1:
            sc_index = sc_indices[0]
        else:
            raise Exception('No measurement in the superconducting branch found')
        return np.array(self.data[sc_indices[0][0]][sc_indices[0][1]].iq_data), sc_indices

    def getZ(self,iq_data,iq_data_sc,subtract_model=False):
        ''' return the impedance by subtraction of the the bias circuit response,
            determined from the response in the superconducting branch.
            This is the Lindeman method
        '''
        # iq_data has shape (nfreq,nrow,2)
        nfreq,nrow,niq = np.shape(iq_data)
        Z = np.empty((nfreq,nrow,niq))
        for ii in range(nrow):
            z_load = (iq_data[:,ii,0]+1j*iq_data[:,ii,1])**-1
            if subtract_model:
                #p = fit_sc_q(iq_data_sc[:,ii,1],x=None,p0=[1,1])
                z_sc = (iq_data_sc[:,ii,0]+1j*iq_data_sc[:,ii,1])**-1
                z_abs = np.real(abs(z_sc))
                z_abs = z_abs[~np.isnan(z_abs)]
                z0=np.mean(z_abs[0:5])
                p = self.fit_sc_q(iq_data_sc[4:,ii,1],x=2*np.pi*np.array(self.data[0][0].frequency_hz)[4:],p0=[1,1])
                z_sc = z0*(1+1j*2*np.pi*np.array(self.data[0][0].frequency_hz)*p[1])
            else:
                z_sc = (iq_data_sc[:,ii,0]+1j*iq_data_sc[:,ii,1])**-1
            z = z_load - z_sc
            Z[:,ii,0]=np.real(z)
            Z[:,ii,1]=np.imag(z)
        return Z

    def plotZ(self, temp_k, db_indices=None,Tc_k=0.16,semilogx=True,f_max_hz=None):
        ''' plot the bias circuit subtracted impedance for all detector bias settings taken at temperature temp_k '''
        assert temp_k in self.temp_list_k, 'Requested temperature is not in temp_list_k'
        temp_index = np.where(np.array(self.temp_list_k)==temp_k)[0]
        if len(temp_index)!=1:
            print('More than one measurement at temperature temp_k.  Analyzing the first measurement')
        temp_index = temp_index[0]
        if db_indices is None:
            db_list = self.db_list[temp_index]
        else:
            db_list = list(np.array(self.db_list[temp_index])[db_indices])
        iqdata_sc, sc_dex = self.get_sc_dataset(Tc_k)
        data = self.data[temp_index] #"data" is a list of SineSweepData objects, one for each db at the requested temp
        num_db = len(data)

        # determine number of independent detector measurements in the mux frame
        if len(set(data[0].row_order)) == 1:
            num_rows = 1
        else:
            num_rows = len(data[0].row_order)

        # loop over rows/detectors, make plots per detector
        for ii in range(num_rows):
            fig, ax = plt.subplots(nrows=2,ncols=2,sharex=False,figsize=(12,8),num=2*ii)
            fig2,ax2 = plt.subplots(1,1,num=2*ii+1)
            row = data[0].row_order[ii]
            for ff in [fig,fig2]:
                ff.suptitle('Row%02d, Temperature = %.1f mK'%(data[0].row_order[ii],temp_k*1000))

            # loop over detector biases
            for jj,db in enumerate(db_list):
                if np.logical_and(db==0,temp_k<Tc_k):
                    continue
                f = data[jj].frequency_hz
                iq_data = np.array(data[jj].iq_data)
                Z = self.getZ(iq_data,iqdata_sc,subtract_model=True)
                if f_max_hz:
                    dex_max = np.argmin(abs(np.array(f)-f_max_hz))
                    f=f[:dex_max]
                    Z=Z[:dex_max,:,:]

                if semilogx:
                    ax[0][0].semilogx(f, Z[:,ii,0],'o-')
                    ax[0][1].semilogx(f, Z[:,ii,1],'o-')
                    ax[1][0].semilogx(f, Z[:,ii,0]**2+Z[:,ii,1]**2,'o-')
                    ax[1][1].semilogx(f, np.unwrap(np.arctan2(Z[:,ii,1],Z[:,ii,0])),'o-')
                else:
                    ax[0][0].plot(f, Z[:,ii,0],'o-')
                    ax[0][1].plot(f, Z[:,ii,1],'o-')
                    ax[1][0].plot(f, Z[:,ii,0]**2+Z[:,ii,1]**2,'o-')
                    ax[1][1].plot(f, np.unwrap(np.arctan2(Z[:,ii,1],Z[:,ii,0])),'o-')

                # plot I vs Q as second plot
                ax2.plot(Z[:,ii,0],Z[:,ii,1],'o-')# plot I vs Q as second plot

            # axes labels
            ax[0][0].set_ylabel('Re{Z}')
            ax[0][1].set_ylabel('Im{Z}')
            ax[1][0].set_ylabel('|Z|$^2$')
            ax[1][1].set_ylabel('Phase')
            ax[1][0].set_xlabel('Freq (Hz)')
            ax[1][1].set_xlabel('Freq (Hz)')
            ax[1][1].legend(tuple(db_list))

            ax2.set_xlabel('Re{Z}')
            ax2.set_ylabel('Im{Z}')
            ax2.set_aspect('equal','box')
            ax2.legend(tuple(db_list))# axes labels

    def fit_sc_q(self,y,x=None,p0=[1,1],plot=False):
        if x is None:
            x = np.array(self.data[0][0].frequency_hz) * 2 * np.pi
        p,foo = leastsq(self.err_func_sc_q, [1,.01], args=(x,y), full_output=0)
        if plot:
            fig,ax = plt.subplots(2,1)
            ax[0].semilogx(x,y,'r.')
            y_fit = self.fit_func_sc_q(p,x)
            y_init = self.fit_func_sc_q(p0,x)
            ax[0].semilogx(x,y_init,'k--')
            ax[0].semilogx(x,y_fit,'k-')
            ax[1].semilogx(x,y-y_fit)
            ax[1].set_xlabel('Angular Frequency (Rad)')
            ax[0].set_ylabel('Response')
            ax[1].set_ylabel('Data - Fit')
        return p

    def fit_func_sc_i(self,p,x):
        ''' x must be angular frequency
            p[0] is overall normalization
            p[1] = tau
        '''
        return p[0]*(1+(p[1]*x)**2)**-1

    def fit_func_sc_q(self,p,x):
        ''' x must be angular frequency
            p[0] = overall normalization
            p[1] = tau
        '''
        return p[0]*x*p[1]/(1+(p[1]*x)**2)

    def err_func_sc_i(self,p,x,y):
        return y - self.fit_func_sc_i(p,x)

    def err_func_sc_q(self,p,x,y):
        return y - self.fit_func_sc_q(p,x)

    def errFuncSCiq(self,p,x,yI,yQ):
        pI = [p[0],p[2]]
        pQ = [p[1],p[2]]
        return yI - self.fitFuncSCi(pI,x) + yQ - self.fitFuncSCq(pQ,x)

    # def analyzeZ(self, temp_k, Tc_k=0.16,semilogx=True,f_max_hz=None):
    #     ''' take the bias circuit subtracted impedance and fit a 1-pole filter response
    #         maybe filter out some noisy frequencies
    #         maybe get real crazy and save a plot
    #         starting with the shell of plotZ
    #     '''
    #     temp_index = np.where(np.array(self.temp_list_k)==temp_k)[0]
    #     if len(temp_index)!=1:
    #         print('More than one measurement at temperature temp_k.  Analyzing the first measurement')
    #     temp_index = temp_index[0]
    #     db_list = self.db_list[temp_index]
    #     iqdata_sc, sc_dex = self.get_sc_dataset(Tc_k)
    #     data = self.data[temp_index] #"data" is a list of SineSweepData objects, one for each db at the requested temp
    #     num_db = len(data)
    #
    #     # determine number of independent detector measurements in the mux frame
    #     if len(set(data[0].row_order)) == 1:
    #         num_rows = 1
    #     else:
    #         num_rows = len(data[0].row_order)
    #
    #     # loop over rows/detectors, make plots per detector
    #     for ii in range(num_rows):
    #         fig, ax = plt.subplots(nrows=2,ncols=2,sharex=False,figsize=(12,8),num=2*ii)
    #         fig2,ax2 = plt.subplots(1,1,num=2*ii+1)
    #         row = data[0].row_order[ii]
    #         for ff in [fig,fig2]:
    #             ff.suptitle('Row%02d, Temperature = %.1f mK'%(data[0].row_order[ii],temp_k*1000))
    #
    #         # loop over detector biases
    #         for jj,db in enumerate(db_list):
    #             if np.logical_and(db==0,temp_k<Tc_k):
    #                 continue
    #             f = np.array(data[jj].frequency_hz)
    #             iq_data = np.array(data[jj].iq_data)
    #             Z = iq_data - iqdata_sc
    #
    #             if f_max_hz:
    #                 dex_max = np.argmin(abs(np.array(f)-f_max_hz))
    #                 f=f[:dex_max]
    #                 Z=Z[:dex_max,:,:]
    #
    #             freqhr = np.logspace(np.log10(f[0]),np.log10(f[-1]),10*len(f)) # high resolution frequency for plotting fit
    #             # Avoid list of  noisy lines
    #             noisy_lines = [] #[43, 54, 110, 175]  # Hz
    #             if noisy_lines:
    #                 for nnll in noisy_lines:
    #                     fmask *= (f < nnll - delta) + (f > nnll  + delta)
    #                 f = copy(f[fmask])
    #
    #             IQfit = sp.optimize.leastsq(self.errFuncSCiq, [-2000, -1000, 1E-3], args=(f * 2 * np.pi, Z[:,ii,0], Z[:,ii,1]), full_output=1)
    #             print(jj,db_list[jj],IQfit[4])
    #             IQfitIpval = [IQfit[0][0],IQfit[0][2]]
    #             IQfitQpval = [IQfit[0][1],IQfit[0][2]]
    #             ifithr = self.fitFuncSCi(IQfitIpval, freqhr * 2 * np.pi)
    #             qfithr = self.fitFuncSCq(IQfitQpval, freqhr * 2 * np.pi)
    #             label = '{}, {:.3f} ms'.format(db_list[jj],IQfit[0][2]*1e3)
    #             if semilogx:
    #                 iplot = ax[0][0].semilogx(f, Z[:,ii,0],marker='o',ls="None")
    #                 thiscolor = iplot[0].get_color()
    #                 ax[0][0].semilogx(freqhr, ifithr, ls='-',color = thiscolor)
    #                 ax[0][1].semilogx(f, Z[:,ii,1],marker='o',ls="None", color=thiscolor)
    #                 ax[0][1].semilogx(freqhr, qfithr, ls='-',color = thiscolor)
    #                 ax[1][0].semilogx(f, Z[:,ii,0]**2+Z[:,ii,1]**2,marker='o',ls="None",color = thiscolor)
    #                 ax[1][0].semilogx(freqhr, ifithr**2+qfithr**2,ls='-',color = thiscolor)
    #                 ax[1][1].semilogx(f, np.unwrap(np.arctan2(Z[:,ii,1],Z[:,ii,0])),marker='o',ls="None",color = thiscolor,label=label)
    #                 ax[1][1].semilogx(freqhr, np.unwrap(np.arctan2(qfithr,ifithr)),ls='-',color = thiscolor)
    #             else:
    #                 iplot = ax[0][0].plot(f, Z[:,ii,0],marker='o',ls="None")
    #                 thiscolor = iplot[0].get_color()
    #                 ax[0][0].plot(freqhr, ifithr, ls='-',color = thiscolor)
    #                 ax[0][1].plot(f, Z[:,ii,1],marker='o',ls="None", color=thiscolor)
    #                 ax[0][1].plot(freqhr, qfithr, ls='-',color = thiscolor)
    #                 ax[1][0].plot(f, Z[:,ii,0]**2+Z[:,ii,1]**2,marker='o',ls="None",color = thiscolor)
    #                 ax[1][0].plot(freqhr, ifithr**2+qfithr**2,ls='-',color = thiscolor)
    #                 ax[1][1].plot(f, np.unwrap(np.arctan2(Z[:,ii,1],Z[:,ii,0])),marker='o',ls="None",color = thiscolor,label=label)
    #                 ax[1][1].plot(freqhr, np.unwrap(np.arctan2(qfithr,ifithr)),ls='-',color = thiscolor)
    #
    #             # plot I vs Q as second plot
    #             iqplot = ax2.plot(Z[:,ii,0],Z[:,ii,1],marker='o',ls="None",label=label)# plot I vs Q as second plot
    #             thiscolor = iqplot[0].get_color()
    #             ax2.plot(ifithr,qfithr,marker=None,ls='-',color=thiscolor)
    #
    #         # axes labels
    #         ax[0][0].set_ylabel('I')
    #         ax[0][1].set_ylabel('Q')
    #         ax[1][0].set_ylabel('I^2+Q^2')
    #         ax[1][1].set_ylabel('Phase')
    #         ax[1][0].set_xlabel('Freq (Hz)')
    #         ax[1][1].set_xlabel('Freq (Hz)')
    #         ax[1][1].legend() #tuple(db_list))
    #
    #         ax2.set_xlabel('I')
    #         ax2.set_ylabel('Q')
    #         ax2.set_aspect('equal','box')
    #
    #         ax2.legend() #tuple(db_list))# axes labels
    #
    #         #embed();sys.exit()

### noise data classes ---------------------------------------------------------------------
@dataclass_json
@dataclass
class NoiseData(DataIO):
    freq_hz: List[Any]
    Pxx: List[Any] = dataclasses.field(repr=False) # averaged PSD with indices [row,sample #]
    column: str # 'A','B','C', or 'D' for velma system
    row_sequence: List[int] # state sequence that maps dfb line period order to mux row select
    num_averages: int
    pre_temp_k: float
    pre_time_epoch_s: float
    dfb_bits_to_A: float # convertion of dfb bits to amps
    rfb_ohm: float
    m_ratio: float
    extra_info: dict

    def plot_avg_psd(self,row_index=None,physical_units=True,fig=None,ax=None):
        if fig is None:
            fig,ax = plt.subplots(1,1)
            fig.suptitle('Column %s averaged noise'%(self.column))
        Pxx = np.array(self.Pxx)
        nrows,nsamples = np.shape(Pxx)
        if row_index is None:
            row_index = list(range(nrows))
        if physical_units:
            m = self.dfb_bits_to_A**2
            ax.set_ylabel('PSD (A$^2$/Hz)')
        else:
            m=1
            ax.set_ylabel('PSD (arb$^2$/Hz)')
        if type(row_index) == int:
            ax.loglog(self.freq_hz,Pxx[row_index,:]*m)
        elif type(row_index) == list:
            for row in row_index:
                ax.loglog(self.freq_hz,Pxx[row,:]*m)
            ax.legend(range(nrows))
        ax.set_xlabel('Frequency (Hz)')
        return fig,ax

@dataclass_json
@dataclass
class NoiseSweepData(DataIO):
    data: List[List[NoiseData]]
    column: str # 'A','B','C', or 'D' for velma system
    row_sequence: List[int] # state sequence that maps dfb line period order to mux row select
    temp_list_k: List[float]
    db_list: List[int]
    signal_column_index: int
    db_cardname: str
    db_tower_channel_str: str
    temp_settle_delay_s: float
    extra_info: dict

    def _handle_fig(self,fig,ax):
        if fig is None:
            fig,ax = plt.subplots(1,1)
        return fig,ax

    def _phys_units(self,physical_units):
        if physical_units:
            y_label_str='PSD (A$^2$/Hz)'
            m=self.data[0][0].dfb_bits_to_A
        else:
            y_label_str='PSD (arb$^2$/Hz)'
            m=1
        return m, y_label_str

    def plot_row_for_temp(self,row_index,temp,bias_indices=None,list=None,physical_units=True,fig=None,ax=None):
        fig,ax = self._handle_fig(fig,ax)
        fig.suptitle('Column %s, Row %02d, T=%d mK'%(self.column,self.row_sequence[row_index],temp*1000))
        temp_m = []
        m, y_label_str = self._phys_units(physical_units)
        dex = self.temp_list_k.index(temp)
        data = self.data[dex]
        if bias_indices==None:
            db_list=self.db_list[dex]
        else:
            db_list=list(np.array(self.db_list)[bias_indices])
        for ii,db in enumerate(db_list):
            df = data[ii]
            temp_m.append(df.pre_temp_k)
            ax.loglog(df.freq_hz,np.array(df.Pxx)[row_index,:]*m**2)
        ax.set_xlabel('Frquency (Hz)')
        ax.set_ylabel(y_label_str)
        ax.legend(db_list)
        print('measured temperatures: ',temp_m)
        return fig, ax 

    def plot_row_single(self,row_index,temp_index,bias_index,physical_units=True,fig=None,ax=None):
        df = data[temp_index][bias_index]
        fig,ax = self._handle_fig(fig,ax)
        m, y_label_str = self._phys_units(physical_units)
        ax.loglog(df.freqs,np.array(df.Pxx)[row_index,:]*m**2)
        return fig, ax 

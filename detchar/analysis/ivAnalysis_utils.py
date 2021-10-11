''' ivAnalysis_utils.py '''

# TO DO / NOTES
# efficiency relative to simulated passband
# Determine some metric for efficiency and report that one number.  what rn_frac?  What cl_temp?
# IVTempSweepData ought to have the coldload setpoint.
# column name for IVColdloadAnalzyeOneRow?
# units for plots?
# plot ranges
# deal with crap rn_frac cuts
# plot titles
# plot sizes
# remove redundant methods that are in different classes IVColdloadOneRow and IVversusADRTempOneRow
# robust data cutting of IV before P vs T fits
# clear nan from P versus Tb fits

import detchar
from detchar.iv_data import IVCircuit
from detchar import IVColdloadSweepData, IVTempSweepData
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k,h,c
from scipy.integrate import quad, simps
from scipy.optimize import leastsq

def smooth(y, box_pts=5):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def Pnu_thermal(nu,T):
    ''' power spectral density (W/Hz) of single mode from thermal source at
        temperature T (in K) and frequency nu (Hz).
    '''
    x = h*nu/(k*T)
    B = h*nu * (np.exp(x)-1)**-1
    return B

def thermalPower(nu1,nu2,T,F=None):
    ''' Calculate the single mode thermal power (in pW) emitted from a blackbody
        at temperature T (in Kelvin) from frequency nu1 to nu2 (in Hz).
        F = F(\nu) is an arbitrary absolute passband defined between nu1 and nu2 with linear
        sampling between nu1 and nu2.  The default is F=None, in which case a
        top hat band is assumed.
    '''
    try:
        if F==None: # case for tophat
            P = quad(Pnu_thermal,nu1,nu2,args=(T))[0] # toss the error
    except: # case for arbitrary passband shape F
        N = len(F)
        nu = np.linspace(nu1,nu2,N)
        integrand = self.Pnu_thermal(nu,T)*F
        P = simps(integrand,nu)
    return

class IVClean():
    ''' Data cleaning methods relevant for IV curves '''
    def is_dac_descending(self,dac):
        result = False
        if dac[1] - dac[0] < 0 :
            result = True
        return result

    def is_iv_inverted(self,dac,fb):
        result = False
        dac_descending = self.is_dac_descending(dac)
        if dac_descending:
            pt1_idx = 1
            pt2_idx = 0
        else:
            pt1_idx = -2
            pt2_idx = -1
        if fb[pt1_idx]-fb[pt2_idx]>0:
            result = True
        return result

    def find_first_zero(self,vec,verbose=False):
        ''' single vector, return index and value where vec crosses zero '''
        ii=0; val = vec[ii]
        while val > 0:
            if ii==len(vec)-1:
                if verbose: print('zero crossing not found')
                ii = None; val == None
                break
            ii=ii+1
            val=vec[ii]
        return ii, val

    def get_turn_index(self,dac,fb,showplot=False):
        ''' return the index corresponding to the IV turnaround for a single IV curve.
        '''

        dfb = np.diff(fb,axis=0)
        if not self.is_iv_inverted(dac,fb):
            dfb=-1*dfb
        dex, val = self.find_first_zero(dfb)
        if showplot:
            fig1 = plt.figure(1) # plot of delta i
            plt.plot(dfb,'o-')
            if dex != None: plt.plot(dex,val,'ro')

            fig2 = plt.figure(2)
            plt.plot(fb,'o-')
            if dex != None: plt.plot(dex,fb[dex],'ro')
            plt.show()
        return dex#,val

    def find_bad_data_index(self,dac,fb,threshold=0.5,showplot=False):
        ''' Return the index where IV curve misbehaves.
            dac and fb(dac) must be in descending order

            Algorithm is to look at fb(dac) for dac values lower than the IV turnaround.
            If the second derivative is positive (ie the slope of the IV curve in transition changes sign),
            the index is flagged and returned.

            If no bad data found, return the last index (such that subsequent method includes all data points)
        '''
        assert dac[1]-dac[0] < 0, ('dac values must be in descending order')
        #print('The threshold is = ', threshold)

        turn_dex = self.get_turn_index(dac,fb,showplot=False)
        if turn_dex == None: return len(dac)

        dfb = np.diff(fb,axis=0)
        norm_dfb = np.mean(dfb[0:10],axis=0)
        x = dfb/norm_dfb # normalize to slope in the normal branch
        ddfb = smooth(np.diff(x,axis=0),3)

        if len(dac)-turn_dex <= 2: # case were IV turn is close to end of data range
            dex = len(dac)
        else:
            ii = turn_dex
            val = ddfb[ii]

            while val < threshold:
                ii+=1
                if ii==len(ddfb): break
                val=ddfb[ii]
            dex = ii+1

            if showplot:
                plt.figure(1)
                plt.xlabel('index')
                plt.ylabel('fb (arb)')
                plt.plot(fb,'bo-')
                plt.plot([turn_dex],[fb[turn_dex]],'ro')
                plt.plot([dex],[fb[dex]],'go')
                plt.plot(fb[0:dex+1],'r*')

                # plt.figure(2)
                # plt.xlabel('index')
                # plt.ylabel('$\Delta$fb')
                # plt.plot(x,'bo-')
                # plt.plot([dex],[x[dex]],'go')
                #
                # plt.figure(3)
                # plt.xlabel('index')
                # plt.ylabel('$\Delta$ $\Delta$ fb')
                # plt.plot(ddfb,'bo-')
                # plt.plot(smooth(ddfb,3))
                # plt.plot([dex-1],[ddfb[dex-1]],'go')
                plt.show()
        return dex

    def remove_NaN(self,arr):
        ''' only works on 1d vector, not array '''
        return arr[~np.isnan(arr)]

    def get_turn_index_arr(self,fb_arr,showplot=False):
        ''' return the indices corresponding to the IV turnaround for a set of IV curves.
            Assumes fb_arr is ordered from highest voltage bias setting to lowest
        '''

        di = np.diff(fb_arr,axis=0) # difference of current array
        #di_rev = di[::-1] # reverse order di_rev[0] corresponse to highest v_bias
        n,m = np.shape(di)
        ivTurnDex = []
        for ii in range(m):
            dex, val = self.find_first_zero(di[:,ii])
            print(dex)
            ivTurnDex.append(dex)

        if showplot:
            fig1 = plt.figure(1) # plot of delta i
            plt.plot(di,'o-')
            for ii in range(m):
                plt.plot(ivTurnDex[ii],fb_arr[ivTurnDex[ii],ii],'ro')

            fig2 = plt.figure(2)
            plt.plot(fb_arr,'o-')
            for ii in range(m):
                plt.plot(ivTurnDex[ii],fb_arr[ivTurnDex[ii],ii],'ro')
            plt.show()

        return ivTurnDex

class IVCurveColumnDataExplore():
    def __init__(self,iv_curve_column_data,iv_circuit=None):
        # fixed globals
        self.n_normal_pts = 10

        self.data = iv_curve_column_data
        self.iv_circuit = self._handle_iv_circuit(iv_circuit)

        # the raw data for one column and dimensionality
        self.x_raw, self.y_raw = self.data.xy_arrays_zero_subtracted_at_dac_high() # raw units
        self.n_pts, self.n_rows = np.shape(self.y_raw)

        # data converted to physical units
        self.phys_units = self._handle_physical_units()
        self.labels = self._handle_labels(self.phys_units)

    # data manipulation methods --------------------------------------------------------
    def _handle_iv_circuit(self, iv_circuit):
        if iv_circuit is not None:
            ivcirc = iv_circuit
        else:
            if self.data.extra_info:
                if 'config' in self.data.extra_info.keys():
                    if 'calnums' in self.data.extra_info['config'].keys():
                        vmax = self._get_voltage_max_from_source(self.data.extra_info['config']['voltage_bias']['source'])
                        ivcirc = IVCircuit(rfb_ohm=self.data.extra_info['config']['calnums']['rfb']+50.0,
                                           rbias_ohm=self.data.extra_info['config']['calnums']['rbias'],
                                           rsh_ohm=self.data.extra_info['config']['calnums']['rjnoise'],
                                           rx_ohm=0,
                                           m_ratio=self.data.extra_info['config']['calnums']['mr'],
                                           vfb_gain=self.data.extra_info['config']['calnums']['vfb_gain']/(2**14-1),
                                           vbias_gain=vmax/(2**16-1))
            else:
                ivcirc = None
        return ivcirc

    def _get_voltage_max_from_source(self,voltage_source='tower'):
        assert voltage_source in ['tower','bluebox'], print('Voltage source ',voltage_source,' not recognized')

        if voltage_source == 'tower':
            vmax = 2.5
        elif voltage_source == 'bluebox':
            vmax = 6.5
        return vmax


    def _handle_physical_units(self):
        y = self.remove_offset(False)
        if self.iv_circuit is not None:
            self.v, self.i = self.convert_to_physical_units(self.x_raw,y)
            phys_units = True
        else:
            self.i = y
            self.v = np.empty((self.n_pts,self.n_rows))
            for ii in range(self.n_rows):
                self.v[:,ii] = self.x_raw
            phys_units = False
        return phys_units

    def _handle_labels(self,phys_units):
        labels = {}
        if phys_units:
            labels['iv']={'x':'Vbias (V)',
                          'y':'Current (A)'}
            labels['vp']={'x':'Vbias (V)',
                          'y':'Power (W)'}
            labels['rp']={'x':'Resistance ($\Omega$)',
                          'y':'Power (W)'}
            labels['vr']={'x':'Vbias (V)',
                          'y':'Resistance ($\Omega$)'}
            labels['dy']={'x':'Vbias (V)',
                          'y':'$\Delta{I}$ (A)'}
            labels['responsivity']={'x':'Vbias (V)',
                          'y':'$\delta{I}$/$\delta{P}$ (V$^{-1})'}

        else:
            labels['iv']={'x':'Vbias (DAC)',
                          'y':'Current (DAC)'}
            labels['vp']={'x':'Vbias (DAC)',
                          'y':'Power (DAC)'}
            labels['rp']={'x':'Resistance (DAC))',
                          'y':'Power (DAC)'}
            labels['vr']={'x':'Vbias (DAC)',
                          'y':'Resistance (DAC)'}
            labels['dy']={'x':'Vbias (DAC)',
                          'y':'$\Delta{I}$ (DAC)'}
            labels['responsivity']={'x':'Vbias (DAC)',
                          'y':'$\delta{I}/\delta{P}$ (DAC)'}
        return labels

    def remove_offset(self,showplot=False):
        x = self.x_raw[::-1][-self.n_normal_pts:]
        yr = self.y_raw[::-1,:]
        yr = yr[-self.n_normal_pts:,:]
        m, b = np.polyfit(x,yr,deg=1)
        y = self.y_raw - b
        if m[0]<0: y = -1*y

        if showplot:
            for ii in range(self.n_rows):
                plt.plot(self.x_raw,y[:,ii])
            plt.show()
        return y

    def convert_to_physical_units(self,x,y):
        assert self.iv_circuit is not None, 'You must supply an iv_circuit to convert to physical units!'
        return self.iv_circuit.to_physical_units(x,y)

    def get_responsivity(self):
        v = np.diff(self.v,axis=0)+self.v[:-1,:]
        di = np.diff(self.i, axis=0)
        dp = np.diff(self.i*self.v, axis=0)
        resp = di/dp
        return v, resp

    # plotting methods --------------------------------------
    def plot_iv(self, fig_num=1):
        fig = plt.figure(fig_num)
        for ii in range(self.n_rows):
            plt.plot(self.v[:,ii],self.i[:,ii],label='%02d'%ii)
        plt.xlabel(self.labels['iv']['x'])
        plt.ylabel(self.labels['iv']['y'])
        plt.legend(loc='upper right')
        return fig

    def plot_responsivity(self,fig_num=1):
        fig = plt.figure(fig_num)
        v,r = self.get_responsivity()
        #for ii in range(self.n_rows):
        plt.plot(v,r)
        plt.xlabel(self.labels['responsivity']['x'])
        plt.ylabel(self.labels['responsivity']['y'])
        plt.legend(tuple(range(self.n_rows)),loc='upper right')
        return fig

    def plot_vipr(self,fig_num=1, figtitle=None):
        #v=v*self.vipr_scaling[0]; i=i*self.vipr_scaling[1]; p=p*self.vipr_scaling[2]; r=r*self.vipr_scaling[3]
        v = self.v
        i = self.i
        p = v*i
        r = v/i

        # fig 1, 2x2 of converted IV
        #fig = plt.figure(fig_num)
        fig, ax = plt.subplots(nrows=2,ncols=2,sharex=False,figsize=(12,8),num=fig_num)
        ax=[ax[0][0],ax[0][1],ax[1][0],ax[1][1]]
        for ii in range(self.n_rows):
            ax[0].plot(v[:,ii],i[:,ii])
            ax[1].plot(v[:,ii],p[:,ii])
            ax[2].plot(r[:,ii],p[:,ii])
            #ax[3].plot(p[:,ii],r[:,ii]/r[-2,ii])
            ax[3].plot(v[:,ii],r[:,ii])

        ax[0].set_xlabel(self.labels['iv']['x'])
        ax[0].set_ylabel(self.labels['iv']['y'])
        ax[1].set_xlabel(self.labels['vp']['x'])
        ax[1].set_ylabel(self.labels['vp']['y'])
        ax[2].set_xlabel(self.labels['rp']['x'])
        ax[2].set_ylabel(self.labels['rp']['y'])
        ax[3].set_xlabel(self.labels['vr']['x'])
        ax[3].set_ylabel(self.labels['vr']['y'])

        #xlabels = ['V (V)','V (V)','P (W)','V (V)']
        #ylabels = ['I (A)', 'P (W)', 'R ($\Omega$)', 'R ($\Omega$)']
        # for ii in range(4):
        #     ax[ii].set_xlabel(xlabels[ii])
        #     ax[ii].set_ylabel(ylabels[ii])
        #     ax[ii].grid()

        # plot range limits
        ax[0].set_xlim((0,np.max(v)*1.1))
        ax[0].set_ylim((0,np.max(i)*1.1))
        ax[1].set_xlim((0,np.max(v)*1.1))
        ax[1].set_ylim((0,np.max(p)*1.1))
        ax[2].set_xlim((0,np.max(r[0,:])*1.1))
        ax[2].set_ylim((0,np.max(p)*1.1))
        ax[3].set_xlim((0,np.max(v)*1.1))
        ax[3].set_ylim((0,np.max(r[0,:])*1.1))
        #ax[3].set_xlim((0,np.max(p)*1.1))
        #ax[3].set_ylim((0,1.1))

        if figtitle is not None:
            fig.suptitle(figtitle)
        return fig

    def plot_dy(self):
        dy = np.diff(self.i,axis=0)
        print(np.shape(dy))
        for ii in range(self.n_rows):
            plt.plot(self.v[0:-1,ii],dy[:,ii],label='%02d'%ii)
        plt.xlabel(self.labels['dy']['x'])
        plt.ylabel(self.labels['dy']['y'])
        plt.legend(loc='upper right')

class IVSetAnalyzeRow(IVClean):
    def __init__(self,dac_values,fb_values_arr,state_list=None,iv_circuit=None,figtitle=None):
        ''' Analyze IV set at different physical conditions for one row.

            dac_values: np_array of dac_values (corresponding to voltage bias across TES),
                        a common dac_value for all IVs is required
            fb_values_arr: N_dac_val x N_sweep numpy array, column ordered
            state_list: list in which items are strings,
                        description of the state of the system for that IV curve
                        used in the legend of plots
        '''
        # options
        self.n_normal_pts = 10
        self.use_ave_offset = True
        self.figtitle=None
        self.vipr_unit_labels = ['($\mu$V)','($\mu$A)','(pW)','(m$\Omega$)']
        self.vipr_scaling = [1e6,1e6,1e12,1e3]

        #
        self.figtitle = figtitle
        self.dacs = dac_values
        self.fb_raw = fb_values_arr
        self.state_list = state_list
        self.n_dac_values, self.num_sweeps = np.shape(self.fb_raw)
        if iv_circuit==None:
            self.to_physical_units = False
        else:
            self.iv_circuit = iv_circuit
            self.to_physical_units = True

        # do conversion to physical units
        self.v,self.i,self.p,self.r = self.get_vipr()
        self.ro = self.r / self.r[0,:]

        self.v,self.i,self.p,self.r,ro = self.remove_bad_data()

    def power_difference_analysis(self,fig_num=1):

        fig,ax = plt.subplots(2,num=fig_num)
        for ii in range(self.num_sweeps):
            ax[0].plot(self.r[:,ii]*1000,self.p[:,ii]*1e12,label=ii)

        ax[0].set_xlabel('Resistance (mOhms)')
        ax[0].set_ylabel('Power (pW)')
        ax[0].legend(tuple(self.state_list))
        ax[0].set_xlim(0,self.r[0,0]*1100)
        ax[1].set_xlim(0,self.r[0,0]*1100)

        dP = np.diff(self.p)
        #dP = self.p[:,1] - self.p[:,0]
        ax[1].plot(self.r[:,0]*1e3,dP*1e12)
        ax[1].set_xlabel('Resistance (mOhms)')
        ax[1].set_ylabel('dP (pW)')

        return fig

    def plot_raw(self,fig_num=1):
        figXX = plt.figure(fig_num)
        for ii in range(self.num_sweeps):
            plt.plot(self.dacs, self.fb_raw[:,ii])
        #plt.plot(self.dacs,self.fb_raw)
        plt.xlabel("dac values (arb)")
        plt.ylabel("fb values (arb)")
        plt.legend(tuple(self.state_list))
        if self.figtitle != None:
            plt.title(self.figtitle)

    def fb_align_and_remove_offset(self,showplot=False):
        fb_align = np.zeros((self.n_dac_values,self.num_sweeps))
        for ii in range(self.num_sweeps): # align fb DC levels to a common value
            dy = self.fb_raw[0,ii]-self.fb_raw[0,0]
            fb_align[:,ii] = self.fb_raw[:,ii]-dy

        # remove offset
        x = self.dacs[::-1][-self.n_normal_pts:]
        y = fb_align[::-1,:] ; y = y[-self.n_normal_pts:,:]
        m, b = np.polyfit(x,y,deg=1)

        if self.use_ave_offset:
            if np.std(b)/np.mean(b) > 0.01:
                print('Warning DC offset of curves differs by > 1\%')
                print('Offset fit: ',np.mean(b),'+/-',np.std(b))
            b = np.mean(b)
        fb_align = fb_align - b
        if m[0]<0: fb_align = fb_align*-1

        if showplot:
            for ii in range(self.num_sweeps):
                plt.plot(self.dacs,fb_align[:,ii])
            plt.show()

        return fb_align

    def get_vipr(self, showplot=False):
        ''' returns the voltage, current, power, and resistance vectors '''
        if self.to_physical_units:
            v,i = self.iv_circuit.to_physical_units(self.dacs,self.fb_align)
        else:
            v = np.zeros((self.n_dac_values,self.num_sweeps))
            for ii in range(self.num_sweeps):
                v[:,ii] = self.dacs
            i=self.fb_align
        p=v*i; r=v/i

        if showplot:
            self.plot_vipr([v,i,p,r])
        return v,i,p,r

    def plot_vipr(self,data_list=None,fig_num=1):

        if data_list==None:
            v=self.v; i=self.i; p=self.p; r=self.r
        else:
            v=data_list[0]; i=data_list[1]; p=data_list[2]; r=data_list[3]
        v=v*self.vipr_scaling[0]; i=i*self.vipr_scaling[1]; p=p*self.vipr_scaling[2]; r=r*self.vipr_scaling[3]

        # fig 1, 2x2 of converted IV
        #fig = plt.figure(fig_num)
        figXX, ax = plt.subplots(nrows=2,ncols=2,sharex=False,figsize=(12,8))
        ax=[ax[0][0],ax[0][1],ax[1][0],ax[1][1]]
        for ii in range(self.num_sweeps):
            ax[0].plot(v[:,ii],i[:,ii])
            ax[1].plot(v[:,ii],p[:,ii])
            ax[2].plot(p[:,ii],r[:,ii])
            #ax[3].plot(p[:,ii],r[:,ii]/r[-2,ii])
            ax[3].plot(v[:,ii],r[:,ii])

        xlabels = ['V %s'%self.vipr_unit_labels[0],'V %s'%self.vipr_unit_labels[0],'P %s'%self.vipr_unit_labels[2],'V %s'%self.vipr_unit_labels[0]]
        ylabels = ['I %s'%self.vipr_unit_labels[1],'P %s'%self.vipr_unit_labels[2], 'R %s'%self.vipr_unit_labels[3],'R %s'%self.vipr_unit_labels[3]]

        #xlabels = ['V (V)','V (V)','P (W)','V (V)']
        #ylabels = ['I (A)', 'P (W)', 'R ($\Omega$)', 'R ($\Omega$)']

        for ii in range(4):
            ax[ii].set_xlabel(xlabels[ii])
            ax[ii].set_ylabel(ylabels[ii])
            ax[ii].grid()

        # plot ranges
        ax[0].set_xlim((0,np.max(v)*1.1))
        ax[0].set_ylim((0,np.max(i)*1.1))
        ax[1].set_xlim((0,np.max(v)*1.1))
        ax[1].set_ylim((0,np.max(p)*1.1))
        ax[2].set_xlim((0,np.max(p)*1.1))
        ax[2].set_ylim((0,np.max(r[0,:])*1.1))
        ax[3].set_xlim((0,np.max(v)*1.1))
        ax[3].set_ylim((0,np.max(r[0,:])*1.1))
        #ax[3].set_xlim((0,np.max(p)*1.1))
        #ax[3].set_ylim((0,1.1))

        if self.figtitle != None:
            figXX.suptitle(self.figtitle)
        if self.state_list != None:
            ax[3].legend(tuple(self.state_list))
        return figXX

    def remove_bad_data(self,threshold=0.5):
        def cut(arr,dexs):
            n,m=np.shape(arr)
            arr_copy = arr.copy()
            for ii in range(m):
                if dexs[ii]==self.n_dac_values: pass
                else: arr_copy[dexs[ii]+1:,ii] = np.ones(self.n_dac_values-dexs[ii]-1)*np.nan
            return arr_copy

        dexs=[]
        for ii in range(self.num_sweeps):
            dexs.append(self.find_bad_data_index(self.dacs,self.fb_raw[:,ii],threshold=threshold,showplot=False))
        self.bad_data_idx = dexs
        v_clean = cut(self.v,dexs)
        i_clean = cut(self.i,dexs)
        r_clean = cut(self.r,dexs)
        ro_clean = cut(self.ro,dexs)
        p_clean = cut(self.p,dexs)
        return v_clean, i_clean, p_clean, r_clean, ro_clean

class IVSetAnalyzeColumn():
    ''' analyze IV curves taken under different physical conditions.
    '''
    def __init__(self,ivcurve_column_data_list,state_list=None,iv_circuit=None):
        ''' ivcurve_column_data_list is a list of IVCurveColumnData instances '''
        self.data_list = ivcurve_column_data_list
        self.state_list = state_list
        self.iv_circuit = iv_circuit
        assert type(self.data_list) == list, 'ivcurve_column_data_list must be of type List'
        self.num_sweeps = len(ivcurve_column_data_list)
        self.dacs, self.fb_raw = self.get_raw_iv()
        self.n_pts = len(self.dacs[0])

    def get_raw_iv(self):
        ''' returns raw dac and feedback values with fb(max(dac))==0 '''
        dacs=[];fbs=[]
        for iv in self.data_list:
            d,f = iv.xy_arrays_zero_subtracted_at_dac_high()
            dacs.append(d) ; fbs.append(f)
        return dacs, fbs

    def get_data_for_row(self,row_index):
        ''' return the raw data for row_index '''
        dac = self.dacs[0] # a cludge for now, might want to allow different dac ranges per IV in future?
        fb_arr = np.zeros((self.n_pts,self.num_sweeps))
        for ii in range(self.num_sweeps):
            fb_arr[:,ii] = self.fb_raw[ii][:,row_index]
        return dac, fb_arr

    def plot_row(self,row_index,to_physical_units=True):
        dac, fb = self.get_data_for_row(row_index)
        iv_set = IVSetAnalyzeRow(dac,fb,state_list=self.state_list,iv_circuit=self.iv_circuit,figtitle='Row%02d'%row_index)
        iv_set.plot_raw(fig_num=1)
        iv_set.plot_vipr(fig_num=2)
        plt.show()

class IVversusADRTempOneRow(IVSetAnalyzeRow):
    ''' analyze thermal transport from IV curve set from one row in which ADR temperature is varied '''
    def __init__(self,dac_values,fb_values_arr, temp_list_k, normal_resistance_fractions=[0.8,0.9],iv_circuit=None):
        ''' dac_values: np_array of dac_values (corresponding to voltage bias across TES),
                        a common dac_value for all IVs is required
            fb_values_arr: N_dac_val x N_sweep numpy array, column ordered in which columns are for different adr temperatures
            temp_list_k: adr temperature list in K, must match column order
        '''

        self.temp_list_k = temp_list_k
        self.rn_fracs = normal_resistance_fractions
        self.num_rn_fracs = len(self.rn_fracs)
        temp_list_k_str = []
        for ii in range(len(temp_list_k)):
            temp_list_k_str.append(str(temp_list_k[ii]))
        super().__init__(dac_values,fb_values_arr,temp_list_k_str,iv_circuit)
        self.v_clean, self.i_clean, self.r_clean, self.p_clean, self.ro_clean = self.remove_bad_data(threshold=.000000001)
        self.p_at_rnfrac = self.get_value_at_rn_frac(self.rn_fracs,self.p_clean,self.ro_clean)
        print(self.p_at_rnfrac)
        self.pfits = self.fit_pvt_for_all_rn_frac()


    def get_value_at_rn_frac(self,rn_fracs,arr,ro):
        '''
        Return the value of arr at fraction of Rn.
        input:
        rn_fracs: fraction of Rn values to be evaluated (NOT PERCENTAGE RN).
        arr: NxM array to determine the Rn fraction at
        ro: NxM normalized resistance

        arr and ro must be same shape
        return: len(rn_fracs) x M array of the interpolated values

        '''
        # ensure rn_fracs is a np.array
        if type(rn_fracs)!=np.ndarray:
            rn_fracs = np.array(rn_fracs)
        assert len(np.where(rn_fracs>1)[0])==0, ('rn_fracs values must be < 1')
        n,m=np.shape(arr)
        result = np.zeros((len(rn_fracs),m))
        for ii in range(m):
            x = IVClean().remove_NaN(ro[:,ii])
            y = IVClean().remove_NaN(arr[:,ii])
            YY = np.interp(rn_fracs,x[::-1],y[::-1])

            # over write with NaN for when data does not extend to fracRn
            ro_min = np.min(x)
            toCut = np.where(rn_fracs<ro_min)[0]
            N = len(toCut)
            if N >0:
                YY[0:N] = np.zeros(N)*np.NaN
            result[:,ii] = YY
        return result

    def plot_pr(self,fig_num=1):
        pPlot = self.get_value_at_rn_frac([0.995],arr=self.p,ro=self.ro)

        # FIG1: P versus R/Rn
        fig = plt.figure(fig_num)
        plt.plot(self.ro_clean, self.p_clean,'-') # plots for all Tbath
        plt.plot(self.rn_fracs,self.p_at_rnfrac,'ro')
        plt.xlim((0,1.1))
        try:
            #plt.ylim((np.min(p_at_rnfrac[~np.isnan(p_at_rnfrac)])*0.9,1.25*np.max(pPlot[~np.isnan(pPlot)])))
            plt.ylim((0,1.25*np.max(pPlot[~np.isnan(pPlot)])))
        except:
            pass
        plt.xlabel('Normalized Resistance')
        plt.ylabel('Power')
        #plt.title(plottitle)
        plt.legend(tuple(self.temp_list_k))
        plt.grid()
        #plt.title(self.figtitle)
        return fig

    def plot_pt(self,fig_num=2):
        # power plateau (evaluated at each rn_frac) versus T_cl
        fig = plt.figure(fig_num)
        llabels=[]
        temp_arr = np.linspace(np.min(self.temp_list_k),np.max(self.temp_list_k),100)
        for ii in range(self.num_rn_fracs):
            if not np.isnan(self.p_at_rnfrac[ii,:]).any():
                plt.plot(self.temp_list_k,self.p_at_rnfrac[ii,:],'o')
                llabels.append(self.rn_fracs[ii])
        for ii in range(self.num_rn_fracs):
            plt.plot(temp_arr,self.ktn_fit_func(self.pfits[ii],temp_arr),'k--')
        plt.xlabel('T$_{b}$ (K)')
        plt.ylabel('TES power plateau')
        plt.legend((llabels))
        #plt.title(self.figtitle)
        plt.grid()
        return fig

    def fit_pvt_for_all_rn_frac(self):
        pfits=[]
        for ii in range(self.num_rn_fracs):
            pfit,pcov = self.fit_pvt(np.array(self.temp_list_k),self.p_at_rnfrac[ii])
            pfits.append(pfit)
        return pfits

    def ktn_fit_func(self,v,t):
        '''
        fit function is P = v[0](v[1]^v[2]-t^v[2])
        '''
        return v[0]*(v[1]**v[2]-t**v[2])

    def fit_pvt(self,t,p,init_guess=[20.e-9,.2,4.0]):
        ''' fits saturation power versus temperature to recover fit parameters K, T and n

            fit function is P = K(T^n-t^n)

            input:
            t: vector of bath temperatures
            p: vector of saturation powers
            init_guess: initial guess parameters for K,T,n in that order

            output:
            fit coefficients
            covarience matrix (diagonals are variance of fit parameters)
        '''
        fitfunc = lambda v,x: v[0]*(v[1]**v[2]-x**v[2])
        errfunc = lambda v,t,p: v[0]*(v[1]**v[2]-t**v[2])-p
        lsq = leastsq(errfunc,init_guess, args=(t,p),full_output=1)
        pfit, pcov, infodict, errmsg, success = lsq
        if success > 4:
            print('Least squares fit failed.  Success index of algorithm > 4 means failure.  Success index = %d'%success)
            print('Error message: ', errmsg)
            # print('Here is a plot of the data:')
            # plt.figure(50)
            # plt.plot(t,p,'bo')
            # plt.plot(t,fitfunc(pfit,t),'k-')
            # plt.plot(t,fitfunc(pfit,t),'r-')
            # plt.legend(('data','fit','init guess'))
            # plt.xlabel('Bath Temperature (K)')
            # plt.ylabel('power (W)')
            # plt.show()
            pcov=np.ones((len(pfit),len(pfit)))
        for ii in range(len(pfit)):
            pfit[ii]=abs(pfit[ii])
        s_sq = (infodict['fvec']**2).sum()/(len(p)-len(init_guess))
        pcov=pcov*s_sq
        return pfit,pcov

class IVColdloadAnalyzeOneRow():
    ''' Analyze a set of IV curves for a single detector taken at multiple
        coldload temperatures and a single bath temperature

        input --
        dac_values: raw dac values corresponding to the voltage bias.
                    Must be the same for all IV curves
        fb_array: N_dac_value x M array, where M is the number of IV curves taken, one for each cold load temperature
        cl_temps_k: list cold load temperatures in K
        bath_temp_k: <number> substrate temperature during measurement
        row_name: <str> for plot labeling only
        dev_name: <str> for plot labeling only
        iv_circuit: instance of iv_data.IVCircuit to convert data to physical units
        predicted_power_w: <list or 1D numpy array> predicted power (in watts) corresponding to each cl_temp, used to determine optical efficiency
        dark_power_w: <list or 1D numpy array> electrical power of dark bolometer corresponding to each cl_temp, used for dark subtraction
    '''

    # to debug
    # axis limits and NaN for plot_vipr

    def __init__(self,dac_values,fb_array,cl_temps_k,bath_temp_k,
                 row_name=None,det_name=None,
                 iv_circuit=None,predicted_power_w=None,dark_power_w=None):
        # fixed globals / options
        self.n_normal_pts=10 # number of points for normal branch fit
        self.use_ave_offset=True # use a global offset to align fb, not individual per curve
        self.rn_fracs = [0.5,0.6,0.7,0.8,0.9] # slices in Rn space to compute electrical power versus temperature
        self.n_rn_fracs = len(self.rn_fracs)

        # other globals
        self.dacs = dac_values
        self.fb = fb_array # NxM array of feedback values.  Columns are per coldload temperature
        self.cl_temps_k = cl_temps_k
        self.bath_temp_k = bath_temp_k
        self.det_name = det_name
        self.row_name = row_name
        self.figtitle = self.det_name+', '+self.row_name+' , Tb = %.1f mK'%(self.bath_temp_k*1000)
        self.n_dac_values, self.n_cl_temps = np.shape(self.fb)
        self.fb_align = self.fb_align_and_remove_offset() # 2D array of aligned feedback
                                                          # (i.e. the appropriate DC offset has been removed)
        # convert to physical units if iv_circuit provided
        if iv_circuit is not None:
            self.iv_circuit = iv_circuit
            self.to_physical_units = True
        else:
            self.to_physical_units = False

        # do analysis of v,i,r,p vectors.  Place main results as globals to class
        self.v,self.i,self.p,self.r = self.get_vipr(showplot=False)
        self.ro = self.r / self.r[0,:]
        self.v_orig, self.i_orig, self.p_orig, self.r_orig, self.ro_orig = self.v, self.i, self.p, self.r, self.ro
        self.remove_bad_data()
        self.p_at_rnfrac = self.get_value_at_rn_frac(self.rn_fracs,self.p,self.ro) # n_rn_fracs x n_cl_temps

        # get change in power versus change in temperature ("infinitesimal", thus lower case d)
        self.cl_dT_k = np.diff(np.array(self.cl_temps_k))
        self.dp_at_rnfrac = np.diff(self.p_at_rnfrac)

        # get change in power relative to P(T=T_cl_index), (not infinitesimal thus big D)
        self.cl_DT_k, self.Dp_at_rnfrac, self.T_cl_index = self.get_Delta_pt()

        # handle darks
        self.dark_analysis, self.dark_power_w, self.dark_Dp_w, self.dark_dp_w = self._handle_dark(dark_power_w)

        # predicted power
        self.analyze_eta, self.predicted_power_w, self.predicted_Dp_w, self.predicted_dp_w = self._handle_prediction(predicted_power_w)

        # get efficiency
        if self.analyze_eta: self.get_efficiency_at_rnfrac()

        # plotting stuff
        self.colors = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']

    # handle methods ----------------------------------------------------------
    def _handle_prediction(self,predicted_power_w):
        if predicted_power_w is not None:
            assert len(predicted_power_w) == self.n_cl_temps, 'Hey, the dimensions of the predicted power do not match the number of cold load temperatures!'
            p_in = predicted_power_w
            Dp, dp = self.get_power_difference_1D(p_in, self.T_cl_index)
            analyze_eta = True
        else:
            p_in = Dp = dp = None
            analyze_eta = False
        return analyze_eta, p_in, Dp, dp

    def _handle_dark(self, dark_power_w):
        if dark_power_w is None:
            dark_analysis = False
            dark_Dp_w = dark_dp_w = None
        else:
            dark_analysis = True
            dark_Dp_w, dark_dp_w = self.get_power_difference_1D(dark_power_w, self.T_cl_index)
        return dark_analysis, dark_power_w, dark_Dp_w, dark_dp_w

    # helper methods -----------------------------------------------------------

    def fb_align_and_remove_offset(self,showplot=False):
        # this method ought to be placed in another general class
        fb_align = np.zeros((self.n_dac_values,self.n_cl_temps))
        for ii in range(self.n_cl_temps): # align fb DC levels to a common value
            dy = self.fb[0,ii]-self.fb[0,0]
            fb_align[:,ii] = self.fb[:,ii]-dy

        # remove offset
        x = self.dacs[::-1][-self.n_normal_pts:]
        y = fb_align[::-1,:] ; y = y[-self.n_normal_pts:,:]
        m, b = np.polyfit(x,y,deg=1)

        if np.std(b)/np.mean(b) > 0.01:
            print('Warning DC offset of curves differs by > 1\%')
            print('Offset fit: ',np.mean(b),'+/-',np.std(b))
        if self.use_ave_offset: b = np.mean(b)
        fb_align = fb_align - b
        if m[0]<0: fb_align = fb_align*-1
        if showplot:
            for ii in range(self.n_cl_temps):
                plt.plot(self.dacs,fb_align[:,ii])
            plt.show()
        return fb_align

    def removeNaN(self,arr):
        ''' only works on 1d vector, not array '''
        return arr[~np.isnan(arr)]

    def remove_bad_data(self):
        ''' remove poor data (typically in superconducting transition).
            Used primarily to avoid throwing off determination of power at a given Rn
        '''
        def cut(arr,dexs):
            n,m=np.shape(arr)
            arr_copy = arr.copy()
            for ii in range(m):
                if dexs[ii]==self.n_dac_values: pass
                else: arr_copy[dexs[ii]+1:,ii] = np.ones(self.n_dac_values-dexs[ii]-1)*np.nan
            return arr_copy

        dexs=[]
        for ii in range(self.n_cl_temps):
            dexs.append(IVClean().find_bad_data_index(self.dacs,self.fb[:,ii],threshold=0.5,showplot=False))
        self.bad_data_idx = dexs
        self.v = cut(self.v,dexs)
        self.i = cut(self.i,dexs)
        self.r = cut(self.r,dexs)
        self.ro = cut(self.ro,dexs)
        self.p = cut(self.p,dexs)

    def update_T_cl_index(self,T_cl_index):
        self.cl_DT_k, self.Dp_at_rnfrac, self.T_cl_index = self.get_Delta_pt(cl_index = T_cl_index)
        if self.dark_analysis:
            self.dark_Dp_w = self.dark_power_w - self.dark_power_w[self.T_cl_index]
        if self.analyze_eta:
            self.predicted_Dp_w = self.predicted_power_w - self.predicted_power_w[T_cl_index]
            self.eta_Dp = self.get_efficiency()

    def power_subtraction(self,dP_w,dP_w_vec):
        ''' dP_w is an N x M array with N rows of %rn cuts over M coldload temps
            dP_w_vec (typically a dark detector response) is a 1 x M array
        '''
        return dP_w - dP_w_vec

    def calc_power_differences(self,T_cl_index):
        if self.analyze_eta: # power predictions
            self.predicted_power_Dp, self.predicted_power_dp = self.get_power_difference_1D(self.predicted_power_w,T_cl_index)
        self.dp_at_rnfrac, self.Dp_at_rnfrac = self.get_power_difference_2D(self.p_at_rnfrac,T_cl_index)

    # gets ---------------------------------------------------------------------
    def get_power_difference_1D(self,power_w, T_cl_index):
        ''' return the '''
        Dp = power_w - power_w[T_cl_index]
        dp = np.diff(power_w)
        return Dp, dp

    def get_power_difference_2D(self,arr,T_cl_index):
        Dp = arr - arr[:,T_cl_index]
        dp = np.diff(arr)
        return Dp, dp

    def get_vipr(self,showplot=False):
        ''' returns the voltage, current, power, and resistance vectors '''
        if self.to_physical_units:
            v,i = self.iv_circuit.to_physical_units(self.dacs,self.fb_align)
        else:
            v = np.zeros((self.n_dac_values,self.n_cl_temps))
            for ii in range(self.n_cl_temps):
                v[:,ii] = self.dacs
            i=self.fb_align
        p=v*i; r=v/i

        if showplot:
            self.plot_vipr([v,i,p,r])
        return v,i,p,r

    def get_value_at_rn_frac(self,rn_fracs,arr,ro):
        '''
        Return the value of arr at fraction of Rn.
        input:
        rn_fracs: fraction of Rn values to be evaluated (NOT PERCENTAGE RN).
        arr: NxM array to determine the Rn fraction at
        ro: NxM normalized resistance

        arr and ro must be same shape
        return: len(rn_fracs) x M array of the interpolated values

        This method used to determine the electrical power at some Rn fraction.
        It is used to make the global variable self.p_at_rnfrac, a 2D array
        Each row of this array is the electrical power determined at each cl_temps_k, and for
        a single cut in Rn.

        '''
        # ensure rn_fracs is a np.array
        if type(rn_fracs)!=np.ndarray:
            rn_fracs = np.array(rn_fracs)
        assert len(np.where(rn_fracs>1)[0])==0, ('rn_fracs values must be < 1')
        n,m=np.shape(arr)
        result = np.zeros((len(rn_fracs),m))
        for ii in range(m):
            x = self.removeNaN(ro[:,ii])
            y = self.removeNaN(arr[:,ii])
            YY = np.interp(rn_fracs,x[::-1],y[::-1])

            # over write with NaN for when data does not extend to fracRn
            ro_min = np.min(x)
            toCut = np.where(rn_fracs<ro_min)[0]
            N = len(toCut)
            if N >0:
                YY[0:N] = np.zeros(N)*np.NaN
            result[:,ii] = YY
        return result

    def get_Delta_pt(self,rn_fracs=None,p_at_rnfrac=None,cl_index=None):
        if cl_index == None: dex = np.argmin(self.cl_temps_k)
        else: dex = cl_index
        if p_at_rnfrac==None: p_at_rnfrac=self.p_at_rnfrac
        if rn_fracs==None: rn_fracs=self.rn_fracs

        DT_k = np.array(self.cl_temps_k)-self.cl_temps_k[dex]
        DP_w = (p_at_rnfrac.transpose() - p_at_rnfrac[:,dex]).transpose()
        return DT_k, DP_w, dex

    def get_pt_delta_for_rnfrac(self,rnfrac):
        assert rnfrac in self.rn_fracs, ('requested rnfrac = ',rnfrac, 'not in self.rn_fracs = ',self.rn_fracs)
        dex = self.rn_fracs.index(rnfrac)
        return self.dP_w[dex]

    def get_efficiency_at_rnfrac(self):
        ''' '''
        self.eta_Dp_arr = self.Dp_at_rnfrac / self.predicted_Dp_w
        self.eta_dp_arr = self.dp_at_rnfrac / self.predicted_dp_w
        if self.dark_analysis:
            self.eta_Dp_arr_darksubtracted = (self.Dp_at_rnfrac - self.dark_Dp_w) / self.predicted_Dp_w
            self.eta_dp_arr_darksubtracted = (self.dp_at_rnfrac - self.dark_dp_w) / self.predicted_dp_w
        else:
            self.eta_Dp_arr_darksubtracted = self.eta_dp_arr_darksubtracted = None


    def get_eta_mean_std(self,eta):
        n,m = np.shape(eta) # n = %rn cut index, m = Tcl index
        dexs=[] # rn cuts w/out np.nan entries
        for ii in range(n):
            if not np.isnan(eta[ii,1:]).any():
                dexs.append(ii)

        eta_m = np.mean(eta[dexs,1:],axis=0)
        eta_std = np.std(eta[dexs,1:],axis=0)
        return eta_m, eta_std

    # plotting methods ---------------------------------------------------------
    def plot_raw(self,fb_align_dc_level=True,fig_num=1):
        fig = plt.figure(fig_num)
        for ii, cl_temp in enumerate(self.cl_temps_k):
            if fb_align_dc_level:
                dy = self.fb[0,ii]-self.fb[0,0]
            else: dy=0
            plt.plot(self.dacs, self.fb[:,ii]-dy)
        plt.xlabel('DAC values')
        plt.ylabel('Feedback values')
        plt.title(self.figtitle+'  raw IV')
        plt.legend((self.cl_temps_k),loc='upper right')
        return fig

    def plot_vipr(self,fig_num=1):
        # fig 1, 2x2 of converted IV
        #fig = plt.figure(fig_num)
        figXX, ax = plt.subplots(nrows=2,ncols=2,sharex=False,figsize=(12,8))
        ax=[ax[0][0],ax[0][1],ax[1][0],ax[1][1]]
        for ii in range(self.n_cl_temps):
            ax[0].plot(self.v[:,ii],self.i[:,ii])
            ax[1].plot(self.v[:,ii],self.p[:,ii])
            ax[2].plot(self.p[:,ii],self.r[:,ii])
            #ax[3].plot(p[:,ii],r[:,ii]/r[-2,ii])
            ax[3].plot(self.v[:,ii],self.r[:,ii])
        # xlabels = ['V ($\mu$V)','V ($\mu$V)','P (pW)','V ($\mu$V)']
        # ylabels = ['I ($\mu$A)', 'P (pW)', 'R (m$\Omega$)', 'R (m$\Omega$)']
        xlabels = ['V (V)','V (V)','P (W)','V (V)']
        ylabels = ['I (A)', 'P (W)', 'R ($\Omega$)', 'R ($\Omega$)']

        for ii in range(4):
            ax[ii].set_xlabel(xlabels[ii])
            ax[ii].set_ylabel(ylabels[ii])
            ax[ii].grid()

        # plot ranges
        # ax[0].set_xlim((0,np.max(self.v)*1.1))
        # ax[0].set_ylim((0,np.max(self.i)*1.1))
        # ax[1].set_xlim((0,np.max(self.v)*1.1))
        # ax[1].set_ylim((0,np.max(self.p)*1.1))
        # ax[2].set_xlim((0,np.max(self.p)*1.1))
        # ax[2].set_ylim((0,np.max(self.r[0,:])*1.1))
        # ax[3].set_xlim((0,np.max(self.v)*1.1))
        # ax[3].set_ylim((0,np.max(self.r[0,:])*1.1))
        #ax[3].set_xlim((0,np.max(p)*1.1))
        #ax[3].set_ylim((0,1.1))

        figXX.suptitle(self.figtitle+'  IV, PV, RP, RV')
        ax[3].legend(tuple(self.cl_temps_k))
        return figXX

    def plot_pr(self,fig_num=1):
        pPlot = self.get_value_at_rn_frac([0.995],arr=self.p,ro=self.ro)

        # FIG1: P versus R/Rn
        fig = plt.figure(fig_num)
        plt.plot(self.ro, self.p,'-') # plots for all Tbath
        plt.plot(self.rn_fracs,self.p_at_rnfrac,'ro')
        plt.xlim((0,1.1))
        try:
            #plt.ylim((np.min(p_at_rnfrac[~np.isnan(p_at_rnfrac)])*0.9,1.25*np.max(pPlot[~np.isnan(pPlot)])))
            plt.ylim((0,1.25*np.max(pPlot[~np.isnan(pPlot)])))
        except:
            pass
        plt.xlabel('Normalized Resistance')
        plt.ylabel('Power')
        #plt.title(plottitle)
        plt.legend((self.cl_temps_k))
        plt.grid()
        plt.title(self.figtitle)
        return fig

    def plot_pt(self,fig_num=1):
        # power plateau (evaluated at each rn_frac) versus T_cl
        fig = plt.figure(fig_num)
        llabels=[]
        for ii in range(len(self.rn_fracs)):
            if not np.isnan(self.p_at_rnfrac[ii,:]).any():
                plt.plot(self.cl_temps_k,self.p_at_rnfrac[ii,:],'o-')
                llabels.append(self.rn_fracs[ii])
        plt.xlabel('T$_{cl}$ (K)')
        plt.ylabel('TES power plateau')
        plt.legend((llabels))
        plt.title(self.figtitle)
        plt.grid()
        return fig

    def plot_pt_delta(self,fig_num=1, dp_at_rnfrac_dark_subtracted=None):
        ''' plot change in saturation power relative to minimum coldload temperature '''
        fig = plt.figure(fig_num)
        legend_vals = []
        if self.analyze_eta:
            plt.plot(self.dcl_temps_k,self.predicted_dp_w,'k-',label='$\Delta{P}_{pred}$')
        jj=0
        for ii in range(len(self.rn_fracs)):
            if not np.isnan(self.dp_at_rnfrac[ii,:]).any():
                plt.plot(self.dcl_temps_k,self.dp_at_rnfrac[ii,:],'o-',color=self.colors[jj],label=str(self.rn_fracs[ii]))
                try:
                    if len(dp_at_rnfrac_dark_subtracted) > 0:
                        plt.plot(self.dcl_temps_k,dp_at_rnfrac_dark_subtracted[ii,:],'o--',color=self.colors[jj],label='_nolegend_')
                except:
                    pass
                jj+=1
        plt.xlabel('T$_{cl}$ - %.1f K'%self.cl_temps_k[self.T_cl_index])
        plt.ylabel('P$_o$ - P')
        plt.legend()
        plt.grid()
        plt.title(self.figtitle)
        return fig

    def plot_efficiency(self, fig_num=1, eta_dark_subtracted=None):
        fig = plt.figure(fig_num)
        plt.plot(self.dcl_temps_k,self.eta.transpose())
        plt.xlabel('T$_{cl}$ - %.1f K'%self.cl_temps_k[self.T_cl_index])
        plt.ylabel('Efficiency')
        plt.legend(self.rn_fracs)
        plt.grid()
        plt.title(self.figtitle)
        return fig

    # def plot_efficiency(self,cl_dT_k, eta, rn_fracs, fig_num=1, eta_dark_subtracted=None):
    #     fig = plt.figure(fig_num)
    #     jj=0
    #     for ii in range(len(rn_fracs)):
    #         if not np.isnan(eta[ii,1:]).any():
    #             plt.plot(cl_dT_k,eta[ii,:],'o-',color=self.colors[jj], label=str(rn_fracs[ii]))
    #             try:
    #                 if len(eta_dark_subtracted) > 0:
    #                     plt.plot(cl_dT_k,eta_dark_subtracted[ii,:],'o--',color=self.colors[jj],label='_nolegend_')
    #             except:
    #                 pass
    #             jj+=1
    #     eta_m, eta_std = self.get_eta_mean_std(eta)
    #     eta_m_ds, eta_std_ds = self.get_eta_mean_std(eta_dark_subtracted)
    #     plt.errorbar(cl_dT_k[1:],eta_m,eta_std,color='k',linewidth=2,ecolor='k',elinewidth=2,label='mean')
    #     plt.errorbar(cl_dT_k[1:],eta_m_ds,eta_std_ds,color='k',linewidth=2,ecolor='k',elinewidth=2,label='mean ds',linestyle='--')
    #     plt.xlabel('T$_{cl}$ - %.1f K'%self.cl_temps_k[self.T_cl_index])
    #     plt.ylabel('Efficiency')
    #     plt.legend()
    #     plt.grid()
    #     plt.title(self.figtitle)
    #     return fig

    def plot_full_analysis(self,showfigs=False,savefigs=False):
        ''' Make plots of the full analysis:
            1) raw IV, one curve per coldload temperature
            2) vipr, (2x2 plot if IV, PV, RP, RV), on curve per coldload temperature
            3) P versus R, with cuts at rnfracs
            4) P versus T_cl

        '''
        figs = []
        figs.append(self.plot_raw(True,fig_num=1)) # raw
        figs.append(self.plot_vipr(fig_num=2)) # 2x2 of converted data
        figs.append(self.plot_pr(fig_num=3))
        if not np.isnan(self.p_at_rnfrac).all():
            figs.append(self.plot_pt(fig_num=4))
            if self.dark_analysis:
                figs.append(self.plot_pt_delta(fig_num=5, dp_at_rnfrac_dark_subtracted=self.dP_w_darksubtracted))
            else:
                figs.append(self.plot_pt_delta(fig_num=5))
        if self.analyze_eta:
            figs.append(self.plot_efficiency(fig_num=6))
        if savefigs:
            fig_appendix=['raw','vipr','pr','pt','dpt','eta']
            for ii,fig in enumerate(figs):
                fig.savefig(self.row_name+'_%d_'%ii+fig_appendix[ii]+'.png')
        if showfigs: plt.show()
        for fig in figs:
            plt.close(fig)
            #fig.clf()

class IVColdloadSweepAnalyzer():
    ''' Class to analyze a coldload IV sweep '''
    def __init__(self,filename_json,detector_map=None,iv_circuit=None):
        self.df = IVColdloadSweepData.from_file(filename_json)
        self.filename = filename_json
        self.data = self.df.data
        self.det_map = detector_map
        self.iv_circuit = iv_circuit
        #self.data is a list of IVTempSweepData, one per coldload temperature setpoint
        #self.data[0].data is a list of IVCurveColumnData, one per bath temperature setpoint
        #self.data[ii].data[jj], ii is the coldload setpoint temperature index
        #                        jj is the bath temperature setpoint index

        # globals about coldload temperatures
        self.set_cl_temps_k = self.df.set_cl_temps_k
        self.max_cl_temp_k = np.max(self.set_cl_temps_k)
        self.min_cl_temp_k = np.min(self.set_cl_temps_k)
        self.n_cl_temps = len(self.set_cl_temps_k)
        self.pre_cl_temps_k = self.df.extra_info['pre_cl_temps_k']
        self.post_cl_temps_k = self.df.extra_info['post_cl_temps_k']
        self.cl_therm_index = 0
        self.measured_cl_temps_k = self.get_measured_coldload_temps(self.cl_therm_index)

        # globals about bath temperatures
        self.set_bath_temps_k = self.data[0].set_temps_k
        self.n_bath_temps = len(self.set_bath_temps_k)

        # globals about IV
        self.dac_values = np.array(self.data[0].data[0].dac_values)
        self.n_dac_values, self.n_rows = np.shape(self.data[0].data[0].fb_values)

        # devices
        foo, self.n_rows = np.shape(self.data[0].data[0].fb_values_array())
        self.row_index_list = list(range(self.n_rows))

    def print_info(self):
        # Too add in future:
        # 1) date/time of start/end of data
        # 2) Did data complete?
        print('Coldload set temperatures: ',self.set_cl_temps_k)
        print('Measured coldload temperatures: ',self.measured_cl_temps_k)
        print('ADR set temperatures: ',self.set_bath_temps_k)
        print('use plot_measured_cl_temps() and plot_measured_bath_temps() to determine if set temperatures were achieved')

    def _package_cl_temp_to_list(self):
        cl_temp_list = []
        cl_temp_list.append(list(self.measured_cl_temps_k[:,0]))
        cl_temp_list.append(list(self.measured_cl_temps_k[:,1]))
        return cl_temp_list

    def get_measured_coldload_temps(self,index=0):
        return 0.5*np.array(self.pre_cl_temps_k)[:,index] + 0.5*np.array(self.post_cl_temps_k)[:,index]

    def get_cl_sweep_dataset_for_row(self,row_index,bath_temp_index=0,cl_indices=None):
        if cl_indices==None:
            cl_indices = list(range(self.n_cl_temps))
        n_cl = len(cl_indices)
        fb = np.zeros((self.n_dac_values,n_cl))
        for ii in range(n_cl):
            fb[:,ii] = self.data[ii].data[bath_temp_index].fb_values_array()[:,row_index]
        return self.dac_values, fb

    def plot_measured_cl_temps(self):
        plt.figure(1)
        plt.xlabel('Setpoint Temperature (K)')
        plt.ylabel('Measured Temperature (K)')
        #cl_temp_list = self._package_cl_temp_to_list()
        plt.plot(self.set_cl_temps_k,self.pre_cl_temps_k,'*')
        plt.plot(self.set_cl_temps_k,self.post_cl_temps_k,'*')
        plt.plot(list(range(self.max_cl_temp_k+1)),'b--')
        plt.legend(('ChA pre','ChB pre','ChA post','ChB post'))
        plt.grid()

        plt.figure(2)
        plt.xlabel('Time (arb)')
        plt.ylabel('Temperature (K)')
        x = list(range(self.n_cl_temps))
        plt.plot(x,self.set_cl_temps_k,'ko-')
        plt.plot(x,self.pre_cl_temps_k,'*')
        plt.plot(x,self.post_cl_temps_k,'*')
        plt.legend(('Setpoint','ChA pre','ChB pre','ChA post','ChB post'))
        plt.show()

    def plot_measured_bath_temps(self):
        pts = [['o','*'],['o','*'],['o','*']]
        colors = ['b','g','r','c','m','y','k']
        for ii in range(self.n_cl_temps):
            for jj in range(self.n_bath_temps):
                if ii==0:
                    plt.axhline(self.set_bath_temps_k[jj],color='k',linestyle='--')
                print("cl temp: ",self.set_cl_temps_k[ii],
                      " Bath temp = ",self.data[ii].data[jj].nominal_temp_k,
                      " Pre IV temp = ",self.data[ii].data[jj].pre_temp_k,
                      " Post IV temp = ",self.data[ii].data[jj].post_temp_k)
                plt.plot([ii],[self.data[ii].data[jj].pre_temp_k],marker=pts[jj][0],color=colors[jj])
                plt.plot([ii],[self.data[ii].data[jj].post_temp_k],marker=pts[jj][1],color=colors[jj])
        plt.xlabel('Coldload index')
        plt.ylabel('Bath Temperature (K)')
        #plt.legend(('pre','post'))
        plt.show()

    def plot_single_iv(self,row_index,cl_temp_index,bath_temp_index):
        #cl_temp_index = self.get_cl_temp_index(cl_temp)
        #bath_temp_index = self.get_bath_temp_index(bath_temp)
        x = self.data[cl_temp_index].data[bath_temp_index].dac_values
        y = self.data[cl_temp_index].data[bath_temp_index].fb_values_array()[:,row_index]
        plt.figure(1)
        plt.xlabel('DAC values')
        plt.ylabel('Feedback values')
        plt.plot(x,y,'-')
        plt.title('Row index = %d, CL_temp_index = %.1f K, Tb_index = %d mK'%(row_index,cl_temp_index,bath_temp_index))
        plt.show()

    def plot_cl_temp_sweep_for_row(self,row_index,bath_temp_index,cl_indices=None):
        if cl_indices==None:
            cl_indices = list(range(self.n_cl_temps))
        x,fb_arr = self.get_cl_sweep_dataset_for_row(row_index,bath_temp_index,cl_indices)
        plt.figure(1)
        for ii in range(len(cl_indices)):
            dy = fb_arr[0,ii]-fb_arr[0,0]
            plt.plot(self.dac_values, fb_arr[:,ii]-dy)
        plt.xlabel('DAC values')
        plt.ylabel('Feedback values')
        plt.title('Row index = %d, Tb = %d mK'%(row_index,self.set_bath_temps_k[bath_temp_index]*1000))
        plt.legend((np.array(self.set_cl_temps_k)[cl_indices]),loc='upper right')
        plt.show()

    def plot_sweep_analysis_for_row(self,row_index,bath_temp_index,cl_indices,showfigs=True,savefigs=False):
        dacs,fb = self.get_cl_sweep_dataset_for_row(row_index=row_index,bath_temp_index=bath_temp_index,cl_indices=cl_indices)
        if self.det_map != None:

            device_dict = {'Row%02d'%row_index: self.det_map['Row%02d'%row_index]['devname']}
            keys = self.det_map['Row%02d'%row_index].keys()
            passband_dict = {}
            if 'freq_edges_ghz' in keys:
                passband_dict['freq_edges_ghz']=self.det_map['Row%02d'%row_index]['freq_edges_ghz']
            if 'passband_ghz' in keys:
                passband_dict['passband_ghz']=self.det_map['Row%02d'%row_index]['passband_ghz']
        else:
            device_dict=None ; passband_dict = {}

        iva = IVColdloadAnalyzeOneRow(dacs,fb,
                                      cl_temps_k=list(np.array(self.set_cl_temps_k)[cl_indices]),# put in measured values here!
                                      bath_temp_k=self.set_bath_temps_k[bath_temp_index],
                                      device_dict=device_dict,
                                      iv_circuit=self.iv_circuit,
                                      passband_dict=passband_dict)
        iva.plot_full_analysis(showfigs,savefigs)

    def plot_pt_delta_diff(self,row_index,dark_row_index,bath_temp_index,cl_indices):
        ''' plot the difference in the change in power verus the change in cold load temperature between two bolometers.
            This is often useful for dark subtraction
        '''
        dacs,fb = self.get_cl_sweep_dataset_for_row(row_index=row_index,bath_temp_index=bath_temp_index,cl_indices=cl_indices)
        dacs,fb_dark = self.get_cl_sweep_dataset_for_row(row_index=dark_row_index,bath_temp_index=bath_temp_index,cl_indices=cl_indices)

        iva = IVColdloadAnalyzeOneRow(dacs,fb,
                                      cl_temps_k=list(np.array(self.set_cl_temps_k)[cl_indices]),# put in measured values here!
                                      bath_temp_k=self.set_bath_temps_k[bath_temp_index],
                                      device_dict=None,
                                      iv_circuit=self.iv_circuit,
                                      passband_dict=None)

        iva_dark = IVColdloadAnalyzeOneRow(dacs,fb_dark,
                                      cl_temps_k=list(np.array(self.set_cl_temps_k)[cl_indices]),# put in measured values here!
                                      bath_temp_k=self.set_bath_temps_k[bath_temp_index],
                                      device_dict=None,
                                      iv_circuit=self.iv_circuit,
                                      passband_dict=None)

        n_rfrac, n_clTemps = np.shape(iva.dP_w)
        for ii in range(n_rfrac):
            plt.plot(np.array(self.set_cl_temps_k)[cl_indices],iva.dP_w[ii,:],'bo-')
            plt.plot(np.array(self.set_cl_temps_k)[cl_indices],iva_dark.dP_w[ii,:],'ko-')
            plt.plot(np.array(self.set_cl_temps_k)[cl_indices],iva.dP_w[ii,:]-iva_dark.dP_w[ii,:],'bo--')

        plt.show()

    def full_analysis(self,bath_temp_index,cl_indices,showfigs=False,savefigs=False,rn_fracs=None,dark_rnfrac=0.7):
        if self.det_map != None:
            # first collect dark responses for each pixel
            dark_keys, dark_indices = self.det_map.get_keys_and_indices_of_type(type_str='dark')
            dark_dPs = {}
            for idx in dark_indices:
                row_dict = self.det_map.map_dict['Row%02d'%idx]
                dacs,fb = self.get_cl_sweep_dataset_for_row(row_index=idx,bath_temp_index=bath_temp_index,cl_indices=cl_indices)
                iva_dark = IVColdloadAnalyzeOneRow(dacs,fb,
                                                   cl_temps_k=list(np.array(self.set_cl_temps_k)[cl_indices]),
                                                   bath_temp_k=self.set_bath_temps_k[bath_temp_index],
                                                   device_dict=self.det_map.get_onerow_device_dict(idx),
                                                   iv_circuit=self.iv_circuit,
                                                   passband_dict=None)
                dark_dPs[str(row_dict['position'])]=iva_dark.get_pt_delta_for_rnfrac(dark_rnfrac)

            # now loop over all rows
            for row in self.row_index_list:
                row_dict = self.det_map.map_dict['Row%02d'%(row)]
                print('Row%02d'%(row),': ',row_dict)
                dacs,fb = self.get_cl_sweep_dataset_for_row(row_index=row,bath_temp_index=bath_temp_index,cl_indices=cl_indices)
                if row_dict['type']=='optical':
                    dark_dP = dark_dPs[str(row_dict['position'])]
                    passband_dict={'freq_edges_ghz':self.det_map.map_dict['Row%02d'%row]['freq_edges_ghz']}
                else:
                    dark_dP = None; passband_dict=None
                iva = IVColdloadAnalyzeOneRow(dacs,fb,
                                              cl_temps_k=list(np.array(self.set_cl_temps_k)[cl_indices]),
                                              bath_temp_k=self.set_bath_temps_k[bath_temp_index],
                                              device_dict=self.det_map.get_onerow_device_dict(row),
                                              iv_circuit=self.iv_circuit,
                                              passband_dict=None,#passband_dict,
                                              dark_dP_w=dark_dP)
                if rn_fracs is not None:
                    iva.rn_fracs = rn_fracs
                iva.plot_full_analysis(showfigs,savefigs)

if __name__ == "__main__":

    # circuit parameters
    iv_circuit = IVCircuit(rfb_ohm=5282.0+50.0,
                           rbias_ohm=10068.0,
                           rsh_ohm=0.0662,
                           rx_ohm=0,
                           m_ratio=8.259,
                           vfb_gain=1.017/(2**14-1),
                           vbias_gain=6.5/(2**16-1))

    path = '/home/pcuser/data/lbird/20210320/'
    fname = 'lbird_hftv0_ColumnA_ivs_20210413_1618354151.json'
    row_index = 23

    # construct fb_arr versus Tbath for a single row
    df = IVTempSweepData.from_file(path+fname)
    dac, fb = df.data[0].xy_arrays()
    n_sweeps = len(df.set_temps_k)
    fb_arr = np.zeros((len(dac),n_sweeps))
    for ii in range(n_sweeps):
        dac, fb = df.data[ii].xy_arrays()
        fb_arr[:,ii] = fb[:,row_index]

    iv_tsweep = IVversusADRTempOneRow(dac_values=dac,fb_values_arr=fb_arr[:,:-2], temp_list_k=df.set_temps_k[:-2], normal_resistance_fractions=[0.985,0.99],iv_circuit=iv_circuit)
    iv_tsweep.plot_raw(1)
    iv_tsweep.plot_vipr(None,2)
    iv_tsweep.plot_pr(fig_num=3)
    iv_tsweep.plot_pt(fig_num=4)
    print(iv_tsweep.pfits[0])
    plt.show()

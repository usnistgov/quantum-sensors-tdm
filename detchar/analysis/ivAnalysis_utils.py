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
        print('The threshold is = ', threshold)

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

class IVSetAnalyzeOneRow(IVClean):
    def __init__(self,dac_values,fb_values_arr,state_list=None,iv_circuit=None):
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
        
    def plot_raw(self,fig_num=1):
        plt.figure(fig_num)
        for ii in range(self.num_sweeps):
            plt.plot(self.dacs, self.fb_raw[:,ii])
        #plt.plot(self.dacs,self.fb_raw)
        plt.xlabel("dac values (arb)")
        plt.ylabel("fb values (arb)")
        plt.legend(tuple(self.state_list))

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
            for ii in range(self.n_cl_temps):
                plt.plot(self.dacs,fb_align[:,ii])
            plt.show()

        return fb_align

    def get_vipr(self,showplot=False):
        ''' returns the voltage, current, power, and resistance vectors '''
        self.fb_align = self.fb_align_and_remove_offset(showplot=False)

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
        return v_clean, i_clean, r_clean, p_clean, ro_clean 

class IVversusADRTempOneRow(IVSetAnalyzeOneRow):
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
            # pylab.figure(50)
            # pylab.plot(t,p,'bo')
            # pylab.plot(t,fitfunc(pfit,t),'k-')
            # pylab.plot(t,fitfunc(pfit,t),'r-')
            # pylab.legend(('data','fit','init guess'))
            # pylab.xlabel('Bath Temperature (K)')
            # pylab.ylabel('power (W)')
            # pylab.show()
            pcov=np.ones((len(pfit),len(pfit)))
        for ii in range(len(pfit)):
            pfit[ii]=abs(pfit[ii])
        s_sq = (infodict['fvec']**2).sum()/(len(p)-len(init_guess))
        pcov=pcov*s_sq
        return pfit,pcov

# class IVSetColumnAnalyze():
#     ''' analyze IV curves taken under different physical conditions. 
#     '''
#     def __init__(self,ivcurve_column_data_list,state_list=None,iv_circuit=None):
#         ''' ivcurve_column_data_list is a list of IVCurveColumnData instances '''
#         self.data_list = ivcurve_column_data_list
#         self.state_list = state_list
#         self.iv_circuit = iv_circuit
#         assert type(self.data_list) == list, 'ivcurve_column_data_list must be of type List'
#         self.num_sweeps = len(ivcurve_column_data_list)
#         self.dacs, self.fb_raw = self.get_raw_iv()
#         self.n_pts = len(self.dacs[0])

#     def get_raw_iv(self):
#         ''' returns raw dac and feedback values with fb(max(dac))==0 '''
#         dacs=[];fbs=[]
#         for iv in self.data_list:
#             d,f = iv.xy_arrays_zero_subtracted_at_dac_high()
#             dacs.append(d) ; fbs.append(f)
#         return dacs, fbs

#     def get_data_for_row(self,row_index):
#         ''' return the raw data for row_index '''
#         dac = self.dacs[0] # a cludge for now, might want to allow different dac ranges per IV in future?
#         fb_arr = np.zeros((self.n_pts,self.num_sweeps))
#         for ii in range(self.num_sweeps):
#             fb_arr[:,ii] = self.fb_raw[ii][:,row_index]
#         return dac, fb_arr
        
#     def plot_row(self,row_index,to_physical_units=True):
#         dac, fb = self.get_data_for_row(row_index)
#         iv_set = IVSetAnalyzeOneRow(dac,fb,state_list=self.state_list,iv_circuit=self.iv_circuit)
#         iv_set.plot_raw(fig_num=1)
#         iv_set.plot_vipr(fig_num=2)
#         plt.show()
                
class IVColdloadAnalyzeOneRow():
    ''' Analyze a set of IV curves for a single detector taken at multiple
        coldload temperatures and a single bath temperature
    '''

    def __init__(self,dac_values,fb_array,cl_temps_k,bath_temp_k,
                device_dict=None,iv_circuit=None,passband_dict=None,
                dark_dP_w=None):
        self.dacs = dac_values
        self.fb = fb_array # NxM array of feedback values.  Columns are per coldload temperature
        self.fb_align = None
        self.cl_temps_k = cl_temps_k
        self.bath_temp_k = bath_temp_k
        self.det_name, self.row_name = self._handle_device_dict(device_dict)
        self.n_dac_values, self.n_cl_temps = np.shape(self.fb)

        # fixed globals
        self.n_normal_pts=10 # number of points for normal branch fit
        self.use_ave_offset=True # use a global offset to align fb, not individual per curve
        self.rn_fracs = [0.5,0.6,0.7,0.8,0.9] # slices in Rn space to compute delta Ps
        if iv_circuit==None:
            self.to_physical_units = False
        else:
            self.iv_circuit = iv_circuit
            self.to_physical_units = True

        self.figtitle = self.det_name+', '+self.row_name+' , Tb = %.1f mK'%(self.bath_temp_k*1000)

        # do analysis, place main results as globals to class
        self.v,self.i,self.p,self.r = self.get_vipr(showplot=False)
        self.ro = self.r / self.r[0,:]
        self.remove_bad_data()
        self.p_at_rnfrac = self.get_value_at_rn_frac(self.rn_fracs,self.p_cl,self.ro_cl)
        self.dark_dP_w = self._handle_dark(dark_dP_w) # a vector (not 2D array)
        self.cl_dT_k, self.dP_w, self.T_cl_index = self.get_delta_pt()
        if self.do_dark_analysis: self.dP_w_darksubtracted = self.power_subtraction(self.dP_w,self.dark_dP_w)

        # predicted power
        self.freq_edges_ghz=self.passband_sim_ghz=None
        self.power_cl_tophat=self.power_cl_sim_passband=self.power_cl_tophat_delta=self.power_cl_sim_passband_delta=self.eta_tophat=self.eta_passband_sim=None
        self._handle_prediction(passband_dict)

        # plotting stuff
        self.colors = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']

    def power_subtraction(self,dP_w,dP_w_vec):
        ''' dP_w is an N x M array with N rows of %rn cuts over M coldload temps
            dP_w_vec (typically a dark detector response) is a 1 x M array
        '''
        return dP_w - dP_w_vec

    def get_pt_delta_for_rnfrac(self,rnfrac):
        assert rnfrac in self.rn_fracs, ('requested rnfrac = ',rnfrac, 'not in self.rn_fracs = ',self.rn_fracs)
        dex = self.rn_fracs.index(rnfrac)
        return self.dP_w[dex]

    def _handle_dark(self, dark_dP_w):
        self.dP_w_darksubtracted = None
        self.do_dark_analysis = False
        try:
            if dark_dP_w == None:
                ddp_w = None
        except:
            assert len(dark_dP_w) == self.n_cl_temps, ('Length of dark_dP_w must equal number of cold load temperatures')
            ddp_w = dark_dP_w
            self.do_dark_analysis = True
        return ddp_w

    def _handle_device_dict(self,device_dict):
        if device_dict==None:
            det_name = 'unknown'; row_name = 'unknown'; f_edges = None
        else:
            assert type(device_dict)==dict, ('device_dict either None or must be of type dictionary')
            row_name = [*device_dict][0]; det_name = device_dict[row_name]
        return det_name, row_name

    def _handle_prediction(self,passband_dict):
        self.prediction, self.frequency_edges_ghz, self.passband_sim_ghz = self._handle_passband(passband_dict)
        if self.prediction[0] != 0:
            self.power_cl_tophat = self.get_predicted_thermal_power_tophat(self.cl_temps_k,f_edges_ghz=self.frequency_edges_ghz)
            self.power_cl_tophat_delta = np.array(self.power_cl_tophat) - self.power_cl_tophat[self.T_cl_index]
            self.eta_tophat = self.get_efficiency(self.power_cl_tophat_delta, self.dP_w)
            if self.do_dark_analysis:
                self.eta_tophat_ds = self.get_efficiency(self.power_cl_tophat_delta, self.dP_w_darksubtracted)
        if self.prediction[1] != 0:
            self.power_cl_sim_passband = self.get_predicted_thermal_power_simpassband(self.cl_temps_k,passband_ghz=self.passband_sim_ghz)
            self.self.power_cl_sim_passband_delta = np.array(self.power_cl_sim_passband) - self.power_cl_sim_passband[self.T_cl_index]
            self.eta_passband_sim = self.get_efficiency(self.power_cl_sim_passband_delta, self.dP_w)
            if self.do_dark_analysis:
                self.eta_passband_sim_ds = self.get_efficiency(self.power_cl_sim_passband_delta, self.dP_w_darksubtracted)

    def _handle_passband(self,passband_dict):
        prediction=[0,0]
        if passband_dict==None or passband_dict=={}:
            freq_edges_ghz = None ; passband_sim_ghz = None
        else:
            assert type(passband_dict) == dict, ('passband_dict must be of type dictionary')
            keys = passband_dict.keys()
            if 'freq_edges_ghz' in keys:
                freq_edges_ghz = passband_dict['freq_edges_ghz']
                if freq_edges_ghz != None: prediction[0]=1
            else:
                freq_edges_ghz = None
            if 'passband_sim_ghz' in keys:
                passband_sim_ghz = passband_dict['passband_sim_ghz']
                prediction[1]=1
            else:
                passband_sim_ghz = None
        return prediction, freq_edges_ghz, passband_sim_ghz

    def get_predicted_thermal_power_tophat(self,cl_temps_k,f_edges_ghz):
        ''' calculate thermal power at cl_temps assuming a tophat passband from f1 to f2 (ghz) '''
        p_tophat = []
        for cl_temp_k in cl_temps_k:
            p_tophat.append(thermalPower(f_edges_ghz[0]*1e9,f_edges_ghz[1]*1e9,T=cl_temp_k,F=None))
        return p_tophat

    def get_predicted_thermal_power_simpassband(self,cl_temps,passband_ghz):
        ''' calculate thermal power at cl_temps assuming the normalized passband supplied passband_ghz '''
        p = []
        for cl_temp_k in cl_temps_k:
            p.append(thermalPower(1,10e12,T=cl_temp_k,F=passband_sim_ghz))
        return p

    def removeNaN(self,arr):
        ''' only works on 1d vector, not array '''
        return arr[~np.isnan(arr)]

    def fb_align_and_remove_offset(self,showplot=False):
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
        #self.fb_align = fb_align
        if showplot:
            for ii in range(self.n_cl_temps):
                plt.plot(self.dacs,fb_align[:,ii])
            plt.show()
        return fb_align

    def get_vipr(self,showplot=False):
        ''' returns the voltage, current, power, and resistance vectors '''
        if self.fb_align==None:
            self.fb_align = self.fb_align_and_remove_offset(showplot=False)

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

    def remove_bad_data(self):
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
        self.v_cl = cut(self.v,dexs)
        self.i_cl = cut(self.i,dexs)
        self.r_cl = cut(self.r,dexs)
        self.ro_cl = cut(self.ro,dexs)
        self.p_cl = cut(self.p,dexs)

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

    def get_delta_pt(self,rn_fracs=None,p_at_rnfrac=None,cl_index=None):
        if cl_index == None: dex = np.argmin(self.cl_temps_k)
        else: dex = cl_index
        if p_at_rnfrac==None: p_at_rnfrac=self.p_at_rnfrac
        if rn_fracs==None: rn_fracs=self.rn_fracs

        dT_k = np.array(self.cl_temps_k)-self.cl_temps_k[dex]
        #p_at_rnfrac[ii,min_dex]-p_at_rnfrac[ii,:]
        dP_w = np.zeros(np.shape(p_at_rnfrac))
        for ii in range(len(rn_fracs)): # must be a better way...
            dP_w[ii,:] = p_at_rnfrac[ii,dex] - p_at_rnfrac[ii,:]
        return dT_k, dP_w, dex

    def get_efficiency(self,dP,dP_m):
        # dexs = [i for i, dP in enumerate(dP) if dP == 0]
        # eta = np.ones(len(dP))*np.nan
        # for ii in range(len(dP)):
        #     if ii in dexs:
        #         pass
        #     else: eta[ii]==dP_m[ii]/dP[ii]
        return dP_m/dP


    def plot_raw(self,fb_align_dc_level=True,fig_num=1):
        fig = plt.figure(fig_num)
        for ii, cl_temp in enumerate(self.cl_temps_k):
            if fb_align_dc_level:
                dy = self.fb[0,ii]-self.fb[0,0]
            else: dy=0
            plt.plot(self.dacs, self.fb[:,ii]-dy)
        plt.xlabel('DAC values')
        plt.ylabel('Feedback values')
        plt.title(self.figtitle)
        plt.legend((self.cl_temps_k),loc='upper right')
        return fig

    def plot_vipr(self,data_list=None,fig_num=1):

        if data_list==None:
            v=self.v; i=self.i; p=self.p; r=self.r
        else:
            v=data_list[0]; i=data_list[1]; p=data_list[2]; r=data_list[3]

        # fig 1, 2x2 of converted IV
        #fig = plt.figure(fig_num)
        figXX, ax = plt.subplots(nrows=2,ncols=2,sharex=False,figsize=(12,8))
        ax=[ax[0][0],ax[0][1],ax[1][0],ax[1][1]]
        for ii in range(self.n_cl_temps):
            ax[0].plot(v[:,ii],i[:,ii])
            ax[1].plot(v[:,ii],p[:,ii])
            ax[2].plot(p[:,ii],r[:,ii])
            #ax[3].plot(p[:,ii],r[:,ii]/r[-2,ii])
            ax[3].plot(v[:,ii],r[:,ii])
        # xlabels = ['V ($\mu$V)','V ($\mu$V)','P (pW)','V ($\mu$V)']
        # ylabels = ['I ($\mu$A)', 'P (pW)', 'R (m$\Omega$)', 'R (m$\Omega$)']
        xlabels = ['V (V)','V (V)','P (W)','V (V)']
        ylabels = ['I (A)', 'P (W)', 'R ($\Omega$)', 'R ($\Omega$)']

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

        figXX.suptitle(self.figtitle)
        ax[3].legend(tuple(self.cl_temps_k))
        return figXX

    def plot_pr(self,rn_fracs,p_at_rnfrac,p,ro,fig_num=1):
        pPlot = self.get_value_at_rn_frac([0.995],arr=p,ro=ro)

        # FIG1: P versus R/Rn
        fig = plt.figure(fig_num)
        plt.plot(ro, p,'-') # plots for all Tbath
        plt.plot(rn_fracs,p_at_rnfrac,'ro')
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

    def plot_pt(self,rn_fracs,p_at_rnfrac,p,ro,fig_num=1):
        # power plateau (evaluated at each rn_frac) versus T_cl
        fig = plt.figure(fig_num)
        llabels=[]
        for ii in range(len(rn_fracs)):
            if not np.isnan(p_at_rnfrac[ii,:]).any():
                plt.plot(self.cl_temps_k,p_at_rnfrac[ii,:],'o-')
                llabels.append(rn_fracs[ii])
        plt.xlabel('T$_{cl}$ (K)')
        plt.ylabel('TES power plateau')
        plt.legend((llabels))
        plt.title(self.figtitle)
        plt.grid()
        return fig

    def plot_pt_delta(self,cl_dT_k, dp_at_rnfrac, rn_fracs, fig_num=1, dp_at_rnfrac_dark_subtracted=None):
        ''' plot change in saturation power relative to minimum coldload temperature '''
        fig = plt.figure(fig_num)
        legend_vals = []
        if self.prediction[0]==1: # include tophat passband prediction
            plt.plot(self.cl_dT_k,self.power_cl_tophat_delta,'k-',label='$\Delta{P}_{calc}$ (top hat)')
        if self.prediction[1]==1: # include simulated passband prediction
            plt.plot(self.cl_dT_k,self.power_cl_sim_passband_delta,'k--',label='$\Delta{P}_{calc}$ (sim passband)')
        jj=0
        for ii in range(len(rn_fracs)):
            if not np.isnan(dp_at_rnfrac[ii,:]).any():
                plt.plot(cl_dT_k,dp_at_rnfrac[ii,:],'o-',color=self.colors[jj],label=str(rn_fracs[ii]))
                try:
                    if len(dp_at_rnfrac_dark_subtracted) > 0:
                        plt.plot(cl_dT_k,dp_at_rnfrac_dark_subtracted[ii,:],'o--',color=self.colors[jj],label='_nolegend_')
                except:
                    pass
                jj+=1
        plt.xlabel('T$_{cl}$ - %.1f K'%self.cl_temps_k[self.T_cl_index])
        plt.ylabel('P$_o$ - P')
        plt.legend()
        plt.grid()
        plt.title(self.figtitle)
        return fig

    def get_eta_mean_std(self,eta):
        n,m = np.shape(eta) # n = %rn cut index, m = Tcl index
        dexs=[] # rn cuts w/out np.nan entries
        for ii in range(n):
            if not np.isnan(eta[ii,1:]).any():
                dexs.append(ii)

        eta_m = np.mean(eta[dexs,1:],axis=0)
        eta_std = np.std(eta[dexs,1:],axis=0)
        return eta_m, eta_std

    def plot_efficiency(self,cl_dT_k, eta, rn_fracs, fig_num=1, eta_dark_subtracted=None):
        fig = plt.figure(fig_num)
        jj=0
        for ii in range(len(rn_fracs)):
            if not np.isnan(eta[ii,1:]).any():
                plt.plot(cl_dT_k,eta[ii,:],'o-',color=self.colors[jj], label=str(rn_fracs[ii]))
                try:
                    if len(eta_dark_subtracted) > 0:
                        plt.plot(cl_dT_k,eta_dark_subtracted[ii,:],'o--',color=self.colors[jj],label='_nolegend_')
                except:
                    pass
                jj+=1
        eta_m, eta_std = self.get_eta_mean_std(eta)
        eta_m_ds, eta_std_ds = self.get_eta_mean_std(eta_dark_subtracted)
        plt.errorbar(cl_dT_k[1:],eta_m,eta_std,color='k',linewidth=2,ecolor='k',elinewidth=2,label='mean')
        plt.errorbar(cl_dT_k[1:],eta_m_ds,eta_std_ds,color='k',linewidth=2,ecolor='k',elinewidth=2,label='mean ds',linestyle='--')
        plt.xlabel('T$_{cl}$ - %.1f K'%self.cl_temps_k[self.T_cl_index])
        plt.ylabel('Efficiency')
        plt.legend()
        plt.grid()
        plt.title(self.figtitle)
        return fig

    def plot_full_analysis(self,showfigs=False,savefigs=False):
        figs = []
        figs.append(self.plot_raw(True,fig_num=1)) # raw
        figs.append(self.plot_vipr(data_list=None,fig_num=2)) # 2x2 of converted data
        figs.append(self.plot_pr(self.rn_fracs,self.p_at_rnfrac,self.p,self.ro,fig_num=3))
        if not np.isnan(self.p_at_rnfrac).all():
            figs.append(self.plot_pt(self.rn_fracs,self.p_at_rnfrac,self.p,self.ro,fig_num=4))
            if self.do_dark_analysis:
                figs.append(self.plot_pt_delta(self.cl_dT_k, self.dP_w, self.rn_fracs,fig_num=5, dp_at_rnfrac_dark_subtracted=self.dP_w_darksubtracted))
            else:
                figs.append(self.plot_pt_delta(self.cl_dT_k, self.dP_w, self.rn_fracs,fig_num=5))
        if self.prediction[0]==1:
            if self.do_dark_analysis:
                figs.append(self.plot_efficiency(self.cl_dT_k, self.eta_tophat, self.rn_fracs, fig_num=6, eta_dark_subtracted=self.eta_tophat_ds))
            else:
                figs.append(self.plot_efficiency(self.cl_dT_k, self.eta_tophat, self.rn_fracs, fig_num=6))
        if self.prediction[1]==1:
            if self.do_dark_analysis:
                figs.append(self.plot_efficiency(self.cl_dT_k, self.eta_passband_sim, self.rn_fracs, fig_num=7, eta_dark_subtracted=self.eta_passband_sim_ds))
            else:
                figs.append(self.plot_efficiency(self.cl_dT_k, self.eta_passband_sim, self.rn_fracs, fig_num=7))
        if savefigs:
            fig_appendix=['raw','vipr','pr','pt','dpt','eta_top','eta_sim']
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

    def full_analysis(self,bath_temp_index,cl_indices,showfigs=False,savefigs=False,dark_rnfrac=0.7):
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
                                              passband_dict=passband_dict,
                                              dark_dP_w=dark_dP)
                iva.plot_full_analysis(showfigs,savefigs)

class DetectorMap():
    ''' Class to map readout channels to detector characteristics '''
    def __init__(self,filename=None):
        self.filename = filename
        self.map_dict = self.from_file(self.filename)
        #self.keys = self.map_dict.keys()

    def from_file(self,filename):
        return self._parse_csv_file(filename)

    def _parse_csv_file(self,filename):
        f=open(filename,'r')
        lines = f.readlines()
        header = lines.pop(0)
        header = header.split(',')[0:-1]
        #print(header)
        ncol = len(header)
        map_dict = {}
        for ii, line in enumerate(lines):
            line = line.split(',')[0:-1]
            map_dict['Row%02d'%(int(line[0]))] = {}
            for jj in range(1,ncol):
                map_dict['Row%02d'%(int(line[0]))][str(header[jj])] = str(line[jj])
        map_dict = self._clean_map_dict(map_dict)
        return map_dict

    def _clean_map_dict(self,map_dict):
        for key, val in map_dict.items():
            f_low = map_dict[key].pop('f_low',None)
            f_high = map_dict[key].pop('f_high',None)
            f_low = self._handle_input(f_low,float)
            f_high = self._handle_input(f_high,float)
            if f_low == None or f_high == None:
                map_dict[key]['freq_edges_ghz'] = None
            else:
                map_dict[key]['freq_edges_ghz'] = [f_low,f_high]
            map_dict[key]['position'] = self._handle_input(map_dict[key]['position'],int)
        return map_dict

    def _handle_input(self,val,thetype=int):
        if val=='None':
            val = None
        else: val = thetype(val)
        return val

    def get_onerow_device_dict(self,row_index):
        return {'Row%02d'%row_index:self.map_dict['Row%02d'%row_index]['devname']}

    def get_keys_and_indices_of_type(self,type_str='dark'):
        print('Warning.  Direct row to index mapping assumed (ie index of Row02 is 2)')
        idx=[]; keys=[]
        for key in self.map_dict.keys():
            if self.map_dict[key]['type']==type_str:
                keys.append(key)
                idx.append(int(key.split('Row')[1]))
        return keys, idx

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
    

    

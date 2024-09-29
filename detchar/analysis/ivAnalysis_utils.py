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
import matplotlib.colors as Colors

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
        if F == None: # case for tophat
            P = quad(Pnu_thermal,nu1,nu2,args=(T))[0] # toss the error
    except: # case for arbitrary passband shape F
        N = len(F)
        nu = np.linspace(nu1,nu2,N)
        integrand = self.Pnu_thermal(nu,T)*F
        P = simps(integrand,nu)
    return P

def get_xy_for_row_from_temp_sweep(iv_tempsweep_data,row):
    ''' dac_values is a 1xn vector
        fb_values_arr is an nxm array, where m is the number of temp sweeps
    '''
    dac_values = iv_tempsweep_data.data[0].dac_values
    n=len(dac_values)
    m=len(iv_tempsweep_data.data) # number of IV curves in the temperature sweep
    fb_values_arr = np.empty((n,m))
    for ii in range(m):
        fb_values_arr[:,ii]=iv_tempsweep_data.data[ii].fb_values_array()[:,row]
    return dac_values, fb_values_arr

class IVCommon():
    ''' Common IV analysis methods '''
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
            plt.xlabel('Index')
            plt.ylabel('$\Delta$ fb')
            if dex != None: plt.plot(dex,val,'ro')

            fig2 = plt.figure(2)
            plt.plot(fb,'o-')
            plt.xlabel('Index')
            plt.ylabel('fb')
            if dex != None: plt.plot(dex,fb[dex],'ro')
            plt.show()
        return dex#,val

    #def get_number_of_normal_branch_pts(self,dac,fb,frac_above_turn):
    #    ''' Determine the number of points to use in the normal branch for
    #        dc offset removal in a smart way
    #    '''

    # def find_band_data_index_new(self,dac,fb,threshold,showplot=False):
    #     assert True,'unfinished!!!'
    #     # take derivatives
    #     dfb = np.diff(y)
    #     ddfb = np.diff(dfb)
    
    #     # Define IV curve regimes: superconducting, in transition, normal
    #     sc_idx = np.argmax(abs(ddfb))+1 # find superconducting index
    #     turn_idx = np.argmin(abs(dfb[sc_idx:]))+sc_idx+1
    #     n_idx = int(N-(N-turn_idx)/2) # defined has half way from IV turn-around to highest Vbias point

    def find_bad_data_index(self,dac,fb,threshold=0.5,showplot=False):
        ''' Return the index where IV curve misbehaves.
            dac and fb(dac) must be in descending order

            Algorithm is to look at fb(dac) for dac values lower than the IV turnaround.
            If the second derivative is positive (ie the slope of the IV curve in transition changes sign),
            the index is flagged and returned.

            If no bad data found, return the last index (such that subsequent method includes all data points)
        '''
        success = True
        if dac[1]-dac[0] > 0:
            print('WARNING: the dac values are not in descending order.  find_bad_data_index will fail')
            success = False 

        turn_dex = self.get_turn_index(dac,fb,showplot=False)
        if turn_dex == None: return len(dac), success

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
        return dex, success

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

    def fit_normal_branch(self,dacs,fb_arr,align_dc=True,n_normal_pts=10):
        ''' return the slope and offset of the normal branch of a family of IV curves '''
        n,m=np.shape(fb_arr)
        fb_align = np.zeros((n,m))

        # align fb DC levels to a common value
        if align_dc:
            for ii in range(m):
                dy = fb_arr[0,ii]-fb_arr[0,0]
                fb_align[:,ii] = fb_arr[:,ii]-dy
        else:
            fb_align = np.copy(fb_arr)

        # remove offset
        x = dacs[::-1][-n_normal_pts:]
        y = fb_align[::-1,:] ; y = y[-n_normal_pts:,:]
        m, b = np.polyfit(x,y,deg=1)
        return m,b,fb_align

    def fb_align_and_remove_offset(self,dacs,fb_arr,n_normal_pts=10,use_ave_offset=False,showplot=False):
        ''' Remove DC offset from a set of IV curves and ensure IV is right side up, 
            i.e the slope in the normal branch is positive

            dacs: voltage bias in dac units (1xn array)
            fb_arr: nxm array of all feedback values in dac units
            n_normal_pts: number of normal points to use for fitting normal branch
            use_ave_offset: use the average of all IV slope in normal branch for subtraction.  This might make sense
                            when analyzing a set of IV curves from the same bolometer.
            showplot: if True, plot the figure

        '''

        m,b,fb_align = self.fit_normal_branch(dacs,fb_arr,align_dc=True,n_normal_pts=n_normal_pts)
        if use_ave_offset: 
            print('Using average offset from all IV curves for DC offset removal.')
            if np.std(b)/np.mean(b) > 0.01:
                print('Warning DC offset of curves differs by > 1\%')
                print('Offset fit: ',np.mean(b),'+/-',np.std(b))
            b = np.mean(b)

        fb_align = fb_align - b # remove the DC offset
        # ensure right side up
        dexs = np.where(m<0)[0]
        for dex in dexs:
            fb_align[:,dex]=-1*fb_align[:,dex]
        if showplot:
            for ii in range(m):
                plt.plot(dacs,fb_align[:,ii])
            plt.show()
        return fb_align

    def remove_bad_data(self,v,i,p,r,threshold=0.5):
        ''' 
            Remove bad data from IV curve set on entire column of data, 
            i.e. v,i,p,r is each an nxm array with n data points and m squid channels/detectors

            calls find_bad_data_index for each squid channel, removes bad data deep in transition.
            returns tuple: (cleaned) v,i,p,r and list of indices of bad data
        '''
        def cut(arr,dexs):
            n,m=np.shape(arr)
            arr_copy = arr.copy()
            for ii in range(m): # loop over columns (ie different detector fb responses)
                if dexs[ii]==n: pass
                else: arr_copy[dexs[ii]+1:,ii] = np.ones(n-dexs[ii]-1)*np.nan
            return arr_copy

        dexs=[]
        n,m=np.shape(i)
        for ii in range(m):
            dex, success = self.find_bad_data_index(v[:,ii],i[:,ii],threshold=threshold,showplot=False)
            #print(ii,dex,success)
            if not success:
                print('remove_band_data failed for index ',ii)
            dexs.append(dex)
        v_clean = cut(v,dexs)
        i_clean = cut(i,dexs)
        r_clean = cut(r,dexs)
        p_clean = cut(p,dexs)
        return v_clean, i_clean, p_clean, r_clean, dexs

    def get_vipr(self, dacs, fb_arr, iv_circuit=None, showplot=False):
        ''' returns the voltage, current, power, and resistance vectors for an
            n x m array of IV curves.  If an instance of iv_data.IVCircuit is
            provided, the returned vectors are in physical units.
        '''
        n,m=np.shape(fb_arr)
        if iv_circuit:
            v,i = iv_circuit.to_physical_units(np.array(dacs),fb_arr)
        else:
            v = np.zeros((n,m))
            for ii in range(m):
                v[:,ii] = dacs
            i=fb_arr
        p=v*i; r=v/i

        if showplot:
            self.plot_vipr([v,i,p,r])
        return v,i,p,r

    def plot_vipr_method(self,v,i,p,r,figtitle=None,figlegend=None,row_indices=None):

        if type(row_indices) == int:
            row_indices = [row_indices]
        if row_indices == None:
            n,m=np.shape(i)
            row_indices = list(range(m))

        # fig 1, 2x2 of converted IV
        fig, ax = plt.subplots(nrows=2,ncols=2,sharex=False,figsize=(12,8))
        ax=[ax[0][0],ax[0][1],ax[1][0],ax[1][1]]
        for ii in row_indices:
            ax[0].plot(v[:,ii],i[:,ii])
            ax[1].plot(v[:,ii],p[:,ii])
            ax[2].plot(r[:,ii],p[:,ii])
            ax[3].plot(v[:,ii],r[:,ii])

        # ax[0].set_xlabel(self.labels['iv']['x'])
        # ax[0].set_ylabel(self.labels['iv']['y'])
        # ax[1].set_xlabel(self.labels['vp']['x'])
        # ax[1].set_ylabel(self.labels['vp']['y'])
        # ax[2].set_xlabel(self.labels['rp']['x'])
        # ax[2].set_ylabel(self.labels['rp']['y'])
        # ax[3].set_xlabel(self.labels['vr']['x'])
        # ax[3].set_ylabel(self.labels['vr']['y'])

        # axes labels
        ax[0].set_xlabel('V')
        ax[0].set_ylabel('I')
        ax[1].set_xlabel('V')
        ax[1].set_ylabel('P')
        ax[2].set_xlabel('R')
        ax[2].set_ylabel('P')
        ax[3].set_xlabel('V')
        ax[3].set_ylabel('R')

        # plot range limits
        #ax[0].set_xlim((np.min(v)*1.1,np.max(v)*1.1))
        #ax[0].set_ylim((np.min(i)*1.1,np.max(i)*1.1))
        # ax[1].set_xlim((0,np.max(v)*1.1))
        # ax[1].set_ylim((0,np.max(p)*1.1))
        # ax[2].set_xlim((0,r[0,0]*1.1))
        # ax[2].set_ylim((0,np.max(p)*1.1))
        # ax[3].set_xlim((0,np.max(v)*1.1))
        ax[3].set_ylim((0,np.max(r[0,row_indices])*1.1))

        for ii in range(4):
            ax[ii].grid('on')

        if figtitle:
            fig.suptitle(figtitle)

        if figlegend == None:
            fig.legend(tuple(row_indices))
        else:
            fig.legend(figlegend)

        plt.tight_layout()
        return fig,ax

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
            x = self.remove_NaN(ro[:,ii])
            y = self.remove_NaN(arr[:,ii])
            YY = np.interp(rn_fracs,x[::-1],y[::-1])

            # over write with NaN for when data does not extend to fracRn
            ro_min = np.min(x)
            toCut = np.where(rn_fracs<ro_min)[0]
            N = len(toCut)
            if N >0:
                YY[0:N] = np.zeros(N)*np.NaN
            result[:,ii] = YY
        return result

    def _get_dac_at_rfrac_method_(self,rn_frac_list,dac,ro):
        ''' return the commanded voltage bias in dac units for an input rn_frac (not percentage),
            given the input dac values and N pts x M row normalized resistance array ro.

            The input vector and array dac and ro, are provided as the data is collected 
            during an IV curve: from high values of dac to low.

            Return is a list.  Elements of the list are np.array corresponding to the dac at the rn_frac_list provided.
            ie dac_list[i][j] corresponds to the ith detector and jth rn_frac value given
        '''
        n,m=np.shape(ro)
        dac_list = []
        for ii in range(m):
            dac_list.append(np.interp(rn_frac_list,ro[:,ii][::-1],dac[::-1]))
        return dac_list

    def _handle_figax(self,fig,ax):
        if not fig: fig,ax = plt.subplots()
        return fig,ax

class IVCurveAnalyzeSingle():
    ''' class for analyzing a single IV curve measured in the standard experimental config.
        The standard config assumes a voltage source in series with some large resistance (rbias_ohm)
        produces a current that runs to the shunt/tes/squid input circuit: a shunt resistance (rsh_ohm)
        with a value much less than the TES is wired in parallel with the TES and the input inductance 
        of a squid.  There may be a parasitic resistance (rx_ohm) in series with the TES.  
    '''
    def __init__(self,x,y,rsh_ohm,rx_ohm=0,to_i_bias=1,to_i_tes=1,analyze_on_init=True):
        ''' 
            self.x_raw: the raw x-data (voltage) provided to the class.  This is often in descending order.
            self.y_raw: the raw y-data (current) provided to the class.  This is often in descending order.
            self.x: the x-data in ascending order 
            self.y: the y-data in ascending order and ensured IV curve is "right-side up" 
            self.v_tes, i_tes, p_tes, r_tes: voltage across, current through, power dissipated, and resistance of tes
            self.si: responsivity derived from IV curve
            self.rn: TES normal resistance, defined as the mean of points in the r_tes vector with index greater than the normal resistance index.
                     The normal resistance index is defined as halfway between the IV turn index (where slope is zero) and the maximum V bias 
            self.rl: load resistance
    
        '''
        self.x_raw = np.array(x) # commanded voltage bias
        self.y_raw = np.array(y) # measured current in some arbitrary units
        self.rsh_ohm = rsh_ohm 
        self.rx_ohm = rx_ohm 
        self.to_i_tes = to_i_tes  
        self.to_i_bias = to_i_bias

        # get basic quantities of interest
        # here things are flipped into ascending order in voltage bias
        if analyze_on_init: self.analyze_iv()
        else: self.v_tes=self.i_tes=self.p_tes=self.r_tes=self.si=self.rn=self.rl=self.x=self.y=None

    ### Main analysis methods ----------------------------------------------------------------
    ### ---------------------------------------------------------------------------------

    def analyze_iv(self,plot=False,beta=0):
        ''' based on algorithm in pySmurf from Ari Cukierman '''
        x,y = self.determine_iv_regimes()
        y = self.remove_dc_offset(x,y)
    
        # calculate quantities of interest
        i_bias = x*self.to_i_bias # convert current bias to shunt network to physical units
        i_tes = y*self.to_i_tes
        r_tes = self.rsh_ohm*(i_bias/i_tes - 1)-self.rx_ohm
        v_tes = i_tes*r_tes # voltage over TES
        p_tes = i_tes*v_tes # electrical power on TES

        R_n = np.mean(r_tes[self.normal_idx:]) # normal resistance, is it better to just use the fit?
        R_L = np.mean(r_tes[1:self.sc_idx]) # load resistance, is it better to just use the fit?
    
        # smooth the data
        smooth_dist = 5
        w_len = 2*smooth_dist + 1
        w = (1./float(w_len))*np.ones(w_len) # window
        i_tes_smooth = np.convolve(i_tes, w, mode='same')
        v_tes_smooth = np.convolve(v_tes, w, mode='same')
        r_tes_smooth = v_tes_smooth/i_tes_smooth

        # Take derivatives
        di_tes = np.diff(i_tes_smooth)
        dv_tes = np.diff(v_tes_smooth)

        # Responsivity estimate
        R_L_smooth = np.ones(len(r_tes_smooth))*R_L
        R_L_smooth[:self.sc_idx] = dv_tes[:self.sc_idx]/di_tes[:self.sc_idx]
        r_tes_smooth_noStray = r_tes_smooth - R_L_smooth
        i0 = i_tes_smooth[:-1]
        r0 = r_tes_smooth_noStray[:-1]
        rL = R_L_smooth[:-1]
        si_etf = -1./(i0*r0)

        si = -(1./i0)*( dv_tes/di_tes - (r0+rL+beta*r0) ) / \
            ( (2.*r0-rL+beta*r0)*dv_tes/di_tes - 3.*rL*r0 - rL**2 )

        # pass these vectors to globals
        self.v_tes=v_tes; self.i_tes=i_tes; self.p_tes=p_tes; self.r_tes=r_tes; self.si=si; self.rn=R_n; self.rl=R_L; self.x=x; self.y=y; self.si_etf=si_etf

        if plot: 
            fig,ax = self.plot_raw()
            fig,ax = self.plot()

    def determine_iv_regimes(self,plot=False):
        ''' Place data in ascending order, make IV curve right-side-up, determine 3 IV regimes: superconducting, in-transition, normal. 
            based on algorithm in pySmurf from Ari Cukierman

            return: x,y,[sc_idx,turn_idx,normal_idx] 
            x: (voltage) in ascending order
            y: (current) in ascending order that is right-side up.  Offset is not subtracted
            sc_idx: index that marks the end of the superconducting regime
            turn_idx: index that marks the slope=0 point
            normal_idx: index that marks the normal branch.  Indices > normal_idx are in the normal branch
        '''
        N=len(self.x_raw)
    
        # place in ascending order
        if self.x_raw[1]-self.x_raw[0] < 0:
            x=np.copy(self.x_raw[::-1])
            y=np.copy(self.y_raw[::-1])
        else:
            x=np.copy(self.x_raw) 
            y=np.copy(self.y_raw) 
        
        # determine IV polarity. if negative, flip
        pval = np.polyfit(x[-10:],y[-10:],1)
        if pval[0] < 0: y=y*-1
        
        # take derivatives
        dfb = np.diff(y)
        ddfb = np.diff(dfb)
    
        # Define IV curve regimes: superconducting, in transition, normal
        sc_idx = np.argmax(abs(ddfb))+1 # superconducting index determined from maximum of 2nd derivative 
        turn_idx = np.argmin(abs(dfb[sc_idx:]))+sc_idx+1 # "turn index" where slope = 0
        n_idx = int(N-(N-turn_idx)/2) # defined has half way from IV turn-around to highest Vbias point

        self.sc_idx=sc_idx; self.turn_idx=turn_idx; self.normal_idx = n_idx

        if plot:
        # plot raw data
            colors = list(Colors.TABLEAU_COLORS)
            fig,ax=plt.subplots()
            ax.plot(x,y,'o',color=colors[0])
            #ax.plot(x[:sc_idx+1],y[:sc_idx+1],'ko')
            ax.plot(x[sc_idx],y[sc_idx],'r.')
            ax.plot(x[turn_idx],y[turn_idx],'r.')
            ax.plot(x[n_idx],y[n_idx],'r.')
            ax.axvspan(xmin=x[0],xmax=x[sc_idx],alpha=0.1,color='r')
            ax.axvspan(xmin=x[sc_idx],xmax=x[turn_idx],alpha=0.1)
            ax.axvspan(xmin=x[turn_idx],xmax=x[n_idx],alpha=0.1,color='y')
            ax.set_xlabel('V [dac]')
            ax.set_ylabel('I [dac]')
            return x,y,fig,ax

        else: return x,y

    def remove_dc_offset(self,x,y,plot=False):
        ''' remove DC offset of IV curve. x,y must be provided in ascending order and 
            right-side up.  This is done within self.determine_iv_regimes
        '''
        
        if self.sc_idx > self.normal_idx:
            print('WARNING: superconducting branch found at higher voltage bias than normal branch.  Setting sc branch to index 1.')
            self.sc_idx = 1 

        # fit normal regime, remove the offset
        p_norm = np.polyfit(x[self.normal_idx:],y[self.normal_idx:],1)
        if self.sc_idx == 0: 
            print('WARNING: no superconducting branch found.')
            y-=p_norm[1] # subtract arbitrary offset using normal branch
            p_sc = None
            
        else:
            # fit superconducting branches 
            p_sc = np.polyfit(x[:self.sc_idx+1],y[:self.sc_idx+1],1)
            y-=p_norm[1] # subtract arbitrary offset using normal branch
            offset_diff = abs(100*(p_norm[1]-p_sc[1])/p_norm[1])
            if offset_diff > 5: 
                print('superconducting and normal branch offsets differ by: %.2f%%.  Applying separate DC offset to superconducting branch.'%(offset_diff))
                y[0:self.sc_idx+1]-=p_sc[1]-p_norm[1] 

        self.p_norm=p_norm; self.p_sc = p_sc 

        if plot:
            fig,ax=plt.subplots()
            ax.plot(x,y,'o-')
            ax.plot(x[:self.sc_idx+1],y[:self.sc_idx+1],'ro')
            ax.plot(x[self.sc_idx],y[self.sc_idx],'go')
            ax.plot(x[normal_index:],y[normal_index:],'ro')
            ax.plot(x,np.polyval([p_norm[0],0],x),linestyle='--',color='k')
            if p_sc is not None: ax.plot(x[:self.sc_idx],np.polyval([p_sc[0],0],x[:self.sc_idx]),linestyle='--',color='k')
        return y

    ### helper methods ----------------------------------------------------------------
    ### ---------------------------------------------------------------------------------

    def get_virp_at_rfrac(self,rfrac):
        idx = np.argmin(abs(self.r_tes/self.rn - rfrac)) 
        return self.v_tes[idx], self.i_tes[idx], self.r_tes[idx], self.p_tes[idx] 

    def get_dac_at_rfrac(self,rfrac):
        idx = np.argmin(abs(self.r_tes/self.rn - rfrac)) 
        return self.x[idx]

    def get_r_for_dac(self,dac,frac=True):
        idx = np.argmin(abs(self.x-dac))
        if frac: return self.r_tes[idx]/self.rn
        else: return self.r_tes[idx]

    ### plotting methods ----------------------------------------------------------------
    ### ---------------------------------------------------------------------------------
    def plot_raw(self,fig=None,ax=None):
        colors = list(Colors.TABLEAU_COLORS)
        if not fig: fig,ax=plt.subplots()
        ax.plot(self.x,self.y,color=colors[0])
        ax.plot(self.x[self.sc_idx],self.y[self.sc_idx],'r.')
        ax.plot(self.x[self.turn_idx],self.y[self.turn_idx],'r.')
        ax.plot(self.x[self.n_idx],self.y[self.n_idx],'r.')
        if self.p_norm is not None: ax.plot(self.x,np.polyval([self.p_norm[0],0],self.x),linestyle='--',color=colors[1])
        if self.p_sc is not None: ax.plot(self.x[:self.sc_idx],np.polyval([self.p_sc[0],0],self.x[:self.sc_idx]),linestyle='--',color=colors[1])
        ax.axvspan(xmin=self.x[0],xmax=self.x[self.sc_idx],alpha=0.1,color='r')
        ax.axvspan(xmin=self.x[self.sc_idx],xmax=self.x[self.turn_idx],alpha=0.1)
        ax.axvspan(xmin=self.x[self.turn_idx],xmax=self.x[self.normal_idx],alpha=0.1,color='y')
        ax.set_xlabel('V [dac]')
        ax.set_ylabel('I [dac]')
        return fig,ax 

    def plot(self,fig=None,ax=None):
        colors = list(Colors.TABLEAU_COLORS)
        if not fig: fig,ax=plt.subplots(3,1)
        ax[0].plot(self.v_tes,self.i_tes,color=colors[0])
        ax[0].plot(self.v_tes[self.sc_idx],self.i_tes[self.sc_idx],'r.')
        ax[0].plot(self.v_tes[self.turn_idx],self.i_tes[self.turn_idx],'r.')
        ax[0].plot(self.v_tes[self.normal_idx],self.i_tes[self.normal_idx],'r.')
        if self.p_norm is not None: ax[0].plot(self.v_tes,np.polyval([self.p_norm[0],0],self.x)*self.to_i_tes,linestyle='--',color=colors[1])
        if self.p_sc is not None: ax[0].plot(self.v_tes[:self.sc_idx],np.polyval([self.p_sc[0],0],self.x[:self.sc_idx])*self.to_i_tes,linestyle='--',color=colors[1])
        ax[0].set_ylabel('I')
            
        ax[1].plot(self.v_tes,self.r_tes*1e3,color=colors[0])
        ax[1].axvspan(xmin=self.v_tes[self.sc_idx],xmax=self.v_tes[self.turn_idx],alpha=0.1)
        ax[1].set_ylabel('R (m$\Omega$)')
            
        ax[2].plot(self.v_tes[self.sc_idx:-1],self.si[self.sc_idx:],color=colors[0])
        ax[2].plot(self.v_tes[self.sc_idx:-1],self.si_etf[self.sc_idx:],linestyle='--',color=colors[1])
        ax[2].set_ylabel('$S_{I}$')
        ax[2].set_ylim((self.si[self.sc_idx+1]*1.1,1))
        ax[2].set_xlabel('V')

        for ii in range(3):
            ax[ii].axvspan(xmin=self.v_tes[0],xmax=self.v_tes[self.sc_idx],alpha=0.1,color='r')
            ax[ii].axvspan(xmin=self.v_tes[self.sc_idx],xmax=self.v_tes[self.turn_idx],alpha=0.1)
            ax[ii].axvspan(xmin=self.v_tes[self.turn_idx],xmax=self.v_tes[self.normal_idx],alpha=0.1,color='y')
        plt.tight_layout()
        return fig,ax

class IVCurveColumnDataExplore(IVCommon):
    ''' Explore IV data taken on a single column at a single bath temperature.  '''
    def __init__(self,iv_curve_column_data,iv_circuit=None):
        # fixed globals
        self.n_normal_pts = 10

        self.data = iv_curve_column_data
        self.iv_circuit = self._handle_iv_circuit(iv_circuit)

        # the raw data for one column and dimensionality
        self.x_raw, self.y_raw = self.data.xy_arrays_zero_subtracted_at_dac_high() # raw units
        # x_raw is in desending order (highest Vbias to lowest, as is typically done in TES IV curves)
        # y_raw is also the response to the bias in decending order.

        # data converted to physical units
        self.fb_align = self.fb_align_and_remove_offset(self.x_raw,self.y_raw,n_normal_pts=self.n_normal_pts,
                                                        use_ave_offset=False,showplot=False)
        self.v,self.i,self.p,self.r = self.get_vipr(self.data.dac_values, self.fb_align, iv_circuit=self.iv_circuit, showplot=False)
        # the high bias to low bias order is preserved.  i.e. v[ii,:] > v[ii+1,:]
        self.ro = self.r / self.r[0,:]
        self.is_data_clean=False
        foo, self.n_rows = np.shape(self.fb_align)
        self.labels = self._handle_labels(iv_circuit)

    # data manipulation methods --------------------------------------------------------
    def _handle_iv_circuit(self, iv_circuit):
        if iv_circuit != None:
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

    def _handle_labels(self,iv_circuit):
        labels = {}
        if iv_circuit is not None:
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

    def clean_data(self,threshold=1):
        self.v, self.i, self.p, self.r, dexs = self.remove_bad_data(self.v,self.i,self.p,self.r,threshold=threshold)
        self.ro = self.r / self.r[0,:]
        self.is_data_clean=True

    def get_responsivity(self):
        v = np.diff(self.v,axis=0)/2+self.v[:-1,:]
        di = np.diff(self.i, axis=0)
        dp = np.diff(self.i*self.v, axis=0)
        resp = di/dp
        return v, resp

    def get_dac_at_rfrac(self,rn_frac_list):
        return self._get_dac_at_rfrac_method_(rn_frac_list,self.x_raw,self.ro)

    # plotting methods --------------------------------------
    def __handle_row_input__(self,row):
        ''' magic method to allow plotting functions to accept a single row to plot as input 
            or a list.  NOTE THAT ROW HERE MEANS THE ROW INDEX!!!
        '''
        if type(row)==int:
            row=[row]
        elif row == 'all':
            row=list(range(self.n_rows))
        return row

    def plot_raw(self,row='all',include_legend=True,fig=None,ax=None):
        fig,ax = self._handle_figax(fig,ax)  
        row = self.__handle_row_input__(row)
        for ii in row:
            ax.plot(self.x_raw,self.y_raw[:,ii])
        plt.xlabel('Vb [dac]')
        plt.ylabel('Vfb [dac]')
        if include_legend: plt.legend(row)
        return fig,ax

    def plot_vipr_for_row(self,row=None,figtitle=None):
        ''' row can be list of rows or an integer.  If None, plot them all '''
        fig, ax = self.plot_vipr_method(self.v,self.i,self.p,self.r,figtitle=figtitle,figlegend=None,row_indices=row)
        return fig,ax

    def plot_iv(self,row='all',fig=None,ax=None):
        fig,ax = self._handle_figax(fig,ax) 
        row = self.__handle_row_input__(row)
        for ii in row:
            plt.plot(self.v[:,ii],self.i[:,ii],label='%02d'%ii)
        plt.xlabel(self.labels['iv']['x'])
        plt.ylabel(self.labels['iv']['y'])
        plt.legend(loc='upper right')
        return fig,ax

    def plot_responsivity(self,row='all',fig=None,ax=None):
        fig,ax = self._handle_figax(fig,ax) 
        row = self.__handle_row_input__(row)
        v,r = self.get_responsivity()
        for ii in row:
            plt.plot(v[:,ii],r[:,ii])
        plt.xlabel(self.labels['responsivity']['x'])
        plt.ylabel(self.labels['responsivity']['y'])
        plt.legend(tuple(range(self.n_rows)),loc='upper right')
        return fig,ax

    def plot_dy(self,row='all',fig=None,ax=None):
        fig,ax = self._handle_figax(fig,ax) 
        row = self.__handle_row_input__(row)
        dy = np.diff(self.i,axis=0)
        for ii in row:
            plt.plot(self.v[0:-1,ii],dy[:,ii],label='%02d'%ii)
        plt.xlabel(self.labels['dy']['x'])
        plt.ylabel(self.labels['dy']['y'])
        plt.legend(loc='upper right')
        return fig,ax

    def plot_prn_v_dac(self,row='all',fig=None,ax=None):
        fig,ax = self._handle_figax(fig,ax) 
        row = self.__handle_row_input__(row)
        for ii in row:
            ax.plot(self.x_raw,self.ro[:,ii])
        ax.set_ylim(0,1.1)
        ax.set_xlabel('V [dac]')
        ax.set_ylabel('Rn frac')
        ax.legend(row)
        return fig, ax

class IVSetAnalyzeRow(IVCommon):
    def __init__(self,dac_values,fb_values_arr,state_list=None,iv_circuit=None,figtitle=None,use_IVCurveAnalyzeSingle=True):
        ''' Analyze IV set at different physical conditions for one row.
            This class is useful for inspection of IV families or for determining the power
            difference between two states (like a 300K to 77K IV chop).
            Use get_xy_for_row_from_temp_sweep() to package the data from the IVTempSweepData
            format into the dac_values and fb_values_arr required here.
            The reason the IVTempSweepData object itself is not passed is for generality.

            Input:

            dac_values: np_array of dac_values (corresponding to voltage bias across TES),
                        a common dac_value for all IVs is required
            fb_values_arr: N_dac_val x N_sweep numpy array, column ordered
            state_list: list in which items are strings,
                        description of the state of the system for that IV curve
                        used in the legend of plots
            iv_circuit: instance of iv_data.IVCircuit used to convert data to physical units
            figtitle: title of figure
            use_IVCurveAnalyzeSingle: <bool> if True use IVCurveAnalyzeSingle to get v,i,p,r
        '''
        # options
        self.n_normal_pts = 10
        self.use_ave_offset = False
        #self.vipr_unit_labels = ['($\mu$V)','($\mu$A)','(pW)','(m$\Omega$)']
        #self.vipr_scaling = [1e6,1e6,1e12,1e3]

        #
        self.figtitle = figtitle
        self.dacs = dac_values
        self.fb_raw = fb_values_arr
        self.state_list = state_list
        self.n_dac_values, self.num_sweeps = np.shape(self.fb_raw)
        self.iv_circuit = iv_circuit

        # do standard IV analysis
        #self.fb_align = self.fb_align_and_remove_offset() # 2D array of aligned feedback
        if not use_IVCurveAnalyzeSingle:
            self.fb_align = self.fb_align_and_remove_offset(self.dacs,self.fb_raw,self.n_normal_pts,
                                                        use_ave_offset=self.use_ave_offset,showplot=False)
            self.v,self.i,self.p,self.r = self.get_vipr(self.dacs, self.fb_align, iv_circuit=self.iv_circuit, showplot=False)

        else:
            print('IV analysis through IVCurveAnalyzeSingle')
            assert iv_circuit, 'if use_IVCurveAnalyzeSingle=True, an IVCircuit object must be supplied'
            ivs = []
            for ii in range(self.num_sweeps):
                ivs.append(IVCurveAnalyzeSingle(x=self.dacs,y=self.fb_raw[:,ii],rsh_ohm=iv_circuit.rsh_ohm,rx_ohm=iv_circuit.rx_ohm,
                                                to_i_bias=iv_circuit.to_i_bias,to_i_tes=iv_circuit.to_i_tes))
            self.fb_align, self.v, self.i, self.p, self.r = self._package_iv_globals_(ivs)

    def _package_iv_globals_(self,ivs):
        result = []
        for iv in ivs:
            result_ii=[]
            for foo in [iv.y,iv.v_tes,iv.i_tes,iv.p_tes,iv.r_tes]:
                result_ii.append(foo)
            result.append(result_ii)
        result=np.array(result) # shape of result : num_sweeps x num_params x num_dacs
        fb_align = result[:,0,:].transpose()[::-1,:]
        v = result[:,1,:].transpose()[::-1,:]
        i = result[:,2,:].transpose()[::-1,:]
        p = result[:,3,:].transpose()[::-1,:]
        r = result[:,4,:].transpose()[::-1,:]
        return fb_align, v, i, p, r

    def power_difference_analysis(self,fig=None,ax=None):

        if not fig: fig,ax = plt.subplots(2)
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

        return fig,ax

    def normal_branch_subtraction(self,showplot=True):
        m,b,fb_align = self.fit_normal_branch(self.dacs,self.fb_raw,align_dc=True,n_normal_pts=self.n_normal_pts)
        arr=np.empty((self.n_dac_values,self.num_sweeps))
        for jj in range(self.num_sweeps):
            y = fb_align[:,jj]-(np.array(self.dacs)*m[jj]+b[jj])
            arr[:,jj]=y
            if showplot: plt.plot(self.dacs,y)
        if showplot:
            plt.xlabel('dac')
            plt.ylabel('normal slope subtracted fb')
            plt.legend(self.state_list)
            plt.title(self.figtitle)
            plt.show()

        return y

    def plot_raw(self):
        figXX = plt.figure()
        for ii in range(self.num_sweeps):
            plt.plot(self.dacs, self.fb_raw[:,ii])
        #plt.plot(self.dacs,self.fb_raw)
        plt.xlabel("dac values (arb)")
        plt.ylabel("fb values (arb)")
        plt.legend(tuple(self.state_list))
        if self.figtitle != None:
            plt.title(self.figtitle)

    def plot_vipr(self):
        fig,ax = self.plot_vipr_method(self.v,self.i,self.p,self.r, figtitle=self.figtitle,figlegend=self.state_list)
        return fig,ax

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
        iv_set.plot_raw(fignum=1)
        iv_set.plot_vipr(fignum=2)
        plt.show()

class IVversusADRTempOneRow(IVSetAnalyzeRow):
    ''' analyze thermal transport from IV curve set from one row in which ADR temperature is varied '''
    def __init__(self,dac_values,fb_values_arr, temp_list_k, normal_resistance_fractions=[0.8,0.9],iv_circuit=None,
                 figtitle=None,use_IVCurveAnalyzeSingle=True):
        ''' dac_values: np_array of dac_values (corresponding to voltage bias across TES),
                        a common dac_value for all IVs is required
            fb_values_arr: N_dac_val x N_sweep numpy array, column ordered in which columns are for different adr temperatures
            temp_list_k: adr temperature list in K, must match column order
            normal_resistance_fractions: determine power at these cuts in Rn fraction space
            iv_circuit: instance of iv_data.IVCircuit used to convert the data to physical units
        '''

        self.temp_list_k = temp_list_k
        self.rn_fracs = normal_resistance_fractions
        self.num_rn_fracs = len(self.rn_fracs)
        temp_list_k_str = []
        for ii in range(len(temp_list_k)):
            temp_list_k_str.append(str(temp_list_k[ii]))
        super().__init__(dac_values,fb_values_arr,temp_list_k_str,iv_circuit,figtitle,use_IVCurveAnalyzeSingle)
        self.ro = self.r / self.r[0,:]
        self.v_clean, self.i_clean, self.p_clean, self.r_clean, dexs = self.remove_bad_data(self.v,self.i,self.p,self.r,threshold=1)
        self.ro_clean = self.r_clean / self.r_clean[0,:]
        self.p_at_rnfrac = self.get_value_at_rn_frac(self.rn_fracs,self.p_clean,self.ro_clean)
        #print(self.p_at_rnfrac)
        self.pfits = self.fit_pvt_for_all_rn_frac()

    def plot_pr(self):
        pPlot = self.get_value_at_rn_frac([0.995],arr=self.p,ro=self.ro)

        # FIG1: P versus R/Rn
        fig = plt.figure()
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

    def plot_pt(self,include_fits=True):
        # power plateau (evaluated at each rn_frac) versus T_cl
        fig = plt.figure()
        llabels=[]
        temp_arr = np.linspace(np.min(self.temp_list_k),np.max(self.temp_list_k),100)
        for ii in range(self.num_rn_fracs):
            if not np.isnan(self.p_at_rnfrac[ii,:]).any():
                plt.plot(self.temp_list_k,self.p_at_rnfrac[ii,:],'o')
                llabels.append('%.3f'%(self.rn_fracs[ii]))

        if include_fits:
            for ii in range(self.num_rn_fracs):
                plt.plot(temp_arr,self.ktn_fit_func(self.pfits[ii],temp_arr),'k--')
        plt.xlabel('T$_{b}$ (K)')
        plt.ylabel('TES power plateau')
        plt.legend((llabels))
        #plt.title(self.figtitle)
        plt.grid()
        return fig

    def fit_pvt_for_all_rn_frac(self):
        ''' Fit power versus Tbath curves to P=K(T^n-Tb^n) for each rn_fraction cut.
            Returns pfits, a num_rn_fracs x 3 array.
            Rows are for each Rn cut; columns are for K,T,n in that order
        '''
        pfits=np.empty((self.num_rn_fracs,3))
        for ii in range(self.num_rn_fracs):
            pfit,pcov = self.fit_pvt(np.array(self.temp_list_k),self.p_at_rnfrac[ii])
            pfits[ii,:]=pfit
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
        if pcov is not None:
            pcov=pcov*s_sq
        return pfit,pcov

    def plot_fits(self):
        K=self.pfits[:,0]
        T=self.pfits[:,1]
        n=self.pfits[:,2]
        G=n*K*T**(n-1)

        vec = [K,T,n,G]
        yaxis_label=['K','T','n','G']
        # fig 1, 2x2 of converted IV
        fig, ax = plt.subplots(nrows=2,ncols=2,sharex=False,figsize=(12,8))
        ax=[ax[0][0],ax[0][1],ax[1][0],ax[1][1]]

        for ii, v in enumerate(vec):
            ax[ii].plot(self.rn_fracs,v,'ko-')
            ax[ii].set_xlabel('Rn_frac')
            ax[ii].set_ylabel(yaxis_label[ii])
            ax[ii].grid('on')

        if self.figtitle:
            fig.suptitle(self.figtitle)

        return fig

class IVColdloadAnalyzeOneRow(IVCommon):
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
        rn_fracs: <list of rn fractions to use for electrical power cuts.  If None: default list used.

        The data products (power at fraction of rn) are NxM, where N are the number or rn frac and M
        is the number of cold load temperatures.  Thus self.p_at_rnfrac[0] gives the power plateau at
        each temperature for the 0th cut in rn fraction.  The transpose of this (MxN) may be better
        since plot(x,y) can plot all vectors in y.  Thus matplotlib plot function is "column" oriented.
    '''

    def __init__(self,dac_values,fb_array,cl_temps_k,bath_temp_k,
                 row_name=None,det_name=None,
                 iv_circuit=None,predicted_power_w=None,dark_power_w=None,rn_fracs=None):
        # fixed globals / options
        self.n_normal_pts=10 # number of points for normal branch fit
        self.use_ave_offset=False # use a global offset to align fb, not individual per curve
        self.rn_fracs = self._handle_rn_fracs(rn_fracs) # slices in Rn space to compute electrical power versus temperature
        self.rn_fracs_legend = self._make_rn_fracs_legend_()
        self.n_rn_fracs = len(self.rn_fracs)
        self.bad_data_threshold = 1

        # main raw data inputs
        self.dacs = dac_values
        self.fb = fb_array # NxM array of feedback values.  Columns are per coldload temperature
        self.cl_temps_k = cl_temps_k
        self.cl_temps_k_legend = self._make_cl_temps_k_legend_()
        self.bath_temp_k = bath_temp_k

        # other useful globals
        self.variable_defs = self.variable_definitions()
        self.row_name, self.det_name = self._handle_row_det_name(row_name,det_name)
        self.figtitle = self.det_name+', '+self.row_name+' , Tb = %.1f mK'%(self.bath_temp_k*1000)
        self.n_dac_values, self.n_cl_temps = np.shape(self.fb)

        # do analysis of v,i,r,p vectors.  Place main results as globals to class
        self.fb_align = self.fb_align_and_remove_offset(self.dacs,self.fb,self.n_normal_pts,use_ave_offset=self.use_ave_offset,showplot=False) # remove DC offset
        v,i,p,r = self.get_vipr(self.dacs, self.fb_align, iv_circuit, showplot=False)
        ro = r / r[0,:]
        self.v_orig, self.i_orig, self.p_orig, self.r_orig, self.ro_orig = v,i,p,r,ro
        self.v, self.i, self.p, self.r, self.bad_data_idx = self.remove_bad_data(v,i,p,r,threshold=self.bad_data_threshold)
        self.ro = self.r/self.r[0,:]
        self.p_at_rnfrac = self.get_value_at_rn_frac(self.rn_fracs,self.p,self.ro) # n_rn_fracs x n_cl_temps

        # get change in power versus change in temperature ("infinitesimal", thus lower case d)
        self.cl_dT_k = np.diff(np.array(self.cl_temps_k))
        self.dp_at_rnfrac = np.diff(self.p_at_rnfrac)*-1

        # get change in power relative to P(T=T_cl_index), (not infinitesimal thus big D)
        self.cl_DT_k, self.Dp_at_rnfrac, self.T_cl_index = self.get_Delta_pt()

        # handle darks
        self.dark_analysis, self.dark_power_w, self.dark_Dp_w, self.dark_dp_w = self._handle_power_input(dark_power_w)
        if self.dark_analysis:
            self.dark_Dp_w = -1*self.dark_Dp_w; self.dark_dp_w=-1*self.dark_dp_w # a stupid thing so that _handle_power_input works for both darks and prediction

        # predicted power
        self.analyze_eta, self.predicted_power_w, self.predicted_Dp_w, self.predicted_dp_w = self._handle_power_input(predicted_power_w)

        # get efficiency
        if self.analyze_eta:
            self.get_efficiency_at_rnfrac()
            self.eta_for_rfrac = self.get_efficiency_from_fit(self.Dp_at_rnfrac)
            if self.dark_analysis:
                Dp_at_rnfrac_darksubtracted = self.Dp_at_rnfrac - self.dark_Dp_w
                self.eta_for_rfrac_darksubtracted = self.get_efficiency_from_fit(Dp_at_rnfrac_darksubtracted)

        # plotting stuff
        self.colors = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']

    # handle methods ----------------------------------------------------------
    def _handle_row_det_name(self,row_name,det_name):
        if row_name == None:
            row_name = 'Row XX'
        if det_name == None:
            det_name = 'Det XX'
        return row_name, det_name

    def _handle_rn_fracs(self,rn_fracs):
        if rn_fracs == None:
            return [0.3,0.4,0.5,0.6,0.7,0.8]
        else:
            return rn_fracs

    def _handle_power_input(self,p_in):
        if p_in is not None:
            assert len(p_in) == self.n_cl_temps, 'Hey, the dimensions of the input power do not match the number of cold load temperatures!'
            Dp, dp = self.get_power_difference_1D(p_in, self.T_cl_index)
            analyze = True
        else:
            Dp = dp = None
            analyze = False
        return analyze, p_in, Dp, dp

    def _make_cl_temps_k_legend_(self):
        XX = []
        for temp in self.cl_temps_k:
            XX.append(str('%.1fK'%temp))
        return XX

    def _make_rn_fracs_legend_(self):
        XX = []
        for r in self.rn_fracs:
            XX.append(str('%.02f'%r))
        return XX

    # helper methods -----------------------------------------------------------
    def variable_definitions(self):
        defs = {}
        defs['dacs'] = 'Common DAC values (voltage applied to detector bias line in computer units) for all measurements.'
        defs['n_dac_values'] = 'number of dac values'
        defs['rn_fracs'] = 'Fraction of normal resistance cuts used for determining electrical power on bolometer.'
        defs['n_rn_fracs'] = 'number of normal resistance fraction cuts'
        defs['fb'] = 'raw feedback array.  Array of dimensions n_dac_values x num_rows'
        defs['fb_align'] = 'Feedback array with DC offset removed'
        defs['cl_temps_k'] = 'Cold load temperatures in Kelvin'
        defs['n_cl_temps'] = 'Number of coldload temperatures'
        defs['bath_temp_k'] = 'The common ADR temperature for all measurements.'
        defs['det_name'] = 'The detector name/id'
        defs['row_name'] = 'The multiplexer row name'
        defs['i'] = 'current array n_dac_values x num_rows.  This array may be truncted to cut out bad data points.'
        defs['v'] = 'voltage array n_dac_values x num_rows. This array may be truncted to cut out bad data points.'
        defs['p'] = 'power array n_dac_values x num_rows. This array may be truncted to cut out bad data points.'
        defs['r'] = 'resistance array n_dac_values x num_rows. This array may be truncted to cut out bad data points.'
        defs['ro'] = 'normalized resistance array n_dac_values x num_rows. This array may be truncted to cut out bad data points.'
        defs['i_orig'] = 'untruncated, original current array n_dac_values x num_rows.'
        defs['v_orig'] = 'untruncated, originalvoltage array n_dac_values x num_rows.'
        defs['p_orig'] = 'untruncated, originalpower array n_dac_values x num_rows.'
        defs['r_orig'] = 'untruncated, originalresistance array n_dac_values x num_rows.'
        defs['ro_orig'] = 'untruncated, original normalized resistance array n_dac_values x num_rows.'
        defs['p_at_rnfrac'] = 'n_rn_fracs x n_cl_temps 2D array of electrical power evaluated at each rn_fracs.  Formatted as row-like in rn_frac cuts, that is the first index selects which cut in rn_frac'
        defs['cl_dT_k'] = 'The difference of the cl_temps_k vector.  The temperature difference between successive coldload temperatures'
        defs['dp_at_rnfrac'] = 'n_rn_fracs x n_cl_temps-1 2D array of the change in electrical power between successive coldload temperature at each rn_fracs.  '\
                               'Used in the `differential` method of determining the optical efficiency.'\
                               'Formatted as row-like in rn_frac cuts, that is the first index selects which cut in rn_frac'
        defs['cl_DT_k'] = 'The difference of cl_temps_k relative to a fixed coldload temperature defined by cl_temps_k[T_cl_index]'
        defs['Dp_at_rnfrac'] = 'n_rn_fracs x n_cl_temps 2D array of difference of electrical power relative to the power at a fixed coldload '\
                               'temperature defined by cl_temps_k[T_cl_index], evaluated at each rn_fracs.  Formatted as row-like in rn_frac cuts, that is the first index selects which cut in rn_frac'
        defs['T_cl_index'] = 'Index of coldload temperature used for the `fixed reference method` of determining optical efficiency'
        defs['dark_analysis'] = 'A boolean to determine of dark subtraction should be applied'
        defs['dark_power_w'] = 'Dark electrical power to be subtracted from the measured electrical power if dark_analysis = True'
        defs['dark_Dp_w'] = 'Difference of dark electrical power relative to the power measured at fixed coldload temperature cl_temps_k[T_cl_index]'
        defs['dark_dp_w'] = 'change in dark electrical power between successive coldload temperatures.'
        defs['analyze_eta'] = 'Boolean to determine if the efficiency analysis should be run'
        defs['predicted_power_w'] = '1 x n_cl_temps array of the prediction for optical power emitted toward the detector at each coldload temperature'
        defs['predicted_Dp_w'] = '1 x n_cl_temps array of the prediction for difference of the optical power emitted '\
                                 'toward the detector at each coldload temperature relative to the power predicted at T_cl = cl_temps_k[T_cl_index]. '\
                                 'Used in the `fixed reference` method.'
        defs['predicted_dp_w'] = '1 x n_cl_temps-1 array of the prediction for change in optical power emitted '\
                                 'toward the detector at successive coldload temperatures. '\
                                 'Used in the `differential` method.'
        defs['eta_Dp_arr'] = '2D optical efficiency array of dimensions n_rn_fracs x cl_temps_k, using the `fixed reference` method.'
        defs['eta_dp_arr'] = '2D optical efficiency array of dimensions n_rn_fracs x cl_temps_k-1, using the `differential` method.'
        defs['eta_Dp_arr_darksubtracted'] = '2D dark subtracted optical efficiency array of dimensions n_rn_fracs x cl_temps_k, using the `fixed reference` method.'
        defs['eta_dp_arr_darksubtracted'] = '2D dark subtracted optical efficiency array of dimensions n_rn_fracs x cl_temps_k-1, using the `differential` method.'
        defs['eta_mean'] = 'mean optical efficiency for each rn_frac (1 x n_rn_fracs)'
        defs['eta_std'] = 'standard deviation of efficiency for each rn_frac (1 x n_rn_fracs)'
        defs['eta_mean_darksubtracted'] = 'dark power subtracted mean optical efficiency for each rn_frac (1 x n_rn_fracs)'
        defs['eta_std_darksubtracted'] = 'dark subtracted standard deviation of efficiency for each rn_frac (1 x n_rn_fracs)'
        defs['DT_eta'] = 'Difference in coldload temperature from reference.  The x axis for eta.'
        return defs

    def update_T_cl_index(self,T_cl_index):
        self.cl_DT_k, self.Dp_at_rnfrac, self.T_cl_index = self.get_Delta_pt(cl_index = T_cl_index)
        if self.dark_analysis:
            self.dark_Dp_w = self.dark_power_w - self.dark_power_w[self.T_cl_index]
        if self.analyze_eta:
            self.predicted_Dp_w = self.predicted_power_w - self.predicted_power_w[T_cl_index]
            self.eta_Dp = self.get_efficiency_at_rnfrac()

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

    def get_Delta_pt(self,rn_fracs=None,p_at_rnfrac=None,cl_index=None):
        if cl_index == None: dex = np.argmin(self.cl_temps_k)
        else: dex = cl_index
        if p_at_rnfrac==None: p_at_rnfrac=self.p_at_rnfrac
        if rn_fracs==None: rn_fracs=self.rn_fracs

        DT_k = np.array(self.cl_temps_k)-self.cl_temps_k[dex]
        DP_w = (p_at_rnfrac[:,dex] - p_at_rnfrac.transpose()).transpose()
        return DT_k, DP_w, dex

    def get_efficiency_at_rnfrac(self,method='fixed reference'):
        '''
        create six global variables that quantify the optical efficiency, and one for the dT vector.
        Global variables are:
        DT_eta : BB temp - To.
        eta_Dp_arr(_darksubtracted): optical efficiency from power relative to a fixed T_cl temp (fixed method).
                                     Array has dimensions n_rfrac  x n_clTemps
        eta_dp_arr(_darksubtracted): optical efficiency from change in power from neighboring T_cl data points (differential method).
                                     Array has dimensions n_rfrac  x n_clTemps - 1
        eta_mean(_darksubtracted): 1x n_clTemps vector.  mean of all temperature points
        eta_std(_darksubtracted): 1x n_clTemps vector.  Standard deviation of all temperature points

        Input "method" is either 'fixed reference' or 'differential' to determine eta_mean and eta_std
        '''
        DT = self.cl_DT_k-self.cl_DT_k[self.T_cl_index]
        self.DT_eta = DT[np.where(DT!=0)[0]] # vector for difference in cold load temp where \eta is evaluated
        self.eta_Dp_arr = np.delete(self.Dp_at_rnfrac,self.T_cl_index,1) / self.predicted_Dp_w[np.where(self.predicted_Dp_w!=0)[0]]
        self.eta_dp_arr = self.dp_at_rnfrac / self.predicted_dp_w
        if method == 'fixed reference':
            self.eta_mean = self.eta_Dp_arr.mean(1)
            self.eta_std = self.eta_Dp_arr.std(1)
        elif method == 'differential':
            self.eta_mean = self.eta_dp_arr.mean(1)
            self.eta_std = self.eta_dp_arr.std(1)
        else:
            self.eta_mean = self.eta_std = None
            print('Unknown efficiency method: ',method)

        if self.dark_analysis:
            self.eta_Dp_arr_darksubtracted = (np.delete(self.Dp_at_rnfrac,self.T_cl_index,1) - np.delete(self.dark_Dp_w,self.T_cl_index)) / self.predicted_Dp_w[np.where(self.predicted_Dp_w!=0)[0]]
            self.eta_dp_arr_darksubtracted = (self.dp_at_rnfrac - self.dark_dp_w) / self.predicted_dp_w
            if method == 'fixed reference':
                self.eta_mean_darksubtracted = self.eta_Dp_arr_darksubtracted.mean(1)
                self.eta_std_darksubtracted = self.eta_Dp_arr_darksubtracted.std(1)
            elif method == 'differential':
                self.eta_mean_darksubtracted = self.eta_dp_arr_darksubtracted.mean(1)
                self.eta_std_darksubtracted = self.eta_dp_arr_darksubtracted.std(1)
            else:
                self.eta_mean_darksubtracted = self.eta_std_darksubtracted = None
                print('Unknown efficiency method: ',method)
        else:
            self.eta_Dp_arr_darksubtracted = self.eta_dp_arr_darksubtracted = None
            self.eta_mean_darksubtracted = self.eta_std_darksubtracted = None

    def get_efficiency_from_fit(self,Dp_at_rnfrac,showplot=False):
        ''' determine optical efficiency by minimizing dp_predicted * eta - dp_measured.
            Return np.array of length number of rfracs
        '''

        # fitfunc = lambda v,dp_predict: v[0]*dp_predict+v[1]
        # errfunc = lambda v,dp_predict,dp_meas: v[0]*dp_predict+v[1]-dp_meas

        fitfunc = lambda v,dp_predict: v*dp_predict
        errfunc = lambda v,dp_predict,dp_meas: v*dp_predict-dp_meas

        if showplot:
            fig,ax=plt.subplots(2,1)
            ax[0].set_ylabel('$\Delta{P}$ [W]')
            ax[1].set_ylabel('residuals')
            ax[1].set_xlabel('T$_{cl}$ - %.1f K'%self.cl_temps_k[self.T_cl_index])

        etas = []
        for ii in range(self.n_rn_fracs):
            #lsq = leastsq(errfunc,[1,.1], args=(self.predicted_Dp_w,self.Dp_at_rnfrac[ii,:]),full_output=1)
            lsq = leastsq(errfunc,[1], args=(self.predicted_Dp_w,Dp_at_rnfrac[ii,:]),full_output=1)
            pfit, pcov, infodict, errmsg, success = lsq
            etas.append(pfit[0])

            if showplot:
                ax[0].plot(self.cl_DT_k,Dp_at_rnfrac[ii,:],'o')
                ax[0].plot(self.cl_DT_k,self.predicted_Dp_w,'k-')
                ax[0].plot(self.cl_DT_k,fitfunc(pfit,self.predicted_Dp_w),'k--')
                ax[1].plot(self.cl_DT_k,errfunc(pfit,self.predicted_Dp_w,Dp_at_rnfrac[ii,:]),'o--')
        return np.array(etas)

    def get_power_vector_for_rnfrac(self,rnfrac):
        assert rnfrac in self.rn_fracs, ('requested rnfrac = ',rnfrac, 'not in self.rn_fracs = ',self.rn_fracs)
        dex = self.rn_fracs.index(rnfrac)
        return self.p_at_rnfrac[dex,:]

    # plotting methods ---------------------------------------------------------
    def plot_raw(self):
        fig, ax = plt.subplots(nrows=1,ncols=2,sharex=False,figsize=(12,8))
        for ii, cl_temp in enumerate(self.cl_temps_k):
            ax[0].plot(self.dacs,self.fb[:,ii])
            ax[1].plot(self.dacs,self.fb_align[:,ii])
        ax[0].set_xlabel('DAC values')
        ax[0].set_ylabel('Feedback values')
        ax[1].set_xlabel('DAC values')
        ax[1].set_ylabel('Feedback values (offset removed)')

        fig.suptitle(self.figtitle+'  raw IV')
        ax[0].legend((self.cl_temps_k_legend),loc='upper right')
        return fig

    def plot_vipr(self):
        figXX, ax = plt.subplots(nrows=2,ncols=2,sharex=False,figsize=(10.5,8))
        ax=[ax[0][0],ax[0][1],ax[1][0],ax[1][1]]
        for ii in range(self.n_cl_temps):
            ax[0].plot(self.v_orig[:,ii],self.i_orig[:,ii],'-',color=self.colors[ii])
            ax[1].plot(self.v_orig[:,ii],self.p_orig[:,ii],'-',color=self.colors[ii])
            ax[2].plot(self.p_orig[:,ii],self.r_orig[:,ii],'-',color=self.colors[ii])
            ax[3].plot(self.v_orig[:,ii],self.r_orig[:,ii],'-',color=self.colors[ii])
            ax[0].plot(self.v[:,ii],self.i[:,ii],'.',color=self.colors[ii],label = '_nolegend_')
            ax[1].plot(self.v[:,ii],self.p[:,ii],'.',color=self.colors[ii],label = '_nolegend_')
            ax[2].plot(self.p[:,ii],self.r[:,ii],'.',color=self.colors[ii],label = '_nolegend_')
            ax[3].plot(self.v[:,ii],self.r[:,ii],'.',color=self.colors[ii],label = '_nolegend_')
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
        for ii in [2,3]:
            ax[ii].set_ylim(0,1.2*self.r[0,0])

        figXX.suptitle(self.figtitle+'  IV, PV, RP, RV')
        ax[0].legend(tuple(self.cl_temps_k_legend),loc='upper right')
        return figXX

    def plot_pr(self):
        pPlot = self.get_value_at_rn_frac([0.995],arr=self.p,ro=self.ro)

        # FIG1: P versus R/Rn
        fig = plt.figure()
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
        plt.legend((self.cl_temps_k_legend))
        plt.grid()
        plt.title(self.figtitle+' P vs R')
        return fig

    def plot_pt(self):
        # power plateau (evaluated at each rn_frac) versus T_cl
        fig = plt.figure()
        llabels=[]
        for ii in range(len(self.rn_fracs)):
            if not np.isnan(self.p_at_rnfrac[ii,:]).any():
                plt.plot(self.cl_temps_k,self.p_at_rnfrac[ii,:],'o-')
                llabels.append('%.2f'%(self.rn_fracs[ii]))
        plt.xlabel('T$_{cl}$ (K)')
        plt.ylabel('TES power plateau')
        plt.legend((llabels))
        plt.title(self.figtitle+' P vs T')
        plt.grid()
        return fig

    def plot_DpDt(self, include_prediction=True, include_darksubtraction=True):
        ''' plot change in saturation power relative to one fixed coldload temperature '''
        fig = plt.figure()
        legend_vals = []
        if include_prediction and self.analyze_eta:
            plt.plot(self.cl_DT_k,self.predicted_Dp_w,'k-',label='$\Delta{P}_{pred}$')
        jj=0
        for ii in range(len(self.rn_fracs)):
            if not np.isnan(self.Dp_at_rnfrac[ii,:]).any():
                plt.plot(self.cl_DT_k,self.Dp_at_rnfrac[ii,:],'o-',color=self.colors[jj],label='%.2f'%(self.rn_fracs[ii]))
                if include_darksubtraction and self.dark_analysis:
                    plt.plot(self.cl_DT_k,self.Dp_at_rnfrac[ii,:]-self.dark_Dp_w,'o--',color=self.colors[jj],label='_nolegend_')
                jj+=1
        plt.xlabel('T$_{cl}$ - %.1f K'%self.cl_temps_k[self.T_cl_index])
        plt.ylabel('P$_o$ - P')
        plt.legend()
        plt.grid()
        plt.title(self.figtitle + ' Dp vs T')
        return fig

    def plot_dpdt(self, include_prediction=True, include_darksubtraction=True):
        ''' plot change in saturation power relative to change in coldload temperature '''
        fig = plt.figure()
        legend_vals = []
        x = self.cl_dT_k/2 + self.cl_temps_k[0:-1] # midpoint between sampled coldload temperatures
        if include_prediction and self.analyze_eta:
            plt.plot(x,self.predicted_dp_w,'k-',label='$dP_{pred}$')
        jj=0
        for ii in range(len(self.rn_fracs)):
            if not np.isnan(self.dp_at_rnfrac[ii,:]).any():
                plt.plot(x,self.dp_at_rnfrac[ii,:],'o-',color=self.colors[jj],label='%.2f'%(self.rn_fracs[ii]))
                if include_darksubtraction and self.dark_analysis:
                    plt.plot(x,self.dp_at_rnfrac[ii,:]-self.dark_dp_w,'o--',color=self.colors[jj],label='_nolegend_')
                jj+=1
        plt.xlabel('T$_{cl}$')
        plt.ylabel('dP')
        plt.legend()
        plt.grid()
        plt.title(self.figtitle + ' dp vs T')
        return fig

    def plot_power_change_vs_temperature(self,include_prediction=True, include_darksubtraction=True):
        fig, ax = plt.subplots(1,2,figsize=(10,5))
        x = self.cl_dT_k/2 + self.cl_temps_k[0:-1] # midpoint between sampled coldload temperatures

        # ax[0]: fixed reference point method; ax[1]: differential method
        if include_prediction and self.analyze_eta:
            ax[0].plot(self.cl_DT_k,self.predicted_Dp_w,'k-',label='$\Delta{P}_{pred}$')
            ax[1].plot(x,self.predicted_dp_w,'k-',label='$dP_{pred}$')

        jj=0
        for ii in range(len(self.rn_fracs)):
            if not np.isnan(self.dp_at_rnfrac[ii,:]).any():
                ax[0].plot(self.cl_DT_k,self.Dp_at_rnfrac[ii,:],'o-',color=self.colors[jj],label='%.2f'%(self.rn_fracs[ii]))
                ax[1].plot(x,self.dp_at_rnfrac[ii,:],'o-',color=self.colors[jj],label=str(self.rn_fracs[ii]))
                if include_darksubtraction and self.dark_analysis:
                    ax[0].plot(self.cl_DT_k,self.Dp_at_rnfrac[ii,:]-self.dark_Dp_w,'o--',color=self.colors[jj],label='_nolegend_')
                    ax[1].plot(x,self.dp_at_rnfrac[ii,:]-self.dark_dp_w,'o--',color=self.colors[jj],label='_nolegend_')
                jj+=1

        ax[0].set_xlabel('T$_{cl}$ - %.1f K'%self.cl_temps_k[self.T_cl_index])
        ax[0].set_ylabel('P$_o$ - P')
        ax[0].grid('on')
        ax[0].set_title('Fixed Reference Method')

        ax[1].set_xlabel('T$_{cl}$ K')
        ax[1].set_ylabel('dP (W)')
        ax[1].grid('on')
        ax[1].set_title('Differential Method')

        ax[0].legend(loc='best')#self.rn_fracs)
        plt.suptitle(self.figtitle+' dp vs T')
        return fig

    def plot_efficiency(self, include_darksubtraction=True):
        fig, ax = plt.subplots(1,2,figsize=(10,5))
        x = self.cl_dT_k/2 + self.cl_temps_k[0:-1] # midpoint between sampled coldload temperatures

        # ax[0]: fixed reference point method; ax[1]: differential method
        jj=0
        for ii in range(len(self.rn_fracs)):
            if not np.isnan(self.dp_at_rnfrac[ii,:]).any():
                ax[0].plot(self.DT_eta,self.eta_Dp_arr[ii,:],'o-',color=self.colors[jj],label='%.2f'%(self.rn_fracs[ii]))
                ax[1].plot(x,self.eta_dp_arr[ii,:],'o-',color=self.colors[jj],label='%.2f'%(self.rn_fracs[ii]))
                if include_darksubtraction and self.dark_analysis:
                    ax[0].plot(self.DT_eta,self.eta_Dp_arr_darksubtracted[ii,:],'o--',color=self.colors[jj],label='_nolegend_')
                    ax[1].plot(x,self.eta_dp_arr_darksubtracted[ii,:],'o--',color=self.colors[jj],label='_nolegend_')
                jj+=1

        ax[0].set_xlabel('T$_{cl}$ - %.1f K'%self.cl_temps_k[self.T_cl_index])
        ax[0].set_ylabel('Optical Efficiency')
        ax[0].grid('on')
        ax[0].set_title('Fixed Reference Method')
        ax[0].set_ylim(0,1)

        ax[1].set_xlabel('T$_{cl}$ K')
        ax[1].grid('on')
        ax[1].set_title('Differential Method')
        ax[1].set_ylim(0,1)
        ax[1].legend()
        plt.suptitle(self.figtitle+ ' efficiency')
        return fig

    def plot_mean_efficiency(self):
        assert self.analyze_eta, 'Analyze_eta=False.  Did you provide a power prediction?'
        fig, ax = plt.subplots(nrows=1,ncols=1)
        ax.plot(self.rn_fracs,self.eta_for_rfrac,'o-',color=self.colors[0],label='fit')
        labels=['fixed ref','diff']
        for ii,XX in enumerate([self.eta_Dp_arr,self.eta_dp_arr]):
            ax.errorbar(self.rn_fracs,XX.mean(1),XX.std(1),linestyle='-',color=self.colors[ii+1],label=labels[ii])
        if self.dark_analysis:
            ax.plot(self.rn_fracs,self.eta_for_rfrac_darksubtracted,'o--',color=self.colors[0])
            for ii,XX in enumerate([self.eta_Dp_arr_darksubtracted,self.eta_dp_arr_darksubtracted]):
                ax.errorbar(self.rn_fracs,XX.mean(1),XX.std(1),linestyle='--',color=self.colors[ii+1])

        ax.set_xlabel('Rn fraction')
        ax.set_ylabel('Optical Efficiency')
        ax.legend()
        ax.set_ylim(0,1.1)
        return fig

    def plot_full_analysis(self,include_darksubtraction=True,showfigs=False,savefigs=False,path=''):
        ''' Make plots of the full analysis:
            1) raw IV, one curve per coldload temperature
            2) vipr, (2x2 plot if IV, PV, RP, RV), on curve per coldload temperature
            3) P versus R, with cuts at rnfracs
            4) P versus T_cl
            5) dP vs dT (1 x 2 using both methods)
            6) efficiency
            7) mean efficiency versus %rn

        '''
        if include_darksubtraction and self.dark_analysis:
            include_ds = True
        else:
            include_ds = False

        figs = []
        figs.append(self.plot_raw()) # raw
        figs.append(self.plot_vipr()) # 2x2 of converted data
        figs.append(self.plot_pr())
        if not np.isnan(self.p_at_rnfrac).all():
            figs.append(self.plot_pt())
            figs.append(self.plot_power_change_vs_temperature(include_prediction=True, include_darksubtraction=include_ds))
            if self.analyze_eta:
                figs.append(self.plot_efficiency(include_darksubtraction=include_ds))
                figs.append(self.plot_mean_efficiency())
        else:
            print('nan found in p_at_rnfrac.  I can not plot power change versus temperature')

        if savefigs:
            fig_appendix=['raw','vipr','pr','pt','dpt','eta']
            for ii,fig in enumerate(figs):
                print(ii, fig_appendix[ii])
                fig.savefig(self.row_name+'_%d_'%ii+fig_appendix[ii]+'.png')
        if showfigs: plt.show()
        #for fig in figs:
            #plt.close(fig)
            #fig.clf()

class IVColdloadSweepAnalyzer():
    ''' Class to analyze a column of data in a coldload IV sweep '''
    def __init__(self,filename_json,detector_map=None,iv_circuit="from file"):
        self.df = IVColdloadSweepData.from_file(filename_json)
        self.filename = filename_json
        self.data = self.df.data
        self.det_map = detector_map
        self.iv_circuit = self._handle_iv_circuit(iv_circuit)
        self.row_sequence = self.df.extra_info['config']['detectors']['Rows']

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

    # helper magic methods -----------------------------------------------------
    def _handle_iv_circuit(self,iv_circuit):
        if iv_circuit == "from file":
            cal = self.df.extra_info['config']['calnums']
            rfb_ohm = cal['rfb']+50.0
            rbias_ohm = cal['rbias']
            rsh_ohm = cal['rjnoise']
            mr = cal['mr']
            vfb_gain = cal['vfb_gain']/(2**14-1)
            source = self.df.extra_info['config']['voltage_bias']['source']

            if source == 'tower':
                vb_max = 2.5
            elif source == 'bluebox':
                vb_max=6.5
            else:
                assert False, 'What is the maximum voltage of your detector voltage bias source?  I need to know to calibrate to physical units'

            iv_circuit = IVCircuit(rfb_ohm=rfb_ohm,
                                   rbias_ohm=rbias_ohm,
                                   rsh_ohm=rsh_ohm,
                                   rx_ohm=0,
                                   m_ratio=mr,
                                   vfb_gain=vfb_gain,
                                   vbias_gain=vb_max/(2**16-1))
        else:
            pass
        return iv_circuit

    def _package_cl_temp_to_list(self):
        cl_temp_list = []
        cl_temp_list.append(list(self.measured_cl_temps_k[:,0]))
        cl_temp_list.append(list(self.measured_cl_temps_k[:,1]))
        return cl_temp_list

    def _row_to_sequence_index(self,row_index):
        ''' Because the TDM electronics has an arbitrary state sequence, the row order
            can be anything.  This method finds the position of row_index within the
            arbitrary state sequence self.row_sequence.  Mostly useful in get_cl_sweep_dataset_for_row()
        '''
        assert row_index in self.row_sequence, print('Row %d'%row_index, ' is not in the row_sequence:',self.row_sequence)
        return self.row_sequence.index(row_index)

    def get_measured_coldload_temps(self,index=0):
        return 0.5*np.array(self.pre_cl_temps_k)[:,index] + 0.5*np.array(self.post_cl_temps_k)[:,index]

    def get_cl_sweep_dataset_for_row(self,row_index,bath_temp_index=0,cl_indices=None):
        if cl_indices==None:
            cl_indices = list(range(self.n_cl_temps))
        fb = np.zeros((self.n_dac_values,len(cl_indices)))
        for ii,cl_idx in enumerate(cl_indices):
            dex = self._row_to_sequence_index(row_index)
            fb[:,ii] = self.data[cl_idx].data[bath_temp_index].fb_values_array()[:,dex]
        return self.dac_values, fb

    def print_info(self):
        # Too add in future:
        # 1) date/time of start/end of data
        # 2) Did data complete?
        print('Coldload set temperatures: ',self.set_cl_temps_k)
        print('Measured coldload temperatures: ',self.measured_cl_temps_k)
        print('ADR set temperatures: ',self.set_bath_temps_k)
        print('use plot_measured_cl_temps() and plot_measured_bath_temps() to determine if set temperatures were achieved')

    def sweep_analysis_for_row(self,row_index,bath_temp_index,
                                    cl_indices=None,rn_fracs=None,predicted_power_w=None,
                                    dark_power_w='auto',dark_rn_frac=0.7):
        if cl_indices == None:
            cl_indices = list(range(len(self.set_cl_temps_k)))
        dacs,fb = self.get_cl_sweep_dataset_for_row(row_index=row_index,bath_temp_index=bath_temp_index,cl_indices=cl_indices)

        if self.det_map:
            row_name = 'Row%02d'%row_index
            det_name = self.det_map.get_devname_from_row_index(row_index)
            position = self.det_map.map_dict[row_name]['position']
            band = self.det_map.map_dict[row_name]['band']
            typ = self.det_map.map_dict[row_name]['type']
            if typ == 'dark':
                dark_power_w=None

            if dark_power_w is None:
                pass 
            elif type(dark_power_w) == int: # assume that an integer is a row index
                darkbolo_index = dark_power_w 
                foo = self.sweep_analysis_for_row(darkbolo_index,bath_temp_index,cl_indices,rn_fracs=[dark_rn_frac],dark_power_w=None)
                dark_power_w = foo.get_power_vector_for_rnfrac(dark_rn_frac)
            elif type(dark_power_w)==str:
                assert dark_power_w == 'auto','unknown string input for dark_power_w '%(dark_power_w) # default is to find the appropriate dark
                darkbolo_index = self.det_map.get_row_from_position_band_pol_type(position,band,None,'dark')
                if darkbolo_index == None:
                    print('A dark bolometer at position %d for band %s has not been found using the auto method.  Defining dark_power_w=None'%(position,band))
                    dark_power_w = None
                else: 
                    print('Dark bolometer for position %d and band %s is row index %2d'%(position,band,darkbolo_index))
                    foo = self.sweep_analysis_for_row(darkbolo_index,bath_temp_index,cl_indices,rn_fracs=[dark_rn_frac],dark_power_w=None)
                    dark_power_w = foo.get_power_vector_for_rnfrac(dark_rn_frac)
            
        else:
            row_name=det_name=dark_power_w=None

        iva = IVColdloadAnalyzeOneRow(dacs,fb,
                                      cl_temps_k = np.array(self.post_cl_temps_k)[cl_indices,0],
                                      bath_temp_k=self.set_bath_temps_k[bath_temp_index],
                                      row_name=row_name, det_name=det_name,
                                      iv_circuit=self.iv_circuit,
                                      predicted_power_w=predicted_power_w,dark_power_w=dark_power_w,rn_fracs=rn_fracs)
        return iva

    def full_analysis(self,bath_temp_index,cl_indices,showfigs=False,savefigs=False,rn_fracs=None,dark_rnfrac=0.7,
                      skipsquidchannels=True):
        assert self.det_map != None,'Must provide a detector map in order to do the full analysis'

        dark_rows = self.det_map.get_row_nums_from_keyval_list([['type','dark']])
        if skipsquidchannels:
            row_indices = self.det_map.get_row_nums_from_keyval_list([['type','optical']]) # optical rows
            row_indices.extend(dark_rows)
            row_indices = sorted(row_indices)
        else:
            row_indices = self.row_index_list

        # first collect dark responses for each pixel and place in dark_Ps dictionary
        dark_indices = []
        for row in dark_rows:
            try: dark_indices.append(self.row_sequence.index(row))
            except: pass
        dark_Ps = {}
        for idx in dark_indices:
            row_name = 'Row%02d'%idx
            row_dict = self.det_map.map_dict[row_name]
            dacs,fb = self.get_cl_sweep_dataset_for_row(row_index=idx,bath_temp_index=bath_temp_index,cl_indices=cl_indices)
            iva_dark = IVColdloadAnalyzeOneRow(dacs,fb,
                                               cl_temps_k=list(np.array(self.set_cl_temps_k)[cl_indices]),
                                               bath_temp_k=self.set_bath_temps_k[bath_temp_index],
                                               row_name=row_name, det_name=self.det_map.get_devname_from_row_index(idx),
                                               iv_circuit=self.iv_circuit,
                                               predicted_power_w=None, dark_power_w=None)
            dark_Ps[str(row_dict['position'])]=iva_dark.get_power_vector_for_rnfrac(dark_rnfrac)

        # now loop over all rows
        for row in row_indices:
            row_name = 'Row%02d'%row
            row_dict = self.det_map.map_dict['Row%02d'%(row)]
            print('Row%02d'%(row),': ',row_dict)
            dacs,fb = self.get_cl_sweep_dataset_for_row(row_index=row,bath_temp_index=bath_temp_index,cl_indices=cl_indices)
            if row_dict['type']=='optical':
                dark_P = dark_Ps[str(row_dict['position'])]
            else:
                dark_P = None

            iva = IVColdloadAnalyzeOneRow(dacs,fb,
                                          cl_temps_k=list(np.array(self.set_cl_temps_k)[cl_indices]),
                                          bath_temp_k=self.set_bath_temps_k[bath_temp_index],
                                          row_name=row_name, det_name=self.det_map.get_devname_from_row_index(row),
                                          iv_circuit=self.iv_circuit,
                                          predicted_power_w=None,
                                          dark_power_w=dark_P)

            if rn_fracs is not None:
                iva.rn_fracs = rn_fracs
            iva.plot_full_analysis(include_darksubtraction=False,showfigs=showfigs,savefigs=savefigs)

    # plotting methods ---------------------------------------------------------
    def plot_measured_cl_temps(self):
        plt.figure()
        plt.xlabel('Setpoint Temperature (K)')
        plt.ylabel('Measured Temperature (K)')
        #cl_temp_list = self._package_cl_temp_to_list()
        plt.plot(self.set_cl_temps_k,self.pre_cl_temps_k,'*')
        plt.plot(self.set_cl_temps_k,self.post_cl_temps_k,'*')
        plt.plot(list(range(self.max_cl_temp_k+1)),'b--')
        plt.legend(('ChA pre','ChB pre','ChA post','ChB post'))
        plt.grid()

        plt.figure()
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
        plt.figure()
        plt.xlabel('DAC values')
        plt.ylabel('Feedback values')
        plt.plot(x,y,'-')
        plt.title('Row index = %d, CL_temp_index = %.1f K, Tb_index = %d mK'%(row_index,cl_temp_index,bath_temp_index))
        plt.show()

    def plot_cl_temp_sweep_for_row(self,row_index,bath_temp_index,cl_indices=None):
        if cl_indices==None:
            cl_indices = list(range(self.n_cl_temps))
        x,fb_arr = self.get_cl_sweep_dataset_for_row(row_index,bath_temp_index,cl_indices)
        plt.figure()
        for ii in range(len(cl_indices)):
            dy = fb_arr[0,ii]-fb_arr[0,0]
            plt.plot(self.dac_values, fb_arr[:,ii]-dy)
        plt.xlabel('DAC values')
        plt.ylabel('Feedback values')
        plt.title('Row index = %d, Tb = %d mK'%(row_index,self.set_bath_temps_k[bath_temp_index]*1000))
        plt.legend((np.array(self.set_cl_temps_k)[cl_indices]),loc='upper right')
        plt.show()

    def plot_sweep_analysis_for_row(self,row_index,bath_temp_index,cl_indices=None,\
                                    rn_fracs=None,showfigs=True,savefigs=False,
                                    predicted_power_w=None,
                                    dark_power_w='auto',dark_rn_frac=0.7,
                                    include_darksubtraction=True):
        iva = self.sweep_analysis_for_row(row_index,bath_temp_index,
                                    cl_indices,rn_fracs,predicted_power_w,dark_power_w,dark_rn_frac)

        if dark_power_w is None and include_darksubtration:
            print('include_darksubtration set to True, but no dark_power_w provided or found.  Dark subtraction not included.')
            include_darksubtraction=False
        
        iva.plot_full_analysis(include_darksubtraction,showfigs,savefigs)
        return iva 


    def _predicted_power(self,cl_temps,f_edges_ghz):
        predicted_power_w = []
        for t in cl_temps:
            predicted_power_w.append(thermalPower(f_edges_ghz[0]*1e9,f_edges_ghz[1]*1e9,t))
        return np.array(predicted_power_w)

    def plot_DpDt_for_rows(self,row_list,bath_temp_index=0,cl_indices=None,rn_frac=0.8,legend=True,
                                predicted_power_w=None,print_efficiency=True):
        ''' plot the change in power for change in cold load temperature for rows in
            row_list.
        '''
        if cl_indices == None:
            cl_indices = list(range(len(self.set_cl_temps_k)))

        for row in row_list:
            if row not in self.row_sequence:
                print('Row%02d was not measured.  Will not plot'%row)
                row_list.remove(row)

        ivas=[]
        legend_txt=[]
        for ii,row in enumerate(row_list):
            if self.det_map:
                row_name = 'Row%02d'%row
                det_name = self.det_map.get_devname_from_row_index(row)
                legend_txt.append(det_name)
                if self.det_map.map_dict[row_name]['freq_edges_ghz'] is not None:
                    predicted_power_w = self._predicted_power(np.array(self.set_cl_temps_k)[cl_indices],self.det_map.map_dict[row_name]['freq_edges_ghz'])
            else:
                row_name=det_name=None
                legend_txt.append(row)
            dacs,fb = self.get_cl_sweep_dataset_for_row(row,bath_temp_index=bath_temp_index,cl_indices=cl_indices)
            iva = IVColdloadAnalyzeOneRow(dacs,fb,
                                          cl_temps_k=np.array(self.set_cl_temps_k)[cl_indices],# put in measured values here!
                                          bath_temp_k=self.set_bath_temps_k[bath_temp_index],
                                          row_name=row_name, det_name=det_name,
                                          iv_circuit=self.iv_circuit,
                                          predicted_power_w=predicted_power_w,dark_power_w=None,rn_fracs=[rn_frac])
            ivas.append(iva)
        fig,ax = plt.subplots(1,1)
        for iva in ivas:
            ax.plot(iva.cl_DT_k,iva.Dp_at_rnfrac[0,:],'o-')#,color=self.colors[jj],label=str(self.rn_fracs[ii]))
        if predicted_power_w is not None:
            bands = []
            ls=['solid','dotted','dashed','dashdot']
            for ii,row in enumerate(row_list):
                band = self.det_map.map_dict['Row%02d'%row]['band']
                if np.logical_and(band is not None, band not in bands):
                    ax.plot(ivas[ii].cl_DT_k,ivas[ii].predicted_Dp_w,color='k',linestyle=ls[len(bands)])
                    legend_txt.append(band+' prediction')
                    bands.append(band)
        ax.set_xlabel('T$_{cl}$ - %.1f K'%iva.cl_temps_k[iva.T_cl_index])
        ax.set_ylabel('P$_o$ - P')

        if legend: ax.legend(legend_txt)
        ax.grid()

        if np.logical_and(print_efficiency,predicted_power_w is not None):
            for ii,row in enumerate(row_list):
                if ivas[ii].analyze_eta:
                    print('%s, %s, eta = %.3f'%(ivas[ii].row_name,ivas[ii].det_name,ivas[ii].eta_mean[0].mean()))
        return fig, ax

    def plot_DpDt_for_position(self,position,bath_temp_index=0,cl_indices=None,rn_frac=0.8,legend=True):
        row_list = self.det_map.rows_in_position(position,True,True)
        fig,ax = self.plot_DpDt_for_rows(row_list,bath_temp_index,cl_indices,rn_frac,legend)
        return fig,ax

    def plot_pt_delta_diff(self,row_index,dark_row_index,bath_temp_index,cl_indices=None):
        ''' plot the difference in the change in power verus the change in cold load temperature between two bolometers.
            This is often useful for dark subtraction
        '''
        if cl_indices==None:
            cl_indices = list(range(self.n_cl_temps))

        dacs,fb = self.get_cl_sweep_dataset_for_row(row_index=row_index,bath_temp_index=bath_temp_index,cl_indices=cl_indices)
        dacs,fb_dark = self.get_cl_sweep_dataset_for_row(row_index=dark_row_index,bath_temp_index=bath_temp_index,cl_indices=cl_indices)

        iva = IVColdloadAnalyzeOneRow(dacs,fb,
                                      cl_temps_k=list(np.array(self.set_cl_temps_k)[cl_indices]),# put in measured values here!
                                      bath_temp_k=self.set_bath_temps_k[bath_temp_index],
                                      iv_circuit=self.iv_circuit)

        iva_dark = IVColdloadAnalyzeOneRow(dacs,fb_dark,
                                      cl_temps_k=list(np.array(self.set_cl_temps_k)[cl_indices]),# put in measured values here!
                                      bath_temp_k=self.set_bath_temps_k[bath_temp_index],
                                      iv_circuit=self.iv_circuit)

        fig, ax = plt.subplots(1,1)
        ax.plot(np.array(self.set_cl_temps_k)[cl_indices],iva.Dp_at_rnfrac.transpose(),'bo-')
        ax.plot(np.array(self.set_cl_temps_k)[cl_indices],iva_dark.Dp_at_rnfrac.transpose(),'ko-')
        ax.plot(np.array(self.set_cl_temps_k)[cl_indices],iva.Dp_at_rnfrac.transpose()-iva_dark.Dp_at_rnfrac.transpose(),'bo--')
        ax.set_xlabel('T$_{cl}$')
        ax.set_ylabel('dP')
        return fig

    def plot_DpDt_for_position2(self,position,bath_temp_index=0,cl_indices=None,rfrac=0.8):
        ''' plot the summary change in power per change in cold load temperature (from reference temperature)
            for all detectors within a pixel at position.  This method requires a complete detector map.

            This is redundant with plot_DpDt_for_position.  Aesthetic differences.
        '''
        #self.det_map.print_data_for_position(position)
        assert self.det_map != None, 'Must have a detector map to plot_DpDt_for_position'
        bath_temp_index=0
        if cl_indices == None:
            cl_indices = list(range(len(self.set_cl_temps_k)))
        bands = self.det_map.get_bands_for_position(position)

        # get order: darks, lowest frequency A,B highest frequency A, B
        row_indices = self.det_map.get_row_nums_from_keyval_list([['position',position],['type','dark']])
        for band in bands:
            for pol in ['A','B']:
                row_indices.extend(self.det_map.get_row_nums_from_keyval_list([['position',position],['band',str(band)],['polarization',pol]]))

        fig,ax = plt.subplots()
        ax.set_ylabel('P$_o$ - P')
        low_band_color = '#1f77b4'
        high_band_color = '#ff7f0e'
        dark_color = "tab:grey"#('k',0.5) #'#2ca02c'
        A_style = 'o'
        B_style = 'v'
        dark_count = 0
        for ii, row_dex in enumerate(row_indices):
            if self.det_map.map_dict['Row%02d'%row_dex]['type'] not in ['Optical','optical']:
                ppower = None
            else:
                ppower = self.get_predicted_power_for_row(row_dex,cl_indices)

            iva_row = self.sweep_analysis_for_row(row_dex,bath_temp_index,cl_indices,
                                                  rn_fracs=np.array([rfrac]),predicted_power_w=ppower)

            # poorly written below to handling plotting visuals
            if self.det_map.map_dict['Row%02d'%row_dex]['type']=='dark':
                dark_count=dark_count+1
                color = dark_color
                if dark_count>1:
                    ls = '*'
                else:
                    ls='+'
            elif self.det_map.map_dict['Row%02d'%row_dex]['band'] == str(bands[0]):
                color = low_band_color
            elif self.det_map.map_dict['Row%02d'%row_dex]['band'] == str(bands[1]):
                color = high_band_color

            if self.det_map.map_dict['Row%02d'%row_dex]['polarization'] == 'A':
                ls = A_style
            elif self.det_map.map_dict['Row%02d'%row_dex]['polarization'] == 'B':
                ls = B_style

            # now actually plot it
            ax.plot(iva_row.cl_DT_k,iva_row.Dp_at_rnfrac[0,:],label=self.det_map.map_dict['Row%02d'%row_dex]['devname'],color=color,marker=ls)
            if ii==0:
                ax.set_xlabel('T$_{cl}$ - %.1f K'%iva_row.cl_temps_k[iva_row.T_cl_index])

            if self.det_map.map_dict['Row%02d'%row_dex]['type']=='optical':
                print(self.det_map.map_dict['Row%02d'%row_dex]['devname'], 'eta = ',iva_row.eta_Dp_arr[~np.isnan(iva_row.eta_Dp_arr)].mean())

        # add tophat prediction
        for band in bands:
            if band == bands[0]:
                color = low_band_color
            elif band == bands[1]:
                color = high_band_color
            rr = self.det_map.get_row_nums_from_keyval_list([['position',position],['band',str(band)]])
            p = self.get_predicted_power_for_row(rr[0],cl_indices)
            ax.plot(iva_row.cl_DT_k,p-p[0],color=color,label='tophat %s band'%band,linestyle='-',alpha=0.5)

        ax.legend(fontsize=8)
        ax.grid()
        fig.suptitle('Position %d '%position+' '.join(self.det_map.map_dict['Row%02d'%row_indices[-1]]['devname'].split(' ')[0:2]))
        return fig,ax

    def get_predicted_power_for_row(self,row_index,cl_indices):
        ''' return top-hat-band predicted power to be used to determine optical efficiency '''
        [fstart,fend] = self.det_map.map_dict['Row%02d'%row_index]['freq_edges_ghz']
        predicted_power_w = []
        for t in np.array(self.post_cl_temps_k)[cl_indices,0]:
            predicted_power_w.append(thermalPower(fstart*1e9,fend*1e9,t))
        return np.array(predicted_power_w)

#######################################

def iv_tempsweep_quicklook(filename,row_index,use_config=True,temp_indices=None,rn_fracs=None,
                           cal_params_dict=None, use_IVCurveAnalyzeSingle=True):

    df = IVTempSweepData.from_file(filename) # df = "data frame"
    cfg = df.data[0].extra_info['config']

    print('State sequence: ',cfg['detectors']['Rows'])
    print('Row Select: ',cfg['detectors']['Rows'][row_index])

    if cfg['voltage_bias']['source'] == 'tower':
        vb_max = 2.5
    elif cfg['voltage_bias']['source'] == 'bluebox':
        vb_max = 6.5
    else:
        assert False,'unknown voltage bias'

    if temp_indices == None:
        temp_indices = range(len(df.set_temps_k))
    temp_list_k = np.array(df.set_temps_k)[temp_indices]

    if rn_fracs == None:
        rn_fracs=[0.5,0.6,0.7,0.8]

    # circuit parameters to convert to physical units
    if use_config:
        cal = cfg['calnums']
        rfb_ohm = cal['rfb']+50.0
        rbias_ohm = cal['rbias']
        rsh_ohm = cal['rjnoise']
        mr = cal['mr']
        vfb_gain = cal['vfb_gain']/(2**14-1)
        vbias_gain = vb_max/(2**16-1)

    else:
        assert cal_params, 'if use_config=False, must provide calibration numbers in cal_params_dict'
        rfb_ohm = cal_params_dict['rfb_ohm']
        rbias_ohm = cal_params_dict['rbias_ohm']
        rsh_ohm = cal_params_dict['rsh_ohm']
        rx_ohm = cal_parmas_dict['rx_ohm']
        mr = cal_params_dict['mr']
        vfb_gain = cal_params_dict['fb_v_per_bit'] #1.017/(2**14-1)
        vbias_gain = cal_params_dict['vdac_v_per_bit'] #v_max / 2**16-1

    iv_circuit = IVCircuit(rfb_ohm=rfb_ohm,
                           rbias_ohm=rbias_ohm,
                           rsh_ohm=rsh_ohm,
                           rx_ohm=0,
                           m_ratio=mr,
                           vfb_gain=vfb_gain,
                           vbias_gain=vbias_gain)

    # construct fb_arr versus Tbath for a single row
    dac, fb = df.data[0].xy_arrays()
    fb_arr = np.zeros((len(dac),len(temp_indices)))
    for ii,dex in enumerate(temp_indices):
        dac, fb = df.data[dex].xy_arrays()
        fb_arr[:,ii] = fb[:,row_index]

    # do the analysis
    iv_tsweep = IVversusADRTempOneRow(dac_values=dac,fb_values_arr=fb_arr, 
                                      temp_list_k=temp_list_k, 
                                      normal_resistance_fractions=rn_fracs,
                                      iv_circuit=iv_circuit,
                                      use_IVCurveAnalyzeSingle=use_IVCurveAnalyzeSingle)
    # plot
    iv_tsweep.plot_raw()
    iv_tsweep.plot_vipr()
    iv_tsweep.plot_pr()
    iv_tsweep.plot_pt()
    iv_tsweep.plot_fits()
    print(iv_tsweep.pfits[:])
    return iv_tsweep.pfits,iv_tsweep.r_clean[0:10,:].mean()
    #plt.show()

def iv_chop(file1,file2,row_index,temp_indices=[0,0],state_list=None,iv_circuit=None):
    ''' file 1 - file 2 '''
    iv1=IVTempSweepData.from_file(file1)
    iv2=IVTempSweepData.from_file(file2)
    if iv_circuit == None: iv_circuit=iv_circuit_from_file(iv1)
    ivchop = IVSetAnalyzeColumn([iv1.data[temp_indices[0]],iv2.data[temp_indices[1]]],state_list=state_list,iv_circuit=iv_circuit) #instance of IVSetAnalyzeColumn
    ivchop.plot_row(row_index,to_physical_units=True)

def iv_circuit_from_file(iv_temp_sweep_data):
    if type(iv_temp_sweep_data) is str:
        df = IVTempSweepData.from_file(iv_temp_sweep_data) # df = "data frame"
    else:
        df = iv_temp_sweep_data
    cfg = df.data[0].extra_info['config']
    cal = cfg['calnums']
    rfb_ohm = cal['rfb']+50.0
    rbias_ohm = cal['rbias']
    rsh_ohm = cal['rjnoise']
    mr = cal['mr']
    vfb_gain = cal['vfb_gain']/(2**14-1)
    source = cfg['voltage_bias']['source']

    if source == 'tower':
        vb_max = 2.5
    elif source == 'bluebox':
        vb_max=6.5
    else:
        assert False, 'What is the maximum voltage of your detector voltage bias source?  I need to know to calibrate to physical units'

    iv_circuit = IVCircuit(rfb_ohm=rfb_ohm,
                           rbias_ohm=rbias_ohm,
                           rsh_ohm=rsh_ohm,
                           rx_ohm=0,
                           m_ratio=mr,
                           vfb_gain=vfb_gain,
                           vbias_gain=vb_max/(2**16-1))
    return iv_circuit

def make_iv_circuit(extra_info):
    assert 'config' in list(extra_info.keys()), 'no configuration file found'
    cfg = extra_info['config']
    cal = cfg['calnums']
    rfb_ohm = cal['rfb']+50.0
    rbias_ohm = cal['rbias']
    rsh_ohm = cal['rjnoise']
    mr = cal['mr']
    vfb_gain = cal['vfb_gain']/(2**14-1)
    source = cfg['voltage_bias']['source']
    if source == 'tower':
        vb_max = 2.5
    elif source == 'bluebox':
        vb_max=6.5
    else:
        assert False, 'What is the maximum voltage of your detector voltage bias source?  I need to know to calibrate to physical units'

    iv_circuit = IVCircuit(rfb_ohm=rfb_ohm,
                           rbias_ohm=rbias_ohm,
                           rsh_ohm=rsh_ohm,
                           rx_ohm=0,
                           m_ratio=mr,
                           vfb_gain=vfb_gain,
                           vbias_gain=vb_max/(2**16-1))

    return iv_circuit

if __name__ == "__main__":
    fname = '/Users/hubmayr/tmp/20230609/uber_omt_ColumnA_ivs_20230613_1686672234.json'
    ivcl = IVColdloadSweepAnalyzer(fname)
    predicted_power_w = []
    for t in ivcl.set_cl_temps_k:
        predicted_power_w.append(thermalPower(nu1=77.0e9,nu2=108.0e9,T=t,F=None))
    ivcl.plot_sweep_analysis_for_row(row_index=17,bath_temp_index=0,cl_indices=None,\
                                     showfigs=True,savefigs=False,predicted_power_w=np.array(predicted_power_w))

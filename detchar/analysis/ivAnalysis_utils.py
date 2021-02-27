''' ivAnalysis_utils.py '''

# NOTES
# cutting bad data in IV
# error handle if there is no IV turn-around
# dark subtraction
# Determine some metric for efficiency and report that one number.  what rn_frac?  What cl_temp?
# IVTempSweepData ought to have the coldload setpoint.
# column name for IVColdloadAnalzyeOneRow?
# units for plots?
# plot ranges
# deal with crap rn_frac cuts
# plot titles
# plot sizes

import detchar
from detchar.iv_data import IVCircuit
from detchar import IVColdloadSweepData
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k,h,c
from scipy.integrate import quad, simps

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
    return P

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
        ''' return the indicies corresponding to the IV turnaround for a set of IV curves.
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

    def find_bad_data_index(self,dac,fb,showplot=False):
        ''' Return the index where IV curve misbehaves.
            ASSUME dac and fb(dac) are in descending order
        '''
        # find IV turn
        # at lower voltage bias than IV turn, make sure second derivative is < 0.

        turn_dex = self.get_turn_index(dac,fb,showplot=False)
        if turn_dex == None: return len(dac)
        dfb = np.diff(fb,axis=0)
        if self.is_dac_descending(dac):
            norm_dfb = np.mean(dfb[0:10],axis=0)
        else:
            norm_dfb = np.mean(dfb[::-1][0:10],axis=0)
        x = dfb/norm_dfb
        ddfb = smooth(np.diff(x,axis=0),3)
        ii = turn_dex; val = ddfb[ii]
        while val < 0.5:
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
            plt.plot(fb[0:dex],'r*')

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
        ''' return the indicies corresponding to the IV turnaround for a set of IV curves.
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

    def find_bad_data_index_arr(self,fb_array, threshold=50,showplot=False):
        ''' Return the index where IV curve misbehaves.  '''
        #    often times when TES latches, the data is not useful, and
        #    in fact problematic when trying to determine P at fracRn, since
        #    power at fracRn can be erroneously double valued.  This method
        #    finds the index where this happens

        # first find indicies where delta i is larger than some threshold
        di = np.diff(fb_array,axis=0)
        norm_di = np.mean(di[-10:,:],axis=0) # positive definite
        n,m = np.shape(di)
        dexs=[]
        for ii in range(m):
            alldexs = np.where(abs(di[:,ii])>threshold*norm_di[ii])
            if len(alldexs[0]) == 0:
                dexs.append(None)
            else:
                dexs.append(np.max(alldexs[0])) # assume the highest vbias is what we want
        self.badDataIndicies = dexs

        if showplot: #for debuggin purposes
            plt.xlabel('index')
            plt.ylabel('current (%s)'%self.i_units)
            for ii in range(m):
                plt.plot(self.i[:,ii],'*-')
                plt.plot(dexs[ii],self.i[dexs[ii],ii],'ro')
                plt.show()
                input('%d'%ii)

    def remove_bad_data(self,PLOT=False):
        ''' fill bad data with np.nan '''
        i_orig = self.i.copy()
        for ii in range(self.n_sweeps):
            if self.badDataIndicies[ii] != None:
                self.v[0:self.badDataIndicies[ii]+1,ii] = np.ones(self.badDataIndicies[ii]+1)*np.nan
                self.i[0:self.badDataIndicies[ii]+1,ii] = np.ones(self.badDataIndicies[ii]+1)*np.nan
                self.p[0:self.badDataIndicies[ii]+1,ii] = np.ones(self.badDataIndicies[ii]+1)*np.nan
                self.r[0:self.badDataIndicies[ii]+1,ii] = np.ones(self.badDataIndicies[ii]+1)*np.nan
                self.ro[0:self.badDataIndicies[ii]+1,ii] = np.ones(self.badDataIndicies[ii]+1)*np.nan
        if PLOT:
            plt.plot(i_orig,'b*')
            plt.plot(self.i,'ro')
            plt.show()



class IVColdloadAnalyzeOneRow():
    ''' Analyze a set of IV curves for a single detector taken at multiple
        coldload temperatures and a single bath temperature
    '''

    def __init__(self,dac_values,fb_array,cl_temps_k,bath_temp_k,device_dict=None,iv_circuit=None,passband_dict=None):
        self.dacs = dac_values
        self.n_dac_values = len(self.dacs)
        self.fb = fb_array # NxM array of feedback values.  Columns are per coldload temperature
        self.fb_align = None
        self.cl_temps_k = cl_temps_k
        self.bath_temp_k = bath_temp_k
        self.det_name, self.row_name = self._handle_device_dict(device_dict)
        self.n_dac_values, self.n_cl_temps = np.shape(self.fb)

        # fixed globals
        self.n_normal_pts=10 # number of points for normal branch fit
        self.use_ave_offset=True # use a global offset to align fb, not individual per curve
        self.rn_fracs = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] # slices in Rn space to compute delta Ps
        if iv_circuit==None:
            self.to_physical_units = False
        else:
            self.iv_circuit = iv_circuit
            self.to_physical_units = True

        self.figtitle = self.det_name+', '+self.row_name+' , Tb = %.1f mK'%(self.bath_temp_k*1000)

        # do analysis, place main results as globals to class
        self.v,self.i,self.p,self.r = self.get_vipr(showplot=False)
        self.ro = self.r / self.r[0,:]
        self.p_at_rnfrac = self.get_value_at_rn_frac(self.rn_fracs,self.p,self.ro)
        self.cl_dT_k, self.dP_w, self.T_cl_index = self.get_delta_pt()
        # predicted power
        self.freq_edges_ghz=self.passband_sim_ghz=None
        self.power_cl_tophat=self.power_cl_sim_passband=self.power_cl_tophat_delta=self.power_cl_sim_passband_delta=self.eta_tophat=self.eta_passband_sim=None
        self._handle_prediction(passband_dict)

    def _handle_device_dict(self,device_dict):
        if device_dict==None:
            det_name = 'unknown'; row_name = 'unknown'; f_edges = None
        else:
            assert type(device_dict)==dict, ('device_dict either None or must be of type dictionary')
            row_name = [*device_dict][0]; det_name = device_dict[row_name]
        return det_name, row_name

    def _handle_prediction(self,passband_dict):
        self.prediction, self.frequency_edges_ghz, self.passband_sim_ghz = self._handle_passband(passband_dict)
        if self.prediction[0] !=0:
            self.power_cl_tophat = self.get_predicted_thermal_power_tophat(self.cl_temps_k,f_edges_ghz=self.frequency_edges_ghz)
            self.power_cl_tophat_delta = np.array(self.power_cl_tophat) - self.power_cl_tophat[self.T_cl_index]
            self.eta_tophat = self.get_efficiency(self.power_cl_tophat_delta, self.dP_w)
        if self.prediction[1] !=0:
            self.power_cl_sim_passband = self.get_predicted_thermal_power_simpassband(self.cl_temps_k,passband_ghz=self.passband_sim_ghz)
            self.self.power_cl_sim_passband_delta = np.array(self.power_cl_sim_passband) - self.power_cl_sim_passband[self.T_cl_index]
            self.eta_passband_sim = self.get_efficiency(self.power_cl_sim_passband_delta, self.dP_w)

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
        fig, ax = plt.subplots(nrows=2,ncols=2,sharex=False,figsize=(12,8))
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

        fig.suptitle(self.figtitle)
        ax[3].legend(tuple(self.cl_temps_k))
        return fig

    def plot_pr(self,rn_fracs,p_at_rnfrac,p,ro,fig_num=1):
        pPlot = self.get_value_at_rn_frac([0.999],arr=p,ro=ro)

        # FIG1: P versus R/Rn
        fig = plt.figure(fig_num)
        plt.plot(ro, p,'-') # plots for all Tbath
        plt.plot(rn_fracs,p_at_rnfrac,'ro')
        plt.xlim((0,1.1))
        plt.ylim((np.min(p_at_rnfrac[~np.isnan(p_at_rnfrac)])*0.9,1.25*np.max(pPlot[~np.isnan(pPlot)])))
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
        for ii in range(len(rn_fracs)):
            plt.plot(self.cl_temps_k,p_at_rnfrac[ii,:],'o-')
        plt.xlabel('T$_{cl}$ (K)')
        plt.ylabel('TES power plateau')
        plt.legend((rn_fracs))
        plt.title(self.figtitle)
        plt.grid()
        return fig

    def plot_pt_delta(self,cl_dT_k, dp_at_rnfrac, rn_fracs, fig_num=1):
        ''' plot change in saturation power relative to minimum coldload temperature '''
        fig = plt.figure(fig_num)
        legend_vals = []
        if self.prediction[0]==1: # include tophat passband prediction
            plt.plot(self.cl_dT_k,self.power_cl_tophat_delta,'k-')
            legend_vals.append('$\Delta{P}_{calc}$ (top hat)')
        if self.prediction[1]==1: # include simulated passband prediction
            plt.plot(self.cl_dT_k,self.power_cl_sim_passband_delta,'k--')
            legend_vals.append('$\Delta{P}_{calc}$ (sim passband)')
        for ii in range(len(rn_fracs)):
            plt.plot(cl_dT_k,dp_at_rnfrac[ii,:],'o-')
            legend_vals.append(str(rn_fracs[ii]))
        plt.xlabel('T$_{cl}$ - %.1f K'%self.cl_temps_k[self.T_cl_index])
        plt.ylabel('P$_o$ - P')
        plt.legend((legend_vals))
        plt.grid()
        plt.title(self.figtitle)
        return fig

    def plot_efficiency(self,cl_dT_k, eta, rn_fracs, fig_num=1):
        fig = plt.figure(fig_num)
        legend_vals = []
        for ii in range(len(rn_fracs)):
            plt.plot(cl_dT_k,eta[ii,:],'o-')
            legend_vals.append(str(rn_fracs[ii]))
        plt.errorbar(cl_dT_k,np.mean(eta,axis=0),np.std(eta,axis=0),color='k',linewidth=4,ecolor='k',elinewidth=4)
        legend_vals.append('mean')
        plt.xlabel('T$_{cl}$ - %.1f K'%self.cl_temps_k[self.T_cl_index])
        plt.ylabel('Efficiency')
        plt.legend((legend_vals))
        plt.grid()
        plt.title(self.figtitle)
        return fig

    def plot_full_analysis(self,showfigs=False,savefigs=False):
        figs = []
        figs.append(self.plot_raw(True,fig_num=1)) # raw
        figs.append(self.plot_vipr(data_list=None,fig_num=2)) # 2x2 of converted data
        figs.append(self.plot_pr(self.rn_fracs,self.p_at_rnfrac,self.p,self.ro,fig_num=3))
        figs.append(self.plot_pt(self.rn_fracs,self.p_at_rnfrac,self.p,self.ro,fig_num=4))
        figs.append(self.plot_pt_delta(self.cl_dT_k, self.dP_w, self.rn_fracs,fig_num=5))
        if self.prediction[0]==1:
            figs.append(self.plot_efficiency(self.cl_dT_k, self.eta_tophat, self.rn_fracs, fig_num=6))
        if self.prediction[1]==1:
            figs.append(self.plot_efficiency(self.cl_dT_k, self.eta_passband_sim, self.rn_fracs, fig_num=7))
        if savefigs:
            fig_appendix=['raw','vipr','pr','pt','dpt','eta_top','eta_sim']
            for ii,fig in enumerate(figs):
                fig.savefig(self.row_name+'_%d_'%ii+fig_appendix[ii]+'.png')
        if showfigs: plt.show()
        for fig in figs:
            fig.clf()

class IVColdloadSweepAnalyzer():
    ''' Class to assess data quality of coldload sweep '''
    # plot cold load measured temperatures
    # plot measured bath temperatures
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

    def get_cl_sweep_dataset_for_row(self,row_index,bath_temp_index=0,cl_indicies=None):
        if cl_indicies==None:
            cl_indicies = list(range(self.n_cl_temps))
        n_cl = len(cl_indicies)
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

    def plot_cl_temp_sweep_for_row(self,row_index,bath_temp_index,cl_indicies=None):
        if cl_indicies==None:
            cl_indicies = list(range(self.n_cl_temps))
        x,fb_arr = self.get_cl_sweep_dataset_for_row(row_index,bath_temp_index,cl_indicies)
        plt.figure(1)
        for ii in range(len(cl_indicies)):
            dy = fb_arr[0,ii]-fb_arr[0,0]
            plt.plot(self.dac_values, fb_arr[:,ii]-dy)
        plt.xlabel('DAC values')
        plt.ylabel('Feedback values')
        plt.title('Row index = %d, Tb = %d mK'%(row_index,self.set_bath_temps_k[bath_temp_index]*1000))
        plt.legend((np.array(self.set_cl_temps_k)[cl_indicies]),loc='upper right')
        plt.show()

    def plot_sweep_analysis_for_row(self,row_index,bath_temp_index,cl_indicies,showfigs=True,savefigs=False):
        dacs,fb = self.get_cl_sweep_dataset_for_row(row_index=row_index,bath_temp_index=bath_temp_index,cl_indicies=cl_indicies)
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
                                      cl_temps_k=list(np.array(self.set_cl_temps_k)[cl_indicies]),# put in measured values here!
                                      bath_temp_k=self.set_bath_temps_k[bath_temp_index],
                                      device_dict=device_dict,
                                      iv_circuit=self.iv_circuit,
                                      passband_dict=passband_dict)
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

if __name__ == "__main__":
    filename_json = 'lbird_hftv0_coldload_sweep_20210203.json'
    dm = DetectorMap('detector_map.csv')
    cl_indicies = [0,1,2,3,4,5,6,7,8]
    row_index = 2 # list(range(24))
    bath_temp_index=0

    # circuit parameters
    iv_circuit = IVCircuit(rfb_ohm=5282.0+50.0,
                           rbias_ohm=10068.0,
                           rsh_ohm=0.0662,
                           rx_ohm=0,
                           m_ratio=8.259,
                           vfb_gain=1.017/(2**14-1),
                           vbias_gain=6.5/(2**16-1))
    df = IVColdloadSweepAnalyzer(filename_json,dm.map_dict,iv_circuit) #df is the main "data format" of the coldload temperature sweep
    dacs,fb = df.get_cl_sweep_dataset_for_row(row_index=row_index,bath_temp_index=bath_temp_index,cl_indicies=cl_indicies)
    iva = IVColdloadAnalyzeOneRow(dacs,fb,
                                  cl_temps_k=list(np.array(df.set_cl_temps_k)[cl_indicies]),# put in measured values here!
                                  bath_temp_k=df.set_bath_temps_k[bath_temp_index],
                                  device_dict=None,
                                  iv_circuit=None,
                                  passband_dict=None)
    x = iva.dacs
    n,m = np.shape(iva.fb)
    ivc = IVClean()
    for ii in range(m):
        y = iva.fb[:,ii]
        #y_smooth = smooth(y)
        #plt.plot(x,y)
        #plt.plot(x,y_smooth)
        #plt.show()
        dex = IVClean().find_bad_data_index(x,y,showplot=True)
    #y=y*-1
    #x=x[::-1];y=y[::-1]

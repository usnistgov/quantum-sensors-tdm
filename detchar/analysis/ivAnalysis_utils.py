''' ivAnalysis_utils.py '''

# NOTES
# IVTempSweepData ought to have the coldload setpoint.
# column name for IVColdloadAnalzyeOneRow?
# units for plots?
# plot ranges
# deal with crap rn_frac cuts

import detchar
from detchar.iv_data import IVCircuit
from detchar import IVColdloadSweepData
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k,h,c
from scipy.integrate import quad, simps

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
            P = quad(self.Pnu_thermal,nu1,nu2,args=(T))[0] # toss the error
    except: # case for arbitrary passband shape F
        N = len(F)
        nu = np.linspace(nu1,nu2,N)
        integrand = self.Pnu_thermal(nu,T)*F
        P = simps(integrand,nu)
    return P

class IVColdloadAnalyzeOneRow():
    ''' Analyze a set of IV curves for a single detector taken at multiple
        coldload temperatures and a single bath temperature
    '''

    def __init__(self,dac_values,fb_array,cl_temps_k,bath_temp_k,device_dict=None,iv_circuit=None,passband=None):
        self.dacs = dac_values
        self.n_dac_values = len(self.dacs)
        self.fb = fb_array # NxM array of feedback values.  Columns are per coldload temperature
        self.fb_align = None
        self.cl_temps_k = cl_temps_k
        self.bath_temp_k = bath_temp_k
        self.det_name, self.row_name = self._handle_device_dict(device_dict)
        self.n_dac_values, self.n_cl_temps = np.shape(self.fb)
        self.f_edges_ghz, self.passband_sim_ghz = self._handle_passband() # for efficiency calculation

        # fixed globals
        self.n_normal_pts=10 # number of points for normal branch fit
        self.use_ave_offset=True # use a global offset to align fb, not individual per curve
        self.rn_fracs = rn_fracs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] # slices in Rn space to compute delta Ps
        if iv_circuit==None:
            self.to_physical_units = False
        else:
            self.to_physical_units = True

        # do analysis
        self.p_tophat, self.p_sim = self.get_predicted_thermal_power()
        self.v,self.i,self.p,self.r = self.get_vipr(showplot=False)
        self.ro = self.r / self.r[0,:]
        self.p_at_rnfrac = self.get_value_at_rn_frac(self.rn_fracs,self.p,self.ro)

    def _handle_passband(self,passband):
        if passband==None:
            f_edges_ghz = None ; passband_sim_ghz = None
        else:
            assert type(passband) == dict, ('passband must be of type dictionary')
            keys = passband.keys()
            if 'f_edges_ghz' in keys:
                f_edges_ghz = passband['f_edges_ghz']
            else:
                f_edges_ghz = None
            if 'passband_sim_ghz' in keys:
                passband_sim_ghz = passband['passband_sim_ghz']
            else:
                passband_sim_ghz = None
        return f_edges_ghz, passband_sim_ghz

    def get_predicted_thermal_power(self):
        ''' calculate thermal power from coldload in two cases.
            1) relative to tophat passband response between f1 and f2 defined in self.f_edges_ghz
            2) relative to simulated absolute bandpass response defined in self.passband.ghz
        '''
        if self.f_edges_ghz != None:
            p_tophat = []
            for ii in range(self.n_cl_temps):
                p_tophat.append(thermalPower(self.f_edges_ghz[0]*1e9,self.f_edges_ghz[1]*1e9,T=self.n_cl_temps[ii],F=None)
        else:
            p_tophat = None
        if self.passband_sim_ghz != None:
            p_sim = []
            for ii in range(self.n_cl_temps):
                p_sim.append(thermalPower(1,10e12,T=self.n_cl_temps[ii],F=self.passband_sim_ghz)
        else:
            p_sim = None

        return p_tophat, p_sim





        #     Ptherm = []
        #     for ii in range(self.n_sweeps):
        #         Ptherm.append(self.thermalPower(nu1=nu1*1e9,nu2=nu2*1e9,T=self.sweepTempsMeasured[ii,0],F=None))
        #     Ptherm=np.array(Ptherm)
        #     dPtherm = Ptherm - Ptherm[min_dex]
        #     plt.plot(self.sweepTempsMeasured[:,0]-self.sweepTempsMeasured[min_dex,0],dPtherm,'k-')

    def _handle_device_dict(self,device_dict):
        if device_dict==None:
            det_name = 'unknown'; row_name = 'unknown'; f_edges = None
        else:
            row_name = device_dict.keys()[0]; det_name = device_name[row_name]
            row_name = device_name[0]; det_name = device_name[1]
        return det_name, row_name

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
            v,i = iv_circuit.to_physical_units(self.dacs,self.fb_align)
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

    def plot_raw(self,fb_align_dc_level=True,fig_num=1):
        plt.figure(fig_num)
        for ii, cl_temp in enumerate(self.cl_temps_k):
            if fb_align_dc_level:
                dy = self.fb[0,ii]-self.fb[0,0]
            else: dy=0
            plt.plot(self.dacs, self.fb[:,ii]-dy)
        plt.xlabel('DAC values')
        plt.ylabel('Feedback values')
        plt.title(self.det_name+', '+self.row_name+' , Tb = %.1f mK'%(self.bath_temp_k*1000))
        plt.legend((self.cl_temps_k),loc='upper right')

    def plot_vipr(self,data_list=None,fig_num=1):
        if data_list==None:
            v=self.v; i=self.i; p=self.p; r=self.r
        else:
            v=data_list[0]; i=data_list[1]; p=data_list[2]; r=data_list[3]

        # fig 1, 2x2 of converted IV
        fig1, ax = plt.subplots(nrows=2,ncols=2,sharex=False,figsize=(12,8))
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

        fig1.suptitle(self.det_name+', '+self.row_name+' , Tb = %.1f mK'%(self.bath_temp_k*1000))
        ax[3].legend(tuple(self.cl_temps_k))
        #plt.show()

    def plot_pr(self,rn_fracs,p_at_rnfrac,p,ro,fig_num=1):
        pPlot = self.get_value_at_rn_frac([0.999],arr=p,ro=ro)

        # FIG1: P versus R/Rn
        fig1 = plt.figure(fig_num)
        plt.plot(ro, p,'-') # plots for all Tbath
        plt.plot(rn_fracs,p_at_rnfrac,'ro')
        plt.xlim((0,1.1))
        plt.ylim((np.min(p_at_rnfrac[~np.isnan(p_at_rnfrac)])*0.9,1.25*np.max(pPlot[~np.isnan(pPlot)])))
        plt.xlabel('Normalized Resistance')
        plt.ylabel('Power')
        #plt.title(plottitle)
        plt.legend((self.cl_temps_k))
        plt.grid()

    def plot_pt(self,rn_fracs,p_at_rnfrac,p,ro,fig_num=1):
        # power plateau (evaluated at each rn_frac) versus T_cl
        fig1 = plt.figure(fig_num)
        for ii in range(len(rn_fracs)):
            plt.plot(self.cl_temps_k,p_at_rnfrac[ii,:],'o-')
        plt.xlabel('T$_{cl}$ (K)')
        plt.ylabel('TES power plateau')
        plt.legend((rn_fracs))
        #plt.title(plottitle)
        plt.grid()

    def plot_pt_delta(self,rn_fracs,p_at_rnfrac,p,ro,fig_num=1):
        ''' plot change in saturation power relative to minimum coldload temperature '''
        fig1 = plt.figure(fig_num)
        min_dex = np.argmin(self.cl_temps_k)
        for ii in range(len(rn_fracs)):
            plt.plot(np.array(self.cl_temps_k)-self.cl_temps_k[min_dex],p_at_rnfrac[ii,min_dex]-p_at_rnfrac[ii,:],'o-')

        # if type(nu)==str:
        #     if nu=='useRowMap':
        #         if 'nu' in self.rowMap['Row%02d'%row].keys():
        #             nu1,nu2 = self.rowMap['Row%02d'%row]['nu']
        #             doPrediction=True
        #         else:
        #             doPrediction=False
        # elif type(nu)==list:
        #     nu1,nu2 = nu
        #     doPrediction=True
        # elif type(nu)==type(None):
        #     doPrediction=False
        # if doPrediction: # calculate the predicted power from blackbody and plot it.
        #     Ptherm = []
        #     for ii in range(self.n_sweeps):
        #         Ptherm.append(self.thermalPower(nu1=nu1*1e9,nu2=nu2*1e9,T=self.sweepTempsMeasured[ii,0],F=None))
        #     Ptherm=np.array(Ptherm)
        #     dPtherm = Ptherm - Ptherm[min_dex]
        #     plt.plot(self.sweepTempsMeasured[:,0]-self.sweepTempsMeasured[min_dex,0],dPtherm,'k-')
        plt.xlabel('T$_{cl}$ - %.1f K'%self.cl_temps_k[min_dex])
        plt.ylabel('P$_o$ - P')
        plt.legend((rn_fracs))
        plt.grid()
        #plt.title(plottitle)

        # # FIG4: efficiency versus delta T (only if nu provided)
        # if doPrediction:
        #     fig4 = plt.figure(4)
        #     for ii in range(len(fracRns)):
        #         eta = (p_at_fracR[ii,min_dex]-p_at_fracR[ii,:])/dPtherm
        #         plt.plot(self.sweepTempsMeasured[:,0]-self.sweepTempsMeasured[min_dex,0],eta,'o-')
        #     plt.xlabel('T$_{cl}$ - %.1f K'%self.sweepTemps[min_dex])
        #     plt.ylabel('$\eta$')
        #     plt.legend((fracRns))
        #     plt.grid()
        #     plt.title(plottitle)
        #     print(plottitle,' \eta=', eta[1])
        #
        # if savePlots:
        #     fig1.savefig('RowIndex%02d_1_rp.png'%row)
        #     fig2.savefig('RowIndex%02d_2_pt.png'%row)
        #     fig3.savefig('RowIndex%02d_3_dpdt.png'%row)
        #     if doPrediction:
        #         fig4.savefig('RowIndex%02d_4_eta.png'%row)
        #     plt.close('all')
        # if displayPlots:
        #     plt.show()

    def plot_full_analysis(self):
        self.plot_raw(True,fig_num=1) # raw
        self.plot_vipr(data_list=None,fig_num=2) # 2x2 of converted data
        self.plot_pt(self.rn_fracs,self.p_at_rnfrac,self.p,self.ro,fig_num=3)
        self.plot_pt_delta(self.rn_fracs,self.p_at_rnfrac,self.p,self.ro,fig_num=4)
        plt.show()



    def analyzeColdload(self,col,row,static_temp,sweep_temps='all', threshold=10,
                        fracRns=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                        nu=None,savePlots=False,displayPlots=True):
        ''' Analyze set of IV curves swept through coldload temperature.
            Produces plots:
            1) Ro versus P with fracRns cuts shown
            2) TES power plateau (P) versus cold load temperature (1 curve per R/Rn frac)
            3) Delta P versus Delta T (1 curve per R/Rn frac)
            4) \eta versus Delta T (1 curve per R/Rn frac)

            Input:
            col: column index (in EasyClient returned data array)
            row: row index (in EasyClient returned data array)
            static_temp: bath_temperature (K)
            sweep_temps: list of coldload temperatures to include IVs for.  'all' is an option
            threshold: cut data that is threshold x dVfb in the normal branch
            fracRns: R/Rn cuts to evaluate power plateaus
            nu: [f_start,f_end] (in GHz) used to calculate radiative power from coldload
                if nu=None, no prediction is given, and the 4th plot is not made
                if nu='useRowMap', the frequency range listed in self.rowMap['RowXX']['nu'] will be used.

        '''
        A,B,C,D = self.checkMeasuredTemperatures(tolerance=0.001,verbose=False,Tbbsensor=0)
        result,v,i,r,p = self.get_virp(0,row,.13,'all','coldload') # make main data vectors
        self.findBadDataIndex(threshold=threshold,PLOT=False)      # remove bad data
        self.removeBadData(False)
        p_at_fracR = self.getFracRn(fracRns,arr=self.p,ro=self.ro) # len(fracRn) x len(sweep_temps) array

        pPlot = self.getFracRn([0.999],arr=self.p,ro=self.ro)

        plottitle = 'Row Index%02d'%row
        if type(self.rowMap)==dict:
            plottitle=plottitle+'; '+ self.rowMap['Row%02d'%row]['detector_name']

        # FIG1: P versus R/Rn
        fig1 = plt.figure(1)
        plt.plot(self.ro,self.p,'-') # plots for all Tbath
        plt.plot(fracRns,p_at_fracR,'ro')
        plt.xlim((0,1.1))
        plt.ylim((np.min(p_at_fracR[~np.isnan(p_at_fracR)])*0.9,np.max(pPlot[~np.isnan(pPlot)])))
        plt.xlabel('Normalized Resistance')
        plt.ylabel('Power (%s)'%self.p_units)
        plt.title(plottitle)
        plt.legend((self.sweepTemps))
        plt.grid()

        # FIG2: ES power plateau versus T_cl
        fig2 = plt.figure(2)
        for ii in range(len(fracRns)):
            plt.plot(self.sweepTempsMeasured[:,0],p_at_fracR[ii,:],'o-')
        plt.xlabel('T$_{cl}$ (K)')
        plt.ylabel('TES power plateau (%s)'%self.p_units)
        plt.legend((fracRns))
        plt.title(plottitle)
        plt.grid()

        # FIG3: change in saturation power relative to minimum coldload temperature
        fig3 = plt.figure(3)
        min_dex = np.argmin(self.sweepTempsMeasured[:,0])
        for ii in range(len(fracRns)):
            plt.plot(self.sweepTempsMeasured[:,0]-self.sweepTempsMeasured[min_dex,0],p_at_fracR[ii,min_dex]-p_at_fracR[ii,:],'o-')

        if type(nu)==str:
            if nu=='useRowMap':
                if 'nu' in self.rowMap['Row%02d'%row].keys():
                    nu1,nu2 = self.rowMap['Row%02d'%row]['nu']
                    doPrediction=True
                else:
                    doPrediction=False
        elif type(nu)==list:
            nu1,nu2 = nu
            doPrediction=True
        elif type(nu)==type(None):
            doPrediction=False

        if doPrediction: # calculate the predicted power from blackbody and plot it.
            Ptherm = []
            for ii in range(self.n_sweeps):
                Ptherm.append(self.thermalPower(nu1=nu1*1e9,nu2=nu2*1e9,T=self.sweepTempsMeasured[ii,0],F=None))
            Ptherm=np.array(Ptherm)
            dPtherm = Ptherm - Ptherm[min_dex]
            plt.plot(self.sweepTempsMeasured[:,0]-self.sweepTempsMeasured[min_dex,0],dPtherm,'k-')
        plt.xlabel('T$_{cl}$ - %.1f K'%self.sweepTemps[min_dex])
        plt.ylabel('P$_o$ - P (%s)'%self.p_units)
        plt.legend((fracRns))
        plt.grid()
        plt.title(plottitle)

        # FIG4: efficiency versus delta T (only if nu provided)
        if doPrediction:
            fig4 = plt.figure(4)
            for ii in range(len(fracRns)):
                eta = (p_at_fracR[ii,min_dex]-p_at_fracR[ii,:])/dPtherm
                plt.plot(self.sweepTempsMeasured[:,0]-self.sweepTempsMeasured[min_dex,0],eta,'o-')
            plt.xlabel('T$_{cl}$ - %.1f K'%self.sweepTemps[min_dex])
            plt.ylabel('$\eta$')
            plt.legend((fracRns))
            plt.grid()
            plt.title(plottitle)
            print(plottitle,' \eta=', eta[1])

        if savePlots:
            fig1.savefig('RowIndex%02d_1_rp.png'%row)
            fig2.savefig('RowIndex%02d_2_pt.png'%row)
            fig3.savefig('RowIndex%02d_3_dpdt.png'%row)
            if doPrediction:
                fig4.savefig('RowIndex%02d_4_eta.png'%row)
            plt.close('all')
        if displayPlots:
            plt.show()



class IVColdloadSweepAnalyzer():
    ''' Class to assess data quality of coldload sweep '''
    # plot cold load measured temperatures
    # plot measured bath temperatures
    def __init__(self,filename_json):
        self.df = IVColdloadSweepData.from_file(filename_json)
        self.data = self.df.data
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
        cl_temp_list = self._package_cl_temp_to_list()
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

    def get_cl_temp_index(self,cl_temp):
        print('to be written')

if __name__ == "__main__":
    #filename_json = '/home/pcuser/data/lbird/20201202/lbird_hftv0_coldload_sweep.json'
    filename_json = 'lbird_hftv0_coldload_sweep.json'
    df = IVColdloadSweepAnalyzer(filename_json) #df is the main "data format" of the coldload temperature sweep
    #df.plot_measured_bath_temps() # first check if all measurements taken at intended temperatures
    #plt.show()

    # now analyze one row, one bath temperature of IV coldload sweep
    row_index=1
    device_name = ['Row%02d'%row_index, 'dev name']
    rn_fracs = np.linspace(.1,.9,9)

    dacs,fb = df.get_cl_sweep_dataset_for_row(row_index=row_index,bath_temp_index=1,cl_indicies=list(range(df.n_cl_temps-1)))
    iv_circuit = IVCircuit(rfb_ohm=5282.0+50.0,
                           rbias_ohm=10068.0,
                           rsh_ohm=0.0662,
                           rx_ohm=0,
                           m_ratio=8.259,
                           vfb_gain=1.017/(2**14-1),
                           vbias_gain=6.5/(2**16-1))

    ivcl_onerow = IVColdloadAnalyzeOneRow(dacs,fb,df.set_cl_temps_k[0:-1],df.set_bath_temps_k[1],device_name,iv_circuit)
    ivcl_onerow.plot_full_analysis()
    #ivcl_onerow.plot_p_vs_r(rn_fracs)
    #plt.show()

    # for ii in range(df.n_rows):
    #     df.plot_cl_temp_sweep_for_row(row_index=ii,bath_temp_index=1)

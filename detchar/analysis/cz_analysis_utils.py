''' cz_analysis_utils.py 

software to analyze complex impedance data stored in a CzData class

@author JH, 5/2024

classes: 
CzDataExplore: quicklook to understand what a datafile is and plot raw response
CzSingle: Analyze comnplex Z for a single freuquency sweep
'''

import matplotlib.pyplot as plt
import numpy as np 
from scipy.optimize import leastsq
from datetime import datetime 
import sys 
sys.path.append('/Users/hubmayr/nistgit/nistqsptdm/')
from detchar.iv_data import CzData


## main plotting function used in multiple classes
def plot_quadrature_detection(f,iq, label=None,semilogx=True,fig=None,ax=None):
    ''' Plot data in 2x2 plot of I,Q,amp, phase versus frequency    
        for rows, temp_indices, and bias_indices.  All of these input fields can either be an integer or a list.

        iq has shape (num_freq, 2)
    '''
    if not fig:
        fig, ax = plt.subplots(nrows=2,ncols=2,sharex=False,figsize=(12,8))
    
    if semilogx:
        ax[0][0].semilogx(f,iq[:,0],'o-')
        ax[0][1].semilogx(f,iq[:,1],'o-')
        ax[1][0].semilogx(f,np.sqrt(iq[:,0]**2+iq[:,1]**2),'o-')
        ax[1][1].semilogx(f,np.unwrap(np.arctan2(iq[:,1],iq[:,0])),'o-',label=label)
    else:
        ax[0][0].plot(f,iq[:,0],'o-')
        ax[0][1].plot(f,iq[:,1],'o-')
        ax[1][0].plot(f,np.sqrt(iq[:,0]**2+iq[:,1]**2),'o-')
        ax[1][1].plot(f,np.unwrap(np.arctan2(iq[:,1],iq[:,0])),'o-',label=label)

    # axes labels
    ax[0][0].set_ylabel('I')
    ax[0][1].set_ylabel('Q')
    ax[1][0].set_ylabel('Amp')
    ax[1][1].set_ylabel('Phase')
    ax[1][0].set_xlabel('Freq (Hz)')
    ax[1][1].set_xlabel('Freq (Hz)')
    return fig,ax

class CzDataExplore():
    ''' Explore a complex impedance data set stored in the CzData format.  No fitting or detailed analysis here. '''
    def __init__(self,czdata_filename):
        self.czdata_filename = czdata_filename
        self.cz = CzData.from_file(czdata_filename)
        self.data = self.cz.data # list of list of SineSweepData instances.  
                                 # 1st index is temperature, second index is voltage bias
        self.num_temp = len(self.cz.temp_list_k)
        self.measured_temps = self._get_measured_temperatures_()
        self.row_order = self.cz.data[0][0].row_order

    def print_metadata(self):
        print('Complex impedance data file %s has the following attributes'%self.czdata_filename)
        print('Data start / stop: ',datetime.utcfromtimestamp(self.data[0][0].pre_time_epoch_s).isoformat(), 
                datetime.utcfromtimestamp(self.data[-1][-1].post_time_epoch_s).isoformat())
        print('Rows: ',self.cz.data[0][0].row_order)
        print('Substrate Temperatures (K): ',self.cz.temp_list_k)
        for ii,temp in enumerate(self.cz.temp_list_k):
            print('Detector biases at temperature %dmK: '%(temp*1000), self.cz.db_list[ii])
        print('Modulation frequencies: %.1f Hz -- %.1f Hz in %d steps'
               %(min(self.cz.data[0][0].frequency_hz),max(self.cz.data[0][0].frequency_hz),len(self.cz.data[0][0].frequency_hz)))

    def _get_measured_temperatures_(self):
        measured_temps = []
        for ii in range(self.num_temp):
            foo = []
            for jj in range(len(self.data[ii])):
                foo.append([self.data[ii][jj].pre_temp_k,self.data[ii][jj].post_temp_k])
            measured_temps.append(foo)
        return measured_temps

    # def plot_measured_temperatures(self):
    #     print('to be written')

    def plot_raw(self,row_index,temp_indices=None,bias_indices=None,semilogx=True):
        ''' Plot raw data in 2x2 plot of I,Q,amp, phase versus frequency    
            for rows, temp_indices, and bias_indices.  All of these input fields can either be an 
            integer or a list.
        '''

        if not temp_indices: temp_indices=list(range(self.num_temp))

        for ii in temp_indices: # loop over temp
            fig, ax = plt.subplots(nrows=2,ncols=2,sharex=False,figsize=(12,8))
            fig2, ax2 = plt.subplots(1,1)

            fig.suptitle('Temperature = %.1f mK'%(self.cz.temp_list_k[ii]*1000))
            fig2.suptitle('I-Q, Temperature = %.1f mK'%(self.cz.temp_list_k[ii]*1000))
            if not bias_indices: bias_indices=range(len(self.cz.db_list[ii]))
            for jj in bias_indices: # loop over detector bias
                ss = self.data[ii][jj]
                db = self.cz.db_list[ii][jj]
                n_freq,n_row,foo = np.shape(ss.iq_data)
                iq_data = np.array(ss.iq_data)

                # I,Q,amp,phase versus frequency
                plot_quadrature_detection(f=ss.frequency_hz,iq=iq_data[:,row_index,:],
                                          label='Tb=%dmK, b=%d'%(self.cz.temp_list_k[ii]*1000,db),
                                          semilogx=semilogx,fig=fig,ax=ax)

                # plot I vs Q as second plot
                ax2.plot(iq_data[:,row_index,0],iq_data[:,row_index,1],'o-',label='Tb=%dmK, b=%d'%(self.cz.temp_list_k[ii]*1000,db))

        ax2.set_xlabel('I')
        ax2.set_ylabel('Q')
        ax2.set_aspect('equal','box')
        ax2.legend()

class CzSingle():
    ''' Base class for single complex impedance measurement.  

        The global Z is the tes impedance.  By default it is normalized by the 
    '''
    def __init__(self,freq_hz,iq_data,iq_data_sc,amp,amp_sc,rfb_ohm=1, rfg_ohm=1,
                      rsh_ohm=1,rbias_ohm=207.0,m_ratio=8,fb_dac_per_v=16383,vfb_gain=1.017):
        # make input accessible throughout class
        self.f = freq_hz
        self.iq_data_raw = np.copy(iq_data)
        self.iq_data_sc_raw = np.copy(iq_data_sc) # IQ data in the superconducting branch
        self.amp_volt=amp # amplitude of CZ stimulus from function generator when in superconducting transition state
        self.amp_volt_sc=amp_sc # ampltude of CZ stimulus from function generator when in superconducting state
        self.rfb_ohm=rfb_ohm
        self.rfg_ohm=rfg_ohm # resistance in series with the function generator
        self.rbias_ohm = rbias_ohm # series resistance in detector bias line 
        self.rsh_ohm = rsh_ohm
        self.m_ratio=m_ratio
        self.fb_dac_per_v=fb_dac_per_v
        self.vfb_gain=vfb_gain
        self.arbs_to_A = self.vfb_gain / (self.fb_dac_per_v*self.rfb_ohm*self.m_ratio) # convert raw to current in amps
        self.polarity = self._get_polarity_()

        self.iq_data = self.to_physical_units(self.iq_data_raw)
        self.iq_data_sc = self.to_physical_units(self.iq_data_sc_raw)
        self.Z = self.getZ(self.rsh_ohm)

    def _get_polarity_(self):
        ''' Due to choice of wirebonds, signal can be inverted.  
            This magic method checks the sign of the lowest frequency 
            in-phase component of the superconducting response 
        '''
        if self.iq_data_sc_raw[np.argmin(self.f),0] < 0: # if lowest freq component of I is negative
            return -1
        else:
            return 1

    def to_physical_units(self,iq):
        ''' converts the quadrature data from arbs to current '''
        iqp=np.copy(iq)
        iqp=iqp*self.polarity # flip if IV curve negative
        iqp[:,1]=iqp[:,1]*-1 # flip Q component due to my software lock in convention 
                             # (the 90deg phase shifted copy of reference seems opposite convension)
        iqp = iqp*self.arbs_to_A
        return iqp
    
    def getZ(self,rsh_ohm=1):
        ''' returns Z, the complex impedance.  The algorithm naturally returns Z/Rsh.  Provide a 
            shunt resistance (Rsh) other than 1 to rescale to physical units 
        '''

        A = (self.iq_data[:,0]+1j*self.iq_data[:,1]) * (self.amp_volt/(self.rfg_ohm+self.rbias_ohm))**-1 #/self.iq_data_sc[np.argmin(self.f),0] # normalize to the current measured in the superconducting branch at lowest frequency
        B = (self.iq_data_sc[:,0]+1j*self.iq_data_sc[:,1]) * (self.amp_volt_sc/(self.rfg_ohm+self.rbias_ohm))**-1 #/self.iq_data_sc[np.argmin(self.f),0]
        Z = A**-1-B**-1
        foo = np.zeros([len(Z),2])
        foo[:,0] = Z.real*rsh_ohm
        foo[:,1] = Z.imag*rsh_ohm
        return foo 

    def _make_mask(self,fmin=None,fmax=None):
        if not fmin: fmin=min(self.f)
        if not fmax: fmax=max(self.f)
        mask = (np.array(self.f)>=fmin) & (np.array(self.f)<=fmax)
        return mask

    def plot_complex_plane(self,fig=None,ax=None,label=None,fmin=None,fmax=None):
        if not fig: fig,ax=plt.subplots()
        mask=self._make_mask(fmin,fmax)
        ax.set_aspect('equal', adjustable='box')
        ax.plot(self.Z[mask,0],self.Z[mask,1],'o',label=label)
        ax.set_xlabel('Re(Z)')
        ax.set_ylabel('Im(Z)')
        ax.set_aspect('equal')
        return fig,ax

    def plot(self,fmt='raw',fig=None,ax=None,semilogx=True,label=None):
        if not fig: fig, ax = plt.subplots(nrows=2,ncols=2,sharex=False,figsize=(12,8))
        if fmt=='raw':
            datas = [self.iq_data_raw,self.iq_data_sc_raw]
        elif fmt=='phys':
            datas = [self.iq_data,self.iq_data_sc]
        elif fmt=='trans':
            datas = [self.iq_data/(self.amp_volt/self.rfg_ohm),self.iq_data_sc/(self.amp_volt_sc/self.rfg_ohm)]
        elif fmt=='1/trans':
            datas = [(self.amp_volt/self.rfg_ohm)/self.iq_data,(self.amp_volt_sc/self.rfg_ohm)/self.iq_data_sc]
        elif fmt=='Z':
            datas=[self.Z]
        else:
            assert False, print('format type unknown: ',fmt)
        
        for data in datas:
            plot_quadrature_detection(f=self.f,iq=data,label=label,semilogx=semilogx,fig=fig,ax=ax)
        fig.suptitle(fmt)
        return fig, ax

    def _guess_tau(self,f_max=1e3):
        ind = np.argmin(self.Z[np.array(self.f)<f_max,1]) # minimum of Q 
        return (2*np.pi*self.f[ind])**-1
    
    def fitQ(self,plot=False,fig=None,ax=None,fmin=None,fmax=None,label=None,semilogx=True):
        mask=self._make_mask(fmin,fmax)
        # In fit function below, p[0] = -2\pi\tau_o LG/(1-LG)^2, the overall amplitude, and p[1] = (2\pi * \tau_I)^2 = (2\pi\tau_o/(1-LG))^2
        # as such the ratio of the fit paraemters p[0]/p[1] = -LG/2\pi\tau_o
        fit_func = lambda p,f: p[0]*f/(1+p[1]*f**2) # form is A/1+B*x^2, in which A and B are the free parameters
        optimize_func = lambda p,f,d: d-fit_func(p,f) # data - fit

        tau_est = self._guess_tau()
        p = leastsq(optimize_func, [1,tau_est],args=(np.array(self.f)[mask],self.Z[mask,1]))[0]

        if plot:
            xfit=np.logspace(np.log10(min(self.f)),np.log10(max(self.f)),1000)
            yfit=fit_func(p,xfit)
            if not fig: fig,ax=plt.subplots(2,1)
            if semilogx:
                ax[0].semilogx(np.array(self.f)[mask],self.Z[mask,1],'o',label=label)
                ax[0].semilogx(xfit,yfit,'k--')
                ax[1].semilogx(np.array(self.f)[mask],self.Z[mask,1]-fit_func(p,np.array(self.f)[mask]),'o')
            else:
                ax[0].plot(np.array(self.f)[mask],self.Z[mask,1],'o',label=label)
                ax[0].plot(xfit,yfit,'k--')
                ax[1].plot(np.array(self.f)[mask],self.Z[mask,1]-fit_func(p,np.array(self.f)[mask]),'o')
            ax[0].set_ylabel('Q')    
            ax[1].set_xlabel('Frequency (Hz)')
            ax[1].set_ylabel('Data - fit')
            return p, fig, ax
        else:
            return p
  
class CzSuperConductingBranch():
    ''' Class for superconducting branch '''
    def __init__(self,czdata_filename):
        self.czdata_filename = czdata_filename
        self.cz = CzData.from_file(czdata_filename)
        self.data = self.cz.data # list of list of SineSweepData instances.  
                                 # 1st index is temperature, second index is voltage bias
        self.f = self.data[0][0].frequency_hz
        self.num_temp = len(self.cz.temp_list_k)
        self.measured_temps = self._get_measured_temperatures_()
        self.row_order = self.cz.data[0][0].row_order
    
    def print_metadata(self):
        print('Complex impedance data file %s has the following attributes'%self.czdata_filename)
        print('Data start / stop: ',datetime.utcfromtimestamp(self.data[0][0].pre_time_epoch_s).isoformat(), 
                datetime.utcfromtimestamp(self.data[-1][-1].post_time_epoch_s).isoformat())
        print('Rows: ',self.cz.data[0][0].row_order)
        print('Substrate Temperatures (K): ',self.cz.temp_list_k)
        for ii,temp in enumerate(self.cz.temp_list_k):
            print('Detector biases at temperature %dmK: '%(temp*1000), self.cz.db_list[ii])
        print('Modulation frequencies: %.1f Hz -- %.1f Hz in %d steps'
               %(min(self.cz.data[0][0].frequency_hz),max(self.cz.data[0][0].frequency_hz),len(self.cz.data[0][0].frequency_hz)))

    def _get_measured_temperatures_(self):
        measured_temps = []
        for ii in range(self.num_temp):
            foo = []
            for jj in range(len(self.data[ii])):
                foo.append([self.data[ii][jj].pre_temp_k,self.data[ii][jj].post_temp_k])
            measured_temps.append(foo)
        return measured_temps

    def plot(self,row_index=0,temp_index=0,bias_indices=None):
        if not bias_indices: bias_indices=range(len(self.cz.db_list[temp_index]))
        fig=ax=None 
        for jj in bias_indices: # loop over detector bias
            ss = self.data[temp_index][jj]
            db = self.cz.db_list[temp_index][jj]
            n_freq,n_row,foo = np.shape(ss.iq_data)
            iq = np.array(ss.iq_data)
            fig,ax=plot_quadrature_detection(self.f,iq,row_index,label=db,fig=fig,ax=ax)

    def get_mean_and_std(self,temp_index,debug=True):
        n_freq,n_row,n_quad = np.shape(self.data[temp_index][0].iq_data)
        iq_all = np.zeros((n_freq,n_row,n_quad,len(self.data[temp_index])))
        for ii,ss in enumerate(self.data[temp_index]):
            iq_all[:,:,:,ii] = ss.iq_data 

        iq_all_mean  = iq_all.mean(axis=-1)
        iq_all_std  = iq_all.std(axis=-1)

        return iq_all, iq_all_mean,iq_all_std

        # iq_all_mean_subtracted = np.zeros(np.shape(iq_all))
        # for ii in range(len(self.data[temp_index])):
        #     iq_all_mean_subtracted[:,:,:,ii]=iq_all[:,:,:,ii]-iq_all_mean

        # print(np.where(iq_all_mean_subtracted>5*iq_all_std))

        # if debug:
        #     fig,ax=plt.subplots()
        #     for ii in range(len(self.data[temp_index])):
        #         ax.plot(self.f,iq_all[:,0,0,ii],'.')
        #     ax.errorbar(self.f,iq_all_mean[:,0,0],yerr=iq_all_std[:,0,0],fmt='o-')

        #     fig,ax=plt.subplots()
        #     ax.plot(self.f,iq_all_mean_subtracted[:,0,0],'o-')

        

    
if __name__ == '__main__':
    # cze = CzDataExplore('/Users/hubmayr/projects/uber_omt/data/velma_uber_omt/20240429/colA_cz_20240517_04.json')
    # cze.print_metadata()
    # ddex=1 # detector index
    # mdex=15 # bias index
    # sc_dex = np.where(np.array(cze.cz.db_list[0])==0)[0][0]
    # czs = CzSingle(freq_hz=cze.data[0][mdex].frequency_hz,iq_data=np.array(cze.data[0][mdex].iq_data)[:,ddex,:],
    #                iq_data_sc=np.array(cze.data[0][sc_dex].iq_data)[:,ddex,:],amp=cze.data[0][mdex].amp_volt,
    #                amp_sc = cze.data[0][sc_dex].amp_volt,
    #                rfb_ohm = cze.data[0][0].rfb_ohm+50,
    #                rfg_ohm = cze.data[0][0].rfg_ohm,
    #                rsh_ohm = 450e-6 )
    # for fmt in ['raw','phys','trans','Z']:
    #     czs.plot(fmt)
    # czs.plot_complex_plane()
    # plt.show()

    czsc = CzSuperConductingBranch('/Users/hubmayr/projects/uber_omt/data/velma_uber_omt/20240429/colA_cz_20240517_05.json')
    #czsc.plot()
    czsc.get_average(0)
    plt.show()
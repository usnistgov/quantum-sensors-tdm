''' cz_analysis_utils.py 

software to analyze complex impedance data stored in a CzData class

@author JH, 5/2024
'''

import matplotlib.pyplot as plt
import numpy as np 
from scipy.optimize import leastsq
from datetime import datetime 
import sys 
sys.path.append('/Users/hubmayr/nistgit/nistqsptdm/')
from detchar.iv_data import CzData

class CzSingle():
    ''' Base class for single complex impedance measurement '''
    def __init__(self,freq_hz,iq_data,iq_data_sc,amp,amp_sc,rfb_ohm=1, rfg_ohm=1,
                      rsh_ohm=1,m_ratio=8,fb_dac_per_v=16383,vfb_gain=1.017):
        # make input accessible throughout class
        self.f = freq_hz
        self.iq_data_raw = iq_data
        self.iq_data_sc_raw = iq_data_sc # IQ data in the superconducting branch
        self.amp=amp
        self.amp_sc=amp_sc
        self.rfb_ohm=rfb_ohm
        self.rfg_ohm=rfg_ohm # resistance in series with the function generator
        self.rsh_ohm = rsh_ohm
        self.m_ratio=m_ratio
        self.fb_dac_per_v=fb_dac_per_v
        self.vfb_gain=vfb_gain

        self.Io = self.amp/self.rfg_ohm # amplitude of current stimulus for non SC state
        self.Io_sc = self.amp_sc/self.rfg_ohm # amplitude of current stimulus for SC state

        self.polarity = self._get_polarity_()
        self.iq_data = self.to_physical_units(self.iq_data_raw)
        self.iq_data_sc = self.to_physical_units(self.iq_data_sc_raw)

        self.Z = self.getZ()

    def _get_polarity_(self):
        ''' Due to choice of wirebonds, signal can be inverted.  
            This magic method checks the sign of the lowest frequency 
            in-phase component of the superconducting response 
        '''
        if self.iq_data_sc_raw[np.argmin(self.f),0] < 1:
            return -1
        else:
            return 1

    def to_physical_units(self,iq):
        ''' converts the quadrature data from arbs to current '''
        iq[:,0]=iq[:,0]*self.polarity
        iq = iq/self.fb_dac_per_v*self.vfb_gain/self.rfb_ohm/self.m_ratio
        return iq
    
    def getZ(self,fmt=None):
        ''' returns Z/Rsh '''
        A = self.iq_data[:,0]+1j*self.iq_data[:,1]
        B = self.iq_data_sc[:,0]+1j*self.iq_data_sc[:,1]
        Z = self.rsh_ohm*((A/self.Io)**-1-(B/self.Io_sc)**-1)
        foo = np.zeros([len(Z),2])
        foo[:,0] = Z.real 
        foo[:,1] = Z.imag
        return foo 

    def plot_complex_plane(self):
        fig,ax=plt.subplots()
        ax.set_aspect('equal', adjustable='box')
        ax.plot(self.Z[:,0],self.Z[:,1],'o-')
        ax.set_xlabel('Re(Z)')
        ax.set_ylabel('Im(Z)')

    def plot(self,fmt='raw',semilogx=True,fig=None,ax=None,label=None):
        if not fig: fig, ax = plt.subplots(nrows=2,ncols=2,sharex=False,figsize=(12,8))
        if fmt=='raw':
            datas = [self.iq_data_raw,self.iq_data_sc_raw]
        elif fmt=='phys':
            datas = [self.iq_data,self.iq_data_sc]
        elif fmt=='trans':
            datas = [2*np.sqrt(2)*self.iq_data/self.Io,2*np.sqrt(2)*self.iq_data_sc/self.Io_sc]
        elif fmt=='Z':
            datas=[self.Z]
        else:
            assert False, print('format type unknown: ',fmt)
        
        for data in datas:
            if semilogx:
                ax[0][0].semilogx(self.f,data[:,0],'o-')
                ax[0][1].semilogx(self.f,data[:,1],'o-')
                ax[1][0].semilogx(self.f,np.sqrt(data[:,0]**2+data[:,1]**2),'o-')
                ax[1][1].semilogx(self.f,np.unwrap(np.arctan2(data[:,1],data[:,0])),'o-',label=label)
            else:
                ax[0][0].plot(self.f,data[:,0],'o-')
                ax[0][1].plot(self.f,data[:,1],'o-')
                ax[1][0].plot(self.f,np.sqrt(data[:,0]**2+data[:,1]**2),'o-')
                ax[1][1].plot(self.f,np.unwrap(np.arctan2(data[:,1],data[:,0])),'o-')
        
        # axes labels
        ax[0][0].set_ylabel('I')
        ax[0][1].set_ylabel('Q')
        ax[1][0].set_ylabel('Amp')
        ax[1][1].set_ylabel('Phase')
        ax[1][0].set_xlabel('Freq (Hz)')
        ax[1][1].set_xlabel('Freq (Hz)')
        fig.suptitle(fmt)
        return fig, ax

    def q_func(self,p):
        ''' 
            p[0] = overall normalization
            p[1] = tau
        '''
        return 

    def _guess_tau(self,f_max=1e3):
        ind = np.argmin(self.Z[np.array(self.f)<f_max,1]) # minimum of Q 
        return (2*np.pi*self.f[ind])**-1
    
    def fitQ(self,plot=False,fig=None,ax=None):
        fit_func = lambda p,f: p[0]*2*np.pi*f*p[1]/(1+(p[1]*2*np.pi*f)**2) #p[0] overall normalization, p[1]=tau
        optimize_func = lambda p,f,d: d-fit_func(p,f) # data - fit

        tau_est = self._guess_tau()
        p = leastsq(optimize_func, [1,tau_est],args=(np.array(self.f),self.Z[:,1]))[0]

        if plot:
            if not fig: fig,ax=plt.subplots(2,1)
            ax[0].semilogx(self.f,self.Z[:,1],'o')
            xfit=np.linspace(min(self.f),max(self.f),1000)
            yfit=fit_func(p,xfit)
            ax[0].semilogx(xfit,yfit,'--')
            ax[0].set_ylabel('Q')
            ax[1].semilogx(self.f,self.Z[:,1]-fit_func(p,np.array(self.f)),'o')
            ax[1].set_xlabel('Frequency (Hz)')
            return p, fig, ax
        else:
            return p
  
class CzDataExplore():
    ''' Explore a complex impedance data set stored in the CzData format '''
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

    def plot_measured_temperatures(self):
        print('to be written')

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

                if semilogx:
                    ax[0][0].semilogx(ss.frequency_hz,iq_data[:,row_index,0],'o-')
                    ax[0][1].semilogx(ss.frequency_hz,iq_data[:,row_index,1],'o-')
                    ax[1][0].semilogx(ss.frequency_hz,np.sqrt(iq_data[:,row_index,0]**2+iq_data[:,row_index,1]**2),'o-')
                    ax[1][1].semilogx(ss.frequency_hz,np.unwrap(np.arctan2(iq_data[:,row_index,1],iq_data[:,row_index,0])),'o-',
                    label='Tb=%dmK, b=%d'%(self.cz.temp_list_k[ii]*1000,db))
                else:
                    ax[0][0].plot(ss.frequency_hz,iq_data[:,row_index,0],'o-')
                    ax[0][1].plot(ss.frequency_hz,iq_data[:,row_index,1],'o-')
                    ax[1][0].plot(np.sqrt(ss.frequency_hz,iq_data[:,row_index,0]**2+iq_data[:,row_index,1]**2),'o-')
                    ax[1][1].plot(ss.frequency_hz,np.unwrap(np.arctan2(iq_data[:,row_index,1],iq_data[:,row_index,0])),'o-')

                # plot I vs Q as second plot
                ax2.plot(iq_data[:,row_index,0],iq_data[:,row_index,1],'o-',label='Tb=%dmK, b=%d'%(self.cz.temp_list_k[ii]*1000,db))

        # axes labels
        ax[0][0].set_ylabel('I')
        ax[0][1].set_ylabel('Q')
        ax[1][0].set_ylabel('I^2+Q^2')
        ax[1][1].set_ylabel('Phase')
        ax[1][0].set_xlabel('Freq (Hz)')
        ax[1][1].set_xlabel('Freq (Hz)')
        ax[1][1].legend()

        ax2.set_xlabel('I')
        ax2.set_ylabel('Q')
        ax2.set_aspect('equal','box')
        ax2.legend()

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

        
def plot_quadrature_detection(f,iq,row_index,label=None,semilogx=True,fig=None,ax=None):
    ''' Plot data in 2x2 plot of I,Q,amp, phase versus frequency    
        for rows, temp_indices, and bias_indices.  All of these input fields can either be an integer or a list.

        iq has shape (num_freq, num_det, 2)
    '''
    if not fig:
        fig, ax = plt.subplots(nrows=2,ncols=2,sharex=False,figsize=(12,8))
    
    if semilogx:
        ax[0][0].semilogx(f,iq[:,row_index,0],'o-')
        ax[0][1].semilogx(f,iq[:,row_index,1],'o-')
        ax[1][0].semilogx(f,np.sqrt(iq[:,row_index,0]**2+iq[:,row_index,1]**2),'o-')
        ax[1][1].semilogx(f,np.unwrap(np.arctan2(iq[:,row_index,1],iq[:,row_index,0])),'o-',label=label)
    else:
        ax[0][0].plot(f,iq[:,row_index,0],'o-')
        ax[0][1].plot(f,iq[:,row_index,1],'o-')
        ax[1][0].plot(f,np.sqrt(iq[:,row_index,0]**2+iq[:,row_index,1]**2),'o-')
        ax[1][1].plot(f,np.unwrap(np.arctan2(iq[:,row_index,1],iq[:,row_index,0])),'o-',label=label)

    # axes labels
    ax[0][0].set_ylabel('I')
    ax[0][1].set_ylabel('Q')
    ax[1][0].set_ylabel('Amp')
    ax[1][1].set_ylabel('Phase')
    ax[1][0].set_xlabel('Freq (Hz)')
    ax[1][1].set_xlabel('Freq (Hz)')
    
    return fig,ax
    
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
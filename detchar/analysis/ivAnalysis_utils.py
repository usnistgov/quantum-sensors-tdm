''' ivAnalysis_utils.py '''

# IVTempSweepData ought to have the coldload setpoint.

import detchar
from detchar.iv_data import IVCircuit
from detchar import IVColdloadSweepData
import numpy as np
import matplotlib.pyplot as plt

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

        # for ii, set_cl_temp_k in enumerate(cl_temps):
        #     x,y = self.data[ii].data[bath_temp_index].xy_arrays_zero_subtracted_at_origin()
        #     x,y = self.data[ii].data[bath_temp_index].xy_arrays_zero_subtracted_at_normal_y_intercept(normal_above_fb=9000)
        #     #x = self.data.data[ii].data[bath_temp_index].dac_values
        #     #y = self.data.data[ii].data[bath_temp_index].fb_values_array()[:,row_index]
        #     y=y[:,row_index]
        #     plt.plot(x,y,'-')

        plt.xlabel('DAC values')
        plt.ylabel('Feedback values')
        plt.title('Row index = %d, Tb = %d mK'%(row_index,self.set_bath_temps_k[bath_temp_index]*1000))
        plt.legend((np.array(self.set_cl_temps_k)[cl_indicies]),loc='upper right')
        plt.show()

    def get_cl_temp_index(self,cl_temp):
        print('to be written')

class IVColdloadAnalyzeOneRow():
    ''' Analyze a set of IV curves for a single detector taken at multiple
        coldload temperatures and a single bath temperature
    '''

    def __init__(self,dac_values,fb_array,cl_temps_k,bath_temp_k,device_name=None,iv_circuit=None):
        self.dacs = dac_values
        self.n_dac_values = len(self.dacs)
        self.fb = fb_array # NxM array of feedback values.  Columns are per coldload temperature
        self.fb_align = None
        self.cl_temps_k = cl_temps_k
        self.bath_temp_k = bath_temp_k
        self.det_name, self.row_name = self._handle_device_name(device_name)
        self.n_dac_values, self.n_cl_temps = np.shape(self.fb)

        # details
        self.n_normal_pts=10 # number of points for normal branch fit
        self.use_ave_offset=True # use a global offset to align fb, not individual per curve
        if iv_circuit==None:
            self.to_physical_units = False
        else:
            self.to_physical_units = True


    def _handle_device_name(self,device_name):
        if device_name==None:
            det_name = 'unknown'; row_name = 'unknown'
        else:
            row_name = device_name.keys()[0]; det_name = device_name[row_name]
        return det_name, row_name

    def fb_align_and_remove_offset(self,showplot=False):
        fb_align = np.zeros((self.n_dac_values,self.n_cl_temps))
        for ii in range(self.n_cl_temps): # align fb DC levels to a common value
            dy = self.fb[0,ii]-self.fb[0,0]
            fb_align[:,ii] = self.fb[:,ii]-dy

        # remove offset
        x = self.dacs[::-1][-self.n_normal_pts:]
        y = fb_align[::-1,:] ; y = y[-self.n_normal_pts:,:]
        m, b = np.polyfit(x,y,deg=1)
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

    def plot_raw(self,fb_align_dc_level=True):
        for ii, cl_temp in enumerate(self.cl_temps_k):
            if fb_align_dc_level:
                dy = self.fb[0,ii]-self.fb[0,0]
            else: dy=0
            plt.plot(self.dacs, self.fb[:,ii]-dy)
        plt.xlabel('DAC values')
        plt.ylabel('Feedback values')
        plt.title(self.det_name+', '+self.row_name+' , Tb = %.1f mK'%(self.bath_temp_k*1000))
        plt.legend((self.cl_temps_k),loc='upper right')

    def plot_results(self,to_physical_units=False):
        ''' Plot results IV, PV, RP, RoP '''
        if self.fb_align ==None:
            self.fb_align = self.fb_align_and_remove_offset(showplot=False)

        if self.to_physical_units:
            v,i = iv_circuit.to_physical_units(self.dacs,self.fb_align)
        else:
            v = np.zeros((self.n_dac_values,self.n_cl_temps))
            for ii in range(self.n_cl_temps):
                v[:,ii] = self.dacs
            i=self.fb_align
        p=v*i; r=v/i

        # fig 0: raw IV
        fig0 = plt.figure(1)
        self.plot_raw(True)

        # fig 1, 2x2 of converted IV
        fig1, ax = plt.subplots(nrows=2,ncols=2,sharex=False,figsize=(12,8))
        ax=[ax[0][0],ax[0][1],ax[1][0],ax[1][1]]
        for ii in range(self.n_cl_temps):
            ax[0].plot(v[:,ii],i[:,ii])
            ax[1].plot(v[:,ii],p[:,ii])
            ax[2].plot(p[:,ii],r[:,ii])
            #ax[3].plot(p[:,ii],r[:,ii]/r[-2,ii])
            ax[3].plot(v[:,ii],r[:,ii])
        xlabels = ['V ($\mu$V)','V ($\mu$V)','P (pW)','V ($\mu$V)']
        ylabels = ['I ($\mu$A)', 'P (pW)', 'R (m$\Omega$)', 'R (m$\Omega$)']
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

if __name__ == "__main__":
    #filename_json = '/home/pcuser/data/lbird/20201202/lbird_hftv0_coldload_sweep.json'
    filename_json = 'lbird_hftv0_coldload_sweep.json'
    df = IVColdloadSweepAnalyzer(filename_json)
    #df.plot_single_iv(1,1,0)
    #df.plot_cl_temp_sweep_for_row(2,0)
    #df.plot_measured_bath_temps()
    dacs,fb = df.get_cl_sweep_dataset_for_row(row_index=2,bath_temp_index=1,cl_indicies=list(range(df.n_cl_temps-1)))
    iv_circuit = IVCircuit(rfb_ohm=5282.0+50.0,
                           rbias_ohm=10068.0,
                           rsh_ohm=0.0662,
                           rx_ohm=0,
                           m_ratio=8.259,
                           vfb_gain=1.017/(2**14-1),
                           vbias_gain=6.5/(2**16-1))
    IVColdloadAnalyzeOneRow(dacs,fb,df.set_cl_temps_k[0:-1],df.set_bath_temps_k[1],None,iv_circuit).plot_results()
    plt.show()

    # for ii in range(df.n_rows):
    #     df.plot_cl_temp_sweep_for_row(row_index=ii,bath_temp_index=1)

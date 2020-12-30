''' ivAnalyzer.py 

Data structure is output of ivCurve.py when NOT using the 'legacy' data format.
self.ivdict has keys 'v', 'config', 'iv'
'iv' is a nested list where the first index calls coldload temperature and the second call bath temperatures. ie.
ivdict['iv'][ii][jj]
ii: index for cold load temperature 
jj: index for bath temperature 

@author: jh, late 2020 early 2021

To do:
* IV normalization - from IV turn around? forcing all in set to the same?
* P versus T_CL
* Compare to top-hat single mode
'''

import numpy as np
import matplotlib.pyplot as plt 
import sys, pickle

class ivAnalyzer(object):
    '''
    ivAnalyzer class to analyze a series of IV curves as a function of 
    bath temperature and/or cold load temperature from a single input file.  
    The intended purpose is either to characterize thermal conductance from a 
    data set of IVs versus bath temperature, or optical efficiency measurements
    for a dataset of IVs versus coldload temperature
    '''
    def __init__(self, file, rowMapFile=None,calnums=None):
        ''' file is full path to the data '''
        self.file = file
        self.getPath 
        self.rowMapFile = rowMapFile

        # load the data self.ivdict is the main dictionary structure
        self.ivdict = self.loadPickle(file)
        self.ivdatadict = self.ivdict['iv'] # the main data structure which is a list: self.ivdatadict[Tbb_ii][T_bath_jj]
        self.config = self.ivdict['config']

        # determine if file has data looped over bath temperature or coldload or both
        self.determineTemperatureSweeps()
        self.v_command_orig=self.ivdict['v']
        if (self.v_command_orig[1] - self.v_command_orig[0]) < 0:
            self.v_command = self.v_command_orig[::-1]
            self.v_ascending_order=False
        else:
            self.v_command = self.v_command_orig 
            self.v_ascending_order=True
        self.n_vbias = len(self.v_command)

        # load row to detector map if provided
        self.loadRowMap() 
        self.getNumberOfColsRows()

        print('Loaded %s'%self.file)
        print('Bath temperatures (K) = ',self.TbTemps)
        print('Cold Load Executed = ',self.coldloadExecute)
        if self.coldloadExecute:
            print('Coldload Temperatures (K) = ',self.bbTemps)

        # calibration numbers
        if calnums==None:
            self.Rfb = self.config['calnums']['rfb'] +50.0 # feedback resistance from dfb card to squid 1 feedback (50 Ohm as output impedance of dfb card)
            self.Rb = self.config['calnums']['rbias'] # resistance between voltage bias source and shunt chip  
            self.Rshunt = self.config['calnums']['rjnoise'] # shunt resistance
            self.Mr = self.config['calnums']['mr'] # mutual inductance ratio
            self.vfb_gain = self.config['calnums']['vfb_gain'] # fullscale voltage out of dfB
            if 'Rx' in self.config['calnums'].keys(): # parasitic resistance in series with the TES, a list, one per row
                self.Rx = self.config['calnums']
            else:
                self.Rx = [0]*self.nrow

        self.v_units='$\mu$V'
        self.i_units='$\mu$A'
        self.p_units='pW'
        self.r_units='m$\Omega$'
        self.mult_v = 1e6 # multiplier to voltage axis to convert to units such as uV
        self.mult_i = 1e6 # multiplier to current axis to convert to units such as uA 
        self.mult_r = 1e3 # multiplier to resistance axis to convert to units such as mOhms
        self.sweepType=None

    # helper methods -------------------------------------------------------------------------

    def getNumberOfColsRows(self):
        self.ncol, self.nrow, self.nsamp, err_fb = np.shape(self.ivdatadict[0][0]['data'])

    def loadPickle(self,filename):
        f=open(self.file,'rb')
        d=pickle.load(f)
        f.close()
        return d

    def getPath(self):
        self.path = '/'.join(self.file.split('/')[0:-1])
    
    def loadRowMap(self):
        ''' RowMap is a dictionary that maps rows to detector name.  
        Key is 'RowXX', which has values 'detector' and 'bay'
        '''
        if self.rowMapFile==None:
            self.rowMap=None
        else:
            self.rowMap = self.loadPickle(self.rowMapFile)

    def determineTemperatureSweeps(self):
        ''' determine if looped over coldload temperature or bath temperature, 
            create global variables:
            self.coldloadExecute
            self.bbTemps
            self.n_cl_temps
            self.TbTemps
            self.n_Tb_temps
        ''' 
        if 'coldload' in self.config.keys() and self.config['coldload']['execute']:
            self.coldloadExecute = True 
            self.bbTemps = self.config['coldload']['bbTemperatures']
            self.n_cl_temps = len(self.ivdatadict)
            if self.n_cl_temps != len(self.bbTemps):
                print('WARNING: Number of coldload temperatures in the data structure =%d. Does not match number in config file = %d'%(self.n_cl_temps,len(self.bbTemps)))
        else: 
            self.coldloadExecute = False 
            self.bbTemps=None
            self.n_cl_temps=0 

        self.TbTemps = self.config['runconfig']['bathTemperatures']
        self.n_Tb_temps = len(self.ivdatadict[0])
        if self.n_Tb_temps != len(self.TbTemps):
            print('WARNING: Number of bath temperatures in the data structure =%d. Does not match number in config file = %d'%(self.n_Tb_temps,len(self.TbTemps)))
    
    def getSweepTemps(self,sweep_temps='all',sweepType='bath_temperature'):
        # some error handling
        if sweepType not in ['bath_temperature','coldload']:
            print('unrecognized sweepType: ',sweepType,' Aborting.')
            sys.exit()
        else:
            self.sweepType=sweepType

        if sweepType == 'coldload':
            allSweepTemps = np.array(self.bbTemps) 
        elif sweepType == 'bath_temperature':
            allSweepTemps = np.array(self.TbTemps)
        if sweep_temps=='all':
            self.sweepTempIndicies=list(range(len(allSweepTemps)))
            self.sweepTemps=allSweepTemps
        else:
            self.sweepTempIndicies = [i for i, value in enumerate(allSweepTemps) if value in sweep_temps]
            self.sweepTemps = allSweepTemps[self.sweepTempIndicies]
        self.n_sweeps = len(self.sweepTemps)
    
    def constructVfbArray(self,col,row,static_temp,sweep_temps='all',sweepType='bath_temperature'):
        ''' return the feedback voltage vectors in a single NxM array for sweepType, 
            where N is the number of v_bias points and M is the number of temperatures of sweepType 

            input:
            col: column index in data return (EasyClient structure)
            row: row index in data return
            static_temp: fixed temperature at which to populate return array.
                  If sweepType='coldload', temp is the fixed bath_temperature,
                  whereas if sweepType:'bath_temperature' temp corresponds to the 
                  single cold load temperature at which to construct the return array.
                  If there is no coldload sweep executed for data in self.file, then 
                  this field is ignored.
            sweep_temps: list of sweep temperatures to get the ivs for. 
            sweepType: 'bath_temperature' (default) or 'coldload'.  Selects if the returned 
                        data array populates a sweep over coldload temperatures or bath temperatures
     
            
        '''
        # preparing...
        self.getSweepTemps(sweep_temps,sweepType)
        result = np.zeros((self.n_vbias,self.n_sweeps)) #initialize result

        # two cases
        if sweepType=='coldload':
            if not self.coldloadExecute:
                print('File: ',self.file,' does contain data looped over cold load temperatures.  Aborting.')
                sys.exit()
            
            if static_temp not in self.TbTemps:
                print('Requested bath temperature Tb=',static_temp, ' not in TbTemps = ',self.TbTemps)
                sys.exit()
            Tb_index = self.TbTemps.index(static_temp)
            for ii in range(self.n_sweeps):
                result[:,ii]=self.ivdatadict[self.sweepTempIndicies[ii]][Tb_index]['data'][col,row,:,1]
            
        elif sweepType=='bath_temperature':
            if not self.coldloadExecute:
                cl_index = 0
            else:
                if temp not in self.bbTemps:
                    print('Requested coldload temperature Tbb=',temp, ' not in bbTemps = ',self.bbTemps)
                    sys.exit()
                else:
                    cl_index = self.bbTemps.index(temp)
            
            for ii in range(self.n_sweeps):
                result[:,ii]=self.ivdatadict[cl_index][self.sweepTempIndicies[ii]]['data'][col,row,:,1]
                
        return result 

    # iv analysis steps -----------------
    def makeAscendingOrder(self,y):
        if not self.v_ascending_order:
            y = y[::-1]
        return y
    
    def removeOffset(self,vfb,intercept='normal',debug=False):
        ''' remove iv curve dc offset.  vfb also forced to have positive slope in normal branch. '''
        n,m=np.shape(vfb)

        if intercept=='normal':
            vfb_diff = np.diff(vfb,axis=0) # difference of vectors
            normal_diff = vfb_diff[-10:].mean(axis=0) # average difference of highest Vb points, proxy for normal branch
            # determine index where slope differs from normal slope by tol (=1% by default)
            frac_diff = (vfb_diff - normal_diff) / normal_diff
            tol = 0.001
            dex = np.argmin(abs(frac_diff - tol),axis=0)
        
            # fit "normal" branch
            pvals=np.zeros((m,2))
            for ii in range(m):
                #pval = np.polyfit(self.v[dex[ii]:],vfb[dex[ii]:,ii],1) # index determined by deviation from flat
                # pval[0] = slope, pval[1] = offset
                pval = np.polyfit(self.v_command[-11:-1],vfb[-11:-1,ii],1) # fixed index provided
                pvals[ii,:] = pval
            #pvals = np.polynomial.polynomial.polyfit(self.v[dex:],vfb[dex:,:],1) 
            vfb_corr = vfb - pvals[:,1] 
            
            # if normal branch slope negative, make positive
            if pvals[0,0] < 0:
                vfb_corr = -1*vfb_corr

            # force all in set to have same DC offset
            offset_delta = vfb_corr[-2,:] - vfb_corr[-2,0]
            vfb_corr = vfb_corr - offset_delta
        else:
            print('only intercept=normal currently supported')

        if debug:
            x_fit = np.linspace(0,np.max(self.v_command),100)
            for ii in range(m):
                fig, (ax0,ax1) = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=(6,12))
                ax0.plot(self.v_command,vfb[:,ii],'bo')
                ax0.plot(self.v_command[dex[ii]:],vfb[dex[ii]:,ii],'ro')
                y_fit = np.polyval(pvals[ii,:],x_fit)
                ax0.plot(x_fit,y_fit,'r--')
                ax0.set_ylabel('Vfb raw (V)')
                ax0.legend(('raw','normal','fit'))
                print(pvals[ii,:])
                ax1.plot(self.v_command,vfb_corr[:,ii],'bo')
                y_fit2 = np.polyval([abs(pvals[ii][0]),0],x_fit)
                ax1.plot(x_fit,y_fit2,'r--')
                ax1.set_ylabel('Vfb offset removed (V)')
                ax1.set_xlabel('raw voltage bias (V)')
                ax1.legend(('data','fit'))
                plt.show()
        return vfb_corr, pvals

    def toPhysicalUnitsSimple(self,y):
        # depricated, use toPhysicsUnits
        y_corr = y * self.vfb_gain *(self.Rfb*self.Mr)**-1
        x = self.v_command / self.Rb * self.Rshunt
        return x,y_corr

    def toPhysicalUnits(self,y):
        y_corr = y * self.vfb_gain *(self.Rfb*self.Mr)**-1
        I = self.v_command /self.Rb
        n,m = np.shape(y)
        x = np.zeros((n,m))
        for ii in range(m):
            #v_shunt = (I - y_corr[:,ii])*self.Rshunt # voltage drop across shunt resistor 
            #x[:,ii] = v_shunt - y_corr[:,ii]*self.Rx[ii]
            x[:,ii] = I*self.Rshunt - y_corr[:,ii]*(self.Rshunt+self.Rx[ii])
        return x,y_corr

    def get_virp(self,col,row,static_temp,sweep_temps='all',sweepType='bath_temperature'):
        result = self.constructVfbArray(col,row,static_temp,sweep_temps,sweepType) # assemble the array
        y=self.makeAscendingOrder(result) # put array in ascending voltage bias order
        y,pvals = self.removeOffset(y,intercept='normal',debug=False) # remove the DC offset

        # convert to physical units
        x,y_phys = self.toPhysicalUnits(y)
        #xp,y_phys = self.toPhysicalUnitsSimple(y)
        #x=np.zeros((self.n_vbias,self.n_sweeps))
        #for ii in range(self.n_sweeps):
        #    x[:,ii]=xp
        v = x * self.mult_v 
        i = y_phys * self.mult_i 
        r = v/i*self.mult_r 
        p = i*v

        self.v=v; self.i=i; self.r=r; self.p=p
        return result,v,i,r,p

    # plotting methods ---------------------------------------------------------------------

    def plotRawResponse(self,col,row,temp,sweepType='bath_temperature'):
        result = self.constructVfbArray(col,row,temp,sweepType)
        n,m=np.shape(result)
        for ii in range(m):
            plt.plot(self.v_command_orig,result[:,ii])
        plt.xlabel('V bias raw (V)')
        plt.ylabel('Vfb raw (V)')
        plt.grid()
        plt.title('Row Index = %02d'%row)
        plt.legend(tuple(self.sweepTemps))
        plt.show()

    def plotResults(self,col,row,static_temp,sweep_temps='all',sweepType='bath_temperature'):
        ''' Plot IV results for a series of IV curves for column "col" and row "row"

            fig1: raw IV 
            fig2: 2x2
            a: converted IV
            b: converted PV 
            c: converted RP
            d: converted RoP
        ''' 
        result,v,i,r,p = self.get_virp(col,row,static_temp,sweep_temps,sweepType)

        # fig 0: raw IV
        fig0, (ax0) = plt.subplots(nrows=1,ncols=1,sharex=False,figsize=(6,4))
        for ii in range(self.n_sweeps):
            ax0.plot(self.v_command_orig,result[:,ii])
        ax0.set_xlabel('Raw Voltage (V)')
        ax0.set_ylabel('Raw Feedback Voltage (V)')
        ax0.grid()
        fig0.suptitle('Row Index = %02d'%row)
        ax0.legend(tuple(self.sweepTemps))

        # fig 1, 2x2 of converted IV 
        fig1, ax = plt.subplots(nrows=2,ncols=2,sharex=False,figsize=(12,12))
        ax=[ax[0][0],ax[0][1],ax[1][0],ax[1][1]]
        for ii in range(self.n_sweeps):
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
        ax[2].set_ylim((0,np.max(r[-1,:])*1.1))
        ax[3].set_xlim((0,np.max(v)*1.1))
        ax[3].set_ylim((0,np.max(r[-1,:])*1.1))
        #ax[3].set_xlim((0,np.max(p)*1.1))
        #ax[3].set_ylim((0,1.1))

        fig1.suptitle('Row Index = %02d'%row)
        ax[3].legend(tuple(self.sweepTemps))
        plt.show()



    

     
         

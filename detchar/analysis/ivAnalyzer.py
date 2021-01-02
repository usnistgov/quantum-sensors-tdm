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
from scipy.constants import k,h,c
from scipy.integrate import quad, simps

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
        self.getPath()
        self.rowMapFile = rowMapFile
        self.loadRowMap() # load row to detector map if provided

        # load the data self.ivdict is the main dictionary structure
        self.ivdict = self.loadPickle(self.file)
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

        self.getNumberOfColsRows()

        # calibration numbers
        if calnums==None:
            self.Rfb = self.config['calnums']['rfb'] +50.0 # feedback resistance from dfb card to squid 1 feedback (50 Ohm as output impedance of dfb card)
            self.Rb = self.config['calnums']['rbias'] # resistance between voltage bias source and shunt chip
            self.Rshunt = self.config['calnums']['rjnoise'] # shunt resistance
            self.Mr = self.config['calnums']['mr'] # mutual inductance ratio
            self.vfb_gain = self.config['calnums']['vfb_gain'] # fullscale voltage out of dfB
            if 'Rx' in self.config['calnums'].keys(): # parasitic resistance in series with the TES, a list, one per row
                self.Rx = self.config['calnums']['Rx']
            else:
                self.Rx = [0]*self.nrow
        else:
            self.Rfb = calnums['rfb'] +50.0 # feedback resistance from dfb card to squid 1 feedback (50 Ohm as output impedance of dfb card)
            self.Rb = calnums['calnums']['rbias'] # resistance between voltage bias source and shunt chip
            self.Rshunt = calnums['rjnoise'] # shunt resistance
            self.Mr = calnums['mr'] # mutual inductance ratio
            self.vfb_gain = calnums['vfb_gain'] # fullscale voltage out of dfB
            if 'Rx' in calnums.keys(): # parasitic resistance in series with the TES, a list, one per row
                self.Rx = calnums['Rx']
            else:
                self.Rx = [0]*self.nrow

        # initialize data arrays and define units
        self.v=self.i=self.p=self.r=self.ro=None # initialize the main data arrays
        self.mult_v = 1e6 # multiplier to voltage axis to convert to units such as uV
        self.mult_i = 1e6 # multiplier to current axis to convert to units such as uA
        self.mult_r = 1e3 # multiplier to resistance axis to convert to units such as
        self.v_units='$\mu$V'
        self.i_units='$\mu$A'
        self.p_units='pW'
        self.r_units='m$\Omega$'
        self.sweepType=None

        print('Loaded %s'%self.file)
        print('Bath temperatures (K) = ',self.TbTemps)
        print('Cold Load Executed = ',self.coldloadExecute)
        if self.coldloadExecute:
            print('Coldload Temperatures (K) = ',self.bbTemps)

    # helper methods -------------------------------------------------------------------------

    def getNumberOfColsRows(self):
        self.ncol, self.nrow, self.nsamp, err_fb = np.shape(self.ivdatadict[0][0]['data'])

    def loadPickle(self,filename):
        f=open(filename,'rb')
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
        ''' populate the global variables
            self.sweepTempIndicies
            self.sweepTemps
            self.n_sweeps
        '''
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
        self.result = result # store raw data as global variable
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
            x[:,ii] = I*self.Rshunt - y_corr[:,ii]*(self.Rshunt+self.Rx[ii])
        return x,y_corr

    def get_virp(self,col,row,static_temp,sweep_temps='all',sweepType='bath_temperature'):
        ''' calculate the main data products voltage bias (v), TES current (i), TES resistance (r),
            TES power (p) for a device connected to the multiplexer readout channel on col, row.
            See constructVfbArray docstring for definitions of other inputs to this method.
        '''
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

        self.v=v; self.i=i; self.r=r; self.p=p; self.ro = r/np.mean(r[-10:,:],axis=0)
        return result,v,i,r,p

    def getFracRn(self,fracRns,arr='p',ro=None):
        '''
        Return the value of arr at fraction of Rn.

        input:
        fracRns: fraction of Rn values to be evaluated (NOT PERCENTAGE RN).
        arr: NxM array to determine the Rn fraction at
        ro: NxM normalized resistance

        arr and ro must be same shape

        return: len(fracRns) x M array of the interpolated values

        '''
        # if strings are provided, use the global variables
        if type(arr) == str:
            if arr == 'p':
                arr = self.p
            elif arr == 'i':
                arr = self.i
            elif arr == 'v':
                arr = self.v

        # ensure the normalized resistance array is ok.
        try: ro.any()
        except:
            if ro==None:
                ro = self.ro
            else:
                print('self.ro is not defined.  Either define it or provide ro array to method getFracRn')
                sys.exit()

        # if fracRns is a list, turn it into a np.array
        if type(fracRns)==list:
            fracRns = np.array(fracRns)

        # make sure fracRn is actually a fraction.
        if len(np.where(fracRns>1)[0])>0:
            print('fracRns have values >1.  Fractions of Rn expected, and not percentage Rn')
            print(fracRns)
            sys.exit()

        n,m=np.shape(arr)
        result = np.zeros((len(fracRns),m))
        for ii in range(m):
            x = self.removeNaN(ro[:,ii])
            y = self.removeNaN(arr[:,ii])
            YY = np.interp(fracRns,x,y)

            # over write with NaN for when data does not extend to fracRn
            ro_min = np.min(x)
            toCut = np.where(fracRns<ro_min)[0]
            N = len(toCut)
            if N >0:
                YY[0:N] = np.zeros(N)*np.NaN
            result[:,ii] = YY


        return result

    # data cleaning methods ------------------------------------------------
    def removeNaN(self,arr):
        ''' only works on 1d vector, not array '''
        return arr[~np.isnan(arr)]

    def findFirstZero(self,vec):
        ''' single vector, return index and value where vec crosses zero '''
        ii=0; val = vec[ii]
        while val > 0:
            if ii>len(vec):
                print('iv curve turnaround not found')
                return None
            ii=ii+1
            val=vec[ii]
        return ii, val

    def getIVturnIndex(self,i=None,PLOT=False):
        ''' return the indicies corresponding to the IV turnaround for
            set of IV curves within array `i'
        '''
        try:
            if i==None:
                i=self.i
        except: pass

        di = np.diff(i,axis=0) # difference of current array
        di_rev = di[::-1] # reverse order di_rev[0] corresponse to highest v_bias
        n,m = np.shape(di)
        ivTurnDex = []
        for ii in range(m):
            dex, val = self.findFirstZero(di_rev[:,ii])
            ivTurnDex.append(n-dex)

        if PLOT:
            # fig1 = plt.figure(1) # plot of delta i
            # plt.plot(di,'o-')
            # for ii in range(m):
            #     plt.plot(ivTurnDex[ii],di[ivTurnDex[ii],ii],'ro')

            fig2 = plt.figure(2)
            plt.plot(i,'o-')
            for ii in range(m):
                plt.plot(ivTurnDex[ii],i[ivTurnDex[ii],ii],'ro')
                plt.show()

        return ivTurnDex

    def findBadDataIndex(self,threshold=50,PLOT=False):
        ''' often times when TES latches, the data is not useful, and
            in fact problematic when trying to determine P at fracRn, since
            power at fracRn can be erroneously double valued.  This method
            finds the index where this happens
        '''
        # first find indicies where delta i is larger than some threshold
        di = np.diff(self.i,axis=0)
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

        if PLOT: #for debuggin purposes
            plt.xlabel('index')
            plt.ylabel('current (%s)'%self.i_units)
            for ii in range(m):
                plt.plot(self.i[:,ii],'*-')
                plt.plot(dexs[ii],self.i[dexs[ii],ii],'ro')
                plt.show()
                input('%d'%ii)

    def removeBadData(self,PLOT=False):
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

    # plotting methods ---------------------------------------------------------------------

    def plotRawResponse(self,col,row,static_temp,sweep_temps='all',sweepType='bath_temperature'):
        result = self.constructVfbArray(col,row,static_temp,sweep_temps,sweepType)
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
        fig1, ax = plt.subplots(nrows=2,ncols=2,sharex=False,figsize=(12,8))
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

    def analyzeColdload(self,col,row,static_temp,sweep_temps='all', threshold=10,
                        fracRns=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                        nu=None,savePlots=False):
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
        result,v,i,r,p = self.get_virp(0,row,.13,'all','coldload') # make main data vectors
        self.findBadDataIndex(threshold=threshold,PLOT=False)      # remove bad data
        self.removeBadData(False)
        p_at_fracR = self.getFracRn(fracRns,arr=self.p,ro=self.ro) # len(fracRn) x len(sweep_temps) array

        plottitle = 'Row Index%02d'%row
        if type(self.rowMap)==dict:
            plottitle=plottitle+'; '+ self.rowMap['Row%02d'%row]['detector_name']

        # FIG1: P versus R/Rn
        fig1 = plt.figure(1)
        plt.plot(self.ro,self.p,'-') # plots for all Tbath
        plt.plot(fracRns,p_at_fracR,'ro')
        #plt.xlim((0,1.1))
        #plt.ylim((0,np.max(self.p)))
        plt.xlabel('Normalized Resistance')
        plt.ylabel('Power (%s)'%self.p_units)
        plt.title(plottitle)
        plt.legend((self.sweepTemps))
        plt.grid()

        # FIG2: ES power plateau versus T_cl
        fig2 = plt.figure(2)
        for ii in range(len(fracRns)):
            plt.plot(self.sweepTemps,p_at_fracR[ii,:],'o-')
        plt.xlabel('T$_{cl}$ (K)')
        plt.ylabel('TES power plateau (%s)'%self.p_units)
        plt.legend((fracRns))
        plt.title(plottitle)
        plt.grid()

        # FIG3: change in saturation power relative to minimum coldload temperature
        fig3 = plt.figure(3)
        min_dex = np.argmin(self.sweepTemps)
        for ii in range(len(fracRns)):
            plt.plot(self.sweepTemps-self.sweepTemps[min_dex],p_at_fracR[ii,min_dex]-p_at_fracR[ii,:],'o-')

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
                Ptherm.append(self.thermalPower(nu1=nu1*1e9,nu2=nu2*1e9,T=self.sweepTemps[ii],F=None))
            Ptherm=np.array(Ptherm)
            dPtherm = Ptherm - Ptherm[min_dex]
            plt.plot(self.sweepTemps-self.sweepTemps[min_dex],dPtherm,'k-')
        plt.xlabel('T$_{cl}$ - %.1f K'%self.sweepTemps[min_dex])
        plt.ylabel('P$_o$ - P (%s)'%self.p_units)
        plt.legend((fracRns))
        plt.grid()
        plt.title(plottitle)

        # FIG4: efficiency versus delta T (only if nu provided)
        if doPrediction:
            fig4 = plt.figure(4)
            for ii in range(len(fracRns)):
                plt.plot(self.sweepTemps-self.sweepTemps[min_dex],(p_at_fracR[ii,min_dex]-p_at_fracR[ii,:])/dPtherm,'o-')
            plt.xlabel('T$_{cl}$ - %.1f K'%self.sweepTemps[min_dex])
            plt.ylabel('$\eta$')
            plt.legend((fracRns))
            plt.grid()
            plt.title(plottitle)

        if savePlots:
            fig1.savefig('RowIndex%02d_1_rp.png'%row)
            fig2.savefig('RowIndex%02d_2_pt.png'%row)
            fig3.savefig('RowIndex%02d_3_dpdt.png'%row)
            if doPrediction:
                fig4.savefig('RowIndex%02d_4_eta.png'%row)
        #plt.show()
        plt.close('all')


    def Pnu_thermal(self,nu,T):
        ''' power spectral density (W/Hz) of single mode from thermal source at
            temperature T (in K) and frequency nu (Hz).
        '''
        x = h*nu/(k*T)
        B = h*nu * (np.exp(x)-1)**-1
        return B

    def thermalPower(self,nu1,nu2,T,F=None):
        ''' Calculate the single mode thermal power (in pW) emitted from a blackbody
            at temperature T (in Kelvin) from frequency nu1 to nu2 (in Hz).
            F = F(\nu) is an arbitrary absolute passband defined between nu1 and nu2 with linear
            samplign between nu1 and nu2.  The default is F=None, in which case a
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
        return P*1e12

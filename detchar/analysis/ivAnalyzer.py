''' ivAnalyzer.py 

self.ivdict has keys 'v', 'config', 'iv'
'iv' is a nested list [[[Tbb1,Tb1],[Tbb1,Tb2],...,[Tbb1,TbN]],[[Tbb2,Tb1],[Tbb2,Tb2],...,[Tbb2,TbN]], ... ]

'''

import numpy as np
import matplotlib.pyplot as plt 
import sys, pickle

class ivAnalyzer(object):
    '''
    ivAnalyzer class to analyze a series of IV curves as a function of 
    bath temperature and/or cold load temperature from a single input file.
    '''
    def __init__(self, file, rowMapFile=None,calnums=None):
        ''' file is full path to the data '''
        self.file = file
        self.getPath 
        self.rowMapFile = rowMapFile

        # load the data self.ivdict is the main dictionary structure
        self.ivdict = self.loadPickle(file)
        self.ivdatadict = self.ivdict['iv']
        self.config = self.ivdict['config']

        # determine if file has data looped over bath temperature or coldload or both
        self.determineTemperatureSweeps()
        self.v_orig=self.ivdict['v']
        if (self.v_orig[1] - self.v_orig[0]) < 0:
            self.v = self.v_orig[::-1]
            self.v_ascending_order=False
        else:
            self.v = self.v_orig 
            self.v_ascending_order=True
        self.n_vbias = len(self.v)

        # load row to detector map if provided
        self.loadRowMap() 

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
            self.vfb_gain = self.config['calnums']['vfb_gain']
        
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
            self.n_col_temps=0 

        self.TbTemps = self.config['runconfig']['bathTemperatures']
        self.n_Tb_temps = len(self.ivdatadict[0])
        if self.n_Tb_temps != len(self.TbTemps):
            print('WARNING: Number of bath temperatures in the data structure =%d. Does not match number in config file = %d'%(self.n_Tb_temps,len(self.TbTemps)))
    
    def constructVfbArray(self,col,row,Tb):
        ''' return the feedback voltage at each coldload temperature for a single Tbath '''
        if Tb not in self.TbTemps:
            print('Requested bath temperature Tb=',Tb, ' not in TbTemps = ',self.TbTemps)
            sys.exit()
        Tb_index = self.TbTemps.index(Tb)
        result = np.zeros((self.n_vbias,self.n_cl_temps))
        for ii in range(self.n_cl_temps):
            result[:,ii]=self.ivdatadict[ii][Tb_index]['data'][col,row,:,1]
        return result 

    def plotCLresponse(self,col,row,Tb):
        result = self.constructVfbArray(col,row,Tb)
        for ii in range(self.n_cl_temps):
            plt.plot(self.v_orig,result[:,ii])
        plt.show()

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
                pval = np.polyfit(self.v[-11:-1],vfb[-11:-1,ii],1) # fixed index provided
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
            x_fit = np.linspace(0,np.max(self.v),100)
            for ii in range(m):
                fig, (ax0,ax1) = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=(6,12))
                ax0.plot(self.v,vfb[:,ii],'bo')
                ax0.plot(self.v[dex[ii]:],vfb[dex[ii]:,ii],'ro')
                y_fit = np.polyval(pvals[ii,:],x_fit)
                ax0.plot(x_fit,y_fit,'r--')
                ax0.set_ylabel('Vfb raw (V)')
                ax0.legend(('raw','normal','fit'))
                print(pvals[ii,:])
                ax1.plot(self.v,vfb_corr[:,ii],'bo')
                y_fit2 = np.polyval([abs(pvals[ii][0]),0],x_fit)
                ax1.plot(x_fit,y_fit2,'r--')
                ax1.set_ylabel('Vfb offset removed (V)')
                ax1.set_xlabel('raw voltage bias (V)')
                ax1.legend(('data','fit'))
                plt.show()
        return vfb_corr, pvals

    def toPhysicalUnitsSimple(self,y):
        y_corr = y * self.vfb_gain *(self.Rfb*self.Mr)**-1
        x = self.v / self.Rb * self.Rshunt
        return x,y_corr

    def plotResults(self,col,row,Tb):
        ''' Plot IV results for a series of IV curves for column "col" and row "row"

            fig1: raw IV 
            fig2: 2x2
            a: converted IV
            b: converted PV 
            c: converted RP
            d: converted RoP
        ''' 
        result = self.constructVfbArray(col,row,Tb)
        y=self.makeAscendingOrder(result)
        y,pvals = self.removeOffset(y,intercept='normal',debug=False)
        x,y_phys = self.toPhysicalUnitsSimple(y)
        n,m = np.shape(y)

        # fig 0: raw IV
        fig0, (ax0) = plt.subplots(nrows=1,ncols=1,sharex=False,figsize=(6,4))
        for ii in range(m):
            ax0.plot(self.v_orig,result[:,ii])
        ax0.set_xlabel('Raw Voltage (V)')
        ax0.set_ylabel('Raw Feedback Voltage (V)')
        ax0.grid()

        # fig 1: 2x2 IV, PV, RP, Ro vs P
        v = x*1e6 
        i = y_phys*1e6 
        r = np.zeros((n,m))
        p = np.zeros((n,m))
        for ii in range(m):
            r[:,ii] = v/i[:,ii]*1000
            p[:,ii] = i[:,ii]*v

        fig1, ax = plt.subplots(nrows=2,ncols=2,sharex=False,figsize=(12,12))
        ax=[ax[0][0],ax[0][1],ax[1][0],ax[1][1]]
        for ii in range(m):
            ax[0].plot(v,i[:,ii])
            ax[1].plot(v,p[:,ii])
            ax[2].plot(p[:,ii],r[:,ii])
            ax[3].plot(p[:,ii],r[:,ii]/r[-2,ii])
        xlabels = ['V ($\mu$V)','V ($\mu$V)','P (pW)','P (pW)']
        ylabels = ['I ($\mu$A)', 'P (pW)', 'R (m$\Omega$)', 'R/R$_o$']
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
        ax[3].set_xlim((0,np.max(p)*1.1))
        ax[3].set_ylim((0,1.1))

        plt.show()

    def foo(self,col,row,Tb):
        y=self.constructVfbArray(col,row,Tb)
        y=self.makeAscendingOrder(y)
        y,pvals = self.removeOffset(y,intercept='normal',debug=False)
        plt.plot(pvals[:,0],'bo-') # slope
        plt.plot(pvals[:,1],'ro-') # offset
        plt.legend(('slope','offset'))
        plt.show()




    

     
         
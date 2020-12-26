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
        self.v=self.ivdict['v']
        self.v_ascending_order = False
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
            plt.plot(self.v,result[:,ii])
        plt.show()

    # iv analysis steps -----------------
    def makeAscendingOrder(self,y):
        if (self.v[1] - self.v[0]) < 0:
            self.v_ascending_order=True
            self.v = self.v[::-1]
        y = y[::-1]
        return y
    
    def removeOffset(self,vfb,intercept='normal',debug=False):
        ''' remove iv curve dc offset.  vfb also forced to have positive slope in normal branch. '''
        if not self.v_ascending_order:
            print('array is not ascending order. Abort!')
            sys.exit()

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
                pval = np.polyfit(self.v[dex[ii]:],vfb[dex[ii]:,ii],1) # index determined by deviation from flat
                #pval = np.polyfit(self.v[-10:],vfb[-10:,ii],1) # fixed index provided
                pvals[ii,:] = pval
            #pvals = np.polynomial.polynomial.polyfit(self.v[dex:],vfb[dex:,:],1) 
            vfb_corr = vfb - pvals[:,1] 
            
            # if normal branch slope negative, make positive
            if pvals[0,0] < 0:
                vfb_corr = -1*vfb_corr
        else:
            print('only intercept=normal currently supported')

        if debug:
            x_fit = np.linspace(0,np.max(self.v),100)
            for ii in range(m):
                fig, (ax0,ax1) = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=(6,12))
                #ax0.plot(self.v,vfb,'bo')
                #ax0.plot(self.v[dex[ii]:],vfb[dex[ii]:,ii],'ro')
                y_fit = np.polyval(pvals[ii,:],x_fit)
                ax0.plot(x_fit,y_fit,'r--')
                ax0.set_ylabel('Vfb raw (V)')
                ax0.legend(('raw','normal','fit'))
                print(pvals[ii,:])
                ax1.plot(self.v,vfb_corr[:,ii],'bo')
                y_fit2 = np.polyval([abs(pvals[ii][0]),0],x_fit)
                ax1.plot(x_fit,y_fit2,'r-')
                ax1.set_ylabel('Vfb offset removed (V)')
                ax1.set_xlabel('raw voltage bias (V)')
                ax1.legend(('data','fit'))
                plt.show()
                #plt.clf()
                #plt.cla()
        return vfb_corr, pvals

    def toPhysicalUnitsSimple(self,y):
        y_corr = y * self.vfb_gain *(self.Rfb*self.Mr)**-1
        x = self.v / self.Rb * self.Rshunt
        return x,y_corr

    def foo(self,col,row,Tb):
        y = self.constructVfbArray(col,row,Tb)
        y = self.makeAscendingOrder(y)
        vfb_corr, pvals = self.removeOffset(y,intercept='normal',debug=False)
        x,vfb_corr = self.toPhysicalUnitsSimple(vfb_corr)
        n,m = np.shape(vfb_corr)
        for ii in range(m):
            plt.plot(x*vfb_corr[:,ii]*1e12,x/vfb_corr[:,ii])
            plt.xlabel('Power (pW) ')
            plt.ylabel('Resistance ($\Omega$)')
        plt.show()






    

     
         

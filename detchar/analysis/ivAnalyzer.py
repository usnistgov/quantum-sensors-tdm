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
    def __init__(self, file, rowMapFile=None):
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
        self.n_vbias = len(self.v)

        # load row to detector map if provided
        self.loadRowMap() 

        print('Loaded %s'%self.file)
        print('Bath temperatures (K) = ',self.TbTemps)
        print('Cold Load Executed = ',self.coldloadExecute)
        if self.coldloadExecute:
            print('Coldload Temperatures (K) = ',self.bbTemps)
        
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





    

     
         

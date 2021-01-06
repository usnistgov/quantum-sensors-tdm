#! /usr/bin/env python
'''
ivCurve.py
11/2020
@author JH

Script to take IV curves on a SINGLE COLUMN of detectors as a function of bath temperature 
and temperature controlled cryogenic blackbody (coldload) operated by a cryocon22

How to use:
1) multiplexer setup in cringe
2) turn on data streaming with dastard & dcom.  Select the appropriate fibers, 
which defines which index in ec.getNewData() corresponds to which column.
Default for velma: FiberSelect 0,1,2 and parallel streaming off.
3) close adr_gui
4) create/edit a configuration file 
5) run ./ivCurve.py <config_filename>

requirements:
iv configuration file created
tower configuration file (called within iv configuration file)
pyYAML

to do:
mapping of rows/ra to row index 

Notes:
**IF ALL THREE FIBERS RUNNING**
EasyClient getNewData() method returns following structure:
(column, row (sequence length), sample number, 0/1=error/fb)
The 0th index (column) has perhaps a funny indexing
0: dfbx2 ch1
1: dfbx2 ch2
2: dfb/clk ch1 
This is defined by what fibers are installed where.
'''

# standard python module imports
import sys, os, yaml, time, subprocess, re, pickle
import numpy as np
import matplotlib.pyplot as plt 

# # QSP written module imports
from cringe.cringe_control import CringeControl
from adr_system import AdrSystem
from instruments import BlueBox, Cryocon22  
import tespickle 
from nasa_client import EasyClient

##########################################################################################################
##########################################################################################################
##########################################################################################################

class ivCurve:
    ''' class for your script to send remote commands to cringe'''
    def __init__(self,cfg,cc,ec,vs,adr,ccon,tp,pickle_file,showPlot,verbose):
        # class instances
        self.cc = cc # cringe control
        self.ec = ec # easyClient
        self.vs = vs # voltage source to drive TES voltage bias
        self.adr = adr # adr
        self.ccon = ccon # cryocon
        self.tp = tp # tes pickle  

        # config
        self.cfg = cfg # config file
        self.showPlot = showPlot
        self.verbose = verbose

        # filename of return data structure
        self.pickle_file = pickle_file

        # mux and dfb
        self.colName = self.cfg['detectors']['Column']
        self.rowAddName = self.cfg['detectors']['Rows']
        self.ncol = self.ec.ncol 
        self.nrow = self.ec.nrow 
        self.dfb_col_index = self.cfg['dfb']['dfb_card_index'] # index used for cringe control
        self.ec_col_index = self.cfg['dfb']['ec_col_index'] # index of returned data structure corresponding to desired column
        self.dfb_row_index = range(0,self.nrow) # a list
        self.defineLockedRows()  

        # static globals
        # iv config 
        self.numpts=10000 # number of samples to average per IV
        self.autoRange=True # dynamic range extender
        self.v_autoRange_lim=(0.15,0.85) # limits for autoRange. NOTE >=0.9 high limit produces poor results
        self.sample_delay = 0.05 # time to wait after commanding voltage before sampling
        self.relock_delay = 0.05 # time to wait after relock before sampling

        # blackbody
        if 'coldload' in self.cfg.keys() and self.cfg['coldload']['execute']:
            self.executeColdloadIVs = True
            self.tWaitIV = self.cfg['coldload']['tWaitIV'] # mintues to allow blackbody to settle before measurement
        else:
            self.executeColdloadIVs = False

        self.errorHandling()

    def errorHandling(self):
        if len(self.rowAddName) != self.nrow:
            print('Row addresses: ',self.rowAddName,' incompatible with easyClient number of rows: ',self.nrow)
            sys.exit()

        if self.ec_col_index > self.ncol:
            print('ec_col_index = ',self.ec_col_index, 'incompatible with easyCleint number of columns: ',self.ncol)
            sys.exit()


    # helper methods 
    # -----------------------------------------------------------------------------------------------------------------------------
    # Currently not used but may be useful in future for error handling since much setup is assumed for this script to work
    def findThisProcess(self, process_name ):
        ps = subprocess.Popen("ps -eaf | grep "+process_name, shell=True, stdout=subprocess.PIPE)
        output = ps.stdout.read()
        ps.stdout.close()
        ps.wait()
        return output

    # Currently not used but may be useful in future for error handling since much setup is assumed for this script to work
    def isThisRunning(self, process_name ):
        output = findThisProcess( process_name )
        if re.search('path/of/process'+process_name, output) is None:
            return False
        else:
            return True

    def defineLockedRows(self):
        ''' define which rows show have feedback locked and which rows remain unlocked.  
            This method explicitly unlocks specified channels and locks the others 
            Creates global variables:

            self.dfb_rowindex_unlocked 
            self.dfb_rowindex_locked 
        '''
        self.dfb_rowindex_unlocked = self.cfg['dfb']['dfb_rowindex_unlocked']
        dfb_rowindex_locked = list(range(0,self.nrow))
        if self.dfb_rowindex_unlocked: # if there are rows to leave unlocked 
            for ii in self.dfb_rowindex_unlocked:
                self.cc.set_fb_lock(0,self.dfb_col_index,ii,a_or_b='A') # unlock these specific rows, does this work with Galen's revision????
            self.dfb_rowindex_locked = list(np.delete(dfb_rowindex_locked,self.dfb_rowindex_unlocked))
        else:
            self.dfb_rowindex_locked = dfb_rowindex_locked 
        print('Row indexs to lock: ',self.dfb_rowindex_locked)
        
        # explicitly lock all requested rows:
        for ii in self.dfb_rowindex_locked:
            #self.cc.set_fb_lock(1,self.dfb_col_index,ii,a_or_b='A') # lock these specific rows
            self.cc.relock_fba(self.dfb_col_index,ii)

    def savePickle(self,return_dict):
        f = open(self.pickle_file, 'wb')
        pickle.dump(return_dict, f)
        f.close()
    # data collection methods -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------------------

    def getIVdataPt(self,npts=10000,v_lim=(0.1,0,9),autoRange=True):
        ''' Data collection method called for each IV point.    
            Average npts and return one dfb value for all columns and all rows in array of shape c, 
            where the last index is either 0: error and 1: feedback.  
            If autoRange called, dfb will be relocked if dfb voltage is out of defined range 
            
            Input
            npts: number of points to average, default=10000
            v_lim: (v_low,v_high), limits of autoRange. Only used if autoRange=True
            autoRange: <bool> if True relock when dfb voltage outside range v_lim

            return: data_mean, ar
            data_mean: array of shape (ncol,nrow,2) providing after of npts
            ar: dictionary concerning autoRange with keys:
                 'flag': <bool> if True an autoRange has been applied for at least one row
                 'indicies': <1D array> row indicies for which autoRange has been applied
                 'offset': <array of shape (ncol,nrow,2)> of offsets from autoRange

        '''
        ar={'flag':False,'indicies':[],'offset':np.zeros((self.ec.ncol,self.ec.nrow))}
        flag=False
        data = self.ec.getNewData(minimumNumPoints=npts,exactNumPoints=True,toVolts=True)
        data_mean = np.mean(data,axis=2)
        #data_std = np.std(data,axis=2)
        if self.verbose:
            for ii in range(self.ec.ncol): # this is kind of stupid since only 1 col intended, but maybe for future?
                for jj in range(self.ec.nrow):
                    print('Col ',ii, 'Row ',jj, ': %0.4f +/- %0.4f'%(data_mean[ii,jj,1],data_std[ii,jj,1]))

        
        if autoRange:
            above = np.where(data_mean[self.ec_col_index,:,1]>v_lim[1])[0]
            below = np.where(data_mean[self.ec_col_index,:,1]<v_lim[0])[0]
            dexs = np.sort(np.hstack((above,below)))
            # remove rows defined in self.dfb_rowindex_unlocked from dexs
            if dexs.size:
                if self.dfb_rowindex_unlocked:
                    common_dexs = list(set(dexs).intersection(self.dfb_rowindex_unlocked))
                    dexs = np.setdiff1d(dexs,common_dexs)
            
            if dexs.size:
                print('autoRange dectected.  Relocking feedback on row indicies: ',dexs)
                ar['flag']=True
                ar['indicies']=dexs
                for dex in dexs:
                    self.relock(dex,a_or_b='A')
                time.sleep(self.relock_delay)
                data_new = self.ec.getNewData(minimumNumPoints=npts,exactNumPoints=True,toVolts=True)
                data_new_mean = np.mean(data_new,axis=2)
                #data_new_std = np.std(data,axis=2)
                dv = data_mean - data_new_mean
                #print(dv[self.ec_col_index,:,1][[dexs]])
                ar['offset'][self.ec_col_index,:][[dexs]] = dv[self.ec_col_index,:,1][[dexs]]

        return data_mean, ar 

    # data packaging methods ------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------------------

    def convertToLegacyFormat(self,v_arr,data_ret,pd):
        for ii in range(self.nrow):
            ivdata = np.vstack((v_arr, data_ret[self.ec_col_index,ii,:,1])) # only return the feedback, not error
            if self.executeColdloadIVs:
                iv_dict = self.tp.createIVDictBB(ivdata,bath_temperature_commanded=pd['temp'],\
                                                 bath_temperature_measured=pd['measured_temperature'],\
                                                 bb_temperature_commanded=pd['Tbb_command'],\
                                                 bb_temperature_measured_before=pd['Tbb_i'],\
                                                 bb_temperature_measured_after=pd['Tbb_f'],\
                                                 bb_voltage_measured_before=None,\
                                                 bb_voltage_measured_after=None,\
                                                 feedback_resistance=self.cfg['calnums']['rfb'],\
                                                 bias_resistance=self.cfg['calnums']['rbias'])
            else:
                iv_dict = self.tp.createIVDictHeater(ivdata, temperature=pd['temp'], feedback_resistance=pd['feedback_resistance'],
                                                    heater_voltage=None,heater_resistance=None, 
                                                    measured_temperature=pd['measured_temperature'],
                                                    bias_resistance=pd['bias_resistance'])
            self.tp.addIVRun(self.colName, self.rowAddName[ii], iv_dict)
            self.tp.savePickle()
    
    # adr temperature control methods ---------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------------------

    def setupTemperatureController(self, channel, t_set=0.1,heaterRange=100,heaterResistance=100.0):
        print('Setting up temperature controller to servo mode on channel',channel, 'and regulating to %.1f mK'%(t_set*1000))
        # determine what is the current state of temperature control
        mode = self.adr.temperature_controller.getControlMode() # can be closed, open, off, or zone
        #cur_chan,autscan=adrTempControl.GetScan() # which channel is currently being read? 
        #cur_temp = adrTempControl.getTemperature(cur_chan) # read the current temperature from that channel
        #cur_tset = adrTempControl.getTemperatureSetPoint() # current temperature setpoint for controlling

        print('current mode: ',mode)

        if mode=='off':
            pass
            
        elif mode=='open':
            self.adr.temperature_controller.SetManualHeaterOut(0)
            time.sleep(1)

        elif mode=='closed':
            self.adr.temperature_controller.setTemperatureSetPoint(0) 

        self.adr.temperature_controller.setupPID(exciterange=3.16e-9, therm_control_channel=channel, 
                                                 ramprate=0.05, heater_resistance=heaterResistance,
                                                 heater_range=heaterRange,setpoint=t_set)

    def IsTemperatureStable(self, T_target, Tsensor=1,tol=.005,time_out=180.):
        ''' determine if the servo has reached the desired temperature '''
        
        if time_out < 10:
            print('Time for potential equilibration must be longer than 10 seconds')
            return False
        
        cur_temp=self.adr.temperature_controller.GetTemperature(Tsensor)
        it_num=0
        while abs(cur_temp-T_target)>tol:
            time.sleep(10.)
            cur_temp = self.adr.temperature_controller.GetTemperature(Tsensor)
            print('Current Temp: ' + str(cur_temp))
            it_num=it_num+1
            if it_num>round(int(time_out/10.)):
                print('exceeded the time required for temperature stability: %d seconds'%(round(int(10*it_num))))
                return False
        return True
    
    def overBias(self, Thigh,Tb,Vbias=0.5,Tsensor=1):
        ''' raise Tbath above Tc, overbias bolometer, cool back to base temperature while 
            keeping bolometer in the normal state
        '''
        print('Overbiasing detectors.  Raise temperature to %.1f mK, apply %.3f V, then cool to %.1f mK.'%(Thigh*1000,Vbias,Tb*1000))
        self.adr.temperature_controller.SetTemperatureSetPoint(Thigh) # raise Tb above Tc
        ThighStable = self.IsTemperatureStable(Thigh,Tsensor=Tsensor,tol=0.005,time_out=180.) # determine that it got to Thigh
        if ThighStable:
            print('Successfully raised Tb > %.3f.  Appling detector voltage bias and cooling back down.'%(Thigh))
        else:
            print('Could not get to the desired temperature above Tc.  Current temperature = ', adrTempControl.GetTemperature(Tsensor))
        self.vs.setvolt(Vbias) # voltage bias to stay above Tc
        self.adr.temperature_controller.SetTemperatureSetPoint(Tb) # set back down to Tbath, base temperature
        TlowStable = self.IsTemperatureStable(Tb,Tsensor=Tsensor,tol=0.002,time_out=180.) # determine that it got to Tbath target
        if TlowStable:
            print('Successfully cooled back to base temperature '+str(Tb))
        else:
            print('Could not cool back to base temperature'+str(Tb)+'. Current temperature = ', adrTempControl.GetTemperature(Tsensor))

    # blackbody temperature control methods ---------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------------- 

    def coldloadServoStabilizeWait(self,temp, loop_channel,tolerance,tbbServoMaxTime=5.0,tWaitIV=20.):
        ''' servo coldload to temperature T and wait for temperature to stabilize '''
        if temp>50.0:
            print('Blackbody temperature '+str(temp)+ ' exceeds safe range.  Tbb < 50K')
            sys.exit()
        
        print('Setting BB temperature to '+str(temp)+'K')
        self.ccon.setControlTemperature(temp=temp,loop_channel=loop_channel)
                    
        # wait for thermometer on coldload to reach tbb --------------------------------------------
        is_stable = self.ccon.isTemperatureStable(loop_channel,tolerance)
        stable_num=0
        while not is_stable:
            time.sleep(5)
            is_stable = self.ccon.isTemperatureStable(loop_channel,tolerance)
            stable_num += 1
            if stable_num*5/60. > tbbServoMaxTime:
                break
                
        print('Letting the blackbody thermalize for ',tWaitIV,' minutes.')
        time.sleep(60*tWaitIV)

    # higher-level IV collection methods ------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------------------

    def relock(self,dfb_row_index,a_or_b='A'):
        ''' 
            open and close the feedback loop for row

            dfb_row_index: <integer> index in line period to relock
            a_or_b: <str> select feedback to lock 'A' or 'B'
        '''

        if a_or_b=='A':
            self.cc.relock_fba(col=self.dfb_col_index, row=dfb_row_index) # used with Galen's update to cringe_control
        elif a_or_b=='B':
            self.cc.relock_fbb(col=self.dfb_col_index, row=dfb_row_index) # used with Galen's update to cringe_control
                

        # if row=='all':
        #     for ii in self.dfb_rowindex_locked:
        #     #self.cc.set_fb_lock(0,self.dfb_col_index,row,a_or_b) # unlock, # used with Gene's version of cringe_control
        #     #self.cc.set_fb_lock(1,self.dfb_col_index,row,a_or_b) # relock, # used with Gene's version of cringe_control
        #         if a_or_b=='A':
        #             self.cc.relock_fba(col=self.dfb_col_index, row=ii) # used with Galen's update to cringe_control
        #         elif a_or_b=='B':
        #             self.cc.relock_fbb(col=self.dfb_col_index, row=ii) # used with Galen's update to cringe_control
        # else:
        #     if a_or_b=='A':
        #         self.cc.relock_fba(col=self.dfb_col_index, row=row) # used with Galen's update to cringe_control
        #     elif a_or_b=='B':
        #         self.cc.relock_fbb(col=self.dfb_col_index, row=row) # used with Galen's update to cringe_control
                

    def iv_sweep(self, v_start=0.1,v_stop=0,v_step=0.01,sweepUp=False):
        ''' v_start: initial voltage
            v_stop: final voltage
            v_step: voltage step size
            sweepUp: if True sweeps in ascending voltage
        '''
        # build commanded voltage bias vector
        v_arr = np.arange(v_stop,v_start+v_step,v_step)
        if not sweepUp:
            v_arr=v_arr[::-1]
        
        # set to initial point and relock all
        self.vs.setvolt(v_arr[0])
        time.sleep(self.relock_delay)
        for ii in self.dfb_rowindex_locked:
            self.relock(ii)
        time.sleep(self.relock_delay)

        # take data at all voltage bias points
        N=len(v_arr)
        data_raw = np.zeros((self.ec.ncol,self.ec.nrow,N,2))
        ar_arr = []
        for ii in range(N):
            self.vs.setvolt(v_arr[ii])
            time.sleep(self.sample_delay)
            data, ar = self.getIVdataPt(npts=self.numpts,v_lim=self.v_autoRange_lim,autoRange=self.autoRange)
            data_raw[:,:,ii,:] = data
            ar_arr.append(ar)

        # build corrected arrays
        data_corr=data_raw.copy()
        if self.autoRange:
            for ii in range(N):
                if ar_arr[ii]['flag']:
                    for jj in range(self.nrow):
                        data_corr[self.ec_col_index,jj,ii+1:,1]=data_corr[self.ec_col_index,jj,ii+1:,1] + ar_arr[ii]['offset'][self.ec_col_index,jj] # ??

        if self.showPlot:
            for ii in range(self.ec.ncol):
                fig = plt.figure(ii)
                fig.suptitle('Column %d'%ii)
                for jj in range(self.ec.nrow):
                    plt.subplot(211)
                    if self.autoRange:
                        plt.plot(v_arr,data_corr[ii,jj,:,1],'-')
                        #plt.plot(v_arr,data_raw[ii,jj,:,1],'o-') 
                    else:
                        plt.plot(v_arr,data_raw[ii,jj,:,1],'-')
                    plt.xlabel('V_bias')
                    plt.ylabel('V_fb')
                    plt.subplot(212)
                    if self.autoRange:
                        plt.plot(v_arr,data_corr[ii,jj,:,0])
                    else:
                        plt.plot(v_arr,data_raw[ii,jj,:,0])
                    plt.xlabel('V_bias')
                    plt.ylabel('V_err')
                plt.legend(range(self.ec.nrow))
            plt.show()
        return v_arr, data_corr, data_raw, ar_arr

    def iv_v_tbath(self,V_overbias,Tbb_command=None):
        ''' loop over bath temperatures and collect IV curves '''
        iv_structure = []
        for jj, temp in enumerate(self.cfg['runconfig']['bathTemperatures']): # loop over temperatures
            if temp == 0: 
                print('temp = 0, which is a flag to take an IV curve as is without commanding the temperature controller.')
            elif self.cfg['voltage_bias']['overbias']: # overbias case
                self.overBias(Thigh=self.cfg['voltage_bias']['overbiasThigh'],Tb=temp,Vbias=V_overbias,Tsensor=self.cfg['runconfig']['thermometerChannel'])
                print('Overbias complete.  Now waiting %d seconds before taking IV curve'%(self.cfg['voltage_bias']['overbiasWait']))
                time.sleep(self.cfg['voltage_bias']['overbiasWait'])
            else:
                print('Setting T_bath = %.1f'%(temp*1000))
                self.adr.temperature_controller.SetTemperatureSetPoint(temp)
                stable = self.IsTemperatureStable(temp,Tsensor=self.cfg['runconfig']['thermometerChannel'],
                                            tol=self.cfg['runconfig']['temp_tolerance'],time_out=180.)
                if not stable:
                    print('cannot obtain a stable temperature at %.3f mK !! I\'m going ahead and taking an IV anyway.'%(temp*1000))
                    
            # Grab bath temperature before/after and run IV curve
            Tb_i = self.adr.temperature_controller.GetTemperature(self.cfg['runconfig']['thermometerChannel'])
            if self.executeColdloadIVs:
                Tbb_i = self.ccon.getTemperature()
                print('Running IV at Tbath = %.1f mK and Tbb = %.1f K\n'%(temp*1000,Tbb_command),'#'*80,'\n')
            else:
                Tbb_i=Tbb_f=Tbb_command=None
                print('Running IV at Tbath = %.1f mK'%(Tb_i*1000),'#'*80,'\n')
            
            v_arr, data_corr, data_raw, ar_arr = self.iv_sweep(v_start=self.cfg['voltage_bias']['v_start'], v_stop=self.cfg['voltage_bias']['v_stop'],
                                                              v_step=self.cfg['voltage_bias']['v_step'],
                                                              sweepUp=self.cfg['voltage_bias']['sweepUp'])
            Tb_f = self.adr.temperature_controller.GetTemperature(self.cfg['runconfig']['thermometerChannel'])
            if self.executeColdloadIVs:
                Tbb_f = self.ccon.getTemperature()

            # save the data            
            if self.cfg['runconfig']['dataFormat']=='legacy':
                print('saving IV curve in legacy format')
                pd = {'temp':temp,'feedback_resistance':self.cfg['calnums']['rfb'], 'measured_temperature':Tb_f,
                        'bias_resistance':self.cfg['calnums']['rbias'],'Tbb_i':Tbb_i, 'Tbb_f':Tbb_f,'Tbb_command':Tbb_command}
                self.convertToLegacyFormat(v_arr,data_corr,pd=pd)
                
            else:
                print('Saving data in new format')
                # ret_dict keys: 'v', 'config', 'ivdict'
                # ivdict has structure: ivdict[iv##]: 'Treq', 'Tb_i', 'Tb_f','data','flags' 
                iv_dict={'Treq':temp, 'Tb_i':Tb_i, 'Tb_f':Tb_f, 'data':data_corr, 'data_raw':data_raw, 'autoRangeDict': ar_arr}
                if self.executeColdloadIVs:
                    iv_dict['coldload']={'Tbb_i':Tbb_i, 'Tbb_f':Tbb_f,'Tbb_command':Tbb_command}
                else:
                    iv_dict['coldload']=None
                iv_structure.append(iv_dict)
                #ret_dict = {'v':v_arr,'config':self.cfg,'iv':iv_structure}
                #pickle.dump( ret_dict, open( self.pickle_file, "wb" ) )
        ret_dict = {'v':v_arr,'config':self.cfg,'iv':iv_structure}
        return ret_dict
    
############################################################################################################
############################################################################################################
############################################################################################################
# main method starts here.

def main():

    # error handling
    # check if adr_gui running, dastard commander, cringe?, dcom ...

    # open config file
    with open(sys.argv[1], 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # open tower config file
    with open(cfg['runconfig']['tower_config'], 'r') as ymlfile:
        tcfg = yaml.load(ymlfile)

    # determine column and rows
    baystring = 'Column' + cfg['detectors']['Column']
    if cfg['detectors']['Rows'] == 'all':
        mux_rows = list(range(32)) 
    else:
        mux_rows = list(cfg['detectors']['Rows'])

    # define where the data is store
    if not os.path.exists(cfg['io']['RootPath']):
        print('The path: %s does not exist! Making directory now.'%cfg['io']['RootPath'])
        os.makedirs(cfg['io']['RootPath'])
    localtime=time.localtime()
    thedate=str(localtime[0])+'%02i'%localtime[1]+'%02i'%localtime[2]
    etime = str(time.time()).split('.')[0]
    filename = cfg['io']['RootPath']+cfg['detectors']['DetectorName']+'_'+baystring+'_ivs_'+thedate+'_'+etime
    pickle_file=filename+'.pkl' 

    # defined tower card addresses for needed voltage sources
    bias_channel=tcfg['db_tower_channel_%s'%cfg['detectors']['Column']]
    sq2fb_tower_channel=tcfg['sq2fb_tower_channel_%s'%cfg['detectors']['Column']]

    # punt if you are asking for a temperature higher than 2K
    for t in cfg['runconfig']['bathTemperatures']:
        if t>2.0:
            print('setpoint temperature ',t,' greater than 2.0K.  Not allowed!  Abort!')
            sys.exit()
    
    # handle overbias voltage value; only used if overbias selected.
    if cfg['voltage_bias']['v_overbias']==None:
        V_overbias = cfg['voltage_bias']['v_start']
    else:
        V_overbias = cfg['voltage_bias']['v_overbias']

    
    
    # instanciate needed classes ------------------------------------------------------------------------------------------------------------
    ec = EasyClient() # easy client for streaming data
    adr = AdrSystem(app=None, lsync=ec.lsync, number_mux_rows=ec.nrow, dfb_number_of_samples=ec.nSamp, doinit=False) # adr: temp control, heatswitch, servo vs. ramp switch
    vs = BlueBox(port='vbox', version=cfg['voltage_bias']['source'], address=tcfg['db_tower_address'], channel=bias_channel)
    sq2fb = BlueBox(port='vbox', version='tower', address=tcfg['sq2fb_tower_address'], channel= sq2fb_tower_channel) # obsolete with new electronics
    cc=CringeControl() # cringe control instance, for control of crate

    if cfg['runconfig']['dataFormat']=='legacy':
        tp = tespickle.TESPickle(pickle_file)
    else: tp=None

    if 'coldload' in cfg.keys():
        if cfg['coldload']['execute']:
            ccon = Cryocon22()
            ccon.controlLoopSetup(loop_channel=cfg['coldload']['loop_channel'],control_temp=cfg['coldload']['bbTemperatures'][0],
                                t_channel=cfg['coldload']['t_channel'],PID=cfg['coldload']['PID'], heater_range='low') # setup BB control
        else: ccon=None

    showPlot=cfg['runconfig']['showPlot']
    verbose=False
    IV = ivCurve(cfg,cc,ec,vs,adr,ccon,tp,pickle_file,showPlot,verbose) # class for all sub-methods above

    if cfg['voltage_bias']['source']=='tower' and cfg['voltage_bias']['v_autobias']>2.5:
        print('tower can only source 2.5V.  Switching v_autobias to 2.5V')
        cfg['runconfig']['v_autobias']=2.5

    if cfg['runconfig']['setupTemperatureServo'] and cfg['runconfig']['bathTemperatures'][0] !=0: # initialize temperature servo if asked
        if adr.temperature_controller.getControlMode() == 'closed':
            t_set = adr.temperature_controller.getTemperatureSetPoint()
        else:
            t_set = 0.05
        IV.setupTemperatureController(channel=cfg['runconfig']['thermometerChannel'], t_set=t_set, 
                                      heaterRange=cfg['runconfig']['thermometerHeaterRange'],heaterResistance=100.0)
    
    # -----------------------------------------------------------------------------------------------------
    #Main loop starts here: loop over BB temperatures and bath temperatures, run multiplexed IVs 
    N = len(cfg['runconfig']['bathTemperatures'])
    iv_dict={} 

    print('\n'*5)
    print('\n\nStarting IV acquisition script on ',baystring,'*'*80)
    print('Rows: ',mux_rows,'\nBath temperatures:',cfg['runconfig']['bathTemperatures'],'\nData will be saved in file: ',pickle_file)
    if 'coldload' in cfg.keys(): 
        if cfg['coldload']['execute']:
            print('Coldload Temperatures: ',cfg['coldload']['bbTemperatures'])
            iv_results = []
            for ii, tbb in enumerate(cfg['coldload']['bbTemperatures']): # loop over coadload temps
                if tbb>50.0:
                    print('Blackbody temperature '+str(tbb)+ ' exceeds safe range.  Tbb < 50K')
                    sys.exit()
                elif tbb==0:
                    print('Tbb = 0 is a flag to take IV at current coldload temperature.  No servoing')
                    tWaitIV=0
                else:
                    ccon.setControlState(state='on') # this command not needed every loop.  Too stupid to figure this out now.
                    if ii==0 and cfg['coldload']['immediateFirstMeasurement']: #skip the wait time for 1st measurement
                        tWaitIV = 0
                    else: 
                        tWaitIV = cfg['coldload']['tWaitIV']
                    IV.coldloadServoStabilizeWait(temp=tbb, loop_channel=cfg['coldload']['loop_channel'],
                                        tolerance=cfg['coldload']['tempBB_tolerance'],tWaitIV=tWaitIV)

                ret_dict = IV.iv_v_tbath(V_overbias,Tbb_command=tbb)
                if cfg['runconfig']['dataFormat']!='legacy':
                    iv_results.append(ret_dict['iv'])
                    ret_dict['iv']=iv_results
                    IV.savePickle(ret_dict)
            if cfg['coldload']['turn_off_BB']:
                ccon.setControlState(state='off')
                
        else:
            ret_dict = IV.iv_v_tbath(V_overbias)
            if cfg['runconfig']['dataFormat']!='legacy':
                ret_dict['iv'] = [ret_dict['iv']] # nested list to ease data processing in ivAnalyzer.py
                IV.savePickle(ret_dict)
    else:
        ret_dict = IV.iv_v_tbath(V_overbias)
        if cfg['runconfig']['dataFormat']!='legacy':
            ret_dict['iv'] = [ret_dict['iv']]
            IV.savePickle(ret_dict) # nested list to ease data processing in ivAnalyzer.py

    if cfg['voltage_bias']['setVtoZeroPostIV']:
        vs.setvolt(0)    

    print('Data saved at: ',pickle_file)
            
if __name__ == '__main__': 
    main()

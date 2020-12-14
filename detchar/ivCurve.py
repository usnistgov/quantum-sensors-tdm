#! /usr/bin/env python
'''
ivCurve.py
11/2020
@author JH

Script to take IV curves on a single column of detectors as a function of bath temperature. 
Assumes multiplexer is setup in cringe before running.
Must exit out of adr_gui  

usage:
./ivCurve.py <config_filename>

requirements:
iv configuration file
tower configuration file
pyYAML

to do:
DFB_CARD_INDEX
error handling: Tbath, v_bias
overbias
tesacquire used for unmuxed case. 
what to do about tesacquire, tesanalyze, sweeper, LoadMuxSettings, singleMuxIV 


Notes:
EasyClient getNewData() method returns following structure:
(column, row (sequence length), sample number, 0/1=error/fb)
The 0th index (column) has perhaps a funny indexing
0: dfbx2 ch1
1: dfbx2 ch2
2: dfb/clk ch1 
This is defined by what fibers are installed where.

'''

# standard python module imports
import sys, os, yaml, time, subprocess, re
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
    def __init__(self,cfg,cc,ec,vs,adr,ccon,tp,showPlot,verbose):
        # pass a bunch of other class instances
        self.cc = cc # cringe control
        self.ec = ec # easyClient
        self.vs = vs # voltage source to drive TES voltage bias
        self.adr = adr # adr
        self.ccon = ccon # cryocon
        self.tp = tp # tes pickle  

        # other stuff
        self.cfg = cfg 
        self.rows_not_locked = self.cfg['dfb']['rows_not_locked']
        self.showPlot = showPlot
        self.verbose = verbose

    # I don't use the two below, but may be useful in future for error handling since much setup is assumed for this script to work
    # -----------------------------------------------------------------------------------------------------------------------------
    
    def findThisProcess(self, process_name ):
        ps = subprocess.Popen("ps -eaf | grep "+process_name, shell=True, stdout=subprocess.PIPE)
        output = ps.stdout.read()
        ps.stdout.close()
        ps.wait()
        return output

    def isThisRunning(self, process_name ):
        output = findThisProcess( process_name )
        if re.search('path/of/process'+process_name, output) is None:
            return False
        else:
            return True

    # data collection methods -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------------------

    def getDataAverageAndCheck(self,v_min=0.1,v_max=0.9,npts=10000):
        ''' This method called for each IV point to collect data.  '''
        flag=False
        data = self.ec.getNewData(minimumNumPoints=npts,exactNumPoints=True,toVolts=True)
        data_mean = np.mean(data,axis=2)
        data_std = np.std(data,axis=2)

        if self.verbose:
            for ii in range(self.ec.ncol):
                for jj in range(self.ec.nrow):
                    print('Col ',ii, 'Row ',jj, ': %0.4f +/- %0.4f'%(data_mean[ii,jj,1],data_std[ii,jj,1]))

        a = data_mean[:,:,1][data_mean[:,:,1]>v_max]
        b = data_mean[:,:,1][data_mean[:,:,1]<v_min]
        if a.size: 
            print('Value above ',v_max,' detected')
            # relock here
            flag=True
        if b.size:
            print('Value below ',v_min,' detected')
            # relock here
            flag=True

        # have some error handling about if std/mean > threshold

        return data_mean, flag

    # data packaging methods ------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------------------

    def convertToLegacyFormat(self,v_arr,data_ret,nrows,pd,column,mux_rows,column_index=0):
        for ii in range(nrows):
            ivdata = np.vstack((v_arr, data_ret[column_index,ii,:,1])) # only return the feedback, not error
            iv_dict = self.tp.createIVDictHeater(ivdata, temperature=pd['temp'], feedback_resistance=pd['feedback_resistance'],
                                                    heater_voltage=None,heater_resistance=None, 
                                                    measured_temperature=pd['measured_temperature'],
                                                    bias_resistance=pd['bias_resistance'])
            self.tp.addIVRun(column, mux_rows[ii], iv_dict)
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

    def coldloadControlInit(self):
        print('to be written')

    def coldloadServoStabilizeWait(self,temp, loop_channel,tolerance,postServoBBsettlingTime,tbbServoMaxTime):
        ''' servo coldload to temperature T and wait for temperature to stabilize '''
        if tbb>50.0:
            print('Blackbody temperature '+str(tbb)+ ' exceeds safe range.  Tbb < 50K')
            sys.exit()
        
        print('setting BB temperature to '+str(tbb)+'K')
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
                
        print('Letting the blackbody thermalize for ',postServoBBsettlingTime,' minutes.')
        time.sleep(60*postServoBBsettlingTime)

    # higher-level IV collection methods ------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------------------

    def relock(self,col_index=1,a_or_b='A'):
        for ii in range(self.ec.nrow):
            if self.rows_not_locked !=None and ii in self.rows_not_locked:
                self.cc.set_fb_lock(0,col_index,ii,a_or_b)
            else:
                self.cc.set_fb_lock(0,col_index,ii,a_or_b)
                self.cc.set_fb_lock(1,col_index,ii,a_or_b)

    def iv_sweep(self, v_start=0.1,v_stop=0,v_step=0.01,sweepUp=False):
        ''' v_start: initial voltage
            v_stop: final voltage
            v_step: voltage step size
            sweepUp: if True sweeps in ascending voltage
        '''
        v_arr = np.arange(v_stop,v_start+v_step,v_step)
        if not sweepUp:
            v_arr=v_arr[::-1]
        
        # set to initial point are relock
        self.vs.setvolt(v_arr[0])
        self.relock()

        N=len(v_arr)
        data_ret = np.zeros((self.ec.ncol,self.ec.nrow,N,2))
        flags = np.zeros(N)
        for ii in range(N):
            self.vs.setvolt(v_arr[ii])
            data, flag = self.getDataAverageAndCheck()
            data_ret[:,:,ii,:] = data
            flags[ii] = flag
        
        if self.showPlot:
            for ii in range(self.ec.ncol):
                fig = plt.figure(ii)
                fig.suptitle('Column %d'%ii)
                for jj in range(self.ec.nrow):
                    plt.subplot(211)
                    plt.plot(v_arr,data_ret[ii,jj,:,1])
                    plt.xlabel('V_bias')
                    plt.ylabel('V_fb')
                    plt.subplot(212)
                    plt.plot(v_arr,data_ret[ii,jj,:,0])
                    plt.xlabel('V_bias')
                    plt.ylabel('V_err')
                plt.legend(range(self.ec.nrow))
            plt.show()
        return v_arr, data_ret, flags

    def iv_v_tbath(self,V_overbias):
        ''' loop over bath temperatures and collect IV curves '''
        for jj, temp in enumerate(self.cfg['runconfig']['bathTemperatures']): # loop over temperatures
            if temp == 0: 
                print('temp = 0, which is a flag to take an IV curve as is without commanding the temperature controller.')
            elif self.cfg['voltage_bias']['overbias']: # overbias case
                self.overBias(Thigh=self.cfg['voltage_bias']['overbiasThigh'],Tb=temp,Vbias=V_overbias,Tsensor=self.cfg['runconfig']['thermometerChannel'])
            else:
                self.adr.temperature_controller.SetTemperatureSetPoint(temp)
                stable = self.IsTemperatureStable(temp,Tsensor=self.cfg['runconfig']['thermometerChannel'],
                                            tol=self.cfg['runconfig']['temp_tolerance'],time_out=180.)
                if not stable:
                    print('cannot obtain a stable temperature at %.3f mK !! I\'m going ahead and taking an IV anyway.'%(temp*1000))
                    
            # Grab bath temperature before/after and run IV curve
            Tb_i = self.adr.temperature_controller.GetTemperature(self.cfg['runconfig']['thermometerChannel'])
            v_arr, data_ret, flags = self.iv_sweep(v_start=self.cfg['voltage_bias']['v_start'], v_stop=self.cfg['voltage_bias']['v_stop'],
                                              v_step=self.cfg['voltage_bias']['v_step'],
                                              sweepUp=self.cfg['voltage_bias']['sweepUp'])
            Tb_f = self.adr.temperature_controller.GetTemperature(self.cfg['runconfig']['thermometerChannel'])

            # save the data            
            if self.cfg['runconfig']['dataFormat']=='legacy':
                print('saving IV curve in legacy format')
                pd = {'temp':temp,'feedback_resistance':self.cfg['calnums']['rfb'], 'measured_temperature':Tb_f,
                    'bias_resistance':self.cfg['calnums']['rbias']}
                self.convertToLegacyFormat(v_arr,data_ret,nrows=self.ec.nrow,pd=pd,
                                      column=self.cfg['detectors']['Column'],mux_rows=self.cfg['detectors']['Rows'],column_index=0)
            else:
                print('Saving data in new format')
                # ret_dict keys: 'v', 'config', 'ivdict'
                # ivdict has structure: ivdict[iv##]: 'Treq', 'Tb_i', 'Tb_f','data','flags' 
                iv_dict['iv%02d'%jj]={'Treq':temp, 'Tb_i':Tb_i, 'Tb_f':Tb_f, 'data':data_ret, 'flags':flags}
                ret_dict = {'v':v_arr,'config':cfg,'ivdict':ivdict}
                pickle.dump( ret_dict, open( pickle_file, "wb" ) )

############################################################################################################
############################################################################################################
############################################################################################################
# main method starts here.

def main():

    # error handling
    # check if adr_gui running, dastard commander, cringe?, dcom ...

    verbose=False
    showPlot=True

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
    filename = cfg['io']['RootPath']+cfg['detectors']['DetectorName']+'_'+baystring+'_ivs_'+thedate+'_'+cfg['io']['suffix']
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

    print('\n\nStarting IV acquisition script on ',baystring,'*'*80,'\nRows: ',mux_rows,'\nTemperatures:',cfg['runconfig']['bathTemperatures'],'\nData will be saved in file: ',pickle_file)
    
    # instanciate needed classes ------------------------------------------------------------------------------------------------------------
    ec = EasyClient() # easy client for streaming data
    adr = AdrSystem(app=None, lsync=ec.lsync, number_mux_rows=ec.nrow, dfb_number_of_samples=ec.nSamp, doinit=False) # adr: temp control, heatswitch, servo vs. ramp switch
    vs = BlueBox(port='vbox', version=cfg['voltage_bias']['source'], address=tcfg['db_tower_address'], channel=bias_channel)
    sq2fb = BlueBox(port='vbox', version='tower', address=tcfg['sq2fb_tower_address'], channel= sq2fb_tower_channel)
    cc=CringeControl() # cringe control instance, for control of crate

    if cfg['runconfig']['dataFormat']=='legacy':
        tp = tespickle.TESPickle(pickle_file)
    else: tp=None

    if 'coldload' in cfg.keys():
        if cfg['coldload']['execute']:
            ccon = Cryocon22()
            ccon.controlLoopSetup(loop_channel=cfg['coldload']['loop_channel'],control_temp=cfg['coldload']['bbTemperatures'][0],
                                t_channel=cfg['coldload']['t_channel'],PID=cfg['coldload']['PID'], heater_range=cfg['coldload']['heater_range']) # setup BB control
        else: ccon=None

    IV = ivCurve(cfg,cc,ec,vs,adr,ccon,tp,showPlot,verbose) # class for all sub-methods above

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

    if 'coldload' in cfg.keys(): 
        if cfg['coldload']['execute']:
            for ii, tbb in enumerate(cfg['coldload']['bbTemperatures']): # loop over coadload temps
                if tbb>50.0:
                    print('Blackbody temperature '+str(tbb)+ ' exceeds safe range.  Tbb < 50K')
                    sys.exit()
                elif tbb==0:
                    print('Tbb = 0 is a flag to take a current temperature.  No servoing')
                else:
                    ccon.setControlState(state='on') # this command not needed every loop.  Too stupid to figure this out now.
                    if ii==0 and cfg['coldload']['immediateFirstMeasurement']: #skip the wait time for 1st measurement
                        postServoBBsettlingTime = 0
                    else: postServoBBsettlingTime = cfg['coldload']['postServoBBsettlingTime']
                        
                IV.coldloadServoStabilizeWait(temp=tbb, loop_channel=cfg['coldload']['loop_channel'],
                                        tolerance=cfg['coldload']['temp_tolerance'], 
                                        postServoBBsettlingTime=postServoBBsettlingTime,
                                        tbbServoMaxTime=cfg['coldload']['tbbServoMaxTime'])

                IV.iv_v_tbath(V_overbias)
        else:
            IV.iv_v_tbath(V_overbias)
    else:
        IV.iv_v_tbath(V_overbias)

    if cfg['voltage_bias']['setVtoZeroPostIV']:
        vs.setvolt(0)    
            
if __name__ == '__main__': 
    main()

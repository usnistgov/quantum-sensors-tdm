#! /usr/bin/env python
'''
ivCurve.py
11/2020
@author JH

Script to take IV curves on a single column of detectors as a function of bath temperature. 

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

'''

# standard python module imports
import sys, os, yaml, time
import numpy as np
# from PyQt4 import Qt
# import time

# # QSP written module imports
from adr_system import AdrSystem
from instruments import BlueBox 
from . import tespickle 
# import sweeper4_mod as sweeper
# import adr_system
# import tesacquire
# import tespickle
# sys.path.append('/home/pcuser/gittrunk/nist_lab_internals/DetectorMapping/')
# import LoadMuxSettings
# import singleMuxIV as smIV


########################################################################################################
########################################################################################################
########################################################################################################
# open config file
with open(sys.argv[1], 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

# open tower config file
with open(cfg['mux_settings']['tower_config'], 'r') as ymlfile:
    tcfg = yaml.load(ymlfile)

# define column and rows
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
pickle_file=cfg['io']['RootPath']+cfg['detectors']['DetectorName']+'_'+baystring+'_ivs_'+thedate+'_'+cfg['io']['suffix']+'.pkl' 

DFB_CARD_INDEX=0 #

# defined tower card addresses for needed voltage sources
bias_channel=tcfg['db_tower_channel_%s'%cfg['detectors']['Column']]
sq2fb_tower_channel=tcfg['sq2fb_tower_channel_%s'%cfg['detectors']['Column']]

# overbias settings (drives channels that cannot by autobiased normal by raising and lowering the temp)
OVERBIAS = True

OverBiasThigh=0.22 #0.35 # temp (K) to raise bath temperature #.3
OverBiasVbias=.60 # voltage bias to applied before lowering temperature
postOverBiasWaitPeriod= 60. # time to let Tbath settle after overbias (secs)

# temperature settings
Tsensor=2 #2#1
for t in cfg['runconfig']['bathTemperatures']:
    if t>2.0:
        print('setpoint temperature ',t,' greater than 2.0K.  Not allowed!  Abort!')
        sys.exit()

UseLoadMuxSettings=True
if UseLoadMuxSettings:
    print('loading mux settings from muxbias.csv')
    lms=LoadMuxSettings.LoadMuxSettings('/home/pcuser/gittrunk/nist_lab_internals/DetectorMapping/muxbias.csv')
    #lms=LoadMuxSettings.LoadMuxSettings('/home/pcuser/gittrunk/nist_lab_internals/DetectorMapping/muxbias_201602_4pixArpiBoard.csv')
    rows,radacvalues,sq2fbvalues,IVstart,IVstop=lms.FormatForIV(column,skipzeros=True,skipNotMeasured=True)
    print('rows:', rows)
    print('radacvalues:', radacvalues)
    print('sq2fbvalues:', sq2fbvalues)
    print('IVstart', IVstart)
    print('IVstop',IVstop)
    
    #input('look ok?')
    
else:
    rows =      [0] 
    radacvalues=[7000]
    sq2fbvalues=[2030]

# dfb settings
if cfg['runconfig']['multiplex']==True:
    number_mux_rows = len(mux_rows)
else:
    number_mux_rows = 2

dfb_settling_time = cfg['dfb']['lsync'] - cfg['dfb']['nsamp'] - 1

# IS THIS REALLY NEEDED?  LET'S NOT DO THIS FOR NOW
# # set the number of points to average for a single IV data point.  I typically see 60Hz pickup so I'd like to 
# # average over several 60 Hz periods, that's what num_frames is setup to do
# sampling_interval = 8.e-9*number_mux_rows*cfg['dfb']['lsync']
# num_frames = 2**(int(round(np.log2(1/sampling_interval/60.*8)))) # number of data points in IV point to average
# dwell_time = sampling_interval*num_frames+0.5


##########################################################################################################
##########################################################################################################
##########################################################################################################

def removeZerosIV(iv):
    ''' Removes the zeros put in by jump 
        Looking for bias voltage in iv[0,:] and fb voltage in iv[1,:] '''

    for pnt in range(len(iv[1])-1,0,-1):
        if iv[1,pnt] == 0:
            iv = np.hstack((iv[:,: pnt],iv[:,pnt+1 :]))
    return iv

def IsTemperatureStable(T_target,adr, Tsensor=cfg['runconfig']['thermometerChannel'],tol=.005,time_out=180.):
    ''' determine if the servo has reached the desired temperature '''
    
    if time_out < 10:
        print('Time for potential equilibration must be longer than 10 seconds')
        return False
    
    cur_temp=adr.GetTemperature(Tsensor)
    it_num=0
    while abs(cur_temp-T_target)>tol:
        time.sleep(10.)
        cur_temp = adr.GetTemperature(Tsensor)
        print('Current Temp: ' + str(cur_temp))
        it_num=it_num+1
        if it_num>round(int(time_out/10.)):
            print('exceeded the time required for temperature stability: %d seconds'%(round(int(10*it_num))))
            return False
    return True

def overBias(adrTempControl,voltage_sweep_source,Thigh,Tb,Vbias=0.5,Tsensor=cfg['runconfig']['thermometerChannel']):
    ''' raise Tbath above Tc, overbias bolometer, cool back to base temperature while 
        keeping bolometer in the normal state
    '''
    adrTempControl.SetTemperatureSetPoint(Thigh) # raise Tb above Tc
    ThighStable = IsTemperatureStable(Thigh,adrTempControl,Tsensor=Tsensor,tol=0.005,time_out=180.) # determine that it got to Thigh
    if ThighStable:
        print('Successfully raised Tb > %.3f.  Appling detector voltage bias and cooling back down.'%(Thigh))
    else:
        print('Could not get to the desired temperature above Tc.  Current temperature = ', adrTempControl.GetTemperature(Tsensor))
    #voltage_sweep_source.setvolt(2.5)
    #time.sleep(1.0)
    voltage_sweep_source.setvolt(Vbias) # voltage bias to stay above Tc
    adrTempControl.SetTemperatureSetPoint(Tb) # set back down to Tbath, base temperature
    TlowStable = IsTemperatureStable(Tb,adrTempControl,Tsensor=Tsensor,tol=0.002,time_out=180.) # determine that it got to Tbath target
    if TlowStable:
        print('Successfully cooled back to base temperature '+str(Tb))
    else:
        print('Could not cool back to base temperature'+str(Tb)+'. Current temperature = ', adrTempControl.GetTemperature(Tsensor))
    
    

############################################################################################################
############################################################################################################
############################################################################################################
# main script starts here.

def main():
    print('\n\nStarting IV acquisition script on ',baystring,'*'*80,'\nRows: ',mux_rows,'\nTemperatures:',cfg['runconfig']['bathTemperatures'],'\nData will be saved in file: ',pickle_file)
    # get the current time
    t0 = time.time()
    # instanciate needed classes
    app = Qt.QApplication(sys.argv) # will this work? is it needed?
    adr = AdrSystem(app=app, lsync=cfg['dfb']['lsync'], number_mux_rows=number_mux_rows, dfb_settling_time=dfb_settling_time, \
                                 dfb_number_of_samples=cfg['dfb']['nsamp'], \
                                 doinit=False)

    # Set up the voltage source that creates the voltage bias and the voltage source for the SQ2fb (typically both tower)
    voltage_sweep_source = BlueBox(port='vbox', version=cfg['runconfig']['voltageBiasSource'], address=tcfg['db_tower_address'], channel=bias_channel)
    sq2fb = BlueBox(port='vbox', version='tower', address=tcfg['sq2fb_tower_address'], channel= sq2fb_tower_channel)
    tesacq = tesacquire.TESAcquire(app=app)
    tes_pickle = tespickle.TESPickle(pickle_file)

    if cfg['runconfig']['voltageBiasSource']=='tower' and cfg['runconfig']['v_autobias']>2.5:
        print('tower can only source 2.5V.  Switching v_autobias to 2.5V')
        cfg['runconfig']['v_autobias']=2.5

    # setup crate for unmuxed case
    if not cfg['runconfig']['multiplex']:
        print('Setting up the DFB for non-multiplexed IVs')
        tesacq.dcSet32(0, 0, adr.crate)  # turn on one SQ1 bias with RA8 card, sets all other values to 0
        tesacq.dfBSetSame(adr.crate, DFB_CARD_INDEX, number_mux_rows, dfb_adc, cfg['dfb']['dac'], cfg['dfb']['P'],
                          cfg['dfb']['I'])  # hard coded that it is dfb_card[0] in the crate class
        tesacq.initRAMAll(adr.crate)  # wipe out all the junk in memory for each ra8 card
        tesacq.raClearAll(
            adr.crate)  # set all channels on all ra8 cards to 0 for both high and low.  All channels enabled
        tesacq.pciSetup(pci_mask, pci_firmware_delay, cfg['dfb']['nsamp'])  # setup the PCI card, will this fail?
        # tesacquire.pci.setNumberOfFrames(num_frames)
        print('DMA size =', tesacquire.pci.getDMASize())

    #Initialize sweeper
    sw = sweeper.Sweeper(app, adr.crate.dfb_cards[DFB_CARD_INDEX], voltage_sweep_source, numberofrows = number_mux_rows, pci_intitialized = 'False')
    time.sleep(1)
    sw.set_data_directory(root_path)

    if not LoopOverHeaterVoltage:
        heater_voltage=None

    #Main loop starts here: loop over temperatures, do IV for each TES
    temp_num=0
    for temp in cfg['runconfig']['bathTemperatures']:
        print('IVs at Temperature',temp)
        temp_num=temp_num+1
        if temp == 0:
            print('temp = 0, which is a flag to not servo the temperature.')
        elif cfg['runconfig']['overbias']:
            # We're going to change the temperature to OverBiasThigh, so need to go to temp
            print('')
        else:
            adr.temperature_controller.SetTemperatureSetPoint(temp)
            stable = IsTemperatureStable(temp,adr=adr.temperature_controller, Tsensor=cfg['runconfig']['thermometerChannel'],tol=.005,time_out=180.)
            if not stable:
                print('cannot obtain a stable temperature at %.3f mK !! I\'m going ahead and taking an IV anyway.'%(temp*1000))

        # take the IV curves
        if cfg['runconfig']['multiplex'] == True:
            if cfg['runconfig']['overbias']:
                print('Overbiasing at Temperature', OverBiasThigh)
                overBias(adr.temperature_controller, voltage_sweep_source, Thigh=OverBiasThigh, Tb=temp,
                         Vbias=OverBiasVbias, Tsensor=Tsensor)
                print('Overbias complete.  Letting bath temperature stabilize for ' + str(postOverBiasWaitPeriod) + 's.')
                time.sleep(postOverBiasWaitPeriod)

            print('Entering multiplexed IV function' + '*' * 80)
            v, vfb_array = smIV.singleMuxIV(app=app,voltage_start=cfg['runconfig']['v_start'], voltage_end=cfg['runconfig']['v_stop'],
                                                    voltage_step=cfg['runconfig']['v_step'],
                                                    column=column, rows=mux_rows, rows_not_locked=rows_not_locked,
                                                    ADCoffset=dfb_adc, dfb_dac = cfg['dfb']['dac'], sq2fbvalue=sq2fb_value_muxedIV,
                                                    dfb_numberofsamples=cfg['dfb']['nsamp'], lsync=cfg['dfb']['lsync'], dfb_p=cfg['dfb']['P'],
                                                    dfb_i=cfg['dfb']['I'],
                                                    tes_bias=voltage_sweep_source, sq2fb=sq2fb,
                                                    dfb=adr.crate.dfb_cards[DFB_CARD_INDEX],
                                                    normal_voltage_blast=normal_voltage_blast,
                                                    plotter=plotter)

            print('Exited multiplexed IV function' + '*' * 80)
            bath_temperature_measured = adr.temperature_controller.GetTemperature(cfg['runconfig']['thermometerChannel'])
            # Save the results in the pickle; make same format as unmultiplexed IVs in series
            for ii in range(len(mux_rows)):
                ivdata = np.vstack((v, vfb_array[ii]))
                iv_dict = tes_pickle.createIVDictHeater(ivdata, temp, feedback_resistance, heater_voltage,
                                                        heater_resistance, bath_temperature_measured,bias_resistor)
                tes_pickle.addIVRun(column, mux_rows[ii], iv_dict)
                tes_pickle.savePickle()

        else: # detector by detector in series case
            #Loop over rows
            for i in range(len(rows)):
                row = rows[i]
                if cfg['runconfig']['overbias'] == True: # overbias if needed
                    print('Running Overbias subscript ...')
                    Tsetpoint = adr.temperature_controller.GetTemperatureSetPoint()
                    overBias(adr.temperature_controller,voltage_sweep_source,Thigh=OverBiasThigh,Tb=Tsetpoint,Vbias=OverBiasVbias,Tsensor=cfg['runconfig']['thermometerChannel'])
                    print('Overbias complete.  Letting bath temperature stabilize for '+str(postOverBiasWaitPeriod)+'s.')
                    if OverbiasExternal:
                        function_generator_offset = agilent_33220A.GetOffset()
                    time.sleep(postOverBiasWaitPeriod)
                else:
                    if OverbiasExternal:
                        print('Applying an external voltage with the signal generator now')
                        agilent_33220A.SetFunction('dc')
                        agilent_33220A.SetLoad(bias_resistor)  # adjusts offset and amplitude values displayed
                        agilent_33220A.SetOffset(OverBiasVbiasExternal)  # VDC, change voltage after I set the load
                        agilent_33220A.SetOutput('on')
                        function_generator_offset = agilent_33220A.GetOffset()
                        time.sleep(5.0)
                    else:
                        print('Not overbiasing or applying external voltage this time')
                tempstring = np.str(np.int(temp*1000))
                tempstring = tempstring + 'mK'

                #number_of_ivs = len(tes_pickle.getRuns(bay, row, 'iv'))
                #new_iv_string = 'iv%03i' % number_of_ivs
                new_iv_string = 'Row%02i' % row
                print('Row %d' % row)
                # figure out where to get the S1 and S2 bias settings
                if use_pickle_rowdata is True:
                    rowinfo = tes_pickle.getRowInfo(column, row)
                    if rowinfo is None:
                        print('Did not find row info')
                        radacvalue = radacvalues[row]
                        sq2fbvalue = sq2fbvalues[row]
                    else:
                        radacvalue = rowinfo['radac']
                        sq2fbvalue = rowinfo['sq2fb']
                else:
                    radacvalue = radacvalues[i]
                    sq2fbvalue = sq2fbvalues[i]

                tesacq.dcSet32(row, radacvalue, adr.crate) # turn on one SQ1 bias with RA8 card, sets all other values to 0
                print('Sq2fb ', sq2fbvalue)
                #setVoltDACUnits(self, val)
                sq2fb.setVoltDACUnits(sq2fbvalue) # bias the squid2
                time.sleep(2)

                filestring = 'Temperature_' + tempstring + "_row_%02i" % row

                # measure the average temperature just before taking the measurement
                if UseLoadMuxSettings:
                    voltage_start=IVstart[i]
                    voltage_end=IVstop[i]
                else:
                    voltage_start=cfg['runconfig']['v_start']
                    voltage_end=cfg['runconfig']['v_stop']

                # loop over heater voltages
                for heater_voltage in heater_voltages:
                    if LoopOverHeaterVoltage:
                        print('setting heater to', heater_voltage, 'V')
                        heater_source = bluebox.BlueBox(port='vbox', version='tower', address=heater_address[i],channel=heater_channel[i])
                        heater_source.setvolt(heater_voltage)

                        # take the IV curve.  All the complexity is here.
                    
                    
                    
                    ivreturn = sw.SmartSweep(voltage_start=voltage_start, voltage_end=voltage_end, voltage_step=cfg['runconfig']['v_step'], dwell_time=dwell_time, \
                                             measure_rows=None, jump_start=-1, jump_stop=-1, sweepstring=filestring, iv_string=new_iv_string,\
                                             normal_voltage_blast=normal_voltage_blast,normal_branch_voltage_append=normal_branch_voltage_append)#,plotter=plotter)

                    if LoopOverHeaterVoltage:
                        heater_source.setvolt(0)

                    ivdata = np.vstack((ivreturn[1],ivreturn[2]))
                    measured_temperature=adr.temperature_controller.GetTemperature(cfg['runconfig']['thermometerChannel'])

                    # Save the results in the pickle
                    iv_dict = tes_pickle.createIVDictHeater(ivdata,temp,feedback_resistance,heater_voltage,heater_resistance,measured_temperature,bias_resistor)
                    tes_pickle.addIVRun(column, row, iv_dict)
                    tes_pickle.savePickle()
    voltage_sweep_source.setvolt(0) 
    tesacq.pciStop()
    t_end=time.time()
    print('almost done')
    print('Length of measurement = ',(t_end-t0)/60.,' minutes')
    #adr.temperature_controller.SetTemperatureSetPoint(temps[0])
    print('done')
    #app.quit() #doesn't actually seem to do much
    #print '2'

if __name__ == '__main__': 
    main()

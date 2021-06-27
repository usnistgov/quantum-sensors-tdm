'''
polcal.py

Take Polarization Calibration data by rotating a source behind a grid
taking data at discrete angles

Early version makes following assumptions:
 - Multiplexing already set up before running
 - you have chop signal into 2nd DFB card
 - some manual inputs noted below are actually set manually at devices and you are just entering values for record keeping
 
Notes for setup:
 - center polcal on pixel you want
 - power source as suggested on amplifier, e.g. 7.8V DC, function gen 3Vpp w/ 1.5V offset (one polarity)
 - note you'll see attenuation of Hawkeye source at higher frequencies (notice above 10Hz)
 - check lock in quality with checklockin.py
 - change ndfb buffer size, if necessary (then re-check checklockin.py)
 - get multiplexing running (e.g. MultplexedDetectorBias.py)
 - run this script
 
 To Do in future:
  - make detectorbias actually set the detector bias
  - add spike/noise removal at timestream level before doing demodulation and saving reduced data. We can pick up bursts from communications (wifi, cell phones, etc)
  - produce and save error estimates
  - check and improve how "Nscans" is handled so it returns home before starting another scan.  Suggest just using Nscans=1 until it's tested/improved 


09/16/16 - Created initial basic version based of beammap.py (JA)
'''

import pickle
from time import *
import numpy as np
import sys

import tesacquire
import adr_system
import agilent33220a
#import XY
import rotary_stage as rot
import ndfb_pci_card as pci

#helper functions
def MakeAngleArray(theta_start,theta_throw,theta_step=10.):
    ''' Makes an evenly spaced angle positions in increasing values
        range is inclusive (i.e. goes to theta_throw+theta_start)
    '''
    x1 = np.arange(theta_start,theta_throw+theta_start+theta_step,theta_step)
    x=list(x1)
    return x

def GetDataNBuffers(N=5):
    try:
        x=pci.getData()
        for i in range(N-1):
            x=np.vstack((x,pci.getData()))
        return x
    except:
        print 'Something bad was returned by PCI'
        return None

def MeasureLockInAllRows(tesacquire_object,reference_column,Nbuffers=5,reference_type='square',response_type='sine'):
    data=GetDataNBuffers(Nbuffers)
    if data == None:
        print 'Failed to get data, will try again'
        data=GetDataNBuffers(Nbuffers)
        if data == None:
            print 'Still could not get good PCI data, so something might be seriously wrong.. probably going to crash now'
    return tesacquire_object.tesana.SoftwareLockin(np.transpose(data[:,:,0,1]) , np.transpose(data[:,:,reference_column,0]),reference_type,response_type)

def MeasureLockInAllRows_filter(tesacquire_object,reference_column,Nbuffers=2,reference_type='square',response_type='sine', nrepeat=10, noise_level =None, noise_sigma=5., median=True, verbose=True):
    '''
    Concept: take several (nrepeat) lumps of data (Nbuffers each), and try to remove outliers beyond (noise_sigma*noise_level) from median
    Return mean or median (median=True) of non-outlier samples
    '''
    result = []
    wg_all = []
    for j in range(nrepeat):
        data=GetDataNBuffers(Nbuffers)
        r = np.array(tesacquire_object.tesana.SoftwareLockin(np.transpose(data[:,:,0,1]) , np.transpose(data[:,:,reference_column,0]),reference_type,response_type))
        result.append(r.copy())
    result = np.array(result)  # indexed as: [sample, data type (I, Q, or ref), row]
    result_mag = np.sqrt(result[:,0]**2 + result[:,1]**2)/result[:,2]
    #print np.shape(result) #5 ,3, 32
    #print np.shape(result_mag) #5, 32
    # if not given estimate noise level (but will be biased if a spiked data set is in group, so won't remove as well)
    nrows = len(data[0,:,00])
    if noise_level:
        noise_std = np.repeat(noise_level,nrows)
    else:
        noise_std = np.std(result_mag,axis=0)
    # fill in a standard array of results with the mean or median of non-outliers
    result_final = r*0.  # create blank array of output structure
    for i in range(nrows):
        #pdb.set_trace()
        # flag those that are outliers
        # first iteration to remove ones that may skew median
        wg = np.where(np.abs(result_mag[:,i]-np.median(result_mag[:,i])) < noise_sigma*noise_std[i])[0] # just look at first column which has 5 values
        median_compare = np.median(result_mag[wg,i])
        # now get final coun
        wg = np.where(np.abs(result_mag[:,i]-median_compare) < noise_sigma*noise_std[i])[0]
        #print wg # 0 up to 4 rejected
        
        if median:
            result_final[:,i] = np.median(result[wg,:,i],axis=0) # median of any points that pass the filter (seems like I, Q, ref column recorded, did not change)
            wg_all.append(wg)
        else:
            result_final[:,i] = np.mean(result[wg,:,i],axis=0)
            wg_all.append(wg)
        if verbose:
            if (len(wg) < len(result_mag[:,i])):
                print 'Removed %d outliers on row %d' % ((len(result_mag[:,i])-len(wg)),i)
    return result_final, result, wg_all #result_final_medians # result_final_medians is new, 10/30/19, SW #also return the filter wg???
	
def saveReturnDict(dictionary,filename):
    ''' save the data to filename '''
    f=open(filename,'w')
    pickle.dump(dictionary,f)
    f.close()
                     
# settings ------------------------------------------------
#Options
path='/home/pcuser/data/ali/20191125/polcal/'
lt=localtime()
thedate=str(lt[0])+'%02i'%lt[1]+'%02i'%lt[2]
output_filename = path+'Polcal_'+thedate+'_'+'bay2_5Hz_res5deg_throw360deg_1.pkl' #'test_polcal_filter2.pkl' #pos1_10Hz_res1deg_throw360deg_1.pkl' #pos2_Vb0p49_res1deg_throw360deg_polcal_filter_0.pkl' #Vb0p49_res5deg_throw360deg.pkl'
RELOCK=True # relock each dfb channel at every position
column='ColumnA'

# Set the scan parameters of rotary stage
theta_0 = 0 #0 will start at home (degrees)
theta_amp = 360 # scan range in degrees (currently capped at 800 until you prove you don't spin wires too much)
step = 5. #2. #22.5 #2 #22.5 #22.5 #1#5 # degrees
p_override = []#[0,10,20,21,22,23,24,25,50,75,85,100,110,111,112,113,114,115,135,145,160,185,200,201,202,203,204,205,230,255,265,280,290,291,292,293,294,295,305,320,345,370,380,381,382,383,384,385]  # leave empty normally, but use this to set your own set of angles instead of equally spaced array

# source params
# make sure frequency is high enough for buffer/demodulation routine being used, or else you'll introduce lots of unnecessary error
# but also note that Hawkeye source starts seeing attenuation and is less square-wavey by 10Hz... so... need fix demodulation for better slow chop
chopper_frequency= 5.0 #10.0#
chopper_reference_dfb = 1
# next two probably won't change, but you can play around to see if anything works better
SourceAmp=3.0 #Volts
SourceOffset=1.5 #Volts

# data params
Nscans = 1
N_chop_buffers = 5 #7 #5 #4

# data and analysis choices
#N_chop_buffers = 2
nrepeat = 10 #10 #10 #5 #7  # repeat taking N_chop_buffers at each point, take mean or median
use_median = True  # otherwise use mean
response_type = 'square'  # 'sine' or 'square'
# Filtering?
# set noise_level=None to not reject any samples.  Set to noise level to reject any outlier of the nrepeat set
noise_level = 5e-5 #5e-5 #1e-6 #2e-6 #8e-7#1e-6 #1e-5  # set to noise estimate in magnitude of lockin.  Get idea for this using checkLockIn_jay.py when off source
#Sam: How do you decide this???
noise_sigma = 3. #3. #5. # reject samples beyond noise_level*noise_sigma from median of set


# setup details (just manually entered/stored, doesn't force this to be true... yet)
EccosorbOn=True

# Bias and multiplexing
#RowOrder=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
RowOrder = range(32)#[1,2,3,4,5,22] #[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]#,7,8,9,10,11,12,13,14,15,16,17,18]#,18,19,20,21,22,23]
#RowsNotLocked = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
RowsNotLocked = [6,7,8,11,12,15,22,23,24,31] #[6,7,8,11,12,15,22,23,24,31] #[0,1,4,6,7,8,11,12,15,22,29,30,31] #[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31] #[0,1,4,6,7,8,11,12,15,22,29,30,31] #[0,1,4,6,7,8,11,12,15,22,29,30,31] #[11,12,15]#[0,6,12,18,22,23] #[0,2,4,6,8,10,12,16,23]#,8,10,12,16,17,18]#,18,20,22,23]


# Accounting only.  Following group not actually controlled by code (currently). Must set before hand
# Just stores written value for accounting, so get it right
DetectorBias=0.350  #currently just for show and storage, doesn't actually change bias
BaseTemperature=0.465 # not set by program, just accounting

# future upgrade?
# vestigial from original beammap code. not implemented. could re-implement if have another function generator
#MeasureHeaterResponse=False # use function generator to tickle the heater response.  Linearity check. 

# crate/ timing settings
dfb_card_number=0
pci_mask=0x3
pci_firmware_delay=6
dfb_numberofsamples=16
lsync=100
number_mux_rows=32
dfb_settlingtime = lsync - dfb_numberofsamples - 1

# ------------------------------------------------------------------------------------------------------------

# start the script
print 'Starting polcal scan'
adr=adr_system.AdrSystem(None,lsync=lsync, number_mux_rows=number_mux_rows, dfb_settling_time=dfb_settlingtime,\
                         dfb_number_of_samples=dfb_numberofsamples, doinit=False)

print 'setting up pci card'  
tes=tesacquire.TESAcquire()  
tes.pciSetup(pci_mask, pci_firmware_delay, dfb_numberofsamples)
pci.stopDMA() # stop and start might help with disorder of rows that I'm seeing
sleep(1)
pci.startDMA(pci_mask,pci_firmware_delay,dfb_numberofsamples)
sleep(1)
Nrows=pci.getNumberOfRows()

onebuffer=pci.getData()
datapts,nrows,ncols,ef=np.shape(onebuffer)
print 'number of data points in one buffer = ',datapts
for i in range(nrows):
    print 'row ',i, 'mean feedback value = ', np.mean(onebuffer[:,i,0,1])

# set source
print 'Initializing the function generator.'
agilent_33220A = agilent33220a.Agilent33220A(pad=10)
agilent_33220A.SetFunction(response_type)
agilent_33220A.SetFrequency(chopper_frequency)
agilent_33220A.SetAmplitude(SourceAmp)
agilent_33220A.SetOffset(SourceOffset)
agilent_33220A.SetOutput('on')
sleep(1)

# setup array of angles to measure
if not p_override:
    p=MakeAngleArray(theta_0,theta_amp,step)
else:
    p=p_override
p=p*Nscans
pcommand = list(np.array(p)*100)  # commands are in units of 1/100 degree
if np.max(p) > 800.:
    print 'Warning, you have entered a total rotation greater than 800 degrees!'
    print 'I am not sure that is safe for the wires, so I am exiting'
    print 'If confident wiring is ok, change the code'
    sys.exit()

Npositions=len(p)
print 'Positions to measure:'
print p
#inc=abs(p[1][1]-p[0][1])

print 'Number of positions = ',Npositions

t_init=time()
# go to first position and wait a bit
print 'Moving to initial position'
my_rot = rot.RotaryStage()
my_rot.initialize()
my_rot.rotarymove_absolute(pcommand[0])
sleep(1)

# return_dict={'angles':p,'Response':[],'theta_0':theta_0,'theta_throw':theta_amp,'stepsize':step,\
             # 'chop_frequency':chopper_frequency,'source_amp':SourceAmp,'source_offset':SourceOffset,\
             # 'EccosorbOn':EccosorbOn,\
             # 'DetectorBias':DetectorBias,'BaseTemperature':BaseTemperature,'RowOrder':RowOrder,'Column':column,\
             # 'RowsNotLocked':RowsNotLocked}
return_dict={'angles':p,'Response':[],'NonFilteredResponse':[],'theta_0':theta_0,'theta_throw':theta_amp,'stepsize':step,\
             'chop_frequency':chopper_frequency,'source_amp':SourceAmp,'source_offset':SourceOffset,\
             'EccosorbOn':EccosorbOn,\
             'DetectorBias':DetectorBias,'BaseTemperature':BaseTemperature,'RowOrder':RowOrder,'Column':column,\
             'RowsNotLocked':RowsNotLocked,'NRepeat':nrepeat,'NBuffers':N_chop_buffers,'AppliedFilter':[]}

# loop over the angle positions and take measurements
Npts_remaining=Npositions
for i in range(Npositions):
    t_loop_start=time()
    if i == 0:
        print 'starting scan'
    else:
        print 'moving to theta=',p[i]
        inc = pcommand[i]-pcommand[i-1]
        my_rot.rotarymove_incremental(inc)
    
    # relock at position
    if RELOCK:
        print 'relocking all rows'
#        for row in range(Nrows-2):  # 8/17/16 - commented out and no longer assume everything re-locked except last two
        for row in range(Nrows):  
            if RowOrder[row] not in RowsNotLocked:  # now actually check if it's supposed to re-lock it
                adr.crate.dfb_cards[dfb_card_number].relock(row)
        sleep(.1)
        
    print 'measuring thermal chop'
    #output = MeasureLockInAllRows(tesacquire_object=tes,reference_column=1,Nbuffers=N_chop_buffers,reference_type='square',response_type=response_type)
    # output = MeasureLockInAllRows_filter(tes,reference_column=chopper_reference_dfb,Nbuffers=N_chop_buffers,reference_type='square',response_type=response_type, nrepeat=nrepeat, noise_level=noise_level,noise_sigma=noise_sigma,median=use_median, verbose=True)
    # return_dict['Response'].append(output)
    output, output_raw, filter = MeasureLockInAllRows_filter(tes,reference_column=chopper_reference_dfb,Nbuffers=N_chop_buffers,reference_type='square',response_type=response_type, nrepeat=nrepeat, noise_level=noise_level,noise_sigma=noise_sigma,median=use_median, verbose=True) # this is changed, 10/30/19, SW
    return_dict['Response'].append(output)
    return_dict['NonFilteredResponse'].append(output_raw) # this is new, 10/30/19, SW
    return_dict['AppliedFilter'].append(filter) # this is new, 10/30/19, SW
    looptime=time()-t_loop_start
    Npts_remaining = Npts_remaining - 1
    time_to_finish=(looptime*Npts_remaining)/60.0
    print 'looptime = ',looptime,'s, # of pts remaining = ',Npts_remaining, ' estimated completion time = ',time_to_finish,' minutes'
    print '-'*5
    
return_dict['Response']=np.array(return_dict['Response'])
return_dict['NonFilteredResponse']=np.array(return_dict['NonFilteredResponse']) # this is new, 10/30/19, SW
return_dict['AppliedFilter']=np.array(return_dict['AppliedFilter']) # this is new, 10/30/19, SW
saveReturnDict(return_dict,output_filename)
total_time=(time()-t_init)/60.0
print 'Finished scan in ', total_time,' minutes.  Closing connection to velmex rotation stage.'
# return to starting position to unwind wiring
my_rot.rotarymove_incremental(-pcommand[-1])
my_rot.reset()  # close communication with velmex

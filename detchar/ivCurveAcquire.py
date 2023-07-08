#! /usr/bin/env python
''' ivCurveAquire.py

top-level script to acquire an iv curve data set.
usage:
./ivCurveAquire.py <config_file> <description>

Configuration: setup mux in cringe, ensure adr_gui is running, edit an iv.config file, 
issue command above.

@author JH 
v0: 2/2021
v1: 4/2021 updated to allow plotting and to take IV curve at current temperature 
v2: 5/2023 updated to use the tower card detector bias
'''

import yaml, sys, os
from iv_utils import *
import matplotlib.pyplot as plt

def plot_iv_groups(iv_temp_sweep_inst,num_in_group=8):
    N = len(iv_temp_sweep_inst.set_temps_k) # number of temp sweeps
    for ii in range(N): # loop over temperature sweeps
        df = iv_temp_sweep_inst.data[ii]
        dac,fb_arr = df.xy_arrays_zero_subtracted_at_dac_high()
        n_dac,n_rows = np.shape(fb_arr)
        n_groups = n_rows//num_in_group
        for jj in range(n_groups+1): # plot in groups 
            plt.figure(jj)
            for kk in range(num_in_group):
                row_index = jj*num_in_group+kk
                if row_index>=n_rows: break
                plt.plot(dac,fb_arr[:,row_index],label=row_index)
            plt.xlabel('dac (arb)')
            plt.ylabel('fb (arb)')
            plt.legend()
            plt.grid()
            plt.title('Group %d'%(jj))
        plt.show()

def is_coldload_sweep():
    result = False
    if 'coldload' in cfg.keys():
        if cfg['coldload']['execute']: 
            result = True
    return result

def handle_column(col):
    ''' based on the column, select the appropriate tower card detector bias channel.  
        This "channel" is called "bayname" in the iv_utils software and BAY_NAMES 
        in the towerwidget.py.  So I follow this poor naming convention 
    '''
    assert col in ['a','b','c','d','A','B','C','D'], 'uknown column.  Column must be A,B,C, or D'
    if col.upper() == 'A':
        bayname = "0"
    elif col.upper() == 'B':
        bayname = "1"
    elif col.upper() == 'C':
        bayname = "2"
    elif col.upper() == 'D':
        bayname = "3"
    return bayname

def create_filename():
    baystring = 'Column' + cfg['detectors']['Column']
    if not os.path.exists(cfg['io']['RootPath']):
        print('The path: %s does not exist! Making directory now.'%cfg['io']['RootPath'])
        os.makedirs(cfg['io']['RootPath'])
    localtime=time.localtime()
    thedate=str(localtime[0])+'%02i'%localtime[1]+'%02i'%localtime[2]
    etime = str(time.time()).split('.')[0]
    filename = cfg['io']['RootPath']+cfg['detectors']['DetectorName']+'_'+baystring+'_ivs_'+thedate+'_'+etime
    write_filename = filename+'.json'
    return write_filename

n_args = len(sys.argv)
config_filename = str(sys.argv[1])
if n_args > 2:
    desc = str(sys.argv[2])
else:
    desc=''

# open config file ---------------------------------
with open(config_filename, 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

write_filename = create_filename()

# voltage bias things ------------------------------
if cfg['voltage_bias']['source'] in ['bluebox','BlueBox','Blue Box','blue box']:
    voltage_source = 'bluebox'
else:
    voltage_source = None
if cfg['voltage_bias']['overbias']:
    to_normal_method='overbias'
    overbias_temp_k=cfg['voltage_bias']['overbias_temp_k']
else:
    to_normal_method=None
    overbias_temp_k=None
bayname = handle_column(cfg['detectors']['Column'])
dacs = np.linspace(int(cfg['voltage_bias']['v_start_dac']),int(cfg['voltage_bias']['v_stop_dac']),int(cfg['voltage_bias']['npts']))

# bath temperature things ---------------------------
bath_temps = cfg['runconfig']['bathTemperatures']
# below commented out because it can crash adr_gui.  Need to ask what the set-temp is through adr_gui, and this is currently not in the software.
# if bath_temps==[0]: # take at current temperature only
#     from instruments import Lakeshore370
#     ls370 = Lakeshore370() 
#     bath_temps = [ls370.getTemperatureSetPoint()]

#############
pt_taker = IVPointTaker(db_cardname=cfg['voltage_bias']['db_cardname'], bayname=bayname, voltage_source = voltage_source)
curve_taker = IVCurveTaker(pt_taker, temp_settle_delay_s=cfg['runconfig']['temp_settle_delay_s'], shock_normal_dac_value=65000, zero_tower_at_end=cfg['voltage_bias']['setVtoZeroPostIV'], adr_gui_control=None)
curve_taker.prep_fb_settings(I=cfg['dfb']['i'], fba_offset=cfg['dfb']['dac_a_offset'], ARLoff=True)
ivsweeper = IVTempSweeper(curve_taker, to_normal_method=to_normal_method, overbias_temp_k=overbias_temp_k, overbias_dac_value = cfg['voltage_bias']['v_start_dac'])

if is_coldload_sweep():
    cl_temps_k = cfg['coldload']['cl_temps_k']
    clsweep_taker = IVColdloadSweeper(ivsweeper)
    data = clsweep_taker.get_sweep(dacs, cl_temps_k, bath_temps,
                                   cl_temp_tolerance_k=cfg['coldload']['cl_temp_tolerance_k'],
                                   cl_settemp_timeout_m=10.0,
                                   cl_post_setpoint_waittime_m=cfg['coldload']['cl_post_setpoint_waittime_m'],
                                   skip_first_settle = cfg['coldload']['immediateFirstMeasurement'],
                                   cool_upon_finish = cfg['coldload']['cool_upon_finish'], extra_info={'config': cfg, 'exp_status':desc},
                                   write_while_acquire = True, filename=write_filename)
else:
    data = ivsweeper.get_sweep(dacs, bath_temps, extra_info={'config':cfg,'exp_status':desc,'row_to_period_map':cfg['detectors']['Rows']})
    data.to_file(write_filename,overwrite=True)
    if cfg['runconfig']['show_plot']:
        print('showing plot')
        plot_iv_groups(data)

if cfg['runconfig']['save_data']:
    data.to_file(write_filename,overwrite=True)
    print('data aquistion finished.\nWrote to file:',write_filename)
else:
    is_save = input('You have selected not to save data.  If you would like to save the file, hit 1 and enter.')
    if is_save=='1': 
        data.to_file(write_filename,overwrite=True)
        print('data aquistion finished.\nWrote to file:',write_filename)
    else:
        print('Exiting without saving')
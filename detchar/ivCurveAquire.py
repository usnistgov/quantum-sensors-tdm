#! /usr/bin/env python
''' ivCurveAquire.py

top-level script to acquire an iv curve data set.
usage:
./ivCurveAquire.py <config_file>

@author JH 2/2021
'''

import yaml, sys, os
from iv_utils import *

# open config file
with open(config_filename, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

config_filename = str(sys.argv[1])

def create_filename():
    if not os.path.exists(cfg['io']['RootPath']):
        print('The path: %s does not exist! Making directory now.'%cfg['io']['RootPath'])
        os.makedirs(cfg['io']['RootPath'])
    localtime=time.localtime()
    thedate=str(localtime[0])+'%02i'%localtime[1]+'%02i'%localtime[2]
    etime = str(time.time()).split('.')[0]
    filename = cfg['io']['RootPath']+cfg['detectors']['DetectorName']+'_'+baystring+'_ivs_'+thedate+'_'+etime
    write_filename = filename+'.json'
    return write_filename

write_filename = create_filename()
if cfg['voltage_bias']['source'] in ['bluebox','BlueBox','Blue Box','blue box':
    voltage_source = 'mrk2'
else:
    voltage_source = 'tower'
if cfg['voltage_bias']['overbias']:
    to_normal_method='overbias'
    overbias_temp_k=cfg['voltage_bias']['overbias_temp_k']
else:
    to_normal_method=None
    overbias_temp_k=None

bath_temps = cfg['runconfig']['bathTemperatures']
dacs = np.linspace(int(cfg['voltage_bias']['v_start_dac']),int(cfg['voltage_bias']['v_stop_dac']),int(cfg['voltage_bias']['npts']))

#############
pt_taker = IVPointTaker(db_cardname=cfg['dfb']['dfb_cardname'], bayname=cfg['detectors']['Column'], voltage_source = voltage_source)
curve_taker = IVCurveTaker(pt_taker, temp_settle_delay_s=60, shock_normal_dac_value=65000, zero_tower_at_end=cfg['voltage_bias']['setVtoZeroPostIV'], adr_gui_control=None)
curve_taker.prep_fb_settings(I=cfg['dfb']['i'], fba_offset=cfg['dfb']['dac_a_offset'])
btemp_sweep_taker = IVTempSweeper(curve_taker, to_normal_method=to_normal_method, overbias_temp_k=overbias_temp_k, overbias_dac_value = cfg['voltage_bias']['v_start_dac'])

if 'coldload' in cfg.keys():
    if cfg['coldload']['execute']:
        cl_temps_k = cfg['coldload']['cl_temps_k']
        clsweep_taker = IVColdloadSweeper(btemp_sweep_taker)
        data = clsweep_taker.get_sweep(dacs, cl_temps_k, bath_temps,
                                       cl_temp_tolerance_k=cfg['coldload']['cl_temp_tolerance_k'],
                                       cl_settemp_timeout_m=10.0,
                                       cl_post_setpoint_waittime_m=cfg['coldload']['cl_post_setpoint_waittime_m'],
                                       skip_first_settle = cfg['coldload']['immediateFirstMeasurement'],
                                       cool_upon_finish = True, extra_info={'config': cfg},
                                       write_while_acquire = True, filename=write_filename)
else:
    data = ivsweeper.get_sweep(dacs, bath_temps, extra_info={'config':cfg})

data.to_file(write_filename,overwrite=True)

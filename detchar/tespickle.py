''' tespickle.py 

class for storing TES characterization data into python pickle structure
@author: Doug Bennett
'''
import sys
import time
from time import sleep
import numpy as np
import pylab
import pickle

class TESPickle(object):
    '''
    Pickle functions
    '''
    def __init__(self, pickle_file = '/Users/dbennett/data/texas/runL2/runL2_characterization2.pkl'):
        
        self.pickle_file = pickle_file
        self.gamma = self.loadPickle(self.pickle_file)
        self.tune_pickle = {}
        
    def loadPickle(self, pickle_file = None):
        
        if pickle_file is None:
            pickle_file = self.pickle_file
        
        try:
            f = open(pickle_file, 'rb')
            self.gamma = pickle.load(f)
            f.close()
        except:
            self.gamma = {}

        return self.gamma
    
    def savePickle(self, save_pickle_file=None):
        if save_pickle_file == None:
            pickle_file = self.pickle_file
        else:
            pickle_file = save_pickle_file
        f = open(pickle_file, 'wb')
        pickle.dump(self.gamma, f)
        f.close()
    
    def addBay(self, bay):
        '''
        Check for the existance of bay. Add the bay if it does not exist. 
        '''
        
        baystring = self.baystring(bay)
        if baystring not in self.gamma:
            self.gamma[baystring] = {}    

    def getBays(self):
        '''
        Return keys of existing bays. 
        '''
        baykeys = list(self.gamma.keys())
        baykeys.sort()
           
        return baykeys   
            
    def addRow(self, bay, row):
        '''
        Check for the existance of row in bay. Add the row if it does not exist. 
        '''
        
        self.addBay(bay)
        
        baystring = self.baystring(bay)
        rowstring = self.rowstring(row)
        if rowstring not in self.gamma[baystring]:
            self.gamma[baystring][rowstring] = {}
            #print 'Row %02i did not exist in pickle' % row

    def getRows(self, bay):
        '''
        Return keys of existing rows. 
        '''

        baystring = self.baystring(bay)
        rowkeys = list(self.gamma[baystring].keys())
        if 'BayType' in rowkeys:
            rowkeys.remove('BayType')
        rowkeys.sort()
           
        return rowkeys  

    def newRunString(self, dict):
        '''
        Return the name of a new run
        '''
        
        number_of_runs = len(list(dict.keys()))            
        new_run_string = 'run%03i' % number_of_runs
        
        return new_run_string

    def newIVRunString(self, dict):
        '''
        Return the name of a new run. This separate run string for IVs is a historical artifact and should be phased out
        '''
        
        number_of_runs = len(list(dict.keys()))            
        new_run_string = 'iv%03i' % number_of_runs
        
        return new_run_string

    def addRun(self, bay, row, measurement_type, dict):
        
        self.addMeasurementType(bay, row, measurement_type)
        
        baystring = self.baystring(bay)
        rowstring = self.rowstring(row)
        if measurement_type == 'iv':
            new_run_string = self.newIVRunString(self.gamma[baystring][rowstring][measurement_type])
        else:
            new_run_string = self.newRunString(self.gamma[baystring][rowstring][measurement_type])
        self.gamma[baystring][rowstring][measurement_type][new_run_string] = dict
        
    def getRuns(self, bay, row, measurement_type):
        '''
        Return keys of existing runs. 
        '''
        
        if measurement_type == 'Lselect' or measurement_type == 'type' or measurement_type == 'Pixel':
            return []
        

        baystring = self.baystring(bay)
            
        try:
            if row[0] == 'R':
                #Row is all ready a rowstring
                rowstring = row
            else:
                rowstring = self.rowstring(row)
        except:
            rowstring = self.rowstring(row)
        
        runkeys = list(self.gamma[baystring][rowstring][measurement_type].keys())
        runkeys.sort()
           
        return runkeys 

    def addMeasurementType(self, bay, row, measurement_type):
        self.addRow(bay, row)
        
        baystring = self.baystring(bay)
        rowstring = self.rowstring(row)
        if measurement_type not in self.gamma[baystring][rowstring]:
            self.gamma[baystring][rowstring][measurement_type] = {}

    def getMeasurmentTypes(self, bay, row):
        '''
        Return keys of existing rows. 
        '''
        
        if row == 'BayType':
            return []

        baystring = self.baystring(bay)
            
        if row[0] == 'R':
            #Row is all ready a rowstring
            rowstring = row
        else:
            rowstring = self.rowstring(row)
        
        typekeys = list(self.gamma[baystring][rowstring].keys())
        typekeys.sort()
           
        return typekeys 

    def getRowInfo(self, bay, row):
        '''
        Return keys of existing rows. 
        '''
        
        if row == 'BayType':
            return []

        baystring = self.baystring(bay)
            
        row_strform = str(row)
        if row_strform[0] == 'R':
            #Row is all ready a rowstring
            rowstring = row
        else:
            rowstring = self.rowstring(row)
        
        typekeys = list(self.gamma[baystring][rowstring].keys())
        if 'row_info' in typekeys:
            rowinfo_dict = self.gamma[baystring][rowstring]['row_info']
        else:
            rowinfo_dict = None
           
        return rowinfo_dict 

    def baystring(self, bay):
        if bay[0:3] == 'Bay':
            return bay
        else:
            return 'Bay' + bay

    def rowstring(self, row):
        return 'Row%02i' % row

    def runstring(self, run):
        return 'run%03i' % run

    def createRowInfoDict(self, radac=None, sq2fb=None, dfb_adc=None):
        
        new_dict = {}
        new_dict['radac'] = radac
        new_dict['sq2fb'] = sq2fb
        new_dict['dfb_adc'] = dfb_adc
        
        return new_dict

    def addRowInfoDict(self, bay, row, dict):

        if type(row).__name__ == 'int':
            self.addRow(bay, row)            
            baystring = self.baystring(bay)
            rowstring = self.rowstring(row)
            self.gamma[baystring][rowstring]['row_info'] = dict
        else:
            radacs = dict['radac']
            sq2fbs = dict['sq2fb']
            dfb_adcs = dict['dfb_adc']
            for row_single in row:
                row_dict = self.createRowInfoDict(radacs[row_single], sq2fbs[row_single], dfb_adcs[row_single])
                self.addRow(bay, row_single)            
                baystring = self.baystring(bay)
                rowstring = self.rowstring(row_single)
                self.gamma[baystring][rowstring]['row_info'] = row_dict                

    def createRowInfoDfbDict(self, address=None, channel=None, dfb_adc=None, dfb_dac=None, sq2fb=None, \
                             mux_p=None, mux_i=None, tria=None, trib=None):
        
        new_dict = {}
        new_dict['address'] = address
        new_dict['channel'] = channel
        new_dict['dfb_adc'] = dfb_adc
        new_dict['dfb_dac'] = dfb_dac
        new_dict['sq2fb'] = sq2fb
        new_dict['mux_p'] = mux_p
        new_dict['mux_i'] = mux_i
        new_dict['tria'] = tria
        new_dict['trib'] = trib
        return new_dict

    def addRowInfoDfbDict(self, bay, row, dict):

        self.addRow(bay, row)            
        baystring = self.baystring(bay)
        rowstring = self.rowstring(row)
        self.gamma[baystring][rowstring]['dfb_row_info'] = dict 

    def getRowDfbInfo(self, bay, row):
        '''
        Get the Dfb info for a row and bay. 
        '''
                

        baystring = self.baystring(bay)
            
        row_strform = str(row)
        if row_strform[0] == 'R':
            #Row is all ready a rowstring
            rowstring = row
        else:
            rowstring = self.rowstring(row)
        
        #print 'Baystring, Rowstring: ', baystring, rowstring
        typekeys = list(self.gamma[baystring][rowstring].keys())
        if 'dfb_row_info' in typekeys:
            rowinfo_dict = self.gamma[baystring][rowstring]['dfb_row_info']
        else:
            rowinfo_dict = None
           
        return rowinfo_dict

    def createRowInfoRADict(self, address=None, ch_num=None, row_enabled=None, row_index=None, \
                            ra_low=None, ra_high=None, ra_delay=None, ra_width=None):
        
        new_dict = {}
        new_dict['address'] = address
        new_dict['ch_num'] = ch_num
        new_dict['row_enabled'] = row_enabled
        new_dict['row_index'] = row_index
        new_dict['ra_low'] = ra_low
        new_dict['ra_high'] = ra_high
        new_dict['ra_delay'] = ra_delay
        new_dict['ra_width'] = ra_width
        return new_dict

    def addRowInfoRADict(self, bay, row, dict):

        self.addRow(bay, row)            
        baystring = self.baystring(bay)
        rowstring = self.rowstring(row)
        self.gamma[baystring][rowstring]['ra_row_info'] = dict


    def getRowRAInfo(self, bay, row):
        '''
        Get the RA info for a row and bay. 
        '''
                

        baystring = self.baystring(bay)
            
        row_strform = str(row)
        if row_strform[0] == 'R':
            #Row is all ready a rowstring
            rowstring = row
        else:
            rowstring = self.rowstring(row)
        
        #print 'Baystring, Rowstring: ', baystring, rowstring
        typekeys = list(self.gamma[baystring][rowstring].keys())
        if 'ra_row_info' in typekeys:
            rowinfo_dict = self.gamma[baystring][rowstring]['ra_row_info']
        else:
            rowinfo_dict = None
           
        return rowinfo_dict
    
    def createMixDict(self, v_feedback, v_error, v_mix, mix_parameter):
        md = {}
        
        md['v_feedback']= v_feedback
        md['v_error'] = v_error
        md['v_mix'] = v_mix
        md['mix_parameter'] = mix_parameter
        return md

    def createIVDict(self,ivdata,temperature=None,feedback_resistance=None,laser_voltage=None, \
                     atten_voltage=None,measured_temperature=None,heater_output=None, notes=None, \
                     std_fb=None, fb_hist=None, bin_centers=None):
        new_dict = {}

        new_dict['data'] = ivdata
        new_dict['datetime'] = time.localtime()
        new_dict['temperature'] = temperature
        new_dict['feedback_resistance'] = feedback_resistance
        new_dict['laser_voltage'] =  laser_voltage
        new_dict['atten_voltage'] = atten_voltage
        new_dict['plot_data_in_ivmuxanalyze'] = True
        new_dict['measured_temperature'] = measured_temperature
        new_dict['heater_output'] = heater_output
        new_dict['notes'] = notes
        new_dict['std_fb'] = std_fb
        new_dict['fb_hist'] = fb_hist
        new_dict['bin_centers'] = bin_centers

        return new_dict
    
    def createIVDictHeater(self,ivdata,temperature=None,feedback_resistance=None,heater_voltage=None,heater_resistance=None,measured_temperature=None,bias_resistance=None,function_generator_offset=None):
        new_dict = {}

        new_dict['data'] = ivdata
        new_dict['datetime'] = time.localtime()
        new_dict['temperature'] = temperature
        new_dict['heater_resistance'] = heater_resistance
        new_dict['heater_voltage']= heater_voltage
        new_dict['feedback_resistance'] = feedback_resistance
        new_dict['plot_data_in_ivmuxanalyze'] = True
        new_dict['measured_temperature'] = measured_temperature
        new_dict['bias_resistance'] = bias_resistance
        new_dict['function_generator_offset'] = function_generator_offset
        return new_dict
    
    def createIVDictBB(self,ivdata,bath_temperature_commanded=None,\
                       bath_temperature_measured=None,\
                       bb_temperature_commanded=None,\
                       bb_temperature_measured_before=None,\
                       bb_temperature_measured_after=None,\
                       bb_voltage_measured_before=None,\
                       bb_voltage_measured_after=None,\
                       feedback_resistance=None,\
                       bias_resistance=None,\
                       multiplexedIV=None):
        new_dict = {}
        new_dict['data'] = ivdata
        new_dict['datetime'] = time.localtime()
        new_dict['bath_temperature_commanded'] = bath_temperature_commanded
        new_dict['bath_temperature_measured'] = bath_temperature_measured
        new_dict['bb_temperature_commanded'] = bb_temperature_commanded
        new_dict['bb_temperature_measured_before'] = bb_temperature_measured_before
        new_dict['bb_temperature_measured_after'] = bb_temperature_measured_after
        new_dict['bb_voltage_measured_before'] = bb_voltage_measured_before
        new_dict['bb_voltage_measured_after'] = bb_voltage_measured_after
        new_dict['feedback_resistance'] = feedback_resistance
        new_dict['bias_resistance'] = bias_resistance
        new_dict['plot_data_in_ivmuxanalyze'] = True
        new_dict['multiplexedIV'] = multiplexedIV
        return new_dict

    def createTcDict(self,temperature_sc, temperature_nm, amplitude_sc_trans, amplitude_nm_trans, amplitude_sc, amplitude_nm, feedback_resistance=None, heater_output=None):
        new_dict = {}

        new_dict['datetime'] = time.localtime()
        new_dict['temperature_sc'] = temperature_sc
        new_dict['temperature_nm'] = temperature_nm
        new_dict['amplitude_sc_trans'] = amplitude_sc_trans
        new_dict['amplitude_nm_trans'] = amplitude_nm_trans
        new_dict['amplitude_sc'] = amplitude_sc
        new_dict['amplitude_nm'] = amplitude_nm
        new_dict['feedback_resistance'] = feedback_resistance
        new_dict['heater_output'] = heater_output

        return new_dict

    def createTickleDict(self, temperatures, amplitudes, feedback_resistance, heater_output, mag_voltage, offsets=None):
        new_dict = {}

        new_dict['datetime'] = time.localtime()
        new_dict['temperature'] = temperatures
        new_dict['amplitude'] = amplitudes
        new_dict['feedback_resistance'] = feedback_resistance
        new_dict['heater_output'] = heater_output
        new_dict['mag_voltage'] = mag_voltage
        new_dict['offset'] = offsets

        return new_dict

    def createNoiseDict(self,data,temperature,feedback_resistance,vbias,resistance,percent,noise_param,measured_temperature=None,heater_output=None,notes=''):
        new_dict = {}

        new_dict['data'] = data
        new_dict['datetime'] = time.localtime()
        new_dict['temperature'] = temperature
        new_dict['feedback_resistance'] = feedback_resistance
        new_dict['bias'] = vbias 
        new_dict['resistance'] = resistance
        new_dict['percentRn'] = percent
        new_dict['noise_param'] = noise_param
        new_dict['measured_temperature'] = measured_temperature
        new_dict['heater_output'] = heater_output
        new_dict['notes'] = notes
        

        return new_dict
    
    def createComplexZDict(self, frequency_array, phase_array, amp_array, real_array, imaginary_array, temperature, feedback_resistance, vbias, resistance, percent, fg_amplitude, fg_offset):
        czd = {}

        czd['datetime'] = time.localtime()
        czd['temperature'] = temperature
        czd['feedback_resistance'] = feedback_resistance
        czd['bias'] = vbias 
        czd['resistance'] = resistance
        czd['percentRn'] = percent
        czd['fg_amplitude'] = fg_amplitude
        czd['fg_offset'] = fg_offset
        
         
        czd['frequency_array'] = frequency_array
        czd['phase_array'] = phase_array
        czd['amplitude_array'] = amp_array
        czd['real_array'] = real_array
        czd['imaginary_array'] = imaginary_array

        return czd
    
    def createHeaterResponseDict(self, frequency_array, phase_array, amp_array, rr_array, ss_array, temperature, feedback_resistance,\
                                    vbias, bias_resistance, heater_bias_resistance, fg_amplitude, fg_offset,unlock_array,temperature_stable):
        chd = {}
        
        chd['datetime'] = time.localtime()
        chd['temperature'] = temperature
        chd['feedback_resistance'] = feedback_resistance
        chd['bias'] = vbias 
        chd['bias resistance'] = bias_resistance
        chd['heater bias resistance'] = heater_bias_resistance
        chd['fg_amplitude'] = fg_amplitude
        chd['fg_offset'] = fg_offset
        chd['temperature_stable']=temperature_stable
        
        chd['unlock_array']=unlock_array
        chd['frequency_array'] = frequency_array
        chd['phase_array'] = phase_array
        chd['amplitude_array'] = amp_array
        chd['rr_array'] = rr_array
        chd['ss_array'] = ss_array
        
        return chd

    def createDecayDict(self, pulse, temperature=None,feedback_resistance=None,vbias_low=None,vbias_high=None,pulse_freq=None,high_voltage=None,low_voltage=None,pulse_width=None,edge_time=None,wg_function=None):
        dsd = {}

        dsd['datetime'] = time.localtime()
        dsd['temperature'] = temperature
        dsd['feedback_resistance'] = feedback_resistance
        dsd['vbias_low'] = vbias_low
        dsd['vbias_high'] = vbias_high
        #dsd['resistance'] = resistance
        #dsd['percentRn'] = percent
        dsd['wg_pulse_frq'] = pulse_freq
        dsd['wg_high_voltage'] = high_voltage
        dsd['wg_low_voltage'] = low_voltage
        dsd['wg_pulse_width'] = pulse_width
        dsd['wg_edge_time'] = edge_time
        dsd['wg_function'] = wg_function
                 
        dsd['decay_pulse'] = pulse

        return dsd

    def createLaserPulseDict(self,data,temperature,feedback_resistance,vbias,resistance,percent,pulse_freq,function,high_voltage,low_voltage,pulse_width,edge_time,time_step):
        new_dict = {}

        new_dict['data'] = data
        new_dict['datetime'] = time.localtime()
        new_dict['temperature'] = temperature
        new_dict['feedback_resistance'] = feedback_resistance
        new_dict['bias'] = vbias 
        new_dict['resistance'] = resistance
        new_dict['percentRn'] = percent
        new_dict['pulse_freq'] = pulse_freq
        new_dict['function'] = function
        new_dict['high_voltage'] = high_voltage
        new_dict['low_voltage'] = low_voltage
        new_dict['pulse_width'] = pulse_width
        new_dict['edge_time'] = edge_time  
        new_dict['time_step'] = time_step
        
        return new_dict


    def addMixRun(self, bay, row, mix_dict):

        self.addRun(bay, row, 'mix', mix_dict)

    def addIVRun(self, bay, row, new_dict):
        
        self.addRun(bay, row, 'iv', new_dict)
        
    def addTcRun(self, bay, row, new_dict):
        
        self.addRun(bay, row, 'Tc', new_dict)

    def addTickleRun(self, bay, row, new_dict):
        
        self.addRun(bay, row, 'tickle', new_dict)

    def addComplexZRun(self, bay, row, complex_z_dict):
        
        self.addRun(bay, row, 'complex_z', complex_z_dict)
        
    def addHeaterResponseRun(self, bay, row, HeaterResponseDict):
        
        self.addRun(bay, row, 'HeaterResponse', HeaterResponseDict)
        
    def addNoiseRun(self, bay, row, noise_dict):
        
        self.addRun(bay, row, 'noise', noise_dict)
        
    def addDecayRun(self, bay, row, decay_dict):
        
        self.addRun(bay, row, 'decay', decay_dict)

    def addPulseRun(self, bay, row, new_dict):
        
        self.addRun(bay, row, 'pulses', new_dict)

    def getAnalayzedData(self, bay, row, measurement_type, run):
        
        data = self.gamma[bay][row][measurement_type][run]['data']
        
        return data

    def createLaserResponseDict(self, frequency_array, phase_array, amp_array, real_array, imaginary_array, temperature, feedback_resistance, vbias, resistance, percent, fg_amplitude, fg_offset):
        laserd = {}

        laserd['datetime'] = time.localtime()
        laserd['temperature'] = temperature
        laserd['feedback_resistance'] = feedback_resistance
        laserd['bias'] = vbias 
        laserd['resistance'] = resistance
        laserd['percentRn'] = percent
        laserd['fg_amplitude'] = fg_amplitude
        laserd['fg_offset'] = fg_offset
        
         
        laserd['frequency_array'] = frequency_array
        laserd['phase_array'] = phase_array
        laserd['amplitude_array'] = amp_array
        laserd['real_array'] = real_array
        laserd['imaginary_array'] = imaginary_array

        return laserd

    def addLaserResponseRun(self, bay, row, laser_response_dict):
        
        self.addRun(bay, row, 'laser_responsivity', laser_response_dict)
    
    def getIVDict(self, bay, row, key=None):
        '''
        Get an IV from pickle 
        '''
        
        baystring = self.baystring(bay)
        rowstring = self.rowstring(row)
        if rowstring not in self.gamma[baystring]:
            print('Row %02i did not exist in pickle' % row)
            
        if key is None:
            iv_dict = self.gamma[baystring][rowstring]['iv']
        else:
            iv_dict = self.gamma[baystring][rowstring]['iv'][key]
        
        return iv_dict    

    def getMeasurmentDict(self, bay, row, measurement_string, key=None):
        '''
        Get a measurement from pickle 
        '''
        
        baystring = self.baystring(bay)
        rowstring = self.rowstring(row)
        if rowstring not in self.gamma[baystring]:
            print('Row %02i did not exist in pickle' % row)
            m_dict = None
        else:
            if measurement_string not in self.gamma[baystring][rowstring]:
                print('Row %02i did not have that measurement string' % row)
                m_dict = None
            else:
                if key is None:
                    m_dict = self.gamma[baystring][rowstring][measurement_string]
                else:
                    m_dict = self.gamma[baystring][rowstring][measurement_string][key]
        
        return m_dict

    def getAnalysisDict(self, bay, row, analysis_string, key=None):
        '''
        Get a IV analysis from pickle 
        '''
        
        baystring = self.baystring(bay)
        rowstring = self.rowstring(row)
        if rowstring not in self.gamma[baystring]:
            print('Row %02i did not exist in pickle' % row)
            m_dict = None
        else:
            if 'analysis' not in self.gamma[baystring][rowstring]:
                print('Row %02i did not have an analysis dict' % row)
                m_dict = None
            else:
                if analysis_string not in self.gamma[baystring][rowstring]['analysis']:
                    m_dict = None
                else:
                    m_dict = self.gamma[baystring][rowstring]['analysis'][analysis_string]
        
        return m_dict

    def addRunAnalysis(self, bay, row, measurement_type, run_string, dict):
        
        baystring = self.baystring(bay)
        rowstring = self.rowstring(row)
        self.gamma[baystring][rowstring][measurement_type][run_string]['analysis'] = dict
    
    def getMixDict(self, bay, row, key):
        '''
        Get an IV from pickle 
        '''
        
        baystring = self.baystring(bay)
        rowstring = self.rowstring(row)
        if rowstring not in self.gamma[baystring]:
            self.gamma[baystring][rowstring] = {}
            #print 'Row %02i did not exist in pickle' % row
            
        mix_dict = self.gamma[baystring][rowstring]['mix'][key]
        
        return mix_dict
    
    def addIVRunAnalysis(self, bay, row, run_string, dict):
        
        self.addRunAnalysis(bay, row, 'iv', run_string, dict)
    
    def createIVRunAnalysisDict(self, vtes, ites, rtes, ptes, rs, sc_fit, nm_fit):
        
        new_dict = {}
        new_dict['vtes'] = vtes
        new_dict['ites'] = ites
        new_dict['rtes'] = rtes
        new_dict['ptes'] = ptes
        new_dict['resistances'] = rs
        new_dict['sc_fit_params'] = sc_fit
        new_dict['nm_fit_params'] = nm_fit
        
        return new_dict

    def addAnalysis(self, bay, row, measurement_type, dict):
        
        baystring = self.baystring(bay)
        rowstring = self.rowstring(row)

        if 'analysis' not in self.gamma[baystring][rowstring]:
            self.gamma[baystring][rowstring]['analysis'] = {}
        
        if measurement_type in self.gamma[baystring][rowstring]['analysis']:
            self.gamma[baystring][rowstring]['analysis'][measurement_type].update(dict)
        else:
            self.gamma[baystring][rowstring]['analysis'][measurement_type] = dict

    def addIVAnalysis(self, bay, row, dict):
        
        self.addAnalysis(bay, row, 'iv', dict)

    def createIVAnalysisDict(self, pRn, Gs, Ks, ns, Tcs, temperatures, TbsToReturn, powerAtRns, v_pnt_array=None, i_pnt_array=None, perRnMid=None, alpha=None,intercept=None,use_first_sc_slope_only=None):
        
        analysis_dict = {}
        
        analysis_dict['percentRns'] = pRn
        analysis_dict['Gs_atpRn'] = Gs
        analysis_dict['Ks_atpRn'] = Ks
        analysis_dict['ns_atpRn'] = ns
        analysis_dict['Tcs_atpRn'] = Tcs
        analysis_dict['temperatures'] = temperatures
        analysis_dict['TbsforParams'] = TbsToReturn
        analysis_dict['powers_atpRn'] = powerAtRns
        analysis_dict['v_atpRn'] = v_pnt_array
        analysis_dict['i_atpRn'] = i_pnt_array
        analysis_dict['percentRnsMid'] = perRnMid
        analysis_dict['alpha'] = alpha
        analysis_dict['intercept'] = intercept
        analysis_dict['use_first_sc_slope_only'] = use_first_sc_slope_only
        
        return analysis_dict

    def createPulseAnalysisDict(self, time, scaled_data):
        
        analysis_dict = {}
        
        analysis_dict['time'] = time
        analysis_dict['scaled_data'] = scaled_data
        
        return analysis_dict

    def createNoiseAnalysisDict(self, frequency, scaled_data):
        
        analysis_dict = {}
        
        analysis_dict['frequency'] = frequency
        analysis_dict['scaled_data'] = scaled_data
        
        return analysis_dict

    def createLRAnalysisDict(self, frequency, scaled_data):
        
        analysis_dict = {}
        
        analysis_dict['frequency'] = frequency
        analysis_dict['scaled_data'] = scaled_data
        
        return analysis_dict


    def createCZAnalysisDict(self, frequency, scaled_data):
        
        analysis_dict = {}
        
        analysis_dict['frequency'] = frequency
        analysis_dict['scaled_data'] = scaled_data
        
        return analysis_dict

    def addPulseRunAnalysis(self, bay, row, run_string, dict):
        
        self.addRunAnalysis(bay, row, 'pulses', run_string, dict)

    def addNoiseRunAnalysis(self, bay, row, run_string, dict):
        
        self.addRunAnalysis(bay, row, 'noise', run_string, dict)

    def addLRRunAnalysis(self, bay, row, run_string, dict):
        
        self.addRunAnalysis(bay, row, 'laser_responsivity', run_string, dict)

    def addCZRunAnalysis(self, bay, row, run_string, dict):
        
        self.addRunAnalysis(bay, row, 'complex_z', run_string, dict)
    
    def getPulseAnalayzedData(self, bay, row, measurement_type, run):
        
        time = self.gamma[bay][row][measurement_type][run]['analysis']['time']
        ydata = self.gamma[bay][row][measurement_type][run]['analysis']['scaled_data']
        data = np.vstack((time,ydata))
        
        return data

    def getNoiseAnalayzedData(self, bay, row, measurement_type, run):
        
        freq = self.gamma[bay][row][measurement_type][run]['analysis']['frequency']
        ydata = self.gamma[bay][row][measurement_type][run]['analysis']['scaled_data']
        data = np.vstack((freq,ydata))
        
        return data
    
    def getLRAnalayzedData(self, bay, row, measurement_type, run):
        
        freq = self.gamma[bay][row][measurement_type][run]['analysis']['frequency']
        ydata = self.gamma[bay][row][measurement_type][run]['analysis']['scaled_data']
        data = np.vstack((freq,ydata))
        
        return data

    def getCZAnalayzedData(self, bay, row, measurement_type, run):
        
        freq = self.gamma[bay][row][measurement_type][run]['analysis']['frequency']
        ydata = self.gamma[bay][row][measurement_type][run]['analysis']['scaled_data']
        data = np.vstack((freq,ydata))
        
        return data
    
    def searchPulses(self, bay, row, temperatures = None, percentRns = None, pulse_widths = None, pulse_freqs = None, measured_dates = None):

        baystring = self.baystring(bay)
        rowstring = self.rowstring(row) 
        
        pulse_keys = list(self.gamma[baystring][rowstring]['pulses'].keys())

        if temperatures is None:
            temperatures = []
            for pulse_key in pulse_keys:
                temperatures.append(self.gamma[baystring][rowstring]['pulses'][pulse_key]['temperature'])
        
        if percentRns is None:
            percentRns = []
            for pulse_key in pulse_keys:
                percentRns.append(self.gamma[baystring][rowstring]['pulses'][pulse_key]['percentRn'])
        
        if pulse_widths is None:
            pulse_widths = []
            for pulse_key in pulse_keys:
                pulse_widths.append(self.gamma[baystring][rowstring]['pulses'][pulse_key]['pulse_width'])
        
        if pulse_freqs is None:
            pulse_freqs = []
            for pulse_key in pulse_keys:
                pulse_freqs.append(self.gamma[baystring][rowstring]['pulses'][pulse_key]['pulse_freq'])
                                        
        if measured_dates is None:
            measured_dates = []
            for pulse_key in pulse_keys:
                measured_dates.append(self.gamma[baystring][rowstring]['pulses'][pulse_key]['datetime'])
        
        matching_keys = []
                     
        for pulse_key in pulse_keys:
            if self.gamma[baystring][rowstring]['pulses'][pulse_key]['temperature'] in temperatures \
                and self.gamma[baystring][rowstring]['pulses'][pulse_key]['percentRn'] in percentRns \
                and self.gamma[baystring][rowstring]['pulses'][pulse_key]['pulse_width'] in pulse_widths \
                and self.gamma[baystring][rowstring]['pulses'][pulse_key]['pulse_freq'] in pulse_freqs \
                and self.gamma[baystring][rowstring]['pulses'][pulse_key]['datetime'][:3] in [mdate[:3] for mdate in measured_dates]:
                
                #print pulse_key
                matching_keys.append(pulse_key)
                #print self.gamma[baystring][rowstring]['pulses'][pulse_key]['temperature']
                #print self.gamma[baystring][rowstring]['pulses'][pulse_key]['percentRn']
                #pulse = self.gamma[baystring][rowstring]['pulses'][pulse_key]['data']
                #pylab.plot(pulse[0,:4500],pulse[1,:4500])
                
        #pylab.show()
        return matching_keys


    def searchIVs(self, bay, row, temperatures = None, laser_voltages=None, 
                  atten_voltages=None, plot_data_in_ivmuxanalyze = None, measured_dates = None, 
                  analysis = None, field_coil_voltages = None):

        baystring = self.baystring(bay)
        rowstring = self.rowstring(row)        

        iv_keys = list(self.gamma[baystring][rowstring]['iv'].keys())

        if temperatures is None:
            temperatures = []
            for iv_key in iv_keys:
                temperatures.append(self.gamma[baystring][rowstring]['iv'][iv_key]['temperature'])

        if laser_voltages is None:
            laser_voltages = []
            for iv_key in iv_keys:
                if 'laser_voltage' in list(self.gamma[baystring][rowstring]['iv'][iv_key].keys()): 
                    laser_voltages.append(self.gamma[baystring][rowstring]['iv'][iv_key]['laser_voltage'])
                else: 
                    self.gamma[baystring][rowstring]['iv'][iv_key]['laser_voltage'] = 0
                    laser_voltages.append(self.gamma[baystring][rowstring]['iv'][iv_key]['laser_voltage'])

        if field_coil_voltages is None:
            field_coil_voltages = []
            for iv_key in iv_keys:
                if 'field_coil_voltage' in list(self.gamma[baystring][rowstring]['iv'][iv_key].keys()): 
                    field_coil_voltages.append(self.gamma[baystring][rowstring]['iv'][iv_key]['field_coil_voltage'])
                else: 
                    self.gamma[baystring][rowstring]['iv'][iv_key]['field_coil_voltage'] = 0
                    field_coil_voltages.append(self.gamma[baystring][rowstring]['iv'][iv_key]['field_coil_voltage'])
                    
        if atten_voltages is None:
            atten_voltages = []
            for iv_key in iv_keys:
                if 'atten_voltage' in list(self.gamma[baystring][rowstring]['iv'][iv_key].keys()): 
                    atten_voltages.append(self.gamma[baystring][rowstring]['iv'][iv_key]['atten_voltage'])
                else: 
                    self.gamma[baystring][rowstring]['iv'][iv_key]['atten_voltage'] = 0
                    atten_voltages.append(self.gamma[baystring][rowstring]['iv'][iv_key]['atten_voltage'])
                                
        if plot_data_in_ivmuxanalyze is None:
            plot_data_in_ivmuxanalyze = []
            for iv_key in iv_keys:
                plot_data_in_ivmuxanalyze.append(self.gamma[baystring][rowstring]['iv'][iv_key]['plot_data_in_ivmuxanalyze'])
                        
        if measured_dates is None:
            measured_dates = []
            for iv_key in iv_keys:
                measured_dates.append(self.gamma[baystring][rowstring]['iv'][iv_key]['datetime'])
                        
                        
        matching_keys = []
                     
        for iv_key in iv_keys:
            if self.gamma[baystring][rowstring]['iv'][iv_key]['temperature'] in temperatures \
                and self.gamma[baystring][rowstring]['iv'][iv_key]['laser_voltage'] in laser_voltages \
                and self.gamma[baystring][rowstring]['iv'][iv_key]['atten_voltage'] in atten_voltages \
                and self.gamma[baystring][rowstring]['iv'][iv_key]['field_coil_voltage'] in field_coil_voltages \
                and self.gamma[baystring][rowstring]['iv'][iv_key]['plot_data_in_ivmuxanalyze'] in plot_data_in_ivmuxanalyze \
                and self.gamma[baystring][rowstring]['iv'][iv_key]['datetime'][:3] in [mdate[:3] for mdate in measured_dates]:
                
                if analysis is True:
                    if 'analysis' in list(self.gamma[baystring][rowstring]['iv'][iv_key].keys()):
                        #print iv_key
                        matching_keys.append(iv_key)
                else:
                    #print iv_key
                    matching_keys.append(iv_key)

        return matching_keys

    def searchCZs(self, bay, row, temperatures = None, percentRns = None, measured_dates = None):

        baystring = self.baystring(bay)
        rowstring = self.rowstring(row)        

        keys = list(self.gamma[baystring][rowstring]['complex_z'].keys())

        if temperatures is None:
            temperatures = []
            for key in keys:
                temperatures.append(self.gamma[baystring][rowstring]['complex_z'][key]['temperature'])

        if percentRns is None:
            percentRns = []
            for key in keys:
                percentRns.append(self.gamma[baystring][rowstring]['complex_z'][key]['percentRn'])
                
        if measured_dates is None:
            measured_dates = []
            for key in keys:
                measured_dates.append(self.gamma[baystring][rowstring]['complex_z'][key]['datetime'])
                                
        matching_keys = []
                     
        for key in keys:
            if self.gamma[baystring][rowstring]['complex_z'][key]['temperature'] in temperatures \
                and self.gamma[baystring][rowstring]['complex_z'][key]['percentRn'] in percentRns \
                and self.gamma[baystring][rowstring]['complex_z'][key]['datetime'][:3] in [mdate[:3] for mdate in measured_dates]:
                
                matching_keys.append(key)

        return matching_keys


    def searchLRs(self, bay, row, temperatures = None, percentRns = None, measured_dates = None):

        baystring = self.baystring(bay)
        rowstring = self.rowstring(row)        

        keys = list(self.gamma[baystring][rowstring]['laser_responsivity'].keys())

        if temperatures is None:
            temperatures = []
            for key in keys:
                temperatures.append(self.gamma[baystring][rowstring]['laser_responsivity'][key]['temperature'])

        if percentRns is None:
            percentRns = []
            for key in keys:
                percentRns.append(self.gamma[baystring][rowstring]['laser_responsivity'][key]['percentRn'])
                
        if measured_dates is None:
            measured_dates = []
            for key in keys:
                measured_dates.append(self.gamma[baystring][rowstring]['laser_responsivity'][key]['datetime'])
                                
        matching_keys = []
                     
        for key in keys:
            if self.gamma[baystring][rowstring]['laser_responsivity'][key]['temperature'] in temperatures \
                and self.gamma[baystring][rowstring]['laser_responsivity'][key]['percentRn'] in percentRns \
                and self.gamma[baystring][rowstring]['laser_responsivity'][key]['datetime'][:3] in [mdate[:3] for mdate in measured_dates]:
                
                matching_keys.append(key)

        return matching_keys

    def searchNoise(self, bay, row, temperatures = None, percentRns = None, measured_dates = None):

        baystring = self.baystring(bay)
        rowstring = self.rowstring(row) 
        
        pulse_keys = list(self.gamma[baystring][rowstring]['noise'].keys())

        if temperatures is None:
            temperatures = []
            for pulse_key in pulse_keys:
                temperatures.append(self.gamma[baystring][rowstring]['noise'][pulse_key]['temperature'])
        
        if percentRns is None:
            percentRns = []
            for pulse_key in pulse_keys:
                percentRns.append(self.gamma[baystring][rowstring]['noise'][pulse_key]['percentRn'])
                                        
        if measured_dates is None:
            measured_dates = []
            for pulse_key in pulse_keys:
                measured_dates.append(self.gamma[baystring][rowstring]['noise'][pulse_key]['datetime'])
        
        matching_keys = []
                     
        for pulse_key in pulse_keys:
            if self.gamma[baystring][rowstring]['noise'][pulse_key]['temperature'] in temperatures \
                and self.gamma[baystring][rowstring]['noise'][pulse_key]['percentRn'] in percentRns \
                and self.gamma[baystring][rowstring]['noise'][pulse_key]['datetime'][:3] in [mdate[:3] for mdate in measured_dates]:
                
                #print pulse_key
                matching_keys.append(pulse_key)
                
        return matching_keys

    def LoopOverPickle(self, bay, rows, gamma = None):
        ''' Loops over rows in pickle '''

        numberofbins = 10

        if gamma is None:
            f = open('gamma', 'r')
            gamma = pickle.load(f)
            f.close()

        rn20 = np.zeros(len(rows), float)

        for k in range(len(rows)):
            row = rows[k]
            rowstring = 'Row%2i' % row
            iv = self.GetIVData(gamma, bay, row)

            if len(iv) != 0:
                print(row)
                pylab.plot(iv[0],iv[1], label=rowstring)
                rn20[k] = self.PercentRn(iv[0], iv[1], rn1 = 0.20)

        pylab.legend()
        pylab.show()
        pylab.hist(rn20, numberofbins, normed=False)
        pylab.show()

    def getNewDictFromKeys(self, original_dict, search_keys):
        '''Return a new dictionary with they keys'''
        
        keys = list(original_dict.keys())
        new_dict = {}
        
        for key in keys:
            if key in search_keys:
                new_dict[key]=original_dict[key]
                
        return new_dict

    # def loadAutotunePickle(self, pickle_file):
        
    #     try:
    #         f = open(pickle_file, 'r')
    #         self.tune_pickle = pickle.load(f)
    #         f.close()
    #     except:
    #         print('Could not open tune pickle file')
    #         self.tune_pickle = {}

    #     return self.tune_pickle
    
    # def getAutotuneSettings(self):
        
    #     bays_list = self.tune_pickle['columnLabel']
    #     num_of_bays = len(bays_list)

    #     sq1biases = self.tune_pickle['sq1BiasVoltageSetPoint']
    #     num_of_sq1biases = len(sq1biases)
    #     sq2fbs = self.tune_pickle['sq2FbVoltageSetPoint']
    #     num_of_sq2fbs = len(sq2fbs)

    #     if num_of_sq1biases == 32:
    #         # loop over bays in autotune file
    #         for bay_index in range(num_of_bays):
    #             bay = bays_list[bay_index]
    #             baystring = self.baystring(bay)
    #             # Loop over rows in autotune file for a given bay
    #             for row_index in range(num_of_sq1biases):
    #                 rowstring = self.rowstring(row_index)
    #                 sq1bias = sq1biases[row_index] 
    #                 sq2fb = sq2fbs[bay_index][row_index]
    #                 new_row_dict = self.createRowInfoDict(radac=sq1bias, sq2fb=sq2fb, dfb_adc=None)
    #                 self.addRowInfoDict(bay, row_index, new_row_dict)
    #     else:
    #         print('Less then 32 rows. Need more info?')
        

    # def dragonToPickle(self, dragon_file=None, bays=['D','E','C','F','A','H','B','G'], fb_cards=[2,3,4,5], \
    #                    fb_channels=[1,2]):
    #     """Translate the dragon file to a gamma pickle file."""
        
    #     f = open(dragon_file,'r')
    #     for line in f:
    #         line_array = line.split()
    #         print(line_array)
    #         if line_array[0] =='nrows:':
    #             nrows = int(line_array[1])
    #         elif line_array[0] =='nsamp:':
    #             nsamp = int(line_array[1])
    #         elif line_array[0] =='sett:':
    #             settling_time = int(line_array[1])
    #         elif line_array[0] =='lsync:':
    #             lsync = int(line_array[1])            
    #         elif line_array[0] =='dfb07':
    #             address = int(line_array[1])
    #             channel = int(line_array[2])
    #             fb_card = fb_cards.index(address)
    #             fb_channel = fb_channels.index(channel)
    #             bay_index = fb_card*2+fb_channel
    #             bay = bays[bay_index]
    #             row = int(line_array[3])
    #             adc_offset = int(line_array[4])
    #             tri_a = int(line_array[5])
    #             tri_b = int(line_array[6])
    #             dac_offset = int(line_array[7])
    #             dac_offset_b = int(line_array[8])
    #             mux_p = int(line_array[9])
    #             mux_i = int(line_array[10])
    #             send_mode = int(line_array[11])
    #             dfb_dict = self.createRowInfoDfbDict(address=address, channel=channel, dfb_adc=adc_offset, \
    #                                                  dfb_dac=dac_offset, sq2fb=dac_offset_b, mux_p=mux_p, mux_i=mux_i, \
    #                                                  tria=tri_a, trib=tri_b)
    #             self.addRowInfoDfbDict(bay, row, dfb_dict)
    #         elif line_array[0] =='ra8':
    #             address = int(line_array[1])
    #             ch_num = int(line_array[2])
    #             row_enabled = int(line_array[3])
    #             row_index = int(line_array[4])
    #             ra_low = int(line_array[5])
    #             ra_high = int(line_array[6])
    #             ra_delay = int(line_array[7])
    #             ra_width = int(line_array[8])
    #             ra_dict = self.createRowInfoRADict(address=address, ch_num=ch_num, row_enabled=row_enabled, \
    #                                                row_index=row_index, ra_low=ra_low, ra_high=ra_high, \
    #                                                ra_delay=ra_delay, ra_width=ra_width)
    #             for bay in bays:
    #                 self.addRowInfoRADict(bay, row_index, ra_dict)
    #     self.savePickle()
        
    #     print('nrows: ', nrows)
    #     print('nsamp: ', nsamp)
    #     print('sett: ', settling_time)
    #     print('lsync: ', lsync)
        
    #     f.close()
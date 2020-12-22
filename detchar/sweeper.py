import sys
from time import sleep
import math
from scipy import linspace, append
from numpy import zeros, append
import numpy as np
from scipy import stats
from PyQt4.Qt import *
import PyQt4
from PyQt4 import Qt

import bluebox
import clock_card as clock
import ndfb_pci_card as pci
import dfb_card as dfb
import dataplot_mpl

# edits:
# 2017 - (Austermann) - added normal_branch_voltage_append keyword to SmartSweep().  This allows you to add a normal branch measurement at a far lever arm (at any resolution) without having to sweep though a huge range at high resolution. e.g. add points at [2.0,1.9,1.8,1.7,1.6] V.

class Sweeper(object):
    '''
    Sweeper 2 - Multiplexing voltage sweeper class
    '''

    def __init__( self, app, dfb, voltage_sweep_source, numberofrows = 1, \
                  pci_intitialized = 'False', pci_channel=0, pci_firmware_delay = 6, pci_nsamp = 4, \
                  *args ):
        self.app = app
        self.dfb = dfb
        self.voltage_source = voltage_sweep_source
        
        self.pci_intitialized = pci_intitialized
        # PCI info is only used if the PCI is not initialized by the calling program
        self.pci_mask = 0x0001
        self.pci_firmware_delay = pci_firmware_delay
        self.pci_nsamp = pci_nsamp

        # Defaults in case they are not specified in function call
        self.voltage_start = 2.5
        self.voltage_end = 0
        self.voltage_step = 0.1      
        self.dwell_time = 0.02

        self.numberofrows = numberofrows
        
        #These get changed in SmartSweetpAutoStop()
        self.autorange = 1
        self.dy_cut = 1
        

    def set_data_directory(self, dir):
        self.data_directory = dir
        

    def set_voltage(self, voltage):
        
        self.voltage_source.setvolt(voltage)

    def relock(self, channel=0, dac_value=8000):
        if dac_value is None:
            dac_value=self.dfb.getDAC(channel)
        print('Relocking now')
        self.dfb.unlock(channel)
        #self.dfb.sendRow(channel)
        sleep(.1)
        #self.dfb.setDAC(channel, dac=dac_value)
        self.dfb.lock(channel)
        #self.dfb.sendRow(channel)
        sleep(.1)
    
         
    def get_feedback_voltage(self, column=0):

        datarows = np.zeros(self.numberofrows, float)
        
        data = pci.getData()
        #print '%f %f' % (data[0, 0, 0, 0], data[0, 0, 0, 1])             # [frame, row, column, error/feedback = 0/1]
        #error    = data[0,0,0,0]

        for row in range(0,self.numberofrows):
            datarows[row] = np.average(data[:, row, column, 1])
        
        return datarows

    def get_error_voltage(self, column=0):

        datarows = np.zeros(self.numberofrows, float)
        
        data = pci.getData()
        #print '%f %f' % (data[0, 0, 0, 0], data[0, 0, 0, 1])             # [frame, row, column, error/feedback = 0/1]
        #error    = data[0,0,0,0]

        for row in range(0,self.numberofrows):
            datarows[row] = np.average(data[:, row, column, 0])
        
        return datarows

    def pci_initialize(self, pci_mask, pci_firmware_delay, pci_nsamp):

        pci.initializeCard()
        pci.startDMA(pci_mask, pci_firmware_delay, pci_nsamp)      # (mask, firmware delay, nsamp)
        num_frames = pci.getNumberOfFrames()
        print("Number of frame before = %i" % num_frames)
        pci.setNumberOfFrames(100)
        num_frames = pci.getNumberOfFrames()
        print("Number of frame after  = %i" % num_frames)

    def pci_close(self):
    
        pci.stopDMA()
        pci.closeCard()    

    def DumbSweep(self, voltage_start=None, voltage_end=None, voltage_step=None, dwell_time=None, \
                  measure_rows=None, sweepstring=None, iv_string=''):

        if voltage_start is None:
            voltage_start = self.voltage_start

        if voltage_end is None:
            voltage_end = self.voltage_end

        if voltage_step is None:
            voltage_step = self.voltage_step
            
        if dwell_time is None:
            dwell_time = self.dwell_time        
        

        if measure_rows is None:
            measure_rows = [True for flag in range(self.numberofrows)]

        #print "Creating plot..."
        self.plot = dataplot_mpl.dataplot_mpl(title="Sweeper", x_label="Voltage Bias (V)", y_label="Feedback Voltage (V)")
        # Set plot title
        self.plot.set_title('Sweeper - ' + sweepstring + "_" + iv_string)

        self.curves = []

        for row in range(self.numberofrows):
            row_title = "Row %i" % row
            self.curves.append(self.plot.addLine(name=row_title, xdata=[], ydata=[]))
        self.plot.show()
        b = self.app.processEvents()
        #print "Done creating plot."
        
        if voltage_start > voltage_end and voltage_step > 0:
            voltage_step = -1.0 * voltage_step
        
        numpts = abs(int((voltage_start - voltage_end) / voltage_step)) + 1
        print('number of points = %i' % numpts)

        # Initialize PCI card and start DMA if not done by calling program
        if self.pci_intitialized == 'False':
            self.pci_initialize(self.pci_mask, self.pci_firmware_delay, self.pci_nsamp)

        # Set voltage to start
        #self.set_voltage(2.5)  #drive normal for sure
        #sleep(.5)
        self.set_voltage(voltage_start)

        sleep(0.2)
        self.set_voltage(voltage_start)
        sleep(1)

        voltage_array = zeros(0, dtype=float)
        raw_array = zeros((self.numberofrows, 0), dtype=float)

        voltages = linspace( voltage_start, voltage_end, numpts )
        print('dwell_time is', dwell_time)
        for v in voltages:
            self.set_voltage(v)
            voltage_array = append(voltage_array, v)
            sleep(dwell_time)
            result = self.get_feedback_voltage()
            result = self.get_feedback_voltage()  #Call for another buffer to make sure it is current (consider averaging the buffer)
            #result = self.get_error_voltage()
            results = np.array([result])
            raw_array = append(raw_array, np.transpose(results), axis=1)

            for row in range(self.numberofrows):
                self.curves[row].update_data(voltage_array, raw_array[row])
            self.plot.update()
            b = self.app.processEvents()

        # Save the plot as a PDF
        # Set the file name
        pdf_file_name = self.data_directory + 'iv_' + sweepstring + "_" + iv_string + '.pdf'
        self.plot.exportPDF(pdf_file_name)
            
        # Stop DMA and Close card if PCI was initialized in this loop
        if self.pci_intitialized == 'False':
            self.pci_close()

        return raw_array, voltage_array


    def SmartSweep(self, voltage_start=None, voltage_end=None, voltage_step=None, dwell_time=None, \
                         measure_rows=None, jump_start=-1, jump_stop=-1, sweepstring=None, iv_string='',\
                         normal_voltage_blast=2.5, normal_branch_voltage_append=[]):

        if voltage_start is None:
            voltage_start = self.voltage_start

        if voltage_end is None:
            voltage_end = self.voltage_end

        if voltage_step is None:
            voltage_step = self.voltage_step
            
        if dwell_time is None:
            dwell_time = self.dwell_time        
        

        if measure_rows is None:
            measure_rows = [True for flag in range(self.numberofrows)]

        #print "Creating plot..."
        self.plot = dataplot_mpl.dataplot_mpl(title="Sweeper", x_label="Voltage Bias (V)", y_label="Feedback Voltage (V)")
        # Set plot title
        self.plot.set_title('Sweeper - ' + sweepstring + "_" + iv_string)

        self.curves = []

        for row in range(self.numberofrows):
            row_title = "Row %i" % row
            self.curves.append(self.plot.addLine(name=row_title, xdata=[], ydata=[]))
        self.plot.show()
        b = self.app.processEvents()
        #print "Done creating plot."
        
        if voltage_start > voltage_end and voltage_step > 0:
            voltage_step = -1.0 * voltage_step
        
        numpts = abs(int((voltage_start - voltage_end) / voltage_step)) + 1
        print('number of points = %i' % numpts)

        flux_skip_threshold = 0.80

        #print measure_rows

        # Initialize PCI card and start DMA if not done by calling program
        if self.pci_intitialized == 'False':
            self.pci_initialize(self.pci_mask, self.pci_firmware_delay, self.pci_nsamp)

        upper_voltage_limit = 0.85
        lower_voltage_limit = 0.15
        #voltage_skip = jump_size  Skip 0.1 V to avoid bad spot
        skip_point_high = jump_start
        print('Skip point high %f' % skip_point_high)
        skip_point_low = jump_stop
        print('Skip point low %f' % skip_point_low)
        row = 0
        #skip_prep_relock = jump_start - (1*voltage_step)
        #print 'Relock before skip below %f' % skip_prep_relock

        voltage_array = zeros(0, dtype=float)
        raw_array = zeros((self.numberofrows, 0), dtype=float)
        adjusted_array = zeros((self.numberofrows, 0), dtype=float)
        current_offset = np.zeros(self.numberofrows)
        offset = 0
        unlockFlag = [False for flag in range(self.numberofrows)]
        flux_slip_flag = [False for flag in range(self.numberofrows)]
        unlock_voltage = np.zeros(self.numberofrows)

        after_jump_flag = False

        voltages = linspace( voltage_start, voltage_end, numpts )
        if len(normal_branch_voltage_append) > 0:
            voltages = np.append(normal_branch_voltage_append,voltages)
        v_jump_high_index = np.argmin(abs(voltages-skip_point_high))
        
                # Set voltage to start and lock
        self.set_voltage(normal_voltage_blast)  #drive normal for sure
        sleep(2)
        self.set_voltage(voltages[0])
        sleep(5)
        for row in range(self.numberofrows):
            self.relock(channel = row)
            sleep(.1)

        if skip_point_high > 0:
            skip_prep_relock = voltages[v_jump_high_index-1]
        else:
            skip_prep_relock = -1
        print('Relock before skip at %f' % skip_prep_relock)
        
        for v in voltages:
            if v >= skip_point_low and v <= skip_point_high:
                voltage_array = append(voltage_array, v)
                #result = np.zeros(self.numberofrows, dtype=float)
                result = np.ones(self.numberofrows, dtype=float)*np.NaN
                results = np.array([result])
                raw_array = append(raw_array, np.transpose(results), axis=1)
                adjusted_array = append(adjusted_array, np.transpose(results), axis=1)
                after_jump_flag = True
                continue
            self.set_voltage(v)
            if after_jump_flag is True:
               sleep(dwell_time*10.0)
               after_jump_flag = False
            voltage_array = append(voltage_array, v)
            sleep(dwell_time)
            result = self.get_feedback_voltage()
            result = self.get_feedback_voltage()  #Call for another buffer to make sure it is current (consider averaging the buffer)
            results = np.array([result])
            raw_array = append(raw_array, np.transpose(results), axis=1)
            adjusted_result = result + current_offset
            adjusted_results = np.array([adjusted_result])
            adjusted_array = append(adjusted_array, np.transpose(adjusted_results), axis=1)

        #sweepplot.AddPointPlot(x,y)

            for row in range(self.numberofrows):
                if np.size(adjusted_array[row]) > 1 and skip_point_high > 0 and v > skip_point_high  \
                                                    and abs(adjusted_array[row,-1]-adjusted_array[row,-2]) > flux_skip_threshold \
                                                    and result[row] > upper_voltage_limit and measure_rows[row] is True:
                    flux_slip_flag[row] = True
                    print('Flux Slip and Above Upper Voltage Limit Row %d' %row)
                    self.relock(channel = row)
                    shifted_result = self.get_feedback_voltage()
                    self.set_voltage(voltage_array[-2])
                    sleep(dwell_time)
                    new_result_last_point = self.get_feedback_voltage()
                    offset = raw_array[row,-2] - new_result_last_point[row]
                    current_offset[row] = current_offset[row] + offset
                    adjusted_array[row,-1] = shifted_result[row]+current_offset[row]
                    self.set_voltage(v)
                elif result[row] > 0.98 and measure_rows[row] is True:
                    print("Unlock Detected - Top Rail Row %d" % row)
                    if unlockFlag[row]is False:
                        unlockFlag[row] = True
                        unlock_voltage[row] = v
                        self.relock(channel = row)
                elif result[row] < 0.01 and measure_rows[row] is True:
                    print("Unlock Detected - Bottom Rail Row %d" % row)
                    if unlockFlag[row] is False:
                        unlockFlag[row] = True
                        unlock_voltage[row] = v
                        self.relock(channel = row)
                elif np.size(adjusted_array[row]) > 1 and skip_point_high > 0 and v > skip_point_high \
                        and abs(adjusted_array[row,-1]-adjusted_array[row,-2]) > flux_skip_threshold:
                    flux_slip_flag[row] = True
                    print('Flux Slip Row %d' %row)
                    self.set_voltage(voltage_array[-2])
                    sleep(dwell_time)
                    new_result_last_point = self.get_feedback_voltage()
                    new_result_last_point = self.get_feedback_voltage()
                    offset = raw_array[row,-2] - new_result_last_point[row]
                    current_offset[row] = current_offset[row] + offset
                    adjusted_array[row,-1] = result[row]+current_offset[row]
                    self.set_voltage(v)
                elif result[row] > upper_voltage_limit and result[row] <= 0.98 and measure_rows[row] is True:
                    print("Above upper voltage limit Row %d." % row)
                    self.relock(channel = row)
                    shifted_result = self.get_feedback_voltage()
                    shifted_result = self.get_feedback_voltage()
                    offset = result[row] - shifted_result[row]
                    current_offset[row] = current_offset[row] + offset
                elif result[row] < lower_voltage_limit and result[row] >= 0.01 and measure_rows[row] is True:
                    print("Below lower voltage limit Row %d." % row)
                    self.relock(channel = row)
                    sleep(1)
                    shifted_result = self.get_feedback_voltage()
                    shifted_result = self.get_feedback_voltage() #Call for a new buffer to make sure it is current
                    sleep(1)
                    offset = result[row] - shifted_result[row]
                    current_offset[row] = current_offset[row] + offset
                elif v == skip_prep_relock:
                    print("Relock Before Skip Row %d." % row)
                    self.relock(channel = row)
                    sleep(1)
                    shifted_result = self.get_feedback_voltage()
                    shifted_result = self.get_feedback_voltage()
                    offset = result[row] - shifted_result[row]
                    current_offset[row] = current_offset[row] + offset

            for row in range(self.numberofrows):
                #self.aplot.update(x=voltage_array, y=adjusted_array[row], row=row)            

                #print 'R %d input, output, adjusted, offset voltage = %f, %f, %f, %f' % \
                # (row, v, result[row], adjusted_result[row], current_offset[row])

                #print row, voltage_array, adjusted_result[row]
                self.curves[row].update_data(voltage_array, adjusted_array[row])
            self.plot.update()
            #sleep(0.1)
            b = self.app.processEvents()

        # Save the plot as a PDF
        # Set the file name
        pdf_file_name = self.data_directory + 'iv_' + sweepstring + "_" + iv_string + '.pdf'
        self.plot.exportPDF(pdf_file_name)
            
        # Stop DMA and Close card if PCI was initialized in this loop
        if self.pci_intitialized == 'False':
            self.pci_close()

        #print unlockFlag
        #print unlock_voltage

        return raw_array, voltage_array, adjusted_array, unlockFlag, unlock_voltage, flux_slip_flag



    def SmartSweeperWithJump(self, voltage_start=None, voltage_end=None, voltage_step=None,  dwell_time=None, sweepstring=None, iv_string='', polarity = 'Positive'):

        if voltage_start is None:
            voltage_start = self.voltage_start

        if voltage_end is None:
            voltage_end = self.voltage_end

        if voltage_step is None:
            voltage_step = self.voltage_step

        if dwell_time is None:
            dwell_time = self.dwell_time   

        if sweepstring is None:
            sweepstring = 'a'

        #root_filename = '/home/pcuser/data/02_dtest_ar7/20090827/'
        log_filename = self.data_directory + 'iv_' + sweepstring + "_" + iv_string + '_log.txt'
        log_file = open(log_filename, 'w')

        datapass = 1

        log_file.write('Voltage Step ' + str(voltage_step) + '\n')

        # Initial Sweep
        raw,volt,fbvolt,unlock,unlockv,flux_slip_flag = \
            self.SmartSweep(voltage_start=voltage_start, voltage_end=voltage_end, voltage_step=voltage_step, sweepstring=sweepstring, iv_string=iv_string)
        fbfinal = np.copy(fbvolt)

        log_file.write('Unlock Flags ' + str(unlock) + '\n')
        log_file.write('Unlock Voltages ' + str(unlockv) + '\n')
        log_file.write('Flux Slip Flags ' + str(flux_slip_flag) + '\n')

        dataout = np.vstack((volt,fbvolt))
        dataout = dataout.transpose()
        file_name = self.data_directory + 'iv_' + sweepstring + "_" + iv_string + '_' + str(datapass) + '.dat'
        np.savetxt(file_name, dataout)

        dataout = np.vstack((volt,raw))
        dataout = dataout.transpose()
        file_name = self.data_directory + 'iv_' + sweepstring + "_" + iv_string + '_raw_' + str(datapass) + '.dat'
        np.savetxt(file_name, dataout)

        rn20, rn10 = self.PercentRn(volt, fbvolt, rn1 = 0.20, rn2 = 0.10, polarity = polarity)

        print('Min of Rn20')
        print(min(rn20))
        log_file.write('20% Rn' + str(rn20) + '\n')
        print('Min of Rn10')
        print(min(rn10))
        log_file.write('10% Rn' + str(rn10) + '\n')
        print('Max Unlock')
        print(max(unlockv))

        if max(unlockv) < min(rn10):
            jump_start = min(rn10)
        elif max(unlockv) < min(rn20):
            jump_start = max(unlockv)
        else:
            sorted_unlockv = np.trim_zeros(np.sort(unlockv),'f')
            if len(sorted_unlockv) > 0:  #Make sure that the array is not empty
                jump_start = min(sorted_unlockv)
            else:
                jump_start = -1
                print('Unlock array is all zeros')

        #print 'Jump_start %f' % jump_start
        skip_point = jump_start
        #print "Unlock voltage for next pass %f" % skip_point
        log_file.write('Unlock voltage for next pass ' + str(skip_point) + '\n')

        if min(rn20) < 0:
            jump_stop = 0
        else:
            jump_stop = self.FindJumpStopVoltage(volt, fbvolt, skip_point,polarity = polarity)

        log_file.write('Jump Stop Voltage ' + str(jump_stop) + '\n')

        unlock = [True for row in unlock]
        measured_flag = unlock[:]  #make a copy of unlock flags


        while any(unlock) and datapass < 3:

            datapass = datapass + 1
            raw,volt,fbvolt, unlock, unlockv, flux_slip_flag = \
                self.SmartSweep(voltage_start= voltage_start, voltage_end = voltage_end, voltage_step = voltage_step, \
                                measure_rows = unlock, jump_start = skip_point, jump_stop = jump_stop, sweepstring=sweepstring, iv_string=iv_string)

            log_file.write('Data Pass ' + str(datapass) + '\n')
            log_file.write('Unlock Flags ' + str(unlock) + '\n')
            log_file.write('Unlock Voltages ' + str(unlockv) + '\n')
            log_file.write('Flux Slip Flags ' + str(flux_slip_flag) + '\n')

            dataout = np.vstack((volt,fbvolt))
            dataout = dataout.transpose()
            file_name = self.data_directory + 'iv_'+ sweepstring + "_" + iv_string + '_' + str(datapass) + '.dat'
            np.savetxt(file_name, dataout)

            dataout = np.vstack((volt,raw))
            dataout = dataout.transpose()
            file_name = self.data_directory + 'iv_' + sweepstring + "_" + iv_string + '_raw_' + str(datapass) + '.dat'
            np.savetxt(file_name, dataout)

            print('Unlock flags')
            print(unlock)
            print('Measured_flags')
            print(measured_flag)
            for row in range(len(measured_flag)):
                if measured_flag[row]:
                    print('Replace Row %d ' % row)
                    fbfinal[row,:] = np.copy(fbvolt[row,:])

            rn20, rn10 = self.PercentRn(volt, fbvolt, rn1 = 0.20, rn2 = 0.10, polarity = polarity)

            log_file.write('20% Rn' + str(rn20) + '\n')
            log_file.write('10% Rn' + str(rn10) + '\n')

            
            #   if unlock[k] == False:
            #        for x in fbvolt[k,:]:
            #            if x is np.nan:
            #                unlock[k] = True
            #                unlockv[k] = rn10[k]
            #                exit
                            
            if max(unlockv) < min(rn10):
                jump_start = min(rn10)
            elif max(unlockv) < min(rn20):
                #jump_start = max(unlockv)
                jump_start = min(rn20)
            else:
                jump_start = max(rn20)
                #jump_start = min(unlockv)

            skip_point = jump_start
            print("Unlock voltage for next pass %f" % skip_point)
            log_file.write('Unlock voltage for next pass ' + str(skip_point) + '\n')

            if min(rn20) < 0:
                jump_stop = 0
            else:
                jump_stop = self.FindJumpStopVoltage(volt, fbvolt, skip_point,polarity = polarity)
        
            measured_flag = unlock[:]  #make a copy of unlock flags

        ivdata = np.vstack((volt,fbfinal))
        dataout = ivdata.transpose()
        file_name = self.data_directory + 'iv_' + sweepstring + "_" + iv_string + '_final.dat'
        np.savetxt(file_name, dataout)

        return ivdata, rn20

    def PercentRn(self, volt, fbvolt, rn1=0.20, rn2=0.10, polarity = 'Positive'):
        '''Calculate specified percent of Rn'''
        fbnormal = np.zeros_like(fbvolt)
        fbsuper = np.zeros_like(fbvolt)
        rn1_lookup = np.zeros(self.numberofrows, float)
        rn2_lookup = np.zeros(self.numberofrows, float)
        
        if polarity == 'Negative':
          fbvolt = fbvolt*(-1) #Reflect about 0

        try:
          for row in range(len(fbvolt)):
            cusp_index = np.argmax(fbvolt[row]) #locate the cusp index by the maximum
            inflection_index = np.argmin(fbvolt[row,:cusp_index]) #inflection is minimum to right of maximum
            linear_region_index = int(math.floor(inflection_index/2)) #linear region - twice the the inflection point
            x = volt[:linear_region_index] #bias array in linear region
            y = fbvolt[row,:linear_region_index] #fb array in linear region
            slope,intercept,r,tt,stderr = stats.linregress(x,y) #linear fit
            fbnormal[row] = fbvolt[row,:]-intercept #fb array shifted so normal region extroplates to zero
            fbsuper[row] = fbvolt[row,:] - fbvolt[row,-1] #fb array where superconucting array is shifted to be zero at 0V
            resistance_array = volt/fbnormal[row]
            rn1_index = np.argmin(abs(resistance_array[inflection_index:cusp_index]-rn1/slope))+inflection_index #closest array index to 20% Rn 
            rn1_lookup[row] = volt[rn1_index] #Closest voltage to 20% Rn
            rn2_index = np.argmin(abs(resistance_array[inflection_index:cusp_index]-rn2/slope))+inflection_index
            rn2_lookup[row] = volt[rn2_index]
        except:
          np.ndarray.fill(rn1_lookup,-1)
          np.ndarray.fill(rn2_lookup,-1)

        print('Calculated 20%Rn voltages')
        print(rn1_lookup)
        print('Calculated 10%Rn voltages')
        print(rn2_lookup)

        return rn1_lookup, rn2_lookup

    def FindJumpStopVoltage(self, volt, fbvolt, skip_point, polarity = 'Positive'):
        #Calculate how far to skip
        fbnormal = np.zeros_like(fbvolt)
        fbsuper = np.zeros_like(fbvolt)
        jump_stop_voltage = np.zeros(self.numberofrows, float)
        
        if polarity == 'Negative':
             fbvolt = fbvolt*(-1) #Reflect about 0        

        jump_start_index = np.argmin(abs(volt-skip_point)) #find the index of start of jump

        for row in range(len(fbvolt)):
            cusp_index = np.argmax(fbvolt[row]) #locate the cusp index by the maximum
            inflection_index = np.argmin(fbvolt[row,:cusp_index]) #inflection is minimum to right of maximum
            linear_region_index = int(math.floor(inflection_index/2)) #linear region - twice the the inflection point
            x = volt[:linear_region_index] #bias array in linear region
            y = fbvolt[row,:linear_region_index] #fb array in linear region
            slope,intercept,r,tt,stderr = stats.linregress(x,y) #linear fit
            fbnormal[row] = fbvolt[row,:]-intercept #fb array shifted so normal region extroplates to zero
            fbsuper[row] = fbvolt[row,:] - fbvolt[row,-1] #fb array where superconucting array us shifted to be zero at 0V
            jump_start_fb_voltage = fbnormal[row,jump_start_index] #fb volatage at start of jump
            jump_stop_voltage_index = np.argmin(abs(fbsuper[row,cusp_index:]-jump_start_fb_voltage))+cusp_index #index of jump stop voltage
            jump_stop_voltage[row] = volt[jump_stop_voltage_index] #the ideal jump stop voltage for a given row

        #print 'Calculated Jump Stop Voltages'
        #print jump_stop_voltage

        jump_stop = np.median(jump_stop_voltage) #set jump_stop as median of the jumpo stop of the rows

        return jump_stop

    def CreateVoltageArray(self, sweep_type, voltage_start, voltage_end, voltage_step, number_of_sweeps = 100):
        '''Create an array of voltage values to sweep through'''

        numpts = abs(int((voltage_start - voltage_end) / voltage_step)) + 1
        #print 'numpts = %i' % numpts

        if sweep_type == 'Unidirectional':
            voltages = linspace( voltage_start, voltage_end, numpts )
        elif sweep_type == 'Bidirectional':
            v1 = linspace( voltage_start, voltage_end, numpts )
            v2 = linspace( voltage_end, voltage_start, numpts )
            voltages = append( v1, v2 )
        elif sweep_type == 'Continuous':
            voltages = []
            for x in range(number_of_sweeps):
                v1 = linspace( voltage_start, voltage_end, numpts )
                v2 = linspace( voltage_end, voltage_start, numpts )
                v12 = append(v1, v2)
                voltages = append( voltages, v12 )
        else:
            print("Error: sweep type invalid!")
            voltages = []

        return voltages

    def SmartSweepAutoStop(self, voltage_start=None, voltage_end=None, voltage_step=None, dwell_time=None, \
                         measure_rows=None, jump_start=-1, jump_stop=-1, sweepstring=None, iv_string='',\
                         autorange = 4, dy_cut = -0.07):
        
        self.autorange = autorange
        self.dy_cut = dy_cut

        if voltage_start is None:
            voltage_start = self.voltage_start

        if voltage_end is None:
            voltage_end = self.voltage_end

        if voltage_step is None:
            voltage_step = self.voltage_step
            
        if dwell_time is None:
            dwell_time = self.dwell_time        
        

        if measure_rows is None:
            measure_rows = [True for flag in range(self.numberofrows)]

        #print "Creating plot..."
        self.plot = dataplot_mpl.dataplot_mpl(title="Sweeper", x_label="Voltage Bias (V)", y_label="Feedback Voltage (V)")
        # Set plot title
        self.plot.set_title('Sweeper - ' + sweepstring + "_" + iv_string)

        self.curves = []

        for row in range(self.numberofrows):
            row_title = "Row %i" % row
            self.curves.append(self.plot.addLine(name=row_title, xdata=[], ydata=[]))
        self.plot.show()
        b = self.app.processEvents()
        #print "Done creating plot."
        
        if voltage_start > voltage_end and voltage_step > 0:
            voltage_step = -1.0 * voltage_step
        
        numpts = abs(int((voltage_start - voltage_end) / voltage_step)) + 1
        print('number of points = %i' % numpts)

        flux_skip_threshold = 0.80

        #print measure_rows

        # Initialize PCI card and start DMA if not done by calling program
        if self.pci_intitialized == 'False':
            self.pci_initialize(self.pci_mask, self.pci_firmware_delay, self.pci_nsamp)

        upper_voltage_limit = 0.85
        lower_voltage_limit = 0.15
        #voltage_skip = jump_size  Skip 0.1 V to avoid bad spot
        skip_point_high = jump_start
        print('Skip point high %f' % skip_point_high)
        skip_point_low = jump_stop
        print('Skip point low %f' % skip_point_low)
        row = 0
        #skip_prep_relock = jump_start - (1*voltage_step)
        #print 'Relock before skip below %f' % skip_prep_relock
        
        # Set voltage to start and lock
        self.set_voltage(2.5)  #drive nomrmal for sure
        sleep(.5)
        self.set_voltage(voltage_start)

        sleep(0.2)
        self.set_voltage(voltage_start)
        sleep(5)

        for row in range(self.numberofrows):
            self.relock(channel = row)
            sleep(.1)

        voltage_array = zeros(0, dtype=float)
        raw_array = zeros((self.numberofrows, 0), dtype=float)
        adjusted_array = zeros((self.numberofrows, 0), dtype=float)
        current_offset = np.zeros(self.numberofrows)
        offset = 0
        unlockFlag = [False for flag in range(self.numberofrows)]
        flux_slip_flag = [False for flag in range(self.numberofrows)]
        unlock_voltage = np.zeros(self.numberofrows)

        after_jump_flag = False

        voltages = linspace( voltage_start, voltage_end, numpts )
        v_jump_high_index = np.argmin(abs(voltages-skip_point_high))

        if skip_point_high > 0:
            skip_prep_relock = voltages[v_jump_high_index-1]
        else:
            skip_prep_relock = -1
        print('Relock before skip at %f' % skip_prep_relock)
        

        dadj_arr = []
        for v in voltages:
            if v >= skip_point_low and v <= skip_point_high:
                voltage_array = append(voltage_array, v)
                #result = np.zeros(self.numberofrows, dtype=float)
                result = np.ones(self.numberofrows, dtype=float)*np.NaN
                results = np.array([result])
                raw_array = append(raw_array, np.transpose(results), axis=1)
                adjusted_array = append(adjusted_array, np.transpose(results), axis=1)
                after_jump_flag = True
                continue
            self.set_voltage(v)
            if after_jump_flag is True:
               sleep(dwell_time*10.0)
               after_jump_flag = False
            voltage_array = append(voltage_array, v)
            sleep(dwell_time)
            result = self.get_feedback_voltage()
            result = self.get_feedback_voltage()  #Call for another buffer to make sure it is current (consider averaging the buffer)
            results = np.array([result])
            raw_array = append(raw_array, np.transpose(results), axis=1)
            adjusted_result = result + current_offset
            adjusted_results = np.array([adjusted_result])
            adjusted_array = append(adjusted_array, np.transpose(adjusted_results), axis=1)
            if (len(adjusted_array[0]) > 1):
                    dadj_arr.append(results[0][0]-prev_num)
            prev_num = results[0][0]
        #sweepplot.AddPointPlot(x,y)

            for row in range(self.numberofrows):
                if np.size(adjusted_array[row]) > 1 and skip_point_high > 0 and v > skip_point_high  \
                                                    and abs(adjusted_array[row,-1]-adjusted_array[row,-2]) > flux_skip_threshold \
                                                    and result[row] > upper_voltage_limit and measure_rows[row] is True:
                    flux_slip_flag[row] = True
                    print('Flux Slip and Above Upper Voltage Limit Row %d' %row)
                    self.relock(channel = row)
                    shifted_result = self.get_feedback_voltage()
                    self.set_voltage(voltage_array[-2])
                    sleep(dwell_time)
                    new_result_last_point = self.get_feedback_voltage()
                    offset = raw_array[row,-2] - new_result_last_point[row]
                    current_offset[row] = current_offset[row] + offset
                    adjusted_array[row,-1] = shifted_result[row]+current_offset[row]
                    self.set_voltage(v)
                elif result[row] > 0.98 and measure_rows[row] is True:
                    print("Unlock Detected - Top Rail Row %d" % row)
                    if unlockFlag[row]is False:
                        unlockFlag[row] = True
                        unlock_voltage[row] = v
                        self.relock(channel = row)
                elif result[row] < 0.01 and measure_rows[row] is True:
                    print("Unlock Detected - Bottom Rail Row %d" % row)
                    if unlockFlag[row] is False:
                        unlockFlag[row] = True
                        unlock_voltage[row] = v
                        self.relock(channel = row)
                elif np.size(adjusted_array[row]) > 1 and skip_point_high > 0 and v > skip_point_high \
                        and abs(adjusted_array[row,-1]-adjusted_array[row,-2]) > flux_skip_threshold:
                    flux_slip_flag[row] = True
                    print('Flux Slip Row %d' %row)
                    self.set_voltage(voltage_array[-2])
                    sleep(dwell_time)
                    new_result_last_point = self.get_feedback_voltage()
                    new_result_last_point = self.get_feedback_voltage()
                    offset = raw_array[row,-2] - new_result_last_point[row]
                    current_offset[row] = current_offset[row] + offset
                    adjusted_array[row,-1] = result[row]+current_offset[row]
                    self.set_voltage(v)
                elif result[row] > upper_voltage_limit and result[row] <= 0.98 and measure_rows[row] is True:
                    print("Above upper voltage limit Row %d." % row)
                    self.relock(channel = row)
                    shifted_result = self.get_feedback_voltage()
                    shifted_result = self.get_feedback_voltage()
                    offset = result[row] - shifted_result[row]
                    current_offset[row] = current_offset[row] + offset
                elif result[row] < lower_voltage_limit and result[row] >= 0.01 and measure_rows[row] is True:
                    print("Below lower voltage limit Row %d." % row)
                    self.relock(channel = row)
                    sleep(1)
                    shifted_result = self.get_feedback_voltage()
                    shifted_result = self.get_feedback_voltage() #Call for a new buffer to make sure it is current
                    sleep(1)
                    offset = result[row] - shifted_result[row]
                    current_offset[row] = current_offset[row] + offset
                elif v == skip_prep_relock:
                    print("Relock Before Skip Row %d." % row)
                    self.relock(channel = row)
                    sleep(1)
                    shifted_result = self.get_feedback_voltage()
                    shifted_result = self.get_feedback_voltage()
                    offset = result[row] - shifted_result[row]
                    current_offset[row] = current_offset[row] + offset

            for row in range(self.numberofrows):
                #self.aplot.update(x=voltage_array, y=adjusted_array[row], row=row)            

                #print 'R %d input, output, adjusted, offset voltage = %f, %f, %f, %f' % \
                # (row, v, result[row], adjusted_result[row], current_offset[row])

                #print row, voltage_array, adjusted_result[row]
                self.curves[row].update_data(voltage_array, adjusted_array[row])
            self.plot.update()
            if (len(voltage_array) > self.autorange):
                strt = len(dadj_arr)-self.autorange
                tdaa = dadj_arr[strt:]
                keep_going = self.autodetect(tdaa,dire)
            elif (len(voltage_array) == self.autorange):
                dire = self.checkdir(dadj_arr)
            else:
                keep_going = True
            #sleep(0.1)
            b = self.app.processEvents()
            if (keep_going == False):
                break

        # Save the plot as a PDF
        # Set the file name
        pdf_file_name = self.data_directory + 'iv_' + sweepstring + "_" + iv_string + '.pdf'
        self.plot.exportPDF(pdf_file_name)
            
        # Stop DMA and Close card if PCI was initialized in this loop
        if self.pci_intitialized == 'False':
            self.pci_close()

        #print unlockFlag
        #print unlock_voltage

        return raw_array, voltage_array, adjusted_array, unlockFlag, unlock_voltage, flux_slip_flag

## Take first measurements to determine polarity (direction of plot).
## Number of measurements is determined by self.num_ahead
## Also assigns the first local extrema
    def checkdir(self,dy):
        dy = np.array(dy)
        med_vals = np.median(dy)
        if (med_vals < 0):
            return 1
        if (med_vals > 0):
            return -1

## Just read four values in
    def autodetect(self,dy,dire):
        dy = np.array(dy)
        back_med = np.median(dy)
        if ((dire * back_med) < self.dy_cut): return False
        else: return True

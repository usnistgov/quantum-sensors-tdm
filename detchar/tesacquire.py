import sys
from time import sleep
import math
from numpy import zeros, append
import numpy as np
#from scipy.io import write_array
from scipy import linspace, stats, polyfit, polyval, fftpack
import pylab
import pickle
import array
import matplotlib
import tesanalyze
import dataplot_mpl
from IPython import embed

import crate
import ndfb_pci_card as pci

class TESAcquire():
    '''
    Acquisition functions
    '''
    
    def __init__(self, app = None):
        
        #App for real time plotting functions
        self.app = app
        self.tesana = tesanalyze.TESAnalyze()

        
    def getMixData(self,mix_param, buffer, row=0,column=0,use_mix=False):
        #buffer = pci.getData()
        #print 'You are now using the mix!'
        v_error    = buffer[:,row,column,0]
        v_feedback = buffer[:,row,column,1]
        #correct error for overwrapping
        for point in range(len(v_error)):
            if v_error[point] > 3:
                v_error[point] = v_error[point]-4
        #match up the measured error with the applied FB
        #v_feedback = v_feedback[:-1]
        #v_error = v_error[1:]
        #mix_param = 0.06846
        if use_mix == False:
            mix_param = 0
            #print 'not using mix parameter'
        #print 'mix parameter is ', mix_param
        v_mix = v_feedback + mix_param * v_error
        return v_mix
    
    def getDataBoth(self):
                
        buffer = pci.getData()
        v_error    = buffer[:,0,0,0]
        v_feedback = buffer[:,0,0,1]

        #correct error for overwrapping
        for point in range(len(v_error)):
            if v_error[point] > 3:
                v_error[point] = v_error[point]-4
                
        return v_feedback, v_error

    def getDataBothAllColumns(self):
                
        buffer = pci.getData()
        v_error    = buffer[:,0,:,0]
        v_feedback = buffer[:,0,:,1]
        
        #correct error for overwrapping
        v_error[v_error > 3] = v_error[v_error > 3] - 4.0

        v_feedback = v_feedback.transpose()
        v_error = v_error.transpose()
                
        return v_feedback, v_error

    def getDataAllRows(self):
        
        #print 'Getting Data'
        buffer = pci.getData()
        #print 'Got Data'
        v_error    = buffer[:,:,0,0]
        v_feedback = buffer[:,:,0,1]

        #correct error for overwrapping
        v_error[v_error > 3] = v_error[v_error > 3] - 4.0
        #for point in range(len(v_error)):
        #    if v_error[point] > 3:
        #        v_error[point] = v_error[point]-4
        
        v_feedback = v_feedback.transpose()
        v_error = v_error.transpose()
                
        return v_feedback, v_error
                        
    def measureComplexZ(self, dfb0, lsync, frequency_array, function_generator, mix_param=0.0, use_mix=False, dfb_dac=8000, number_mux_rows=2):
        #Setup plots
        print 'use mix is ', use_mix
        print 'mix param is ', mix_param
        plot1 = dataplot_mpl.dataplot_mpl(width=6, height=5, title="ComplexZ", x_label="Frequency (Hz)", y_label="Real Part (Vfb/Vbias)", scale_type='semilogx')
        plot2 = dataplot_mpl.dataplot_mpl( width=6, height=5, title="ComplexZ", x_label="Frequency (Hz)", y_label="Imaginary Part (Vfb/Vbias)", scale_type='semilogx')        
        curve1 = plot1.addLine(name='Real', xdata=[], ydata=[])
        curve2 = plot2.addLine(name='Imaginary', xdata=[], ydata=[])       
        plot1.show()
        plot2.show()
        b = self.app.processEvents()
        
        tesana = tesanalyze.TESAnalyze()
        
        #freq_array = np.zeros(len(frequency_array))
        amp_array = np.zeros(len(frequency_array))
        phase_array = np.zeros(len(frequency_array))
        real_array = np.zeros(len(frequency_array))
        imaginary_array = np.zeros(len(frequency_array))
        
        index = 0
        for frequency in frequency_array:
            print '#################################'
            print "frequency", frequency
            # Set the function generator
            function_generator.SetFrequency(frequency)
            sleep(2)
            
            # Get data
            buffer_size=pci.getBufferSize()/16
            number_of_columns = pci.getNumberOfColumns()
            #print number_of_columns
            number_of_rows = pci.getNumberOfRows()
            #print number_of_rows
            timestep = 20.0e-9 * (lsync+1) * number_mux_rows #This is a problem we need to pass the number of rows
            number_of_buffers = int(math.ceil(3.0/frequency/(timestep*buffer_size)))
            print number_of_buffers
            #array_size = number_of_buffers*buffer_size
            #v_bias_array = np.zeros(array_size, float)
            #v_response_array = np.zeros(array_size, float)
            
            v_bias_array = np.zeros(0)
            v_response_array = np.zeros(0)
            for num_buffer in range(number_of_buffers):
                buffer = pci.getData()
                #mix_param = 0.08789
                v_fb = buffer[:,0,0,1]
                v_fb = v_fb[:-1]
                v_mix = self.getMixData(mix_param = mix_param, buffer = buffer, use_mix = use_mix)
                v_bias_array = np.hstack((v_bias_array, buffer[:,0,1,0]))
                #v_bias_array = v_bias_array[:-1]      
                #temporary switching to open loop mode and read teh error signal
                #v_response_array = np.hstack((v_response_array, buffer[:,0,0,1]))
                v_response_array = np.hstack((v_response_array, v_mix))
#                pylab.figure()
#                pylab.plot(v_bias_array[:6000], '-o', label ='drive')
#                pylab.plot(v_response_array[:6000], '-s', label = 'mix')
#                pylab.plot(v_fb[:1000], '-x', label = 'fb')
#                pylab.legend()
#                pylab.show()
                #v_response_array = np.hstack((v_response_array, buffer[:,0,0,0]))
            # Check for unlock
            vfb_mean = np.mean(v_response_array)
            print 'vfb_mean is ', vfb_mean
            

                
            if vfb_mean > 0.85 or vfb_mean < 0.15:
                print 'Unlock detected'
                function_generator.SetOutput('off')
                sleep(1)
                self.relock(dfb0, channel=0, dac_value=dfb_dac)
                self.relock(dfb0, channel=1, dac_value=dfb_dac)
                sleep(2)
                function_generator.SetOutput('on')
                sleep(2)
                # Retake data
                v_bias_array = np.zeros(0)
                v_response_array = np.zeros(0)
                for num_buffer in range(number_of_buffers):
                    buffer = pci.getData()
                    v_mix = self.getMixData(mix_param = mix_param, buffer = buffer, use_mix = True)
                    v_bias_array = np.hstack((v_bias_array, buffer[:,0,1,0]))
                    #v_bias_array = v_bias_array[:-1]   
                    v_response_array = np.hstack((v_response_array, v_mix))
                # Check if still unlcoked
                vfb_mean = np.mean(v_response_array)
                if vfb_mean > 0.95 or vfb_mean < 0.01:
                    print 'Still Unlocked detected: Could be serious problem!'
            array_size = len(v_bias_array)    
            #print np.size(v_bias_array), np.size(v_response_array)
            time_array = np.arange(array_size)
            time_array = time_array*timestep
            #print np.size(time_array)
                
            #print '%f %f' % (data[0, 0, 0, 0], data[0, 0, 0, 1])             # [frame, row, column, error/feedback = 0/1]
            #error    = data[0,0,0,0]
    
            # Get Vbias and Vresponse arrays
            #v_bias_array     = data[:, 0, 1, 0]
            #v_response_array = data[:, 0, 0, 1]
            
             
            # Calculate the Complex Z for this frequency
            phase_array[index], amp_array[index], real_array[index], imaginary_array[index] = \
                tesana.calcComplexZFromSine(time_array, v_bias_array, v_response_array, frequency)
            
            #print 'now plotting'    
            curve1.update_data(frequency_array[:index], real_array[:index])
            plot1.update()
            curve2.update_data(frequency_array[:index], imaginary_array[:index])
            plot2.update()
            b = self.app.processEvents()
                
            index += 1
            print '###############################'
        # Return the results
        return phase_array, amp_array, real_array, imaginary_array

    def measureNoiseArray(self, dfb_card, frame_rate, noise_params = [[.25,12,.01],[3.0,50,24],[18.0,400,1000]], noise_cutoff = .01, mix_param=0.0, use_mix=False, use_relocker=True, use_pulse_reject=False, use_baseline_reject=False):
        '''Measure noise with different steps and averages and stitch together'''

        #noise_params = [[.25,12,.01],[3.0,50,24],[18.0,400,1000]]
        #frame_rate = 1/(30.0e-9*41)
        
        pylab.figure()
        
        stitched_Vxx = np.zeros(0)
        stitched_freqs = np.zeros(0)
        averaged_psd_array_list, freqs_list = measureAverageNoiseArray(self, dfb_card, frame_rate=frame_rate, noise_params=noise_params, noise_cutoff = noise_cutoff, mix_param=mix_param, use_mix=use_mix, use_relocker=use_relocker, use_pulse_reject=use_pulse_reject, use_baseline_reject=use_baseline_reject)
        for index in range(len(noise_params)):
            Vxx = averaged_psd_array_list[index]
            freqs = freqs_list[index]
            indicies_greater_then = np.nonzero(freqs>noise_param[index][2])
            start_index =  indicies_greater_then[0][0]
            print start_index
            indexes_kept_stitched_array = np.nonzero(stitched_freqs<freqs[start_index])
            stitched_freqs = np.hstack((stitched_freqs[indexes_kept_stitched_array], freqs[start_index:]))
            stitched_Vxx = np.hstack((stitched_Vxx[indexes_kept_stitched_array], Vxx[start_index:]))
            #pylab.loglog(freqs[start_index:],Vxx[start_index:])
            
        #pylab.loglog(stitched_freqs, stitched_Vxx,'-')
        #pylab.show()
        
        return stitched_Vxx, stitched_freqs

    def measureNoise(self, dfb0, frame_rate, noise_param = [[.25,12,.01],[3.0,50,24],[18.0,400,1000]], noise_cutoff = .0007, mix_param=0.0, use_mix=False):
        '''Measure noise with different steps and averages and stitch together
        
        noise_param[x][0]: fstep (lowest frequency bin)
        noise_param[x][1]: number of averages
        noise_param[x][2]: frequencies greater than this value are averaged
        '''

        #noise_param = [[.25,12,.01],[3.0,50,24],[18.0,400,1000]]
        #frame_rate = 1/(30.0e-9*41)
        
        pylab.figure()
        
        stitched_Vxx = np.zeros(0)
        stitched_freqs = np.zeros(0)
        for index in range(len(noise_param)):
            Vxx, freqs = self.measureAveragedNoise(dfb0, frame_rate = frame_rate, fstep = noise_param[index][0], averages = noise_param[index][1], mix_param=mix_param, use_mix=use_mix)
            indicies_greater_then = np.nonzero(freqs>noise_param[index][2])
            #start_index =  int(noise_param[index][2]*len(freqs))
            start_index =  indicies_greater_then[0][0]
            print start_index
            indexes_kept_stitched_array = np.nonzero(stitched_freqs<freqs[start_index])
            stitched_freqs = np.hstack((stitched_freqs[indexes_kept_stitched_array], freqs[start_index:]))
            stitched_Vxx = np.hstack((stitched_Vxx[indexes_kept_stitched_array], Vxx[start_index:]))
            #pylab.loglog(freqs[start_index:],Vxx[start_index:])
            
        #pylab.loglog(stitched_freqs, stitched_Vxx,'-')
        #pylab.show()
        
        return stitched_Vxx, stitched_freqs

    def measureNoiseAndStitch(self, frame_rate, noise_param = [[.25,12,.01],[3.0,50,24],[18.0,400,1000]], noise_cutoff=0.002, mix_param=0.0, \
                              use_mix=False,RemovePoly=False,order=1,verbose=False):
        '''Measure noise with different steps and averages and stitch together'''

        #noise_param = [[.25,12,.01],[3.0,50,24],[18.0,400,1000]]
        #frame_rate = 1/(30.0e-9*41)
        
        pylab.figure()
        
        stitched_Vxx = np.zeros(0)
        stitched_freqs = np.zeros(0)
        for index in range(len(noise_param)):
            Vxx, freqs = self.measureAveragedNoiseOld(frame_rate = frame_rate, fstep = noise_param[index][0],averages = noise_param[index][1],noise_cutoff=noise_cutoff,\
                                                      mix_param=mix_param,use_mix=use_mix,RemovePoly=RemovePoly,order=order,verbose=verbose)
            indicies_greater_then = np.nonzero(freqs>noise_param[index][2])
            #start_index =  int(noise_param[index][2]*len(freqs))
            start_index =  indicies_greater_then[0][0]
            print start_index
            indexes_kept_stitched_array = np.nonzero(stitched_freqs<freqs[start_index])
            stitched_freqs = np.hstack((stitched_freqs[indexes_kept_stitched_array], freqs[start_index:]))
            stitched_Vxx = np.hstack((stitched_Vxx[indexes_kept_stitched_array], Vxx[start_index:]))
            #pylab.loglog(freqs[start_index:],Vxx[start_index:])
            
        #pylab.loglog(stitched_freqs, stitched_Vxx,'-')
        #pylab.show()
        
        return stitched_Vxx, stitched_freqs

    def measureAverageNoiseArray(self, dfb_card, frame_rate, noise_params, noise_cutoff = .0025, mix_param=0.0, use_mix=False, use_relocker=True, use_pulse_reject=False, use_baseline_reject=False):
        ''' Measure PSD and do different ranges while reusing the data. Use with digital feedback card. 
        PCI card should be initialized and DMA should be running!'''

        max_arraysize = 13107200
        buffer_size = pci.getBufferSize()/8
        num_buffers = int(math.floor(max_arraysize/buffer_size))
        arraysize = buffer_size*num_buffers
        #print 'array size %d' % arraysize

        fs = frame_rate
        
        num_ranges = len(noise_params)
        print 'number of ranges: ',num_ranges
        nfft = np.zeros(num_ranges)
        averages_per_maxarray = np.zeros(num_ranges)
        num_psd_to_avg = np.zeros(num_ranges)
        
        psd_array_list = []
        freqs_list = []
        for range_index in range(num_ranges):
            print noise_params
            fstep=noise_params[range_index][0]
            averages=noise_params[range_index][1]
        
            fftpwr = int(round(math.log(fs/fstep,2))) #find closest power of 2 for fft
            nfft[range_index] = 2**fftpwr
        
            averages_per_maxarray[range_index] = math.floor(arraysize / nfft)
            num_psd_to_avg[range_index] = int(math.ceil(averages/averages_per_maxarray))
            psd_array = np.zeros((num_psd_to_avg[range_index],nfft[range_index]/2+1))
            psd_array_list.append(psd_array)
            freqs_list.append(np.zeros(0))

        #print 'fs %f' % fs
        #print 'nfft ', nfft
        pfstep = fs/nfft
        #print 'frequency step ', pfstep
        print 'num of psd to average ', num_psd_to_avg
        pavg = averages_per_maxarray * num_psd_to_avg
        #print '# of averages ', pavg
        
        max_num_psd_to_avg = num_psd_to_avg.max()

        for index in range(max_num_psd_to_avg): 
            loop_counter = 0
            noise_std = 100.0
            best_noise_buffer = np.zeros(0)
            best_noise_std = 1.0
            no_pulse_flag = True
            while loop_counter < 13 and (noise_std > noise_cutoff or pulse_flag is True or buffer_mismatch is True or relock_flag is True):
                noise_buffer = self.getNoiseBuffers(num_buffers = num_buffers, mix_param = mix_param, use_mix = use_mix)
                noise_std = np.std(noise_buffer)
                noise_mean = np.mean(noise_buffer)
                noise_median = np.median(noise_buffer)
                noise_min = np.min(noise_buffer)
                noise_max = np.max(noise_buffer)
                noise_frank = (noise_max-noise_min)/noise_mean
                section_means, section_stds, section_time = self.statsOfSections(noise_buffer, fs, num_sections=500)
                mean_std = np.mean(section_stds)
                median_std = np.median(section_stds)
                max_std = np.max(section_stds)
                min_std = np.min(section_stds)
                median_mean = np.median(section_means) 
                loop_counter += 1
                print '************'
                print 'loop, mean, median, std ' , loop_counter, noise_mean, noise_median, noise_std
                print 'min, max, frank ' , noise_min, noise_max, noise_frank
                print 'section_time', section_time
                print 'mean_std, median_std, max_std, outlier std', mean_std, median_std, max_std, min_std, max_std > 8*median_std
                print 'mean_of_first, mean_of_last, dif, mistmatch_at edges', section_means[0], section_means[-1], section_means[0]-section_means[-1], abs(section_means[0]-section_means[-1]) > 20*median_std
                
                # Perform checks on the acquired data
                #Set relock flag to True if the meadian of section means is outside of specified range
                if (median_mean > .70 or median_mean < .30) and use_relocker is True:
                    print 'Mean out of desired range, relocking...'
                    self.relock(dfb_card, channel=0)
                    self.relock(dfb_card, channel=1)
                    sleep(6)
                    relock_flag = True
                else:
                    relock_flag = False
                # Check for a pulse or disruption in buffer and if so set pulse flag to True
                if (max_std > 8*median_std) and use_pulse_reject is True:
                    print 'Disruption in buffer, throwing out data'
                    pulse_flag = True
                else:
                    pulse_flag = False
                # Check for a mismatch between begining and end of buffer
                if (abs(section_means[0]-section_means[-1]) > 20*median_std) and use_baseline_reject is True:
                    print 'mismatch between ends of buffer, throwing out data'
                    buffer_mismatch = True
                else:
                    buffer_mismatch = False
                # If the data passes the checks copy it to the the "best" arrays otherwise check to see if it atleast better then the current best   
                if relock_flag is False and pulse_flag is False and buffer_mismatch is False:
                    best_noise_buffer = noise_buffer
                    best_noise_std = noise_std
                elif noise_std < best_noise_std:
                    best_noise_buffer = noise_buffer
                    best_noise_std = noise_std
            # Calculate the PSD for the different ranges and store in the correct index of the overall array
            for range_index in range(num_ranges):
                # Only calculate the PSD if it is need for a given range
                if index < num_psd_to_avg[range_index]: 
                    Vxx, freqs = self.calculatePSD(best_noise_buffer, fs, nfft=nfft[range_index], window=pylab.window_hanning)
                    psd_array_list[range_index][index] = Vxx**2
                    freqs_list[range_index] = freqs  #This does not change for the different averages so we can just keep the values the last time through the loop
        # Average the arrays for in range and store as a list of averaged arrays
        averaged_psd_array_list = []
        for range_index in range(num_ranges):
            Vxx = np.sqrt(np.average(psd_array_list[range_index], axis=0))
            averaged_psd_array_list.append(Vxx)
        print 'Shape of averaged psd array and frequency array', averaged_psd_array_list, freqs_list
        
        return averaged_psd_array_list, freqs_list        

    def measureAveragedNoise(self, dfb0, frame_rate, fstep = 18.0, averages = 10, arraysize=13107200, noise_cutoff = .0025, mix_param=0.0, use_mix=False):
        ''' Measure the PSD with digital feedback card. PCI card should be initialized and DMA should be running!'''
    
        #max_arraysize = 13107200
        buffer_size = pci.getBufferSize()/8
        #num_buffers = int(math.floor(max_arraysize/buffer_size))
        num_buffers = int(math.floor(arraysize / buffer_size))
        arraysize = buffer_size*num_buffers
        #print 'array size %d' % arraysize
        
        fs = frame_rate
        fftpwr = int(round(math.log(fs/fstep,2))) #find closest power of 2 for fft
        
        nfft = 2**fftpwr
        averages_per_maxarray = math.floor(arraysize / nfft)
        num_psd_to_avg = int(math.ceil(averages/averages_per_maxarray))
        
        #print 'fs %f' % fs
        #print 'nfft %d' % nfft
        pfstep = fs/nfft
        #print 'frequency step %f' % pfstep
        #print 'num of psd to average %d' % num_psd_to_avg
        
        # Edited KTC 09072017
        print 'num of psd to average %d' % averages
        pavg = averages_per_maxarray * num_psd_to_avg
        #print '# of averages %d' % pavg
        psd_array = np.zeros(nfft/2+1)
        
        for index in range(averages): 
            loop_counter = 0
            noise_std = 100.0
            best_noise_buffer = np.zeros(0)
            best_noise_std = 1.0
            no_pulse_flag = True
            while loop_counter < 13 and (noise_std > noise_cutoff or pulse_flag is True or buffer_mismatch is True or relock_flag is True):
                noise_buffer = self.getNoiseBuffers(num_buffers = num_buffers, mix_param = mix_param, use_mix = use_mix)
                noise_std = np.std(noise_buffer)
                noise_mean = np.mean(noise_buffer)
                noise_median = np.median(noise_buffer)
                noise_min = np.min(noise_buffer)
                noise_max = np.max(noise_buffer)
                noise_frank = (noise_max-noise_min)/noise_mean
                section_means, section_stds, section_time = self.statsOfSections(noise_buffer, fs, num_sections=500)
                mean_std = np.mean(section_stds)
                median_std = np.median(section_stds)
                max_std = np.max(section_stds)
                min_std = np.min(section_stds)
                median_mean = np.median(section_means) 
                loop_counter += 1
                print '************'
                print 'loop, mean, median, std ' , loop_counter, noise_mean, noise_median, noise_std
                print 'min, max, frank ' , noise_min, noise_max, noise_frank
                print 'section_time', section_time
                print 'mean_std, median_std, max_std, outlier std', mean_std, median_std, max_std, min_std, max_std > 8*median_std
                print 'mean_of_first, mean_of_last, dif, mistmatch_at edges', section_means[0], section_means[-1], section_means[0]-section_means[-1], abs(section_means[0]-section_means[-1]) > 20*median_std
                
                if median_mean > .70 or median_mean < .30:
                    print 'Mean out of desired range, relocking...'
                    self.relock(dfb0, channel=0)
                    self.relock(dfb0, channel=1)
                    self.relock(dfb0, channel=2) #temporary
                    self.relock(dfb0, channel=3) #temporary
                    sleep(6)
                    relock_flag = True
                else:
                    relock_flag = False
                if max_std > 8*median_std:
                    print 'Disruption in buffer, throwing out data'
                    pulse_flag = True
                else:
                    pulse_flag = False
                if abs(section_means[0]-section_means[-1]) > 20*median_std:
                    print 'mismatch between ends of buffer, throwing out data'
                    buffer_mismatch = True
                else:
                    buffer_mismatch = False
                    
                if relock_flag is False and pulse_flag is False and buffer_mismatch is False:
                    best_noise_buffer = noise_buffer
                    best_noise_std = noise_std
                elif noise_std < best_noise_std:
                    best_noise_buffer = noise_buffer
                    best_noise_std = noise_std
            Vxx, freqs = self.calculatePSD(best_noise_buffer, fs, nfft=nfft, window=pylab.window_hanning)
            
            # Edited KTC 09072017
            psd_array += Vxx**2
            
        Vxx = np.sqrt(psd_array/float(averages))
        
        return Vxx, freqs
    
    def statsOfSections(self, noise_buffer, fs, num_sections=100):
        
        num_pts = len(noise_buffer)
        pts_per_section = int(num_pts/num_sections)
        section_time = pts_per_section/fs
        
        section_stds = np.zeros(0)
        section_means = np.zeros(0)
        current_index = 0
        
        while current_index <= num_pts-pts_per_section:
            buffer_section = noise_buffer[current_index:current_index+pts_per_section]
            section_std = np.std(buffer_section)
            section_mean = np.mean(buffer_section)
            section_stds = np.hstack((section_stds,section_std))
            section_means = np.hstack((section_means,section_mean))
            current_index += pts_per_section
            
        return section_means, section_stds, section_time

    def measureAveragedNoiseOld(self, frame_rate, fstep = 18.0, averages = 400, arraysize = 13107200, noise_cutoff = .00085, \
                                mix_param=0.0, use_mix=False,RemovePoly=False,order=1,verbose=False):
        ''' Measure the PSD with digital feedback card. PCI card should be initialized and DMA should be running!'''
    
        max_arraysize = 13107200
        buffer_size = pci.getBufferSize()/8
        num_buffers = int(math.floor(max_arraysize/buffer_size))
        arraysize = buffer_size*num_buffers
        #print 'array size %d' % arraysize
        
        fs = frame_rate
        fftpwr = int(round(math.log(fs/fstep,2))) #find closest power of 2 for fft
        
        nfft = 2**fftpwr
        averages_per_maxarray = math.floor(arraysize / nfft)
        num_psd_to_avg = int(math.ceil(averages/averages_per_maxarray))
        
        #print 'fs %f' % fs
        #print 'nfft %d' % nfft
        pfstep = fs/nfft
        #print 'frequency step %f' % pfstep
        print 'num of psd to average %d' % num_psd_to_avg
        pavg = averages_per_maxarray * num_psd_to_avg
        #print '# of averages %d' % pavg
        
        psd_array = np.zeros((num_psd_to_avg,nfft/2+1))
        for index in range(num_psd_to_avg):
            loop_counter = 0
            noise_std = 1.0
            while loop_counter < 10 and noise_std > noise_cutoff:
                noise_buffer = self.getNoiseBuffers(num_buffers = num_buffers, mix_param = mix_param, use_mix = use_mix)
                noise_std = np.std(noise_buffer)
                noise_mean = np.mean(noise_buffer)
                noise_min = np.min(noise_buffer)
                noise_max = np.max(noise_buffer)
                noise_frank = (noise_max-noise_min)/noise_mean
                loop_counter += 1
                print 'mean, std, min, max, frank, loop ' , noise_mean, noise_std, noise_min, noise_max, noise_frank, loop_counter
            Vxx, freqs = self.calculatePSD(noise_buffer, fs, nfft=nfft, window=pylab.window_hanning,RemovePoly=RemovePoly,order=order,verbose=verbose)
            psd_array[index] = Vxx**2
            
        Vxx = np.sqrt(np.average(psd_array, axis=0))
        
        return Vxx, freqs

    def measureNoiseOld(self, frame_rate, arraysize = 13107200, fstep = 18.0, avg = 400):
        ''' Measure the PSD with digital feedback card. '''
    
        max_arraysize = 13107200
        buffer_size = pci.getBufferSize()/8
        num_buffers = int(math.floor(max_arraysize/buffer_size))
        arraysize = buffer_size*num_buffers
        print 'array size %d' % arraysize
        
        fftpwr = int(round(math.log(arraysize/avg,2)))
        
        nfft = 2**fftpwr
        sampling_freq = nfft*fstep
        
        decimate_factor = int(round(frame_rate/sampling_freq))
        print 'decimation factor %d' % decimate_factor
        
        if decimate_factor < 1:
            print 'Using minimums for noise'
            decimate_factor = 1
            
        fs = frame_rate/decimate_factor
        print 'fs %f' % fs
        print 'nfft %d' % nfft
        pfstep = fs/nfft
        print 'frequency step %f' % pfstep
        pavg = arraysize/nfft
        print '# of averages %d' % pavg
        
        decimated_noise_buffer = self.getDecimatedNoiseBuffers(decimate_factor, num_buffers = 100)
        print 'Length of decimated noise buffer %d' % len(decimated_noise_buffer)
        Vxx, freqs = self.calculatePSD(decimated_noise_buffer, fs, nfft=nfft, window=pylab.window_hanning)
        
        return Vxx, freqs

    def getDecimatedNoiseBuffers(self, decimate, num_buffers = 100):
        ''' Decimate and concatenate noise buffers'''
        
        decimated_buffer = np.zeros(0)
        for index in range(decimate):
            noise_buffer = self.getNoiseBuffers(num_buffers)
            decimated_buffer = np.hstack((decimated_buffer, noise_buffer[::decimate]))
            
        return decimated_buffer
    
    def getNoiseBuffers(self, num_buffers, mix_param=0.0, muxed=False, use_mix=False):
        ''' Takes Buffers for Noise Measurement '''
        stacked_buffer = np.zeros(0)
        print 'getting noise buffers ', num_buffers
        for num_buffer in range(num_buffers):
            buffer = pci.getData()
            #print 'calling vmix!'
            if not muxed:
                v_mix = self.getMixData(mix_param= mix_param, buffer = buffer, use_mix = use_mix)
                #stacked_buffer = np.hstack((stacked_buffer, buffer[:,0,0,1]))
                stacked_buffer = np.hstack((stacked_buffer, v_mix))
            else:
                v_mix = np.array([])
                for irow in range(buffer.shape[1]):
                    imix = self.getMixData(mix_param= mix_param, row=irow, buffer = buffer, use_mix = use_mix)
                    if irow == 0:
                        v_mix = imix
                    else:
                        v_mix = np.vstack((v_mix, imix))
                print 'Shape of buffer assuming muxed', v_mix.shape
                
                if num_buffer == 0:
                    stacked_buffer = v_mix
                else:
                    stacked_buffer = np.hstack((stacked_buffer, v_mix))
        print 'Final data shape', stacked_buffer.shape
        return stacked_buffer
    
    def calculatePSD(self, fbdata, fs, nfft = 262144, window=pylab.window_hanning,RemovePoly=False,order=1,verbose=False):
        ''' Calculate PSD from buffer '''
        
        version_string = matplotlib.__version__
        version = version_string.split('.')
        version_number = 1000*float(version[0])+1*float(version[1])+.1*float(version[2][0])
        
        if RemovePoly:
            fbdata=self.removePolynomial(np.arange(len(fbdata)), fbdata, order,verbose)
            
        Pxx, freqs = pylab.mlab.psd(fbdata, NFFT=nfft, Fs=fs, window=window)
    
        if version_number < 98.4:
            Pxx = Pxx/fs*2  # For older versions of mlab 
            print 'Using old psd version'
    
        Vxx = (Pxx)**.5
        
        return Vxx, freqs

    def measureDecayBiasSwitch(self, crate, lsync, record_length=150000, pretrigger=15000, polarity='falling', dfb_dac=8000):
        ''' Measure the decay times for a bias switch'''
        
        tesana = tesanalyze.TESAnalyze()
        
        #Setup plots
        #plot1 = dataplot_mpl.dataplot_mpl(width=6, height=5, title="ComplexZ", x_label="Frequency (Hz)", y_label="Real Part (Vfb/Vbias)", scale_type='semilogx')
        #plot2 = dataplot_mpl.dataplot_mpl( width=6, height=5, title="ComplexZ", x_label="Frequency (Hz)", y_label="Imaginary Part (Vfb/Vbias)", scale_type='semilogx')        
        #curve1 = plot1.addLine(name='Real', xdata=[], ydata=[])
        #curve2 = plot2.addLine(name='Imaginary', xdata=[], ydata=[])       
        #plot1.show()
        #plot2.show()
        #b = self.app.processEvents()
        
        tesana = tesanalyze.TESAnalyze()
            
        # Get data
        buffer_size=pci.getBufferSize()/16
        number_of_columns = pci.getNumberOfColumns()
        #print number_of_columns
        number_of_rows = pci.getNumberOfRows()
        #print number_of_rows
        timestep = 20.0e-9 * (lsync+1) * 2
        #number_of_buffers = int(math.ceil(3.0/frequency/(timestep*buffer_size)))
        number_of_buffers = 50
        print number_of_buffers
        #array_size = number_of_buffers*buffer_size
        #v_bias_array = np.zeros(array_size, float)
        #v_response_array = np.zeros(array_size, float)
        
        v_trig_array = np.zeros(0)
        v_response_array = np.zeros(0)
        for num_buffer in range(number_of_buffers):
            buffer = pci.getData()
            v_response_array = np.hstack((v_response_array, buffer[:,0,0,1]))
            v_trig_array = np.hstack((v_trig_array, buffer[:,0,1,0]))
        vfb_mean = np.mean(v_response_array)
        vtrig_mean = np.mean(v_trig_array)
        print 'vfb_mean is ', vfb_mean
        print 'vtrig_mean is ', vtrig_mean
        
        array_size = len(v_response_array)    
        #print np.size(v_bias_array), np.size(v_response_array)
        time_array = np.arange(array_size)
        time_array = time_array*timestep
        #print np.size(time_array)       

        data_points = np.vstack((time_array,v_response_array))
        trig_points = np.vstack((time_array,v_trig_array))
        
        trigger_indices = tesana.findEdgeTriggerInBuffer(trig_points, record_size = record_length, pretrigger = pretrigger, polarity = polarity, number_of_averages = 2)
        print trigger_indices
        recorded_pulses = tesana.sliceBufferUsingTriggers(data_points[1], trigger_indices, record_length, pretrigger)
        print 'shape of recorded pulses array', recorded_pulses.shape
#        for k in range(len(recorded_pulses)):
#            pylab.plot(recorded_pulses[k])

        average_pulse = np.zeros_like(recorded_pulses[0])
        number_of_averages = len(recorded_pulses)
        for pnt in range(len(average_pulse)):
            average_pulse[pnt] = np.mean(recorded_pulses[:,pnt])
        baseline = average_pulse[:pretrigger/2]
        subtracted_pulse = -1*(average_pulse-np.mean(baseline))

        tdata = np.arange(len(subtracted_pulse))  #number of samples
        record_time = tdata*timestep - pretrigger*timestep

#        pylab.figure()
#        pylab.plot(record_time, subtracted_pulse)
        
        pulse_out = np.vstack((record_time, subtracted_pulse))
                        
        return pulse_out

    def measureAvergaePulse(self, lsync, pulse_freq, number_mux_rows=2,  pretrigger=30000, percent_of_period=0.80, reject_pileup=False, num_pulses=100, polarity='falling'):

        tesana = tesanalyze.TESAnalyze()
        
        test_data = pci.getData() 
        bs = pci.getBufferSize()
        buffer_points = np.size(test_data[:,0,0,1])
        print "buffer size", bs
        print 'buffer points', buffer_points
        
        buffer_length = buffer_points*20e-9*number_mux_rows*(lsync+1)  # converting to time: clockrate*#mux rows*lysnc
        pulses_per_buffer = pulse_freq*buffer_length
        num_buffers = int(round(num_pulses/pulses_per_buffer))
        print 'pulses per buffer', pulses_per_buffer
        print 'number of buffers', num_buffers
        
        time_step = 20e-9*number_mux_rows*(lsync+1)  # converting to time: clockrate*framerate, framerate = #mux rows*lysnc           
        pnt_per_period = 1.0/pulse_freq/time_step
        record_length =  int(percent_of_period*pnt_per_period)
        
        
        average_pulse_array = np.zeros((0,record_length))
        number_of_averages_array = []
                    
        while num_buffers > 50:
                        
            data_points, mon_points = self.getBuffersWithTrigger(num_buffer=50, time_step=time_step)
            
            trigger_indices = tesana.findEdgeTriggerInBuffer(mon_points, record_size = record_length, pretrigger = pretrigger, polarity = 'rising', number_of_averages = 2)
            recorded_pulses = tesana.sliceBufferUsingTriggers(data_points[1], trigger_indices, record_length, pretrigger) 
            # Drift correct individual pulses
            drift_corrected_pulses = tesana.pulsesDriftCorrect(recorded_pulses,pretrigger=pretrigger, polarity=polarity, percent_of_pretrigger=0.80)    
            # Reject pulse pileup
            if reject_pileup is True:
                drift_corrected_pulses = tesana.rejectPulsePileUp(drift_corrected_pulses, cutoff_high = 3.0, cutoff_low = 10.0, pretrigger = pretrigger, polarity = polarity)
                if len(drift_corrected_pulses) == 0:
                    print 'All pulses rejected! Potential problem'
                    continue
                
            average_pulse = np.zeros_like(drift_corrected_pulses[0])
            number_of_averages = len(drift_corrected_pulses)
            for pnt in range(len(average_pulse)):
                average_pulse[pnt] = np.mean(drift_corrected_pulses[:,pnt])
        
            average_pulse_array = np.vstack((average_pulse_array,average_pulse))
            number_of_averages_array.append(number_of_averages)
            
            num_buffers = num_buffers-50
        
        data_points, mon_points = self.getBuffersWithTrigger(num_buffer=50, time_step=time_step)
        
        trigger_indices = tesana.findEdgeTriggerInBuffer(mon_points, record_size = record_length, pretrigger = pretrigger, polarity = 'rising', number_of_averages = 2)
        recorded_pulses = tesana.sliceBufferUsingTriggers(data_points[1], trigger_indices, record_length, pretrigger) 
        # Drift correct individual pulses
        drift_corrected_pulses = tesana.pulsesDriftCorrect(recorded_pulses,pretrigger=pretrigger, polarity=polarity, percent_of_pretrigger=0.80)  

        if reject_pileup is True:
            drift_corrected_pulses = tesana.rejectPulsePileUp(drift_corrected_pulses, cutoff_high = 3.0, cutoff_low = 10.0, pretrigger = pretrigger, polarity = polarity)

        average_pulse = np.zeros_like(drift_corrected_pulses[0])
        number_of_averages = len(drift_corrected_pulses)
        for pnt in range(len(average_pulse)):
            average_pulse[pnt] = np.mean(drift_corrected_pulses[:,pnt])
        
        average_pulse_array = np.vstack((average_pulse_array,average_pulse))
        number_of_averages_array.append(number_of_averages)
        
        #average the averaged pulses
        average_of_averaged_pulses = np.zeros_like(average_pulse_array[0,:])
        for loop_index in range(len(average_of_averaged_pulses)):
            average_of_averaged_pulses[loop_index] = np.average(average_pulse_array[:,loop_index], weights=number_of_averages_array)
        xdata = np.arange(len(average_of_averaged_pulses))  #number of samples
        time_of_averaged_pulse = xdata*time_step
        
        output_pulse = np.vstack((time_of_averaged_pulse, average_of_averaged_pulses))

        return output_pulse

    def measureSinglePulse(self, datafile, lsync, pulse_freq, number_mux_rows=2, num_pulses= 500, pretrigger=30000, percent_of_period=0.80, decimationfactor = 1):

        tesana = tesanalyze.TESAnalyze()
        
        test_data = pci.getData() 
        bs = pci.getBufferSize()
        buffer_points = np.size(test_data[:,0,0,1])
        print "buffer size", bs
        print 'buffer points', buffer_points
        
        buffer_length = buffer_points*20e-9*number_mux_rows*(lsync+1)  # converting to time: clockrate*#mux rows*lysnc
        pulses_per_buffer = pulse_freq*buffer_length
        num_buffers = int(round(num_pulses/pulses_per_buffer))
        print 'pulses per buffer', pulses_per_buffer
        print 'number of buffers', num_buffers
        
        time_step = 20e-9*number_mux_rows*(lsync+1)  # converting to time: clockrate*framerate, framerate = #mux rows*lysnc           
        pnt_per_period = 1.0/pulse_freq/time_step
        record_length =  int(percent_of_period*pnt_per_period)
        print record_length
        print pretrigger
        
        
        average_pulse_array = np.zeros((0,record_length))
        number_of_averages_array = []
                    
        while num_buffers > 50:
                        
            data_points, mon_points = self.getBuffersWithTrigger(num_buffer=50, time_step=time_step)
            
            trigger_indices = tesana.findEdgeTriggerInBuffer(mon_points, record_size = record_length, pretrigger = pretrigger, polarity = 'rising', number_of_averages = 2)
            if len(trigger_indices) < 1:
                print 'Got no triggers bitch'
                break
            recorded_pulses = tesana.sliceBufferUsingTriggers(data_points[1], trigger_indices, record_length, pretrigger) 

            for k in range(len(recorded_pulses)):
                scaled_pulses = recorded_pulses[k]*65535
                #pylab.plot(scaled_pulses)
                #print 'min value is ',np.min(scaled_pulses)
                #print 'max value is ',np.max(scaled_pulses)
                binary_pulse = array.array('H',scaled_pulses[::decimationfactor])
                datafile.write(binary_pulse)            
            num_buffers = num_buffers-50
        
        data_points, mon_points = self.getBuffersWithTrigger(num_buffer=50, time_step=time_step)
        
        trigger_indices = tesana.findEdgeTriggerInBuffer(mon_points, record_size = record_length, pretrigger = pretrigger, polarity = 'rising', number_of_averages = 2)
        if len(trigger_indices) < 1:
            print 'Got no triggers bitch'
            
        recorded_pulses = tesana.sliceBufferUsingTriggers(data_points[1], trigger_indices, record_length, pretrigger) 

        for k in range(len(recorded_pulses)):
            scaled_pulses = recorded_pulses[k]*65535
            #pylab.plot(scaled_pulses)
            #print 'min value is ',np.min(scaled_pulses)
            #print 'max value is ',np.max(scaled_pulses)
            binary_pulse = array.array('H',scaled_pulses[::decimationfactor])
            datafile.write(binary_pulse)

    def measureVPhi(self, trigVal=0.2, numOfStepsInTri=65536, numOfPhi0InTri=59):
        
        v_feedback, v_error = self.getDataBoth()
        
        triMinIndex = np.argmin(v_feedback[:numOfStepsInTri])
        triMaxIndex = np.argmax(v_feedback[triMinIndex:numOfStepsInTri+triMinIndex])+triMinIndex
        trigIndex = np.argmin(abs(v_feedback[triMinIndex:triMaxIndex]-trigVal))+triMinIndex
        triangle = v_feedback[trigIndex-numOfStepsInTri/numOfPhi0InTri/2 : trigIndex+numOfStepsInTri/numOfPhi0InTri/2]
        vphi_trig = v_error[trigIndex-numOfStepsInTri/numOfPhi0InTri/2 : trigIndex+numOfStepsInTri/numOfPhi0InTri/2]
            
        return triangle, vphi_trig
 
    def measureVPhiRows(self, trigVal=0.2, numOfStepsInTri=65536, numOfPhi0InTri=59):
        '''Take multiplexed data for 1 column and return array with a few phi0 around the trigger value of the triangle. '''
    
        v_feedback_rows, v_error_rows = self.getDataAllRows()
        number_of_rows = len(v_feedback_rows)
        #print 'number_of_rows',number_of_rows

        # Try to get a few phi0 of the error around trigVal in triangle
        try:
            v_feedback = v_feedback_rows[0]
            v_error = v_error_rows[0]
            triMinIndex = np.argmin(v_feedback[:numOfStepsInTri])
            triMaxIndex = np.argmax(v_feedback[triMinIndex:numOfStepsInTri+triMinIndex])+triMinIndex
            trigIndex = np.argmin(abs(v_feedback[triMinIndex:triMaxIndex]-trigVal))+triMinIndex
            triangle = v_feedback[trigIndex-numOfStepsInTri/numOfPhi0InTri/2 : trigIndex+numOfStepsInTri/numOfPhi0InTri/2]
            vphi_trig = v_error[trigIndex-numOfStepsInTri/numOfPhi0InTri/2 : trigIndex+numOfStepsInTri/numOfPhi0InTri/2]
            triangle_rows = np.copy(triangle)
            vphi_trig_rows = np.copy(vphi_trig)
            
            for index in range(number_of_rows-1):
                v_feedback = v_feedback_rows[index+1]
                v_error = v_error_rows[index+1]
                triMinIndex = np.argmin(v_feedback[:numOfStepsInTri])
                triMaxIndex = np.argmax(v_feedback[triMinIndex:numOfStepsInTri+triMinIndex])+triMinIndex
                trigIndex = np.argmin(abs(v_feedback[triMinIndex:triMaxIndex]-trigVal))+triMinIndex
                triangle = v_feedback[trigIndex-numOfStepsInTri/numOfPhi0InTri/2 : trigIndex+numOfStepsInTri/numOfPhi0InTri/2]
                vphi_trig = v_error[trigIndex-numOfStepsInTri/numOfPhi0InTri/2 : trigIndex+numOfStepsInTri/numOfPhi0InTri/2]
                triangle_rows = np.vstack((triangle_rows,triangle))
                vphi_trig_rows = np.vstack((vphi_trig_rows,vphi_trig))
        except:
            print 'Error in trigger and cut, returning whole array'
            triangle_rows = v_feedback_rows
            vphi_trig_rows = v_error_rows

        return triangle_rows, vphi_trig_rows


    def measureTcTickle(self, tempcontrol, amplitude_cutoff, timestep, frequency=100.0, start_temperature=0.120, stop_temperature=0.140, temperature_tolerance=.001, temperture_settle_tol=.0003, Tchannel=1, FullOutput=False):
        ''' Determine Tc by tickle method (sine wave into the detector bias line).  Algorithm uses a binary search to hone in on Tc 
        
            input:
            tempcontrol: lakeshore 370 class 
            amplitude_cutoff: <float> above (below) amplitude_cutoff defined as superconducting (normal)
            timestep: <float> the sampling interval used to convert the x-axis of buffer samples to time
            frequency: <float> frequency of tickle in Hz
            start_temperature: <float> [K] must be below Tc
            stop_temperature: <float> [K] must be above Tc
            temperature_tolerance: <float> [K] find Tc to this tolerance
            temperature_settle_tol: <float> [K] Temperature servo must be this good
            Tchannel: <int> Lakeshore channel to read for temperature servo
            FullOutput: return the ssr as the 5th item in the return tuple 
            
            output: final Tsc, final Tn, final Asc, final An, all*
            
            * if FullOutput: True
            all[i]: iteration
            all[i][0]: measured temperature, measured amplitude at measured temperature, ssr
            
            Tsc: superconducting temperature
            Tn: normal temperature
            Asc: amplitude in superconducting state
            An: amplitude of normal state
            ssr: reduced sum of sqaures of residuals (a goodness of fit metric)
            
        ''' 
        temperature_sc = start_temperature
        temperature_nm = stop_temperature
        amplitude_sc = 0.0
        amplitude_nm = 0.0
        
        all=[]
        while (temperature_nm - temperature_sc) > temperature_tolerance:
            print 'temperature sc = ', temperature_sc
            print 'temperature nm = ', temperature_nm
            measure_temperature = round(((temperature_nm - temperature_sc)/2.0 + temperature_sc)*1.0e6)/1.0e6
            print 'measure temperature = ', measure_temperature
            tempcontrol.SetTemperatureSetPoint(measure_temperature)
            sleep(60)
            current_temperature = tempcontrol.GetTemperature(Tchannel)
            while abs(current_temperature-measure_temperature) > temperture_settle_tol:
                sleep(10)
                current_temperature = tempcontrol.GetTemperature(Tchannel)
                
            amplitude, is_superconducting, ssr, offset = self.measureTickle(amplitude_cutoff, timestep, frequency=frequency, init_guess=[3.14159, .33, .42])
            all.append([current_temperature,amplitude,ssr])
            if is_superconducting is True:
                temperature_sc = measure_temperature
                amplitude_sc = amplitude
            else:
                temperature_nm = measure_temperature
                amplitude_nm = amplitude
        
        if FullOutput:        
            return temperature_sc, temperature_nm, amplitude_sc, amplitude_nm, all
        else:
            return temperature_sc, temperature_nm, amplitude_sc, amplitude_nm
        
    def measureTickle(self, amplitude_cutoff, timestep, frequency=100.0, init_guess=[3.14159, .33, .42]):
        ''' measure the amplitude of a tickle response.  Data is fit to a sine wave.  The fit amplitude is 
            returned
            
            input:
            amplitude_cutoff: <float> defines what is superconducting and normal.
                              A measured amplitude above amplitude_cutoff is deemed superconducting
                              A measured amplitude below amplitude_cutoff is deemed normal
            timestep: <float> The sampling interval used to convert samples in the buffer to time
            frequency: <float> frequency of tickle in Hz
            init_guess: <list> [phase,amplitude,offset]
            
            output: fitted amplitude, is_superconducting (True or False), the reduced sum of squares of residuals of fit
            
        '''
        
        v_feedback, v_error = self.getDataBoth()
        time_array = np.arange(len(v_feedback))*timestep
        
        #v_max = max(v_feedback)
        #v_min = min(v_feedback)
        vfb_std = np.std(v_feedback)
        print 'Std of Vfb is ', vfb_std
        
        #sineparam = self.tesana.fitSineCZ(time_array, v_feedback, frequency, init_guess)
        out = self.tesana.fitSineCZ(time_array, v_feedback, frequency, init_guess, fullout=True)
        amplitude = abs(out[0][1])
        offset = out[0][2]
        error = out[2]['fvec'] # data-fit from scipy.optimize.leastsq output
        ssr = np.sum(error**2)**.5/len(error) # reduced sum of squares of residuals. If dividing by the 1sigma noise would be reduced xi squared
        
        #pylab.plot(error)
        #pylab.show()
        #amplitude = abs(sineparam[1])
        
        print 'Amplitude is ', amplitude
        print 'Normalized RMS of residuals is ', ssr
        if amplitude > amplitude_cutoff:
            is_superconducting = True
        else:
            is_superconducting = False
        
        return amplitude, is_superconducting, ssr, offset        

    def measureTickleMultiColumn(self, amplitude_cutoff, timestep, frequency=100.0, init_guess=[3.14159, .33, .42]):
        ''' measure the amplitude of a tickle response.  Data is fit to a sine wave.  The fit amplitude is 
            returned
            
            input:
            amplitude_cutoff: <float> defines what is superconducting and normal.
                              A measured amplitude above amplitude_cutoff is deemed superconducting
                              A measured amplitude below amplitude_cutoff is deemed normal
            timestep: <float> The sampling interval used to convert samples in the buffer to time
            frequency: <float> frequency of tickle in Hz
            init_guess: <list> [phase,amplitude,offset]
            
            output: fitted amplitude, is_superconducting (True or False), the reduced sum of squares of residuals of fit
            
        '''
        
        v_feedback, v_error = self.getDataBothAllColumns()
        time_array = np.arange(len(v_feedback))*timestep
        
        #v_max = max(v_feedback)
        #v_min = min(v_feedback)
        vfb_std = np.std(v_feedback[0])
        print 'Std of Vfb is ', vfb_std
        
        num_of_columns = len(v_feedback)
        amplitude = np.zeros(num_of_columns)
        is_superconducting = np.zeros(num_of_columns, dtype=bool)
        ssr = np.zeros(num_of_columns)
        offset = np.zeros(num_of_columns)
          
        for column_index in range(len(num_of_columns)):
            v_feedback_col = v_feedback[column_index]
            out = self.tesana.fitSineCZ(time_array, v_feedback, frequency, init_guess, fullout=True)
            amplitude[column_index] = abs(out[0][1])
            offset[column_index] = out[0][2]
            error[column_index] = out[2]['fvec'] # data-fit from scipy.optimize.leastsq output
            ssr[column_index] = np.sum(error[column_index]**2)**.5/len(error[column_index]) # reduced sum of squares of residuals. If dividing by the 1sigma noise would be reduced xi squared
            if amplitude[column_index] > amplitude_cutoff[column_index]:
                is_superconducting[column_index] = True
            else:
                is_superconducting[column_index] = False
        
        print 'Amplitude is ', amplitude
        print 'Normalized RMS of residuals is ', ssr
        
        return amplitude, is_superconducting, ssr, offset

    
    # Crate Control Functions

    def relock(self, dfb_card, channel=0, dac_value=8000):
        print 'Relocking now'
        dfb_card.unlock(channel)
        dfb_card.sendRow(channel)
        sleep(.2)
        dfb_card.setDAC(channel, dac=dac_value)
        dfb_card.lock(channel)
        dfb_card.sendRow(channel)
        sleep(.2)

    def relock2ch(self, dfb_card):
        print 'Relocking now'
        dfb_card.unlock(0)
        dfb_card.unlock(1)
        dfb_card.sendRow(0)
        dfb_card.sendRow(1)
        sleep(.4)
        dfb_card.lock(0)
        dfb_card.lock(1)
        dfb_card.sendRow(0)
        dfb_card.sendRow(1)
        sleep(.2)

    def dcSet32(self, row, radacvalue, ocrate, units='dac'):
        ''' clears all RA8 biases and sets high=low=radacvalue for 
            row on ocrate
        '''
     
        self.raClearAll(ocrate)
        
        print 'DC Set Row ', row, radacvalue 
    
        if row >= 0 and row <= 7:
          ocrate.ra8_cards[0].dcSet(row,radacvalue,units)
          #print 'ra8 address ', ocrate.ra8_cards[0].address
        elif row >= 8 and row <= 15:
          ocrate.ra8_cards[1].dcSet(row-8,radacvalue,units)
        elif row >= 16 and row <= 23:
          ocrate.ra8_cards[2].dcSet(row-16,radacvalue,units)
          #print 'ra8 address ', ocrate.ra8_cards[2].address
        elif row >= 24 and row <= 31:
          ocrate.ra8_cards[3].dcSet(row-24,radacvalue,units)
        else:
          print 'Row out of range' 
    
        self.raDisableAll(ocrate)
        
    def raClearAll(self, ocrate):
    
#        for card in range(len(ocrate.ra8_cards)):
#            ocrate.ra8_cards[card].setLowForAll(0)
#            ocrate.ra8_cards[card].setHighForAll(0)
#            ocrate.ra8_cards[card].enableAll()

        for card in ocrate.ra8_cards:
            card.setLowForAll(0)
            card.setHighForAll(0)
            card.enableAll()
            card.sendAllRows()
        sleep(.5)
        
    def raDisableAll(self, ocrate):
    
#        for card in range(len(ocrate.ra8_cards)):
#            ocrate.ra8_cards[card].disableAll()

        for card in ocrate.ra8_cards:
            card.disableAll()
            card.sendAllRows()
        sleep(.5)

    def initRAMAll(self, ocrate):

        for card in ocrate.ra8_cards:
            card.initRAM()
        sleep(.5)

    def setNumRowsAll(self, ocrate, ra_rows=8):

        for card in ocrate.ra8_cards:
            card.setNumRows(ra_rows)
        sleep(.5)

    def sendRAGlobals(self, ocrate, ra_rows=8):

        for card in ocrate.ra8_cards:
            card.setNumRows(ra_rows)
            num_rows = card.getNumRows()
            #print 'Number of Rows ', num_rows, card
            card.sendGlobals()
        sleep(.5)

    def dfBSetSame(self, ocrate, card, number_mux_rows, dfb_adc, dfb_dac, dfb_p, dfb_i):
        
        for row in range(number_mux_rows):
            ocrate.dfb_cards[card].setADC(row=row,adc=dfb_adc)
            ocrate.dfb_cards[card].setDAC(row=row,dac=dfb_dac)
            ocrate.dfb_cards[card].setP(row=row,p=dfb_p)
            ocrate.dfb_cards[card].setI(row=row,i=dfb_i)
            ocrate.dfb_cards[card].lock(row=row)
        ocrate.dfb_cards[card].sendAll()
 
#        for row in range(number_mux_rows):
#            ocrate.setADC(row=row,adc=dfb_adc)
#            ocrate.setDAC(row=row,dac=dfb_dac)
#            ocrate.setP(row=row,p=dfb_p)
#            ocrate.setI(row=row,i=dfb_i)
#            ocrate.lock(row=row)
#        ocrate.sendAll()
        
    # PCI Functions
    
    def pciSetup(self, pci_mask, pci_firmware_delay, dfb_numberofsamples):
        
        print pci_mask, pci_firmware_delay, dfb_numberofsamples
        pci.initializeCard()
        sleep(.2) 
        pci.startDMA(pci_mask, pci_firmware_delay, dfb_numberofsamples)      # (mask, firmware delay, nsamp)
        #sleep(3)
        print '# of Rows', pci.getNumberOfRows()
        print '# of Cols', pci.getNumberOfColumns()
        print '# of buffers', pci.getNumberOfBuffers()
        print '# of frames', pci.getNumberOfFrames()
        print 'buffer size', pci.getBufferSize()
        print 'DMA size', pci.getDMASize()

    
    def pciStop(self):
        
        pci.stopDMA()
        
    def pciCloseCard(self):
        
        pci.closeCard()
        
    def pciSetNumberofFrames(self,numOfFrames=32768):
        
        pci.setNumberOfFrames(numOfFrames)

    def pciGetNumberofFrames(self):
        
        print '# of frames', pci.getNumberOfFrames()
        
    def getBuffersWithTrigger(self, num_buffer, time_step):
        
        
        v_fb0 = np.zeros(0)
        v_err1 = np.zeros(0)
    
        for num_buffer in range(50):
            data = pci.getData()
            mix_data = self.getMixData(0.0, data, use_mix=False)
            v_fb0 = np.hstack((v_fb0, mix_data))
            v_err1 = np.hstack((v_err1, data[:,0,1,0]))
        xdata = np.arange(len(v_fb0))  #number of samples
        xtime = xdata*time_step
        data_points = np.vstack((xtime,v_fb0))
        mon_points = np.vstack((xtime,v_err1))
        
        return data_points, mon_points

    def ra8Setup(self, ra8_cards, number_mux_rows):
        '''initRAM and disableAll, and set # rows for the ra8 cards'''
        for card in ra8_cards:
            card.initRAM()
            #card.disableAll() #This is not being sent
            card.setNumRows(number_mux_rows)
            card.sendGlobals()
    
    def ra8CardSelect(self, ra8_cards, channel):
        '''selects the ra8 card'''
        card_num      = channel/8
        ra8_card      = ra8_cards[card_num]
        local_channel = channel % 8
        return ra8_card, local_channel
    
    def rowToChannelMap(self, ra8_cards, rows, channels, ra8_highs, ra8_widths, ra8_lows=[]):
        '''maps the selected rows to the ra8 channels, and sets the ra8_high,low,and width values'''
        for x in range(len(rows)):
            row       = rows[x]
            channel   = channels[x]
            ra8_high  = ra8_highs[channel]
            ra8_width = ra8_widths[channel]
            if len(ra8_lows) == len(ra8_highs):
                ra8_low = ra8_lows[channel]
            else:
                ra8_low = 0
            #print 'row, channels, ra8_high, ra8_width, ra8_low are ', row, channel, ra8_high, ra8_width, ra8_low
            ra8_card,local_channel = self.ra8CardSelect(ra8_cards, row)
            ra8_card.mapToRow(local_channel,x)
            ra8_card.setHigh(local_channel,ra8_high)
            ra8_card.setLow(local_channel,ra8_low)
            ra8_card.setDuration(local_channel,ra8_width)
            ra8_card.enable(local_channel)
            ra8_card.sendRow(local_channel)
            
    def removePolynomial(self,x,y,order=1,verbose=False):
        ''' fit y(x) to a polynomial of order "order" and remove it from the data.  This is useful 
            prior to taking PSD for example
        '''
        polycoeffs = polyfit(x,y,order)
        yfit = polyval(polycoeffs,x)
        if verbose:
            print 'Determined following polynomial coefficients: '
            print polycoeffs
            pylab.subplot(211)
            pylab.plot(x, y, 'k.')
            pylab.plot(x, yfit, 'r-')
            pylab.xlabel('x')
            pylab.ylabel('y')
            pylab.legend(('data','%d order poly fit'%order))
            pylab.subplot(212)
            pylab.plot(x,y-yfit,'k.')
            pylab.xlabel('x')
            pylab.ylabel('y with %d order poly removed'%order)
            pylab.show()
            
        return y-yfit
    
    def MeasureDetectorLinearity(self,dfb_card,function_generator,timestep, dfb_dac=8000, Voffset=.1,V_start=0.01,V_stop=.5,V_step=.1,\
                                 frequency=45.0, verbose=False,debug=False):
        ''' measure the linearity of the detector using a sine wave from the Agilent 33220A function generator
            This algorithm was developed for sending a signal to the on island heater; however it could be 
            used for signals down the detector bias line.  This algorithm assumes that you have tuned the detector, but 
            does relock the dfb.  The voltage must go from low to high.  Note the actual data is not saved due to the 
            large length of data.  Instead only the fit parameters are saved.
            
            Input:
               dfb_card: an instance of the digital feedback card class
               function_generator: an instance of the function generator Agilent 33200 class
               timestep: frame_interval=clock_rate*lsync_real*number_mux_rows (this is the sampling interval)
               dfb_dac: the DAC voltage offset to lock at
               Voffset: offset voltage of function generator
               V_start: starting peak to peak amplitude
               V_stop: ending peak to peak amplitude
               V_step: voltage step size (peak to peak)
               frequency: frequency of sine wave to apply
               
             Output: <tuple> input voltage, fitted amplitude, fitted offset, sum of squares of residuals (a goodness of fit
                 metric)
        '''
        if V_stop < V_start:
            print 'V_start > V_end.  Algorithm does not work with voltage amplitudes in descending order.  Abort!'
            return False
        
        if V_start < .01:
            print 'amplitude is below minimum value of function generator.  Must be at least 10mV.  Abort!'
            return False
        # Open the function generator
        fg=function_generator
        fg.SetFunction('sine')
        #sleep(1)
        fg.SetFrequency(frequency)
        #sleep(1)
        fg.SetOffset(Voffset)
        #fg.SetAmplitude(0.0)
        # relock here
        sleep(1)
        #self.relock(dfb_card, channel=0, dac_value=dfb_dac)
        #self.relock(dfb_card, channel=1, dac_value=dfb_dac)
        sleep(1)
        
        fg.SetOutput('on')
        #sleep(1)
        
        # loop over amplitude and measure the response
        V_in = []
        fit_amplitude = []
        fit_offset = []
        residuals = []
        
        V=V_start
        while V < V_stop+V_step:
            # set the amplitude
            fg.SetAmplitude(V)
            sleep(1)
            #grab one buffer of data and fit to a sine wave
            if verbose:
                print 'acquiring one buffer of data'
            v_feedback,v_error = self.getDataBoth()
            time_array = np.arange(len(v_feedback))*timestep
            
            v_max = max(v_feedback)
            v_min = min(v_feedback)
            v_mean = np.mean(v_feedback)
            vfb_std = np.std(v_feedback)
            
            # init_guess: <list> [phase,amplitude,offset]
            out = self.tesana.fitSineCZ(time_array, v_feedback, frequency, init_guess=[3.1415,.3,.5], fullout=True)
            # phase = out[0][0]
            amplitude = abs(out[0][1])
            offset = out[0][2]
            error = out[2]['fvec'] # data-fit from scipy.optimize.leastsq output
            ssr = np.sum(error**2)**.5/len(error) # reduced sum of squares of residuals. If dividing by the 1sigma noise would be reduced xi squared
        
            V_in.append(V)
            fit_amplitude.append(amplitude)
            fit_offset.append(offset)
            residuals.append(ssr)
                
            if verbose:
                print 'v_max = ', v_max, '\nv_min = ', v_min, '\nv_mean = ', v_mean, '\nfit_amplitude = ', amplitude,'\nfit_offset = ', offset, '\nssr = ', ssr 
            if debug:
                pass
                #fit=self.tesana.sineFitFunc(p=out[0], t=time_array, f=frequency)
                #pylab.plot(time_array,v_feedback,'bo')
                #pylab.plot(time_array,fit,'b-')
                #pylab.plot(time_array,v_error)
                #pylab.xlabel('time')
                #pylab.ylabel('response')
                #pylab.legend(('v_feedback','v_error'))
                #pylab.show()
                    
            V=V+V_step
        
        fg.SetOutput('off')
        if debug:
            pylab.plot(V_in,fit_amplitude,'bo')
            pylab.plot(V_in,fit_amplitude,'b-')
            pylab.errorbar(V_in, fit_amplitude, yerr=residuals, fmt='bo')
            pylab.xlabel('applied voltage (Vpp)')
            pylab.ylabel('Fitted V_feedback response amplitude (Vp)')
            pylab.grid()
            pylab.show()
            
        return V_in, fit_amplitude, fit_offset, residuals
    
    def MeasureHeaterFrequencyResponse(self, dfb0, lsync, frequency_array, function_generator, mix_param=0.0, use_mix=False, dfb_dac=8000, 
                                       reference_column=1, integration_factor=10.0):
        #Setup plots
        print 'use mix is ', use_mix
        print 'mix param is ', mix_param
        print 'integration_factor is ', integration_factor
        #plot1 = dataplot_mpl.dataplot_mpl(width=6, height=5, title="ComplexZ", x_label="Frequency (Hz)", y_label="Real Part (Vfb/Vbias)", scale_type='semilogx')
        #plot2 = dataplot_mpl.dataplot_mpl( width=6, height=5, title="ComplexZ", x_label="Frequency (Hz)", y_label="Imaginary Part (Vfb/Vbias)", scale_type='semilogx')        
        #curve1 = plot1.addLine(name='Real', xdata=[], ydata=[])
        #curve2 = plot2.addLine(name='Imaginary', xdata=[], ydata=[])       
        #plot1.show()
        #plot2.show()
        #b = self.app.processEvents()
        
        tesana = tesanalyze.TESAnalyze()
        
        # initialize arrays
        N = len(frequency_array)
        I_array = np.zeros(N)
        Q_array = np.zeros(N)
        v_reference_amp_array = np.zeros(N)
        unlock_array = np.zeros(N)
        
        # all of this calculation could be external to the script to save time/calculation
        buffer_size=pci.getBufferSize()/16
        number_of_columns = pci.getNumberOfColumns()
        number_of_rows = pci.getNumberOfRows()
        timestep = 20.0e-9 * (lsync+1) * number_of_rows
        
        # loop over frequencies and measure detector response
        index = 0   
        for frequency in frequency_array:
            #print '#################################'
            print "Frequency {:.2f} Hz".format(frequency),
            # Set the function generator
            function_generator.SetFrequency(frequency)
            sleep(1)
            function_generator.SetOutput('on')
            sleep(1)
            
            # Get data 
            number_of_buffers = int(math.ceil(integration_factor/frequency/(timestep*buffer_size))) # determining the number of buffers to grab
            v_bias_array = np.zeros(0)
            v_response_array = np.zeros(0)
            for num_buffer in range(number_of_buffers):
                buffer = pci.getData()
                #v_fb = buffer[:,0,0,1]
                #v_fb = v_fb[:-1]
                v_mix = self.getMixData(mix_param = mix_param, buffer = buffer, use_mix = use_mix)
                try:
                    v_bias_array = np.hstack((v_bias_array, buffer[:,0,reference_column,0])) # this is the reference signal
                except:
                    print 'Why does it crash here?'
                    print reference_column
                    print buffer.shape
                v_response_array = np.hstack((v_response_array, v_mix))

            # check if unlocked
            unlock=False
            n_iter=0
            n_attempt=3
            vfb_mean = np.mean(v_response_array)
            #print 'vfb_mean is ', vfb_mean
            while vfb_mean > 0.90 or vfb_mean < 0.10:
                print 'Unlock detected'
                function_generator.SetOutput('off')
                sleep(1)
                self.relock(dfb0, channel=0, dac_value=dfb_dac)
                self.relock(dfb0, channel=1, dac_value=dfb_dac)
                sleep(2)
                function_generator.SetOutput('on')
                sleep(2)
                # Retake data
                v_bias_array = np.zeros(0)
                v_response_array = np.zeros(0)
                for num_buffer in range(number_of_buffers):
                    buffer = pci.getData()
                    v_mix = self.getMixData(mix_param = mix_param, buffer = buffer, use_mix = False)
                    v_bias_array = np.hstack((v_bias_array, buffer[:,0,1,0]))
                    v_response_array = np.hstack((v_response_array, v_mix))

                vfb_mean = np.mean(v_response_array)
                if n_iter==n_attempt or n_iter>n_attempt:
                    print 'the feedback was unlocked in all %d attempts.  This data is bogus'%n_attempt
                    unlock=True
                    break
                n_iter=n_iter+1
                    
            array_size = len(v_bias_array)
            time_array = np.arange(array_size)*timestep
            #embed();sys.exit()
            #Software Lock-in code
            if not unlock:
                I, Q, v_reference_amp = tesana.SoftwareLockin(v_signal=v_response_array, v_reference=v_bias_array, \
                                                              reference_type='square',response_type='sine')
                I_array[index]=I
                Q_array[index]=Q
                v_reference_amp_array[index] = v_reference_amp
                print 'I: {} Q: {}'.format(I,Q)
            else:
                I_array[index]=np.nan
                Q_array[index]=np.nan
                v_reference_amp_array[index]=np.nan       
                print 'The response became unlocked.  Returning no data'
                
            unlock_array[index]=unlock
            
            #print 'now plotting'    
            #curve1.update_data(frequency_array[:index], real_array[:index])
            #plot1.update()
            #curve2.update_data(frequency_array[:index], imaginary_array[:index])
            #plot2.update()
            #b = self.app.processEvents()
                
            index += 1
            #print '###############################'
        function_generator.SetOutput('off')
        return I_array, Q_array, v_reference_amp_array, unlock_array
        
        
    def MeasureLockIn(self, dfb0, lsync, frequency, mix_param=0.0, use_mix=False, dfb_dac=8000,n_trys=5,Debug=False):
        ''' Measure the lock in signal using one dfb card recording the reference signal, and one dfb card for 
            the detectors.  Works for multiplexed signals.
        
            Input:
            dfb0: the detector dfb class instance
            lsync: line sync
            frequency: the reference signal frequency.  (Only used to determine the number of samples taken)
            mix_param: if mixing error and feedback for signal
            use_mix: if True, mix the error and feedback 
            dfb_dac: level to relock if needed
            n_trys: number of relock attempts if an unlock is detected in any row
            
            Output:
            a numpy array that is the length of the number of multiplexed rows.  It is in order of the dfb channel order.
            Each row has: [amp, phi, rr, ss, qq, qs, rs, unlocked]; if unlocked==True the row was unlocked and the data is bad. 
        '''    
        
        # Get data
        buffer_size=pci.getBufferSize()/16
        number_of_columns = pci.getNumberOfColumns()
        number_of_rows = pci.getNumberOfRows()
        timestep = 20.0e-9 * (lsync+1) * number_of_rows
        number_of_buffers = int(math.ceil(3.0/frequency/(timestep*buffer_size))) # determining the number of buffers to grab
        
        NeedData=True
        n_iter=1
        while NeedData: # loop to help with unlocks
            # initialize the arrays
            v_bias_array=[]
            v_response_array=[]
            # grab the buffers
            buffers=[]
            iter_num=1
            #pci.getData() # burn the first buffer
            print 'Grabbing buffer number: ',
            for num_buffer in range(number_of_buffers):
                print iter_num,
                buffers.append(pci.getData())
                iter_num=iter_num+1
            print '\n',    
            # stack the buffers per array and check for unlock
            unlocked_rows=[]
            for row in range(number_of_rows):
                v_bias = np.zeros(0)
                v_response = np.zeros(0)
                
                for num_buffer in range(len(buffers)):
                    v_bias=np.hstack((v_bias,buffers[num_buffer][:,row,1,0])) # this is the reference signal and uses all error
                    v_response=np.hstack((v_response,self.getMixData(mix_param=mix_param, buffer=buffers[num_buffer], row=row, column=0, use_mix=use_mix)))
                    iter_num=iter_num+1
                
                # check if unlocked
                v_response_mean=np.mean(v_response)
                if v_response_mean >0.9 or v_response_mean<0.1:
                    print 'row ', row,' is unlocked'
                    unlocked_rows.append(row)
                # plot for debugging if you like
                
                if Debug:
                    pylab.figure(row+1)
                    pylab.plot(v_bias)
                    pylab.plot(v_response)
                    pylab.legend(('reference','response'))
                    pylab.title('Row '+str(row))
                    
                    #pylab.show()
                v_response_array.append(v_response)
                v_bias_array.append(v_bias)
            if Debug:
                pylab.show()
            N_unlocked = len(unlocked_rows)
            if N_unlocked>0:
                print 'Following rows are unlocked: ', unlocked_rows
                if n_iter > n_trys:
                    print 'the feedback was unlocked in all %d attempts.  This data is bogus'%n_trys
                    break
                # try re-syncing these rows here
                for row in unlocked_rows:
                    self.relock(dfb_card=dfb0, channel=row, dac_value=dfb_dac)
                
            else:
                NeedData=False
            n_iter=n_iter+1
                        
        # now that the data is (presumably) good, lock in
        data_out=[]
        for row in range(number_of_rows):
            output = self.tesana.SoftwareLockin(v_signal=v_response_array[row], v_reference=v_bias_array[row])
            output=list(output)
            if row in unlocked_rows:
                output.append(True)
            else:
                output.append(False)
            data_out.append(output)
        data_out=tuple(data_out)
        return data_out
    
    def MeasureLockInNBuffers(self,number_of_buffers=5, mix_param=0.0, use_mix=False):
        # Get data
        buffer_size=pci.getBufferSize()/16
        number_of_columns = pci.getNumberOfColumns()
        number_of_rows = pci.getNumberOfRows()
        #timestep = 20.0e-9 * (lsync+1) * number_of_rows
        
        buffers=[]
        for num_buffer in range(number_of_buffers):
            buffers.append(pci.getData())
        
        # loop over rows then buffers checking for unlock and lock-in
        dataout=[]
        for row in range(number_of_rows):
            row_output=[]
            for num_buffer in range(number_of_buffers):
                v_lockin=[]    
                vb = buffers[num_buffer][:,row,1,0]
                vr = self.getMixData(mix_param=mix_param, buffer=buffers[num_buffer], row=row, column=0, use_mix=use_mix)
                lock = self.tesana.SoftwareLockin(v_signal=vr, v_reference=vb)
                vr_mean = np.mean(vr)
                lock=list(lock)
                if vr_mean>0.9 or vr_mean<0.1:
                    lock.append(1)
                else:
                    lock.append(0)
                row_output.append(np.array(lock))
            dataout.append(row_output)
        return np.array(dataout)
    
    def LockInPlot(self, lsync, number_of_buffers=5, mix_param=0.0, use_mix=False, reference_column=1, reference_type='sine',response_type='sine'):
        ''' Takes N buffers of data.  Plots the reference, signal and lock-in data.  Does not check for unlocking.  Is there
            a gap in time between buffers?
        '''

        # Get data
        buffer_size=pci.getBufferSize()/16
        number_of_columns = pci.getNumberOfColumns()
        number_of_rows = pci.getNumberOfRows()
        timestep = 20.0e-9 * (lsync+1) * number_of_rows
        
        # Initialize the arrays
        v_bias_array=[]
        v_response_array=[]

        # Grab the buffers
        buffers=[]
        for num_buffer in range(number_of_buffers):
            buffers.append(pci.getData())
        
        t=np.arange(len(buffers)*len(buffers[0]))*timestep
        t_for_lock=np.linspace(0,t[-1],number_of_buffers)
        # Stack the buffers per array and check for unlock
        for row in range(number_of_rows):
            v_bias = np.zeros(0)
            v_response = np.zeros(0)
            
            v_lockin=[]    
            for num_buffer in range(len(buffers)):
                vb = buffers[num_buffer][:,row,reference_column,0]
                vr = self.getMixData(mix_param=mix_param, buffer=buffers[num_buffer], row=row, column=0, use_mix=use_mix)
                v_bias=np.hstack((v_bias,vb)) # This is the reference signal and uses all error.
                v_response=np.hstack((v_response,vr))
                I,Q,v_ref_amp = self.tesana.SoftwareLockin(v_signal=vr, v_reference=vb, reference_type=reference_type,response_type=response_type)
                AMP = np.sqrt(I**2+Q**2)/v_ref_amp
                v_lockin.append(AMP)
            m=np.mean(np.array(v_lockin))
            s=np.std(np.array(v_lockin))

            print 'Signal Amp = ', (np.max(v_response) - np.min(v_response))/2.0
            print 'row', row, 'mean and std: ',m,'+/-',s,' std/mean=',s/m

            pylab.figure(row + 1)
            pylab.subplot(211)
            print 'i changed this'
            pylab.plot(t, v_bias,'.-')
            pylab.plot(t, v_response)
            #if v_bias.size > 50000: # Plotting will break for very large buffers if not utilized.
            #    print 'Buffer size too large to plot full dataset.'
            #    print 'Plotting first 50000 points only.'
            #    pylab.plot(t[:50000], v_bias[:50000],'.-')
            #    pylab.plot(t[:50000], v_response[:50000])
            #else:
            #    pylab.plot(t, v_bias,'.-')
            #    pylab.plot(t, v_response)
            pylab.xlabel('Time (sec)')
            pylab.legend(('reference', 'response'))
            pylab.title('Row ' + str(row) + ' reference and response')
            pylab.subplot(212)
            pylab.plot(t_for_lock, v_lockin, 'o-')
            pylab.xlabel('Time(sec)')
            pylab.title('Row ' + str(row) + 'lock in signal')
        try:
            pylab.show()
        except:
            print 'LockInPlot cannot plot the plot, too many numbers!'

    def __IsEven(self,number):
        if number%2==0:
            x=True
        else:
            x=False
        return x
            
    def rowToChannelMapWithCheck(self, ra8_cards, ra8_channels, dfb_channels, ra8_highs, ra8_widths, ra8_lows=[]):
        ''' maps ra8_channels and their biases to the row order.  Also checks for the Randy criteria: gotta switch 
            even odd even odd...
            
            The naming is different than rowToChanneMap because it makes more sense to Hannes this way
        '''
        # check that it switched between odd and even
        zero_index=dfb_channels.index(0)
        parity=self.__IsEven(ra8_channels[zero_index])
        for i in range(1,len(dfb_channels)):
            dex=dfb_channels.index(i)
            temp_parity=self.__IsEven(ra8_channels[dex])
            if parity == temp_parity:
                print 'Randy criteria not met.  Row ordering must always switch parity'
                return False
            parity = temp_parity
        
        for x in range(len(ra8_channels)):
            row       = ra8_channels[x] # which RA8 channel
            channel   = dfb_channels[x] # order in multiplexing
            ra8_high  = ra8_highs[x]
            ra8_width = ra8_widths[x]
            if len(ra8_lows) == len(ra8_highs):
                ra8_low = ra8_lows[x]
            else:
                ra8_low = 0
            #print 'row, channels, ra8_high, ra8_width, ra8_low are ', row, channel, ra8_high, ra8_width, ra8_low
            ra8_card,local_channel = self.ra8CardSelect(ra8_cards, row)
            ra8_card.mapToRow(local_channel,channel)
            ra8_card.setHigh(local_channel,ra8_high)
            ra8_card.setLow(local_channel,ra8_low)
            ra8_card.setDuration(local_channel,ra8_width)
            ra8_card.enable(local_channel)
            ra8_card.sendRow(local_channel)
            
    def getDataNBuffers(self,N):
        ''' grab N buffers of data '''
        x=pci.getData()
        for i in range(N-1):
            x=np.vstack((x,pci.getData()))
        return x
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

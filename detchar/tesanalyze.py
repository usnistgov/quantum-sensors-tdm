import sys
from time import sleep
import math
from scipy import odr
from numpy import zeros, append
import numpy as np
from scipy import linspace, stats, fftpack
#from scipy.io import write_array
import scipy
import scipy.optimize
import scipy.interpolate
from scipy.signal import hilbert
import pylab
import pickle
from IPython import embed

kB = 1.3806503e-23
#bays = [A,B,C,D,E,F,G,H]
#rows = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29.30,31]

class TESAnalyze():
    '''
    Analysis functions
    '''

    def calcComplexZFromSine(self, time_data, input_data, response_data, frequency, init_guess = [3.14159, .33, .42]):

        #y = sineFitFunc(init_guess, time_data, frequency)
        #pylab.plot(time_data,y)
        #print frequency
        #print np.size(time_data)
        #print np.size(input_data)
        insine = self.fitSineCZ(time_data, input_data, frequency, init_guess)
        init_guess2 = [insine[0], frequency]
        amp = insine[1]
        offset = insine[2]
        insine2 = self.fitFreqSineCZ(time_data, input_data, amp, offset, init_guess2)
        frequency = insine2[1]
        input_phase = insine2[0]
        input_amp = insine[1]
        print 'fit frequency is ', frequency
        outsine = self.fitSineCZ(time_data, response_data, frequency, init_guess)
        # Check for phase outside of the range 0 <= phase < 2 pi
        if input_phase < 0 or input_phase >= 2 * np.pi:
            print "Warning: bias phase is wrong", input_phase
        input_phase = self.fixPhase(input_phase,input_amp)
        print "Fixed phase = ", input_phase
        if outsine[0] < 0 or outsine[0] >= 2 * np.pi:
            print "Warning: response phase is wrong", outsine[0]
        outsine[0] = self.fixPhase(outsine[0],outsine[1])
        print "Fixed phase = ", outsine[0]
    
        #phase_difference =  (outsine[0] - input_phase)
        #phase_difference = self.fixPhase(phase_difference, 1)
        #amplitude_ratio =  abs(outsine[1]/input_amp)
        #real_part = abs(abs(outsine[1]/input_amp)*math.cos(outsine[0] - input_phase))
        real_part = (abs(outsine[1]/input_amp)*math.cos(outsine[0] - input_phase))
        #imaginary_part = abs(abs(outsine[1]/input_amp)*math.sin(outsine[0] - input_phase))
        imaginary_part = (abs(outsine[1]/input_amp)*math.sin(outsine[0] - input_phase))
        amplitude_ratio = math.sqrt(real_part**2+imaginary_part**2)
        phase_difference = math.atan2(imaginary_part,real_part)
        print 'end of calcCZ'
        return phase_difference, amplitude_ratio, real_part, imaginary_part
    
    def fixPhase(self, phase, amplitude):
        # Return a phase between 0 and 2 pi
        print 'correcting for phase:  phase, amp ', phase, amplitude
        
        if amplitude < 0:
            phase -= np.pi
        
#        if phase < 0:
#            phase += 2*np.pi
#
#    
#        if phase > 2 * np.pi:
#            phase = phase % (2 * np.pi)
#            
#        print phase
    
        return phase
    
    def fitSineCZ(self, time, voltage, frequency, init_guess,fullout=False):
        ''' Do a fit on the sine '''
        out = scipy.optimize.leastsq(self.sineErrFunc, init_guess, args=(time, voltage, frequency), full_output=1)
        print 'first sine fit for phase, amp and offset ', out[0]
#        pylab.figure()
#        pylab.plot(time, voltage, '-o', label='data')
#        pylab.plot(time, self.sineFitFunc(out[0], time, frequency), label = 'fit')
#        pylab.title('Fit sine wave')
#        pylab.legend()
#        pylab.figure()
#        res = voltage - self.sineFitFunc(out[0], time, frequency)
#        pylab.plot(time, res, '-x', label='fit residuals')
#        pylab.show()
        if fullout:
            return out
        else:
            return out[0]
     
    def sineFitFunc(self, p, t, f):
        
        phase = p[0]
        amp = p[1]
        offset = p[2]
        v = amp*np.sin(2.*math.pi*f*t+phase)+offset
        
        # Constraints
#        if phase < 0 or phase >= 2 * math.pi:
#            v = v + 999999999
#
#        if amp < 0:
#            v = v + 999999999
        
        return v
    
    def sineErrFunc(self, p, t, y, f):
        
        err =  y - self.sineFitFunc(p,t,f) 
        return err

    def fitFreqSineCZ(self, time, voltage, amp, offset, init_guess):
        ''' Do a fit on the sine '''
        out = scipy.optimize.leastsq(self.sineFreqErrFunc, init_guess, args=(time, voltage, amp, offset), full_output=1)
        print 'second sine fit for phase, freq ', out[0]
#        pylab.figure()
#        pylab.plot(time, voltage, '-o', label='data')
#        pylab.plot(time, self.sineFreqFitFunc(out[0], time, amp, offset), label = 'fit')
#        pylab.title('Fit freq sine wave')
#        pylab.legend()
#        pylab.figure()
#        res = voltage - self.sineFreqFitFunc(out[0], time, amp, offset)
#        pylab.plot(time, res, '-x', label='fit freq residuals')
#        pylab.show()
        return out[0]

    def sineFreqFitFunc(self, p, t, amp, offset):
        
        phase = p[0]
        f = p[1]
        v = amp*np.sin(2.*math.pi*f*t+phase)+offset
        
        # Constraints
#        if phase < 0 or phase >= 2 * math.pi:
#            v = v + 999999999
#
#        if amp < 0:
#            v = v + 999999999
        
        return v
    
    def sineFreqErrFunc(self, p, t, y, amp, offset):
        
        err =  y - self.sineFreqFitFunc(p,t,amp,offset) 
        return err


    def AveragePeriodicPulsesFromBuffer(self, pulse, time_step, period, trigger_level, record_length = None, pretrigger= 500):
        ''' Chops up buffer and returns an average pulse '''

       # pylab.plot(pulse[0],pulse[1])
        #pylab.figure()

        if record_length is None:
            pnt_per_period = period/time_step
            record_length =  int(pnt_per_period-0.02*pnt_per_period)
        else:
            record_length = record_length
        print record_length
        
        num_pulses = int(len(pulse[1])/record_length)
        
        total_index = 0
        pulses = np.zeros((num_pulses,record_length))
        for k in range(num_pulses):
            pulsearray, after_pulse_index, triggered = self.TriggerAndRecord(pulse[1,total_index:], trigger_level, polarity = 'falling', recordsize = record_length, pretrigger = pretrigger)
            #print after_pulse_index, len(pulsearray)
            total_index = total_index+after_pulse_index
            pylab.hold(True)
            if triggered is True:
                pulses[k] = pulsearray
                #pylab.plot(pulses[k])
            else:
                pulses[k] = np.ones(record_length)*np.nan
                 
        #pylab.figure()
   
        average_pulse = np.zeros_like(pulses[0,:])
        number_of_averages = float(np.sum(np.isfinite(pulses[:,k])))  #ignoring NaN
        for k in range(len(average_pulse)):
            average_pulse[k] = np.nansum(pulses[:,k])/float(np.sum(np.isfinite(pulses[:,k]))) #average ignoring NaN
        pulse_time = np.arange(len(average_pulse))*time_step
        
        #pylab.plot(pulse_time,average_pulse)
        #pylab.show()
        pulse_out = np.vstack((pulse_time, average_pulse))
        
        return pulse_out, number_of_averages

    def sliceBufferUsingTriggers(self, fb, trigger_indices, record_size, pretrigger):
        ''' Slices a buffer into records based on trigger indices '''
        
           
        #pylab.figure()

        recorded_pulses = np.zeros((0,record_size))
        for loop_index in range(len(trigger_indices)):
            trigger_index = trigger_indices[loop_index]
            if trigger_index-pretrigger < 0:
                start_index = 0
                print 'start before beginging of buffer'
                continue
            else:    
                start_index = trigger_index-pretrigger
            if trigger_index+record_size-pretrigger > len(fb):
                stop_index = -1
                print 'stop after end of buffer'
                break
            else:
                stop_index = trigger_index+record_size-pretrigger
            recorded_pulses = np.vstack((recorded_pulses,fb[start_index:stop_index]))
            #pylab.plot(fb[start_index:stop_index])
        
        return recorded_pulses
    
    def rejectPulsePileUp(self, pulses, cutoff_high, cutoff_low, pretrigger, polarity = 'falling',plot_rejected=False):
        ''' Scan recorded pulses and drop records with more than one pulse '''
        
        
        
        if polarity == 'falling':
            polarity_correction = -1.0
        else:
            polarity_correction = 1.0
        
        # Try to identify pulses with different mean values
        pulse_areas = np.zeros(0)
        for pulse in pulses:
            #baseline = pulse[:pretrigger/2]
            #subtracted_pulse = polarity_correction*(pulse-np.mean(baseline))
            pulse_area = np.sum(pulse)
            pulse_areas = np.hstack((pulse_areas,pulse_area))
        
        mean_area = np.mean(pulse_areas)
            
        clean_pulses = np.zeros((0,len(pulses[0,:])))
        rejected_pulse_count = 0
        for pulse in pulses:
            #baseline = pulse[:pretrigger/2]
            #subtracted_pulse = polarity_correction*(pulse-np.mean(baseline))
            pulse_area = np.sum(pulse)
            if pulse_area < (cutoff_high*mean_area) and pulse_area >  (1.0/cutoff_low*mean_area):
                #clean_pulses = np.vstack((clean_pulses, pulse))
                clean_pulses = np.vstack((clean_pulses, pulse))
            else:
                print 'Rejecting a pulse: pulse area %f  mean_area %f' %(pulse_area,mean_area)
                if plot_rejected is True:
                    if rejected_pulse_count == 0:
                        #pylab.figure()
                        #pylab.plot(pulse)
                        print 'No pulses'
                rejected_pulse_count += 1

        #if  rejected_pulse_count > 0:
            #if plot_rejected is True:
                #pylab.show()
        
        return clean_pulses    

    def findEdgeTriggerInBuffer(self, buffer, record_size = 20000, pretrigger = 500, polarity = 'falling', trigger_level = None, number_of_averages = 100):
        ''' Looks at a buffer and finds the edges to trigger on. '''
        
        
        t = buffer[0]
        fb = buffer[1]
        
        #pylab.plot(t,fb)
        
        # smooth the data with a running average
        fb_smooth = np.zeros_like(fb)
        fb_cumsum = fb.cumsum()
        
        fb_smooth[number_of_averages:] = (fb_cumsum[number_of_averages:] - fb_cumsum[:-number_of_averages]) / number_of_averages
        
        #pylab.figure()
        #pylab.plot(t[number_of_averages:-number_of_averages],fb_smooth[number_of_averages:-number_of_averages])
        
        slope_array = np.zeros_like(fb_smooth)
        for index in range(len(fb_smooth)-1):
            slope_array[index] = (fb_smooth[index+1]-fb_smooth[index])/(t[index+1]-t[index])
            
        #pylab.figure()
        #pylab.plot(t[number_of_averages:-number_of_averages],slope_array[number_of_averages:-number_of_averages])    
        
        if trigger_level is None:
            if polarity == 'falling':
                trigger_level = slope_array[number_of_averages:-number_of_averages].min()/2.0
            else:
                trigger_level = slope_array[number_of_averages:-number_of_averages].max()/2.0
                #print slope_array.max()
        print trigger_level
        
        found_trigger = True
        start_index = 0
        trigger_indices = []
        while found_trigger is True:
            trigger_index, triggered = self.findIndexOfTigger(slope_array[start_index:], trigger_level, polarity = polarity)
            overall_trigger_index = start_index + trigger_index
            start_index = overall_trigger_index + record_size - pretrigger
            found_trigger = triggered
            if overall_trigger_index < len(slope_array):
                trigger_indices.append(overall_trigger_index)
        
        #print trigger_indices
        #print t[trigger_indices]
        #sleep(5)           
        #pylab.show()
        
        return trigger_indices
    
    def findIndexOfTigger(self, buffer, trigger_level, polarity = 'falling'):
        
        triggered = False
        if polarity == 'rising':
            for pnt in range(len(buffer)):
                if buffer[pnt] > trigger_level:
                    triggered = True
                    trigger_index = pnt
                    break
        else:
            for pnt in range(len(buffer)):
                if buffer[pnt] < trigger_level:
                    triggered = True
                    trigger_index = pnt
                    break
 
        if triggered is False:
            trigger_index = len(buffer)-1
        

        return trigger_index, triggered

    def findTriggerInBuffer(self, buffer, record_size, pretrigger, polarity = 'falling'):
        ''' Triggers on a pulse based on polarity and record size.'''
        
        if polarity == 'rising':
            pulse_extrema = np.max(buffer)
        else:
            pulse_extrema = np.min(buffer)
            
        buffer_average = np.average(buffer)
        
        if polarity == 'rising':
            initial_trigger = buffer_average+(pulse_extrema - buffer_average)/2.0
        else:
            initial_trigger = buffer_average-(buffer_average - pulse_extrema)/2.0  
        print initial_trigger
        
        #tes.TriggerAndRecord(current_buffer, initial_trigge, polarity = 'falling', recordsize = record_size, pretrigger = pretrigger)

        triggerlevel = initial_trigger
        triggered = False
        if polarity == 'rising':
            for pnt in range(len(buffer)):
                if buffer[pnt] > triggerlevel:
                    triggered = True
                    triggerindex = pnt
                    break
        else:
            for pnt in range(len(buffer)):
                if buffer[pnt] < triggerlevel:
                    triggered = True
                    triggerindex = pnt
                    break
                
        baesline_start = triggerindex - pretrigger
        baseline_stop = int(triggerindex - pretrigger/2)
        baseline = buffer[baesline_start:baseline_stop]
        baseline_avg = np.average(baseline)
        baseline_std = np.std(baseline)

        if polarity == 'rising':
            final_trigger = baseline_avg+baseline_std*20.0
        else:
            final_trigger = baseline_avg-baseline_std*20.0  

    def TriggerAndRecord(self, current_buffer, triggerlevel, polarity = 'falling', recordsize = 10000, pretrigger = 500):
        ''' Finds pulses in buffere using a level trigger and returs pulse and stop postion  '''

        data = current_buffer
        triggered = False

        if polarity == 'falling':
            for pnt in range(len(data)):
                if data[pnt] < triggerlevel:
                    triggered = True
                    triggerindex = pnt
                    break
        else:
            for pnt in range(len(data)):
                if data[pnt] > triggerlevel:
                    triggered = True
                    triggerindex = pnt
                    break

        if triggered is True:
            start_pulse_index = triggerindex - pretrigger
            stop_pulse_index = triggerindex + recordsize - pretrigger
            if stop_pulse_index > len(data):
                print 'Pulse record extends past end of array'
                triggered = False
                pulsearray = []
                after_pulse_index = -1
            elif start_pulse_index < 0:
                print 'Pretrigger record extends past end of begining of array'
                triggered = False
                pulsearray = []
                after_pulse_index = stop_pulse_index    
            else:
                pulsearray = data[start_pulse_index : stop_pulse_index]
                after_pulse_index = stop_pulse_index + 1
        else:
            pulsearray = []
            after_pulse_index = []
       
        return pulsearray, after_pulse_index, triggered

    def pulsesDriftCorrect(self, pulses, pretrigger, polarity='falling', percent_of_pretrigger=0.80):
        '''Drift correct individual pulses.'''

        if polarity == 'falling':
            polarity_correction = -1.0
        else:
            polarity_correction = 1.0
        
        drift_corrected_pulses = np.zeros((len(pulses), len(pulses[0]) ))
        for k in range(len(pulses)):
            pulse_indicies = np.arange(len(pulses[k]))
            baseline = np.polyfit(pulse_indicies[:pretrigger*percent_of_pretrigger], pulses[k][:pretrigger*percent_of_pretrigger],1)
            drift_corrected_pulse = polarity_correction*(pulses[k] - (pulse_indicies*baseline[0] + baseline[1]))
            drift_corrected_pulses[k] = drift_corrected_pulse
        
        return drift_corrected_pulses

    def percentRnInterpolated(self,v_array,r_array,rn,rn_percent):
        ''' Return interpolated value of vsweep coresponding to a requested %Rn given resistance.'''
        
        r_desired = rn_percent / 100.0 * rn
        r_previous = 0.0
        found = False
        loop_index = 0
        
        for r in r_array:
            if r_desired < r_previous and r_desired >= r and found is False:
                found = True
                if r == r_desired:
                    #We have an exact match, so get the voltage directl
                    v = v_array[loop_index]
                else:
                    #We need to interpolate
                    v1 = v_array[loop_index-1]
                    v2 = v_array[loop_index]
                    v = v1 + ((r_desired-r_previous)*(v2-v1)/(r-r_previous))
            loop_index=loop_index+1
            r_previous = r
            
        return v

    def PercentRn(self, volt, fbvolt, rn1 = 0.20):
	'''Calculate specified percent of Rn'''
	fbnormal = np.zeros_like(fbvolt) 
	fbsuper = np.zeros_like(fbvolt)
	#rn1_lookup = np.zeros(numberofrows, float)

	cusp_index = np.argmax(fbvolt) #locate the cusp index by the maximum
	inflection_index = np.argmin(fbvolt[:cusp_index]) #inflection is minimum to right of maximum
	linear_region_index = int(math.floor(inflection_index/2)) #linear region - twice the the inflection point
	x = volt[:linear_region_index] #bias array in linear region
	y = fbvolt[:linear_region_index] #fb array in linear region
	slope,intercept,r,tt,stderr = stats.linregress(x,y) #linear fit
	fbnormal = fbvolt-intercept #fb array shifted so normal region extroplates to zero
	fbsuper = fbvolt - fbvolt[-1] #fb array where superconucting array us shifted to be zero at 0V
	resistance_array = volt/fbnormal
        #pylab.plot(volt,resistance_array)
        #pylab.show()
	rn1_index = np.argmin(abs(resistance_array[inflection_index:cusp_index]-rn1/slope))+inflection_index #closest array index to 20% Rn 
	rn1_lookup = volt[rn1_index] #Closest voltage to 20% Rn

	#print 'Calculated 20%Rn voltages'
	print rn1_lookup

	return rn1_lookup

    def PercentRnMuxed(self, volt, fbvolt, rn1 = 0.20, rn2 = 0.10, numberofrows = 2):
	'''Calculate specified percent of Rn'''
	fbnormal = np.zeros_like(fbvolt) 
	fbsuper = np.zeros_like(fbvolt)
	rn1_lookup = np.zeros(numberofrows, float)
	rn2_lookup = np.zeros(numberofrows, float)

	for row in range(len(fbvolt)):
	    cusp_index = np.argmax(fbvolt[row]) #locate the cusp index by the maximum
	    inflection_index = np.argmin(fbvolt[row,:cusp_index]) #inflection is minimum to right of maximum
	    linear_region_index = int(math.floor(inflection_index/2)) #linear region - twice the the inflection point
	    x = volt[:linear_region_index] #bias array in linear region
	    y = fbvolt[row,:linear_region_index] #fb array in linear region
	    slope,intercept,r,tt,stderr = stats.linregress(x,y) #linear fit
	    fbnormal[row] = fbvolt[row,:]-intercept #fb array shifted so normal region extroplates to zero
	    fbsuper[row] = fbvolt[row,:] - fbvolt[row,-1] #fb array where superconucting array us shifted to be zero at 0V
	    resistance_array = volt/fbnormal[row]
            pylab.plot(volt,resistance_array)
            pylab.show()
	    rn1_index = np.argmin(abs(resistance_array[inflection_index:cusp_index]-rn1/slope))+inflection_index #closest array index to 20% Rn 
	    rn1_lookup[row] = volt[rn1_index] #Closest voltage to 20% Rn
	    rn2_index = np.argmin(abs(resistance_array[inflection_index:cusp_index]-rn2/slope))+inflection_index
	    rn2_lookup[row] = volt[rn2_index]

	print 'Calculated 20%Rn voltages'
	print rn1_lookup
	print 'Calculated 10%Rn voltages'
	print rn2_lookup

	return rn1_lookup, rn2_lookup

    def FindVPhiInBuffer(self, trifb, tri_error, rows, fbrows, trigVal, numOfPhi0InTri, tstep):
        '''Given a buffer with triangle and VPhi, returns the vphi around a trigger point'''

        numOfStepsInTri = 2**14*2*fbrows
        tpts = np.arange(0.0,tstep*vphiPts,tstep)

        for j in range(len(rows)):
            row = rows[j]
            triMinIndex = np.argmin(trifb[row,:numOfStepsInTri])
            triMaxIndex = np.argmax(trifb[row,triMinIndex:numOfStepsInTri+triMinIndex])+triMinIndex
            trigIndex = np.argmin(abs(trifb[row,triMinIndex:triMaxIndex]-trigVal))+triMinIndex
            vphi_trig = tri_error[row,trigIndex-numOfStepsInTri/numOfPhi0InTri/2 : trigIndex+numOfStepsInTri/numOfPhi0InTri/2]
            tpts_trig = tpts[trigIndex-numOfStepsInTri/numOfPhi0InTri/2 : trigIndex+numOfStepsInTri/numOfPhi0InTri/2]
            pylab.subplot(8,4,j+1)
            pylab.plot(tpts_trig*100,vphi_trig)
            if j > 3:
	       pylab.xlabel('time (ms)')
            if j == 0 or row == 4:
	       pylab.ylabel('Error Signal (V)')
            pylab.title('Row ' + str(row))
        pylab.show()
        
    def ivConvert(self, vbias, vfb, intercept, rfb, rbias, rsh, rp, mr = 8.66, vfb_max = 0.965, mux_type=None):
        ''' Convert vbias and vfb into ites and vtes '''

        if mux_type == 'star':
            # For star we will use mr variable to hold the conversion in A/V
            ites = (vfb - intercept)*mr
        else:     
            ites = (vfb - intercept)*vfb_max/((rfb)*mr)
        vtes = vbias*rsh/rbias-ites*(rsh+rp)
        rtes = vtes/ites
        ptes = vtes*ites
        
        return vbias, vtes, ites, rtes, ptes

    def itesConvertArray(self, vfb, intercept, rfb, mr = 8.66, vfb_max = 0.965):
        ''' Convert vfb into.'''
        
        ites = (vfb - intercept)*vfb_max/((rfb)*mr)
        
        return ites

    def ivConvertSplit(self, vbias, vfb, intercept, rfb, rbias, rsh, rp, mr = 8.66, vfb_max = 0.965, mux_type=None):
        ''' Convert vbias and vfb into ites and vtes '''
        
        fb_diff = np.diff(vfb)
        nan_indicies = np.where(np.isnan(vfb))
        if len(nan_indicies[0]) > 0:
            # transition data is at values larger then the NaNs
            end_sc = int(min(nan_indicies[0])-1)
        else:       
            neg_slope_array = np.where(fb_diff<0)
            #print 'SC stop'
            #print neg_slope_array
            end_sc =  min(neg_slope_array[0]) - 1
        #print end_sc
        
        yrsh = np.array([7.9e-3,8.08e-3,8.114e-3,8.118e-3,8.124e-3,8.14e-3,8.23e-3,8.414e-3])/32.0
        xib = np.array([1.0e-5,3.0e-5,1.0e-4,3.0e-4,1.0e-3,3.0e-3,1.0e-2,3.0e-2])
        ibz = 7.9e-3/32.0
        frsh = scipy.interpolate.interp1d(xib, yrsh, bounds_error=False, fill_value=ibz)

        
        vbias_sc = vbias[:end_sc]
        vfb_sc = vfb[:end_sc]
        vbias_nm = vbias[end_sc+2:]
        vfb_nm = vfb[end_sc+2:]
        
        if mux_type == 'star':
            # For star we will use mr variable to hold the conversion in A/V
            ites_sc = (vfb_sc - intercept[0])*mr
            ites_nm = (vfb_nm - intercept[1])*mr
        else:
            ites_sc = (vfb_sc - intercept[0])*vfb_max/((rfb)*mr)
            ites_nm = (vfb_nm - intercept[1])*vfb_max/((rfb)*mr)
        
        irsh_sc = vbias_sc/rbias - ites_sc
        irsh_nm = vbias_nm/rbias - ites_nm

        vtes_sc = vbias_sc*rsh/rbias-ites_sc*(rsh+rp)
        vtes_nm = vbias_nm*rsh/rbias-ites_nm*(rsh+rp)        
        vtes_sc_vrsh = vbias_sc*frsh(irsh_sc)/rbias-ites_sc*(rsh+rp)
        vtes_nm_vrsh = vbias_nm*frsh(irsh_nm)/rbias-ites_nm*(rsh+rp)
        
        vbias = np.hstack((vbias_sc,vbias_nm))
        ites = np.hstack((ites_sc,ites_nm))
        vtes = np.hstack((vtes_sc,vtes_nm))
        vtes_vrsh = np.hstack((vtes_sc_vrsh,vtes_nm_vrsh))
        irsh = vbias/rbias - ites
        
        rtes = vtes/ites
        ptes = vtes*ites
        
        return vbias, vtes, ites, rtes, ptes, vtes_vrsh, irsh
    
    def ivFitNormal(self, vbias, vfb, vbias_normal):
        ''' Fit the normal branch of the IV and return the slope and intercept.'''
       
        nm_indicies = np.where(vbias>vbias_normal)
        nm_start = min(nm_indicies[0])
        vbias_nm = vbias[nm_start:]
        vfb_nm = vfb[nm_start:]
        nm_m, nm_b = scipy.polyfit(vbias_nm, vfb_nm, 1)
        vfb_nm_fit = scipy.polyval([nm_m,nm_b],vbias_nm)
        nm_residuals = vfb_nm-vfb_nm_fit
        #print nm_m, nm_b
        #pylab.figure()
        #pylab.plot(vbias_nm,vfb_nm, 'o')
        #pylab.plot(vbias_nm,vfb_nm_fit)
        #pylab.show()
        #pylab.figure()
        #pylab.plot(vbias_nm,nm_residuals)
        
        return nm_m, nm_b
    
    def ivFitSuperconduct(self, vbias, vfb):
        ''' Fit the superconducting branch of the IV and return the slope and intercept. '''

        fb_diff = np.diff(vfb)
        nan_indicies = np.where(np.isnan(vfb))
        if len(nan_indicies[0]) > 0:
            # transition data is at values larger then the NaNs
            end_sc = int(min(nan_indicies[0])-1)
        else:       
            neg_slope_array = np.where(fb_diff<0)
            #print 'SC stop'
            #print neg_slope_array
            end_sc =  min(neg_slope_array[0]) - 2
        #print end_sc
        vbias_sc = vbias[:end_sc]
        vfb_sc = vfb[:end_sc]
        sc_m, sc_b = scipy.polyfit(vbias_sc, vfb_sc, 1)
        vfb_sc_fit = scipy.polyval([sc_m,sc_b],vbias_sc)
        residuals = vfb_sc-vfb_sc_fit
        #print sc_m, sc_b
        #pylab.figure()
        #pylab.plot(vbias_sc,vfb_sc, 'o')
        #pylab.plot(vbias_sc,vfb_sc_fit)
        #pylab.figure()
        #pylab.plot(vbias_sc,residuals)
        
        return sc_m, sc_b
                
    def ivAnalyze(self, ivdata, rfb, rbias, rjnoise, mr = 8.3, mrsh = 0, vfb_max = 0.965, intercept='superconduct',
                  vbias_normal=3.3, sc_slope=None, nm_slope=None, intercept_value=None, mux_type = None):
        '''Analyze raw data to find TES current and voltage '''
        
        #if data is decreasing in vbias, flip it around
        if (ivdata[0,1] - ivdata[0,0]) < 0:
            vbias = ivdata[0,::-1]
            vfb = ivdata[1,::-1]
            print 'Flipping ivdata'
        else:
            vbias = ivdata[0]
            vfb = ivdata[1]
        #pylab.plot(vbias,vfb)
        #pylab.show()
        
        #Fit the normal branch
        #If nm_slope is given by user, user wants to force it to be used.
        if nm_slope is not None:
            nm_m = nm_slope
            if intercept == 'normal' and intercept_value is None:
                trash_m, nm_b = self.ivFitNormal(vbias, vfb, vbias_normal=vbias_normal)
                print 'Get intercept but not slope', nm_b
            else:
                nm_b = None
        else:
            nm_m, nm_b = self.ivFitNormal(vbias, vfb, vbias_normal=vbias_normal)

        #Fit Super Conducting branch
        if intercept == 'only normal':
            # Assume the parasitic is zero and the rsh value is equal to the combination rjnoise
            sc_m = 0
            sc_b = 0
            rp = 0
            rsh = rjnoise
            rn = (rfb*mr/(rbias*nm_m*vfb_max)-1)*rsh
        else:  
            #If sc_slope is given by user, user wants to force it to be used.
            if sc_slope is not None:
                sc_m = sc_slope
                sc_b = 0
                print 'Not measuring sc slope ', sc_m
            else:
                sc_m, sc_b = self.ivFitSuperconduct(vbias, vfb)
            # Using sc_m and nm_m calculate the resistances
            #Using mrsh  
            #rn = rjnoise*rbias*(sc_m-nm_m)/(rbias*nm_m-mrsh*rfb)
            #rsh = rjnoise*(rbias*vfb_max*sc_m-mrsh*rfb)/(rfb*(mr-mrsh))
            #rp = rjnoise*(mr*rfb-rbias*vfb_max*sc_m)/(rfb*(mr-mrsh))
            #Setting mrsh=0
            if mux_type == 'star':
                # For star we will use mr variable to hold the conversion in A/V
                rn = rjnoise*(sc_m-nm_m)/nm_m
                rsh = rjnoise*rbias*sc_m*mr
                rp = rjnoise-rsh
            else:
                rn = rjnoise*(sc_m-nm_m)/nm_m
                rsh = rjnoise*(rbias*vfb_max*sc_m)/(rfb*mr)
                rp = rjnoise*(mr*rfb-rbias*vfb_max*sc_m)/(rfb*mr)
            
                 
        #Group slopes and intercepts
        sc = [sc_m, sc_b]
        nm = [nm_m, nm_b]
            
        #Tempoary overide of R vals
        #rp = 0
        #rsh = 7.9e-3/32.0
        
        rs = [rn, rsh, rp]
        
        #print rsh, rp, rn, nm_m, nm_b
        
        if intercept_value is not None:
            b = intercept_value
        else:
            if intercept == 'normal' or intercept == 'only normal':
                b = nm_b
            elif intercept == 'both':
                b = [sc_b,nm_b]
            else:
                b = sc_b    
        
        #Calculate iv
        if intercept == 'both':
            vbias_copy, vtes, ites, rtes, ptes, vtes_vrsh, irsh = self.ivConvertSplit(vbias, vfb, b, rfb, rbias, rsh, rp, mr, vfb_max, mux_type=mux_type)
            #pylab.plot(vtes*1e6,ites*1e3)
        else:
            vbias_copy, vtes, ites, rtes, ptes = self.ivConvert(vbias, vfb, b, rfb, rbias, rsh, rp, mr, vfb_max, mux_type=mux_type)
            #pylab.plot(vtes*1e6,ites*1e3)
        #pylab.plot(vtes,rtes/rn)
        # Check that normal slope gives same value as rn calc
        #rnm_indicies = np.where(vtes>1.6e-6)
        #rnm_start = min(rnm_indicies[0])
        #print rnm_start
        #vtes_rnm = vtes[rnm_start:]
        #ites_rnm = ites[rnm_start:]
        #rnm_m, rnm_b = scipy.polyfit(vtes_rnm, ites_rnm, 1)
        #nm_m_array[index] = nm_m
        #nm_b_array[index] = nm_b
        #ites_rnm_fit = scipy.polyval([rnm_m,rnm_b],vtes_rnm)
        #nm_residuals = ites_rnm-ites_rnm_fit
        #print 1./rnm_m, rnm_b
        #pylab.plot(vtes_rnm,ites_rnm_fit, '.')
        
        return vtes, ites, rtes, ptes, rs, sc, nm

    def ivAnalyzeLaserLoop(self, ivs_dict, tespkl, bay, row, rfb, rbias, rjnoise, mr, mrsh, vfb_max, pRn, vbias_normal=2.4):
        
        ivkeys = ivs_dict.keys()
        ivkeys.sort()
        
        atten_voltages = np.zeros(len(ivkeys))
        laser_voltages = np.zeros_like(atten_voltages)
        v_pnt_array = np.zeros((len(pRn),len(atten_voltages))) 
        i_pnt_array = np.zeros_like(v_pnt_array) 
        powerAtRns = np.zeros_like(v_pnt_array)     
        rns = np.zeros_like(atten_voltages) 
        
        pylab.figure()
        for index in range(len(ivkeys)):
            key = ivkeys[index]
            ivdata = ivs_dict[key]['data']
            temperature = ivs_dict[key]['temperature']
            pylab.plot(ivdata[0],ivdata[1], '.', label=key)
        pylab.legend()
            
        pylab.figure()
        for index in range(len(ivkeys)):
            key = ivkeys[index]
            ivdata = ivs_dict[key]['data']
            atten_voltage = ivs_dict[key]['atten_voltage']
            laser_voltage = ivs_dict[key]['laser_voltage']
            print key, laser_voltage, atten_voltage
            laser_voltages[index] = laser_voltage
            atten_voltages[index] = atten_voltage
            #pylab.plot(ivdata[0],ivdata[1], '.', label=key)
            vtes, ites, rtes, ptes, rs, sc, nm = self.ivAnalyze(ivdata, rfb, rbias, rjnoise, mr, mrsh, vfb_max, intercept='superconduct', vbias_normal=vbias_normal)
            rn = rs[0]
            rns[index] = rn
            v_pnts, i_pnts, p_pnts = self.ivInterpolate(vtes, ites, rtes, pRn, rn)
            v_pnt_array[:,index] = v_pnts
            i_pnt_array[:,index] = i_pnts
            powerAtRns[:,index] = v_pnts*i_pnts
            iv_analysis_dict = tespkl.createIVRunAnalysisDict(vtes, ites, rtes, ptes, rs, sc, nm)
            tespkl.addIVRunAnalysis(bay, row, key, iv_analysis_dict)
        pylab.xlabel('VTES (uV)')
        pylab.ylabel('ITES (mA)')
        
        return laser_voltages, atten_voltages, powerAtRns
        
    def ivInterpolate(self, vtes, ites, rtes, percentRns, rn, tran_pRn_start=0.090):

        tran_indicies = np.where((rtes/rn)>tran_pRn_start)
        tran_start = min(tran_indicies[0])
        vtes_tran = vtes[tran_start:]
        ites_tran = ites[tran_start:]
        rtes_tran = rtes[tran_start:]
        #pylab.plot(vtes_tran,rtes_tran/rn)
        # Check that the r data being used for interpolation is monotonically increasing
        if np.all(np.diff(rtes_tran) > 0) is False:
            print 'IV %s not monotomicaly increasing in R' % key
            # Find the I and V that corespond to the requested percent Rns
        v_pnts = np.interp(percentRns/100.0*rn, rtes_tran, vtes_tran)
        i_pnts = np.interp(percentRns/100.0*rn, rtes_tran, ites_tran)
        print v_pnts, i_pnts
        p_pnts = v_pnts*i_pnts
        pylab.plot(v_pnts*1e6, i_pnts*1e3, 'o')
        
        return v_pnts, i_pnts, p_pnts

    def ivPRnLinearFit(self, v_pnt_array, i_pnt_array):
        
        pylab.figure()
        for index in range(len(v_pnt_array[:,0])):
            pylab.plot(v_pnt_array[index]*1e6, i_pnt_array[index]*1e3,'o')
            rvt_m, rvt_b = scipy.polyfit(v_pnt_array[index], i_pnt_array[index], 1)
            ites_rvt_fit = scipy.polyval([rvt_m,rvt_b],v_pnt_array[index])
            rvt_residuals = i_pnt_array[index]-ites_rvt_fit
            pylab.plot(v_pnt_array[index]*1e6, ites_rvt_fit*1e3)
            #pylab.plot(v_pnt_array[index], rvt_residuals,'o')
        pylab.title('Fits to percent Rn values')
        pylab.xlabel('VTES (mu)')
        pylab.ylabel('ITES (mA)')        

    def interpolateIVs(self, ivs, ivkeys, pRn):
        '''Get IVs at different temperatures from pickle and interpolate'''
        
            
        temperatures = np.zeros(len(ivkeys))
        v_pnt_array = np.zeros((len(pRn),len(ivkeys))) 
        i_pnt_array = np.zeros_like(v_pnt_array) 
        powerAtRns = np.zeros_like(v_pnt_array) 
        
        for index in range(len(ivkeys)):
            ivkey = ivkeys[index]
            print ivkey
            temperature =  ivs[ivkey]['temperature']
            temperatures[index] = temperature
            temperature_str =  str(temperature)
            itesr = ivs[ivkey]['analysis']['i_tes']
            vtesr = ivs[ivkey]['analysis']['v_tes']
            rtesr = ivs[ivkey]['analysis']['r_tes']
            rn = ivs[ivkey]['analysis']['normal_resistance']
            # Reverse the order of array to be increasing
            ites = itesr[::-1]
            vtes = vtesr[::-1]
            rtes = rtesr[::-1]
            # Check for NaN
            nan_indicies = np.where(np.isnan(vtes))
            if len(nan_indicies[0]) > 0:
                # transition data is at values larger then the NaNs
                start_tran = int(max(nan_indicies[0])+1)
            else:
                #otherwise start at a fixed value
                indicies_greater_then = np.where(rtes/rn > .080)
                start_tran = min(indicies_greater_then[0])
            # slice arrays to get data in the transition and above
            i_tes_tran = ites[start_tran:]
            v_tes_tran = vtes[start_tran:]
            r_tes_tran = rtes[start_tran:]
            # Check that the r data being used for interpolation is monotonically increasing
            if np.all(np.diff(v_tes_tran) > 0) is False:
                print 'IV %s not monotomicaly increasing in V' % ivkey
            # Find the I and V that corespond to the requested percent Rns
            v_pnts = np.interp(pRn/100.0*rn, r_tes_tran, v_tes_tran)
            i_pnts = np.interp(pRn/100.0*rn, r_tes_tran, i_tes_tran)
            # Plot data the I versus V at the requested R for this temperature
            pylab.plot(v_tes_tran,i_tes_tran, label=temperature_str)
            pylab.plot(v_pnts, i_pnts, 'o')
            #Fill arrays [percentRn, temperature]
            v_pnt_array[:,index] = v_pnts
            i_pnt_array[:,index] = i_pnts
            powerAtRns[:,index] = v_pnts*i_pnts
        
        pylab.legend()
        
        return temperatures, powerAtRns, rn


    def fitPowerLaw(self, pRn, temperatures, powerAtRns, init_guess, fitToLast=True, TbsToReturn=[0.080,0.094,0.095] ,plot=True):
        '''Fit power versus T_base with power law.'''

        if plot is True:
            pylab.figure()
        
        Ks = np.zeros(len(pRn))
        ns = np.zeros(len(pRn))
        Tcs = np.zeros(len(pRn))
        Gs = np.zeros(len(pRn))
        Ks_error = np.zeros(len(pRn))
        ns_error = np.zeros(len(pRn))
        Tcs_error = np.zeros(len(pRn))
        
        for index in range(len(pRn)):
            if plot is True:
                pylab.plot(temperatures*1e3,powerAtRns[index]*1e9,'o')
            out = scipy.optimize.leastsq(self.powerlaw_err_func, init_guess, args=(temperatures, powerAtRns[index]), full_output=1)
            Ks[index] = out[0][0]
            ns[index] = out[0][1]
            Tcs[index] = out[0][2]
            temp_pnts = np.linspace(0.040,out[0][2],25)
            if plot is True:
                pylab.plot(temp_pnts*1e3, self.powerlaw_fit_func(out[0],temp_pnts)*1e9)
            #temp_pnts = np.linspace(0.055,output.beta[2],25)
            #pylab.plot(temp_pnts, self.powerlaw_fit_func(output.beta,temp_pnts))
        Gs = ns*Ks*Tcs**(ns-1)


        if fitToLast is True:
            # Find the T and G for a given Tb using the highest n and K in the transistion
            
            # Find Tbs matching temperatues where data was taking
            Tbs = np.zeros(0)
            Tbs_index = np.zeros(0,int)
            for Tb in TbsToReturn:
                Tb_index = np.where(Tb == temperatures)
                if len(Tb_index[0]) > 0:
                    Tbs_index = np.hstack((Tbs_index , int(Tb_index[0][0])))
                    Tbs = np.hstack((Tbs , temperatures[int(Tb_index[0][0])]))
                else:
                    print 'Could not find Tbath=', Tb
            print Tbs
            
            K = Ks[-1]
            n = ns[-1]
            print K, n
            Ts = np.zeros((len(Tbs),len(pRn)))
            Gs_Tb = np.zeros((len(Tbs),len(pRn)))
            for index in range(len(Tbs)):
                Tb = Tbs[index]
                Tb_index = Tbs_index[index]
                Ts[index] = (powerAtRns[:,Tb_index]/K+Tb**(n))**(1./n)
                Gs_Tb[index] = n*K*(Ts[index])**(n-1.)
        
        if plot is True:
            pylab.xlabel('Bath Temperature (mK)')
            pylab.ylabel('Power (nW)')
            f1 = pylab.figure()
            p1 = f1.add_subplot(2,2,1)
            p1.plot(pRn,Ks,'o')
            pylab.title('K values from fits')
            pylab.xlabel('%Rn')
            pylab.ylabel('K')
            p2 = f1.add_subplot(2,2,2)
            p2.plot(pRn,ns,'o')
            pylab.title('n values from fits')
            pylab.xlabel('%Rn')
            pylab.ylabel('n')
            p3 = f1.add_subplot(2,2,3)
            p3.plot(pRn,Tcs,'o', label='T constant')
            for index in range(len(Ts)):
                p3.plot(pRn,Ts[index],'o',label='Tb='+str(Tbs[index]))
            pylab.title('T values from fits')
            pylab.xlabel('%Rn')
            pylab.ylabel('Tc (K)')  
            leg3 = pylab.legend(loc='best')
            try:
                for t in leg3.get_texts():
                    t.set_fontsize('small')
            except:
                print 'Legend error'
                
            p4 = f1.add_subplot(2,2,4)
            p4.plot(pRn,Gs,'o',label='T constant')
            for index in range(len(Ts)):
                p4.plot(pRn,Gs_Tb[index],'o',label='Tb='+str(Tbs[index]))
            pylab.title('G values from fits')
            pylab.xlabel('%Rn')
            pylab.ylabel('G (W/K)')
            leg4 = pylab.legend(loc='best')
            try:
                for t in leg4.get_texts():
                    t.set_fontsize('small')
            except:
                print 'Legend error'         
            
        return Gs_Tb, Ks, ns, Ts #, Ks_error, ns_error, Tcs_error


    def powerlaw_fit_func(self, p, x):
    
        power = p[0]*(p[2]**p[1]-np.power(x,p[1]))
        return power

    def powerlaw_err_func(self, p, x, y):
    
        err =  y - self.powerlaw_fit_func(p,x) 
        return err

    def calculateAlphaFromIV(self, percentRn, Ts_array, rn, plot=True):
        '''Calculates alpha from Tc values for each percent Rn'''
        
        if plot is True:
            pylab.figure()
        
        
        alpha_array = np.zeros((len(Ts_array),len(Ts_array[0])-1))
        Tmid_array = np.zeros_like(alpha_array)
        for index in range(len(Ts_array)):
            Ts = Ts_array[index]
            R = percentRn*rn
            dR = np.diff(R)
            dT = np.diff(Ts)
            dRdT = np.divide(dR,dT)
            Rmid = R[:-1]+dR/2.0
            Tmid = Ts[:-1]+dT/2.0
            alpha = Tmid/Rmid*dRdT
            perRnMid = Rmid/rn
            alpha_array[index] = alpha 
            Tmid_array[index] = Tmid
            
            if plot is True:
                pylab.plot(perRnMid,alpha, 'o')



        if plot is True:
            pylab.title('Alpha from extracted TC values')
            pylab.xlabel('%Rn')
            pylab.ylabel('Alpha')
                
        return alpha_array, perRnMid, Tmid_array
    
    def notchFilter(self,x,y,filter_xs,bins_to_cut_array):
        
        new_x = x
        new_y = y
        
        for loop_index in range(len(filter_xs)):
            filter_x = filter_xs[loop_index]
            bins_to_cut = bins_to_cut_array[loop_index]
            filter_index = np.argmin(np.abs(new_x-filter_x))
            new_x = np.hstack((new_x[:filter_index-bins_to_cut],new_x[filter_index+1+bins_to_cut:]))
            new_y = np.hstack((new_y[:filter_index-bins_to_cut],new_y[filter_index+1+bins_to_cut:]))
            
        return new_x, new_y
    
    def SoftwareLockin(self, v_signal, v_reference, reference_type='square',response_type='sine'):
        ''' lockin algorithm written by J. McMahon 
            input:
                v_signal: the output signal
                v_reference: the reference 
                mix_type: 'square' or 'sine', this affects the normalization
                
            output: tuple
            
                I: real part of transfer function
                Q: imaginary part of transfer function
                v_reference_amp = amplitude of reference signal
                
            NOTES: 
            To get the the amplitude of the locked in signal compute: sqrt(I**2+Q**2)/v_reference_amp
            algorithm checked for normalization on 5/12/2012 
            As of 10/26/2012 this is now compatible with arrays
            
            BUG KNOWN: 'SQUARE' 'SQUARE' CONFIGURATION GIVES THE WRONG AMPLITUDE NORMALIZATION IF 
                       THE PHASE IS DIFFERENT THAN ZERO
                
        '''

        if reference_type=='square':
            v_ref_norm = 1
            IQ_norm = np.pi/2
        elif reference_type=='sine':
            v_ref_norm = 2
            IQ_norm = 2
            
        if response_type == 'sine':
            res_norm = 1
        elif response_type == 'square':
            res_norm = 2/np.pi
        #embed();sys.exit()
        # perhaps crappy way to make a 1D array compatible with the rest of the function (need 1 x Nsamp array)
        theshape=np.shape(v_reference)
        if len(theshape)==1:
            N = len(v_reference)  # number of samples
            dummy = np.empty((1,N))
            dummy[0,:]=v_signal
            v_signal = dummy
            #v_signal = dummy[:N/2] # take half
            dummy = np.empty((1,N))
            dummy[0,:]=v_reference
            v_reference = dummy
            #v_reference = dummy[::2] # double rate
         
        dex = 1
        Nd,N=np.shape(v_reference)
        
        # only take highest power of 2 of the data (faster!)
        N = 2**(int(np.log2(N)))
        v_signal = v_signal[:,0:N]
        v_reference = v_reference[:,0:N]

        # remove the DC offset from signal and reference as well as create hanning window 
        ref_mean = np.mean(v_reference,dex)
        dat_mean = np.mean(v_signal,dex)
        ref=np.empty((Nd,N))
        dat=np.empty((Nd,N))
        hanning_window = np.hanning(N)
        hw = np.empty((Nd,N))
        for i in range(Nd):
            ref[i] = v_reference[i]-ref_mean[i]            
            dat[i] = v_signal[i]-dat_mean[i]
            hw[i] = hanning_window
        
        
        # 90 degree phase shift of reference.  Due to limitation of python2.6.5 scipy.signal.hilbert, I cannot 
        # take the hilbert transform of a matrix, just a 1D array.  I need to construct the matrix.  Technically,
        # I should take the reference of each column instead of using one for all.  However, doing this would 
        # be slow, and the data is sampled so quickly, it probably doesn't matter
        
        rh = hilbert(ref[0]) # now a complex array
        ref_hil = np.empty((Nd,N),complex)
        for ii in range(Nd):
            ref_hil[ii] = rh    
        
        # take FFTs
        fr = np.fft.fft(hw*ref_hil,N)
        fs = np.fft.fft(hw*dat,N)

        # auto and cross correlations (8/3 accounts for power removed due to hanning window)
        rr = fr.conjugate()*fr*8/3. # autocorrelation of reference
        qs = fr.conjugate()*fs*8/3. # cross correlation of 'analytic reference' and signal
        
        v_reference_amp = np.sqrt(np.mean(np.real(rr),dex)/N*v_ref_norm) # somewhat mislabeled, only true if one harmonic present
        
        #v_signal_amp = np.sqrt(ss.real.mean()/N*2) # ditto here 
        I = np.mean(qs.real,dex)/N*np.sqrt(2)*IQ_norm*res_norm # transfer function in phase component
        Q = np.mean(qs.imag,dex)/N*np.sqrt(2)*IQ_norm*res_norm # transfer function out of phase component
        #pylab.plot(v_reference[:100],'.-')
        #pylab.plot(v_signal[:100], '.-')
        #pylab.show()

        return I, Q, v_reference_amp

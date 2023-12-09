'''
fts_utils.py

everything one should need to analyze data taken with the NIST FTS system
@author JH, started 03/2021, significant updates 6/2021 in prep for LTD

Notes:
1) phase correction for individual scans or for average?
2) should phase correction only be applied where there is signal?
3) Definition of "standard" fft packing from scipy.fftpack.fft:

The packing of the result is "standard": If ``A = fft(a, n)``, then
``A[0]`` contains the zero-frequency term, ``A[1:n/2]`` contains the
positive-frequency terms, and ``A[n/2:]`` contains the negative-frequency
terms, in order of decreasingly negative frequency. So ,for an 8-point
transform, the frequencies of the result are [0, 1, 2, 3, -4, -3, -2, -1].
To rearrange the fft output so that the zero-frequency component is
centered, like [-4, -3, -2, -1,  0,  1,  2,  3], use `fftshift`.

'''

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
#import dataclasses
#from dataclasses_json import dataclass_json
#from typing import Any, List
import scipy
import scipy.fftpack
from scipy.fftpack import fftshift, ifftshift
from scipy import signal
import scipy.constants
import os
from scipy.integrate import quad, simpson

icm2ghz = scipy.constants.c/1e7

def load_fts_scan_fromfile(filename):
    ''' current data format is the ascii output of the bluesky software '''
    f=open(filename,'r')
    lines=f.readlines()
    f.close()
    N=len(lines)
    # determine header from data
    for ii in range(N):
        if 'OPD (cm)' in lines[ii]:
            dex = ii
            break
    # put header info into dictionary
    header={}
    for ii in range(dex-1):
        val = lines[ii][:-1].split(':,')
        header[str(val[0])] = str(val[1])

    # parse data into numpy N,2 array
    data = np.empty((N-dex-1,2))
    for ii in range(N-dex-1):
        x=lines[ii+dex+1][:-1].split(',')
        data[ii,0]=float(x[0])
        data[ii,1]=float(x[1])

    x = data[:,0]
    y = data[:,1]
    file = header['File']
    version = header['VERSION']
    file_prefix = header['FILE_PREFIX']
    file_number = int(header['FILE_NUMBER'])
    num_scans = int(header['NUM_SCANS'])
    current_scan = int(header['CURRENT_SCAN'])
    resolution = float(header['RESOLUTION'])
    nyquist = float(header['NYQUIST'])
    num_samples = int(header['SAMPLES'])
    speed = float(header['SPEED'])
    zpd = float(header['ZPD'])
    buffer_size = int(header['BUFFER_SIZE'])
    date = str(header['DATE'])
    juldate = float(header['JULDATE'])
    source = str(header['SOURCE'])
    comment = str(header['COMMENT'])

    return FtsData(x,y,file,version,file_prefix,file_number,num_scans,current_scan,resolution,nyquist,
                   num_samples,speed,zpd,buffer_size,date,juldate,source,comment,'OPD (cm)','Signal (V)')

#@dataclass_json
@dataclass
class FtsData():
    x: np.ndarray
    y: np.ndarray
    file: str
    version: str
    file_prefix: str
    file_number: int
    num_scans: int
    current_scan: int
    resolution: float
    nyquist: float
    num_samples: int
    speed: float
    zpd: float
    buffer_size: int
    date: str
    juldate: float
    source: str
    comment: str
    x_label: str
    y_label: str

    def plot(self):
        plt.figure()
        plt.plot(self.x, self.y)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(f"Interferogram for file: {self.file}")

    def get_spectrum(self,poly_filter_deg=1,plotfig=False):
        f,B = IfgToSpectrum().to_spectrum_simple(self.x,self.y,self.speed,poly_filter_deg=1,plotfig=True)
        return f,B

    def print_metadata(self):
        attrs = vars(self)
        for key in attrs:
            if key not in ['x','y']:
                print(key, '::', attrs[key])


class TimeDomainDataProcessing():
    def remove_poly(self,x,y,deg=1):
        ''' remove polynomial from detector response
        '''
        # p = scipy.polyfit(x,y,deg)
        # y=y-scipy.polyval(p,x)
        p = np.polyfit(x,y,deg)
        y=y-np.polyval(p,x)
        return y

    def notch_frequencies(self,x,y,v,filter_freqs_hz=[60.0,120.0,180.0,240.0,300.0,420.0,480.0,540.0],
                          filter_width_hz=.1,plotfig=False):
        ''' remove power at discrete frequencies '''
        t = x/v # timestream
        samp_int=t[1]-t[0]
        y=y-y.mean()
        f,ffty = IfgToSpectrum().get_fft(t,y,plotfig=False)

        ffty_filt = np.copy(ffty)
        for i in filter_freqs_hz:
            js1 = f[(f>(i-filter_width_hz/2.0)) & (f<(i+filter_width_hz/2.0))] # positive frequencies
            js2 = f[(f<-1*(i-filter_width_hz/2.0)) & (f>-1*(i+filter_width_hz/2.0))] # positive frequencies
            js=np.concatenate((js1,js2))
            for j in js:
                ffty_filt[list(f).index(j)]=0

        y_filt = scipy.fftpack.ifft(ffty_filt)
        if plotfig:
            plt.figure(1)
            plt.plot(f,abs(ffty))
            plt.plot(f,abs(ffty_filt))
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Response (arb)')
            plt.legend(('orig','notched'))

            plt.figure(2)
            plt.plot(x,y)
            plt.plot(x,y_filt)
            plt.xlabel('OPD')
            plt.ylabel('Response (V)')
            plt.legend(('orig','notched'))
            plt.show()
        return y_filt

    def standardProcessing(self,x,y,v,filter_freqs_hz=[60,180],filter_width_hz=0.5,poly_filter_deg=1,plotfig=False):
        if filter_freqs_hz is not None:
            y_filt = self.notch_frequencies(x,y,v,filter_freqs_hz,filter_width_hz,plotfig=False)
        else:
            y_filt = y
        y_filt = self.remove_poly(x,y_filt,deg=poly_filter_deg)
        if plotfig:
            plt.figure(1)
            plt.plot(x,y-np.mean(y))
            plt.plot(x,y_filt)
            plt.xlabel('OPD (cm)')
            plt.ylabel('Response (V)')
            plt.legend(('orig','filtered'))
            plt.show()
        return y_filt

class IfgToSpectrum(TimeDomainDataProcessing):

    # lower level methods ----------------------------------------------------------
    def get_zpd(self,x,y,plotfig=False):
        ''' find the zero path difference from the maximum intensity of the IFG '''
        z=(y-y.mean())**2
        ZPD=x[z==z.max()]
        if len(ZPD)>1:
            print('warning, more than one maximum found.  Estimated ZPD not single valued!  Using first index')
        ZPD = ZPD[0]
        dex=list(x).index(ZPD)
        if plotfig:
            plt.figure(1,figsize = (12,6))
            plt.plot(x,y,'b.-')
            plt.plot(x[dex],y[dex],'ro')
            plt.xlabel('OPD (cm)')
            plt.ylabel('Detector response (V)')
            plt.title('Location of ZPD')
            plt.show()
        return dex,ZPD

    def ascending_to_standard_fftpacking(self,x,y,zpd_index=None):
        if zpd_index==None:
            zpd_dex, zpd_val = self.get_zpd(x,y)
        else:
            zpd_dex = zpd_index
        x_std = np.concatenate((x[zpd_dex:],x[0:zpd_dex][::-1]))
        y_std = np.concatenate((y[zpd_dex:],y[0:zpd_dex][::-1]))
        return x_std, y_std

    def window_and_shift(self,x,y,window="hanning"):
        x = scipy.fftpack.fftshift(x) # packing is now 0,1,2,...N/2,N/2-1,N/2-2,...1
        if window == "hanning":
            y = scipy.fftpack.fftshift(y*np.hanning(len(y))) # packing is now 0,1,2,...N/2,N/2-1,N/2-2,...1
        else:
            y = scipy.fftpack.fftshift(y)
        return x,y

    def interp_angle(self, f_for_interpolation,f_to_interp_from,theta_to_interp_from,debug = False):
        '''
        basically a standard interpolation of an angle behaves poorly when +pi changes to -pi and vice versa
        solution is to plot theta on a unit circle in the complex plane.
        then interpolate the new location in the complex plane from the nearest two points
        then to get the angle from that new point on the complex plane
        this is just a simple linear interpolation

        inputs:
        f_for_interpolation   is the data for which you want to generate new angles from
        f_to_interp_from      is the domain from which you interpolate from
        theta_to_interp_from  is the range from which you will interpolate from

        outputs:
        theta_interpolated    is new angles interpolated from at f_for_interpolation
        '''
        sin_theta = np.sin(theta_to_interp_from)
        cos_theta = np.cos(theta_to_interp_from)
        theta_interpolated = np.array(())
        count = 0
        for f in f_for_interpolation:
            difference = f-f_to_interp_from
            closest_index = np.argmin(np.abs(difference))
            difference[closest_index] = np.max(np.abs(difference))
            second_closest_index = np.argmin(np.abs(difference))
            length_to_closest = np.abs(f-f_to_interp_from[closest_index])
            length_to_second_closest = np.abs(f-f_to_interp_from[second_closest_index])
            total_length = length_to_closest + length_to_second_closest
            sin_theta_interpolated = sin_theta[closest_index]*(length_to_second_closest/total_length)+sin_theta[second_closest_index]*(length_to_closest/total_length)
            cos_theta_interpolated = cos_theta[closest_index]*(length_to_second_closest/total_length)+cos_theta[second_closest_index]*(length_to_closest/total_length)
            theta_interpolated = np.append(theta_interpolated,np.arctan2(sin_theta_interpolated,cos_theta_interpolated))
            if debug:
                if count == 5:
                    print(np.arctan2(sin_theta_interpolated,cos_theta_interpolated))
                    print(theta_interpolated)
                    plt.figure(-23)
                    plt.title(str(f)+","+str(f_to_interp_from[closest_index])+","+str(f_to_interp_from[second_closest_index]))
                    plt.plot(sin_theta[closest_index],cos_theta[closest_index],"o",label = "closest")
                    plt.plot(sin_theta[second_closest_index],cos_theta[second_closest_index],"o",label = "2nd closest")
                    plt.plot(sin_theta_interpolated,cos_theta_interpolated,"o",label = "interpolated")
                    plt.ylim(-1,1)
                    plt.xlim(-1,1)
                    plt.legend()
                    plt.show()

            count = count+1
        return theta_interpolated

    def get_fft(self,x,y,plotfig=False):
        ''' return the FFT of y and the frequencies sampled assuming equally spaced samples in x.
            Packing of return is "standard", i.e. [0, 1, 2, 3, -4, -3, -2, -1] for 8 point transform.
        '''
        samp_int = x[1]-x[0]
        ffty=scipy.fftpack.fft(y)
        f=scipy.fftpack.fftfreq(len(y),samp_int)

        if plotfig:
            plt.figure(1,figsize = (12,6))
            plt.subplot(211)
            plt.title('Time and FFT space')
            plt.plot(x,y)
            plt.subplot(212)
            plt.plot(f,np.abs(ffty))
            plt.plot(f,np.real(ffty))
            plt.plot(f,np.imag(ffty))
            plt.legend(('abs','real','imag'))
            plt.figure(2)
            plt.title('phase')
            plt.plot(f,np.arctan(np.imag(ffty)/np.real(ffty)))
            plt.xlabel('Frequency')
            plt.ylabel('Phase (rad)')
            plt.show()
        return f,ffty

    def fit_angle(self,frequencies,theta,min_poly,max_poly,deg=1,PLOT=False,units="wavenumber"):
        '''
        function to fit angle versus frequency with a polynomial
        probably best to fit in complex plane

        inputs:
        frequencies-
        theta-
        fmin-
        fmax-
        deg-

        outputs:
        fit_dict-
             fit_dict['fit']-             output of curve fit
             fit_dict['fit_result']       y data for the fitted result
             fit_dict['initial_guess']    y data for the inital guess for fitting
        '''

        if units == "wavenumber":
            unit_scale = 1
        elif units == "GHz":
            unit_scale = 30
        elif units == "THz":
            unit_scale = 30/1000.
        else:
            print("please select a valid unit type")
            return

        theta_complex = np.hstack((np.sin(theta),np.cos(theta)))

        print("min_poly,max_poly",min_poly,max_poly)

        frequencies_use = frequencies[(frequencies*unit_scale>min_poly) & (frequencies*unit_scale<max_poly)]
        theta_use = theta[(frequencies*unit_scale>min_poly) & (frequencies*unit_scale<max_poly)]
        theta_complex = np.hstack((np.sin(theta_use),np.cos(theta_use)))


        if deg == 0:
            p0 = np.asarray([np.median(theta_use)])
            upper_bound = (np.pi)
            lower_bound = (-np.pi)
            bounds = (lower_bound,upper_bound)
        elif deg == 1:
            p0 = np.asarray((np.median(theta_use),0))
            upper_bound = (np.pi,np.inf)
            lower_bound = (-np.pi,-np.inf)
            bounds = (lower_bound,upper_bound)
        elif deg == 2:
            p0 = np.asarray([np.median(theta_use),0,0])
            upper_bound = (np.pi,np.inf,np.inf)
            lower_bound = (-np.pi,-np.inf,-np.inf)
            bounds = (lower_bound,upper_bound)
        else:
            print("Please choose a polynomial degree of 0,1, or 2")
            return None

        print(p0)



        fit = curve_fit(poly_angle_complex_plane,frequencies_use,theta_complex,p0 = p0,bounds = bounds)
        initial_guess = poly_mod_2pi(frequencies,p0)
        fit_result = poly_mod_2pi(frequencies,fit[0])
        fit_dict = {"fit":fit,"fit_result":fit_result,"initial_guess":initial_guess}

        if PLOT:
            plt.figure(figsize = (12,6))
            plt.subplot(211)
            plt.title("fit_angle function output")
            plt.xlabel("Fitted domain "+units)
            plt.ylabel("Phase (Radians)")
            plt.plot(np.sort(frequencies*unit_scale),initial_guess[np.argsort(frequencies)],label = "initial guess")
            plt.plot(np.sort(frequencies*unit_scale),fit_result[np.argsort(frequencies)],label = "fit result")
            plt.plot(frequencies*unit_scale,theta,"o",mec = "k",label = "data")
            plt.xlim(min_poly,max_poly)
            plt.legend()
            plt.subplot(212)
            plt.xlabel("Entire domain "+units)
            plt.ylabel("Phase (Radians)")
            plt.plot(np.sort(frequencies*unit_scale),initial_guess[np.argsort(frequencies)],label = "initial guess")
            plt.plot(np.sort(frequencies*unit_scale),fit_result[np.argsort(frequencies)],label = "fit result")
            plt.plot(frequencies*unit_scale,theta,".",label = "data")
            plt.fill([min_poly,max_poly,max_poly,min_poly], [-1.25*np.pi,-1.25*np.pi,1.25*np.pi,1.25*np.pi], 'grey', alpha=0.4)
            #plt.xlim(fmin,fmax)
            plt.ylim(-1.25*np.pi,1.25*np.pi)
            plt.legend()
            plt.show()

        return fit_dict


    # medium level methods --------------------------------------------------------------
    def get_symmetric_ifg(self,x,y,zpd_index=None,x_to_opd=False,
                             fftpacking=False,plotfig=False):
        ''' return only the symmetric portion of an asymmetric IFG given the index of the
            zero path difference (zpd_index).
        '''
        if zpd_index == None:
            zpd_index, zpd_val = self.get_zpd(x,y)
        else:
            zpd_val = x[zpd_index]

        N=zpd_index*2
        #x_sym=x[0:N]
        #y_sym=y[0:N] # length of x_sym, y_sym always odd by construction
        x_sym=x[0:N+1]
        y_sym=y[0:N+1]
        if fftpacking:
            #x_sym,y_sym = self.ascending_to_standard_fftpacking(x_sym,y_sym,zpd_index)
            x_sym = fftshift(x_sym) ; y_sym = fftshift(y_sym)
        if x_to_opd:
            x_sym = x_sym - zpd_val
            x=x-zpd_val

        if plotfig:
            plt.figure(1,figsize = (12,6))
            plt.plot(x,y,label = "Entire interferogram")
            plt.plot(x[zpd_index],y[zpd_index],'ro',label='ZPD')
            plt.plot(x_sym,y_sym,label = "Double-sided interferogram")
            plt.xlabel('OPD (cm)')
            plt.ylabel('Detector response (arb)')
            plt.title('Symmetric portion of IFG')
            plt.legend()
            plt.show()
        return x_sym,y_sym

    def get_zero_padded_high_res_ifg(self, x, y, zpd_index=None, add_linear_weight=True,
                                     fftpacking=True, plotfig=False, debug=False):
        ''' return zero padded ifg in ascending order (from - retardation to +) '''
        if zpd_index == None:
            zpd_index, zpd_val = self.get_zpd(x,y)
        else:
            zpd_val = x[zpd_index]
        N = len(x)
        xp = x - zpd_val # define x-axis as optical path length difference (ie 0 = 0 path length difference)
        dex1 = 2*zpd_index+1 # first index after double-sided ifg
        xpp = np.array(list(xp[dex1:][::-1]*-1) + list(xp))
        if add_linear_weight:
            weight = np.linspace(0,1,dex1)
        else:
            weight = np.ones(dex1)
        yp = np.array([0]*(N-dex1) + list(y[0:dex1]*weight) + list(y[dex1:])) # zero pad LHS of ifg
        if fftpacking:
            xp = fftshift(xp); xpp=fftshift(xpp); yp=fftshift(yp)
        if debug:
            #print(len(xp[0:dex1]), dex1)
            plt.plot(xp,y,'-')
            plt.plot(xp[0:dex1], y[0:dex1],'bo')
            plt.plot(xp[dex1:], y[dex1:],'go')
            plt.axvline(xp[dex1],label='1st pt after dsifg')
            plt.xlabel('OPD (cm)'); plt.ylabel('Response (arb)')
            plt.legend()
            plt.show()

        if plotfig:
            plt.plot(xpp,yp,'b*-')
            dex = np.where(xpp==0)[0]
            plt.plot(xpp[dex],yp[dex],'ro')
            plt.xlabel('OPD (cm)')
            plt.ylabel('Response (arb)')
            plt.show()
        return xpp,yp

    def make_high_res_symmetric_ifg(self,x,y,zpd_index=None,x_to_opd=True,
                                    fftpacking=True,plotfig=False):
        ''' force a symmetric IFG from the single sided IFG by mirroring the IFG '''
        if zpd_index == None:
            zpd_index, zpd_val = self.get_zpd(x,y)
        else:
            zpd_val = x[zpd_index]
        x_cat = np.concatenate((-x[2*zpd_index+1:][::-1],x))
        y_cat = np.concatenate((y[2*zpd_index+1:][::-1],y)) # just mirror the -delta portion not measured
        if fftpacking:
            x_cat = fftshift(x_cat); y_cat = fftshift(y_cat)
        if x_to_opd:
            x_cat = x-zpd_val
        if plotfig:
             plt.plot(x,y,color= "b",linewidth=0.5,label = "raw data")
             plt.plot(x_cat,y_cat,color= "k",linewidth=0.5,label = "concatenated")
             plt.legend()
             plt.show()
        return x_cat,y_cat

    def get_phase_spectrum(self,x,y,zpd_index):
        ''' return frequency and phase spectrum from double sided IFG in fft standard packing '''
        x_ds,y_ds = self.get_symmetric_ifg(x,y,zpd_index,x_to_opd=True,
                                             fftpacking=True,plotfig=False) # packing is "standard"
        f_ds, B_ds = self.get_fft(x_ds, y_ds*fftshift(np.hanning(len(y_ds))), plotfig=False) # apodize ifg before FFT
        phi_ds = np.arctan(np.imag(B_ds)/np.real(B_ds)) # output between -pi/2 and pi/2
        return f_ds, phi_ds

    # high-level methods -------------------------------------------------------------------
    def get_phase_corrected_spectrum(self,x,y,ZPD = None,
                        min_filter=None,
                        max_filter=None,
                        algorithm='Richards',
                        phase_fit_method = "interp",
                        phase_fit_degree = 1,
                        min_poly = 210/30.,
                        max_poly = 300/30.,
                        units = "wavenumber",
                        window = True,
                        debug=False):
        '''
        Full FTS analysis following for single sided IFGs using an algorithm of 'Mertz' or 'Richards'

        written by Jordan Wheeler

        inputs:
        x-                  distances of interferogram points in cm from 0 to total throw L
        y-                  interferogram intensity values
        ZPD-                the location of the white light fringe in same units as x.  if not specified program will try to automatically find it.
        min_filter-         minimum frequency below which to filter signal out of if using method Richards
        max_filter-         maximum frequency above which to filter signal out of if using method Richards
        algorithm-          Mertz or Richards i.e. fix phase in frequency space or interferogram space
        phase_fit_method-   either poly or interp - poly is good for extrapolating the phase correction to frequencies with little S/N
        phase_fit_degree-   the degree of polynomial for fitting the phase if using phase_fit_method = poly can be 0, 1, or 2
        min_poly-           min for which to fit phase with if using phase_fit_method = poly (units wavenumber)
        max_poly-           max for which to fit phase with if using phase_fit_method = poly (units wavenumber)
        units-              the units for plotting can be wavenumber, GHz, or THz default wavenumber cm^-1
        window-             <bool> if true apply hanning window
        debug-              make a bunch of plots or not - always make plots if it is the first run

        outputs:
        f-                  frequencies of in wavenumber of spectrum
        B-                  complex fourier transformed spectrum with signal
                            in the real component and noise in the imaginary component

        '''
        if phase_fit_method != "poly":
            phase_fit_method = "interp"

        if units == "wavenumber":
            unit_scale = 1
        elif units == "GHz":
            unit_scale = icm2ghz
        elif units == "THz":
            unit_scale = icm2ghz/1000
        else:
            print("please select a valid unit type")
            return

        samp_int = x[1]-x[0] # used in plotting later on
        y_orig = y.copy()
        y = self.remove_poly(x,y,deg=1)
        # find ZPD -------------------------------------------------------------
        if ZPD == None:
            ZPD_index, ZPD = self.get_zpd(x,y,plotfig=debug)
        else:
            ZPD_index = np.argmin(np.abs(x-ZPD))
            ZPD = x[ZPD_index]

        # Work on symmetric portion of ifg --------------------------------------------
        # get phase of symmeric portion of ifg ----------------------------------------
        x_sym,y_sym = self.get_symmetric_ifg(x,y,zpd_index=ZPD_index,x_to_opd=False,
                                 fftpacking=False,plotfig=debug)
        if window:
            y_sym = scipy.fftpack.ifftshift(y_sym*np.hanning(len(y_sym))) # packing is now 0,1,2,...N/2,N/2-1,N/2-2,...1
        else:
            y_sym = scipy.fftpack.ifftshift(y_sym)
        x_sym = scipy.fftpack.ifftshift(x_sym) # packing is now 0,1,2,...N/2,N/2-1,N/2-2,...1
        f_sym,S_sym = self.get_fft(x_sym,y_sym,plotfig=debug) #FFTandFrequencies(x_sym,y_sym,plotfig=plotfig,units = units)
        phase_sym = np.angle(S_sym)

        # Work on full (mirrored) double-sided IFG -------------------------------------
        # force a symmetric IFG from the single sided IFG by mirroring the IFG ---------
        # _ds is for "double sided"
        x_ds = np.concatenate((-x[2*ZPD_index+1:][::-1]+ZPD,x-ZPD))
        y_ds = np.concatenate((y[2*ZPD_index+1:][::-1],y)) # mirror the -delta portion not measured
        if window:
            y_ds = scipy.fftpack.ifftshift(np.hanning(len(y_ds))*y_ds)
        else:
            y_ds = scipy.fftpack.ifftshift(y_ds)
        x_ds = scipy.fftpack.ifftshift(x_ds)
        f,S = self.get_fft(x_ds,y_ds,plotfig=debug)

        # interpolate phase information to resolution of mirrored double sided ifg
        interp_func = scipy.interpolate.interp1d(scipy.fftpack.fftshift(f_sym),
                                                 scipy.fftpack.fftshift(phase_sym),
                                                 kind = "linear",
                                                 bounds_error = False,
                                                 fill_value = 0)

        if phase_fit_method == "poly":
            fit_dict = self.fit_angle(f_sym,phase_sym,min_poly,max_poly,deg=phase_fit_degree,plotfig=debug,units=units)
            phase_highres = poly_mod_2pi(f,fit_dict['fit'][0])
        else:
            phase_highres = self.interp_angle(f,f_sym,phase_sym,debug = False)

        # do phase correction ---------------------------------------------------------
        if algorithm=='Mertz': # phase correct in frequency spectrum space
            S_corr = S*np.exp(-1j*phase_highres)

        elif algorithm=='Richards': #phase correct in interferogram space
            if min_filter: # since you are Fourier transforming to correct phase you might as well do a frequency filter
                if max_filter:
                    boxcar = np.zeros(len(y_ds))
                    boxcar[(f*unit_scale>min_filter) & (f*unit_scale<max_filter)] = 1
                    boxcar[(f*unit_scale<-1*min_filter) & (f*unit_scale>-1*max_filter)] = 1
                else:
                    print("please specify both min_filter and max_filter")
            else:
               boxcar = np.ones(len(y_ds))  # do nothing

            phase_ifft = np.fft.ifft(np.exp(-1j*phase_highres)*boxcar) # interpolate between points
            N=len(phase_ifft)
            phase_ifft_sym = scipy.fftpack.fftshift(phase_ifft)

            # convolve in interferogram space and trim extra stuff from convolution
            y_corr = np.real(signal.convolve(scipy.fftpack.fftshift(y_ds),scipy.fftpack.fftshift(phase_ifft)))[N//2:3*N//2]
            if window:
                y_corr = y_corr*np.hanning(len(y_corr)) #appodize should be made into option
            else:
                pass
            f,S_corr = self.get_fft(x_ds,scipy.fftpack.ifftshift(y_corr),plotfig=debug)

        if debug:
            # fig 1: ifg and location of zpd
            plt.figure(1)
            plt.title("Manually specified Zero Path Length difference (ZPD) location")
            plt.plot(x,y)
            plt.plot(x[ZPD_index],y[ZPD_index],"*",label = "ZPD")
            plt.legend()

            # fig 2 symmetric ifg with hanning window applied (if asked)
            plt.figure(2)
            plt.title("hanning window applied")
            plt.plot(x_sym,y_sym,label = 'data')
            plt.plot(x_sym,np.hanning(len(y_sym))*np.max(np.abs(y_sym)),label = 'scaled window')
            plt.plot(x_sym,y_sym*np.hanning(len(y_sym)),label ='windowed data')
            plt.legend()
            plt.xlabel("Path length (cm)")
            plt.ylabel("Power")

            # fig 3: symmetric ifg with proper indexing
            plt.figure(3)
            plt.title("Proper indexing for FFT\nWhite light fringe at 0\nPoint to left of white light fringe at index N")
            plt.plot(y_sym)
            plt.ylabel("Power")
            plt.xlabel("Index")
            #plt.show()

            # fig 4: mirrored double sided ifg
            plt.figure(figsize = (12,6))
            plt.title("mirroring interferogram")
            plt.plot(x-ZPD,y,linewidth = 2,label = "full interferogram")
            plt.plot(x_sym,y_sym,linewidth = 2,label = "symmetric portion")
            plt.plot(x_ds, y_ds,linewidth = 2,label = "mirrored portion")

            # fig 5: check that white light fringe is at index 0
            plt.figure()
            plt.title("Proper indexing for FFT\nWhite light fringe at 0\nPoint to left of white light fringe at index N")
            plt.plot(y_ds)
            plt.ylabel("Power")
            plt.xlabel("Index")
            #plt.show()

            # fig 6: interpolated phase
            plt.figure(figsize = (12,6))
            plt.title("Fitting of phase, currently using method " + phase_fit_method)
            plt.plot(f_sym*unit_scale,theta,"o",mec = "k",label = "theta from symmetric interferogram")
            plt.plot(f*unit_scale,theta_highres_2,".",label = "theta interpolated to higher resolution in complex plane")
            if phase_fit_method == "poly":
                plt.plot(np.sort(f*unit_scale),theta_highres_3[np.argsort(f)],label = "theta fitted polynomial")
            plt.ylim(-5,5)
            plt.xlabel(units)
            plt.legend()
            #plt.show()
        return f,S_corr

    def phase_correction_mertz(self,x,y,zpd_index=None,plotfig=False):
        ''' Generate phase corrected spectrum using method from
            L. Mertz. Rapid scanning fourier transform spectroscopy.
            J. Phys. Coll. C2, Suppl. 3-4, 28:88, 1967.

            return f,B (phase corrected) in stardard fft packing
        '''
        print('WARNING. MERTZ PHASE CORRECTION NOT FULLY TESTED!!!!')
        # Get phase info from double sided IFG
        if zpd_index == None:
            zpd_index, zpd_val = self.get_zpd(x, y, plotfig=False)
        f_ds, phi_ds = self.get_phase_spectrum(x,y,zpd_index)

        # x_ds,y_ds = self.get_symmetric_ifg(x,y,zpd_index,x_to_opd=True,
        #                                      fftpacking=True,plotfig=False) # packing is "standard"
        # f_ds, B_ds = self.get_fft(x_ds, y_ds*fftshift(np.hanning(len(y_ds))), plotfig=False) # apodize ifg before FFT
        # phi_ds = np.arctan(np.imag(B_ds)/np.real(B_ds)) # output between -pi/2 and pi/2
        #phi_ds = np.angle(B_ds,deg=False) # output between -pi and pi

        # Get multiplicative phase correction for high res spectrum
        xp,yp = self.get_zero_padded_high_res_ifg(x, y, zpd_index, add_linear_weight=True,
                                                  fftpacking=True,plotfig=False, debug=False)
        f,B = self.get_fft(xp, yp*fftshift(np.hanning(len(yp))), plotfig=False)
        phi = fftshift(self.interp_angle(f_for_interpolation=ifftshift(f), f_to_interp_from=ifftshift(f_ds), theta_to_interp_from=ifftshift(phi_ds), debug = False))
        plt.plot(phi,'o-')
        plt.show()
        # phase correct
        B_corr = B*np.exp(-1j*phi)

        if plotfig:
            plt.figure(-3,figsize = (12,6))
            plt.subplot(211)
            plt.plot(f,np.real(B_corr),'b-',label='corr real')
            plt.plot(f,np.imag(B_corr),'g-',label='corr imag')
            plt.plot(f,np.real(B),'b--',label='orig real')
            plt.plot(f,np.imag(B),'g--',label='orig imag')
            plt.xlabel('Frequency (icm)')
            plt.ylabel('Spectrum (arb)')
            plt.legend()

            plt.subplot(212)
            plt.plot(f_ds,phi_ds,'b*-')
            plt.plot(f, phi,'g*-')
            plt.plot(f, np.arctan(np.imag(B)/np.real(B)),'c-')
            plt.plot(f, np.arctan(np.imag(B_corr)/np.real(B_corr)),'g-')
            #plt.plot(f, np.angle(S_corr),'ro')#np.arctan(np.imag(S_corr)/np.real(S_corr)),'ro-')
            plt.xlabel('Frequency (icm)')
            plt.ylabel('phase')
            plt.legend(('ds','interp', 'raw', 'corrected'))
            plt.show()
        return f,B_corr

    def phase_correction_forman(self,x,y,zpd_index=None,plotfig=False):
        ''' generate phase corrected spectrum using method in:

            Michael L. Forman, W. Howard Steel, and George A. Vanasse.
            Correction of Asymmetric Interferograms Obtained in Fourier Spectroscopy.
            Journal of the Optical Society of America, 56(1):59–63, 1966.

        '''
        print('FORMAN PHASE CORRECTION METHOD IS UNFINISHED')
        if zpd_index == None:
            zpd_index, zpd_val = self.get_zpd(x, y, plotfig=False)
        f_ds, phi_ds = self.get_phase_spectrum(x,y,zpd_index)
        pcf = scipy.fftpack.ifft(np.exp(-1j*phi_ds))
        Isym = np.real(signal.convolve(fftshift(y),pcf))[N//2:3*N//2]

        if plotfig:
            plt.plot(x,y,label='orig')
            plt.plot(x,Isym,label='phase corr')
            plt.legend()
            plt.show()

    def to_spectrum(self,x,y,poly_filter_deg=1,window="hanning"):
        print('UNFINISHED')

        # get phase from low res, double-sided IFG
        y_filt = self.remove_poly(x,y,poly_filter_deg)
        zpd_index, zpd = self.get_zpd(x,y_filt,plotfig=False)
        x_sym, y_sym = self.get_symmetric_ifg(x,y_filt,zpd_index,plotfig=False)
        x_sym, y_sym = self.window_and_shift(x,y_filt,window)
        f_sym, S_sym = self.get_fft(x_sym, y_sym,plotfig=False)
        theta_lowres=np.angle(S_sym)

        # get high res spectrum
        x_highres, y_highres = self.make_high_res_symmetric_ifg(x,y_filt,plotfig=False)
        x_highres, y_highres = self.window_and_shift(x_highres,y_highres,window)
        f,S = self.get_fft(x_highres,y_highres,plotfig=False)

        # do phase correction
        theta = self.interp_angle(f,f_sym,theta_lowres,debug = False)
        f,B = self.phase_correction_richards(x_highres,y_highres,theta,window,True)

        return f,B

    def to_spectrum_simple(self,x,y,v,poly_filter_deg=1,plotfig=False):

        samp_int=x[1]-x[0]
        N=len(y)
        f=scipy.fftpack.fftfreq(N,samp_int)[0:N//2]*icm2ghz
        y_filt = self.standardProcessing(x,y,v,filter_freqs_hz=None,filter_width_hz=0.5,poly_filter_deg=poly_filter_deg,plotfig=False)
        #y_filt = self.remove_poly(x,y,poly_filter_deg)
        B=np.abs(scipy.fftpack.fft(y_filt)[0:N//2])
        if plotfig:
            plt.plot(f,B,'b-',label='no window')
            plt.xlabel('Frequency (GHz)')
            plt.ylabel('Response (arb)')
            #plt.show()
        return f,B

    def to_spectrum_alt(self,x,y,poly_filter_deg=1,zpd_index=None,plotfig=False):

        y_filt = self.remove_poly(x,y,poly_filter_deg)
        x_highres, y_highres = self.make_high_res_symmetric_ifg(x,y_filt,zpd_index,plotfig=False)
        samp_int=x[1]-x[0]
        N = len(x_highres)
        f=scipy.fftpack.fftfreq(N,samp_int)[0:N//2]*icm2ghz
        B=np.abs(scipy.fftpack.fft(y_highres*np.hanning(N))[0:N//2])
        if plotfig:
            plt.plot(f,B,'b-',label='no window')
            plt.xlabel('Frequency (GHz)')
            plt.ylabel('Response (arb)')
            #plt.show()
        return f,B

    def peak_normalize(self,x,y,x_range=None):
        ''' normalize y to max(y) with in the x_range '''
        if x_range==None:
            x_range=[np.min(x),np.max(x)]
        dex1 = np.argmin(abs(x-x_range[0]))
        dex2 = np.argmin(abs(x-x_range[1]))
        y_max = np.max(y[dex1:dex2])
        return y/y_max

class FtsMeasurement(IfgToSpectrum):
    ''' Analyze a set of identical FTS sweeps '''
    def __init__(self,scan_list):
        self.scan_list = scan_list # holds all info

        # metadata stuff
        self.file_prefix = scan_list[0].file_prefix
        self.comment = scan_list[0].comment
        self.source = scan_list[0].source
        self.file_number_list = self.__get_file_numbers()
        self.speed = self.scan_list[0].speed
        self.num_scans = self.scan_list[0].num_scans
        self.num_samples = self.scan_list[0].num_samples

        # data
        self.x = self.scan_list[0].x
        self.y_mean, self.y_std = self.get_ifg_mean_and_std(remove_poly=True,poly_deg=1)

        # analysis
        self.f, self.S, self.S_mean, self.S_std = self.get_spectra_mean_and_std()
        f,S = self.get_phase_corrected_spectrum(self.x,self.y_mean,ZPD = None,
                            min_filter=None,
                            max_filter=None,
                            algorithm='Richards',
                            phase_fit_method = "interp",
                            phase_fit_degree = 1,
                            min_poly = 210/30.,
                            max_poly = 300/30.,
                            units = "wavenumber",
                            window = True,
                            debug=False)
        N = len(f)
        self.f_phase_corr = f[0:N//2]*icm2ghz
        self.S_phase_corr = S[0:N//2]

    def __get_file_numbers(self):
        file_number_list = []
        for scan in self.scan_list:
            file_number_list.append(scan.file_number)
        return file_number_list

    def plot_ifgs(self,fig_num=1):
        plt.figure(fig_num)
        y_mean_list = []
        for ii, scan in enumerate(self.scan_list):
            #plt.plot(self.x,scan.y,'.',linewidth=0.5,label=scan.current_scan)
            plt.plot(self.x,scan.y,'.',linewidth=0.5,label=scan.current_scan)
            y_mean_list.append(scan.y.mean())
        plt.errorbar(self.x, self.y_mean+np.array(y_mean_list).mean(), self.y_std,
                     color='k',linewidth=1,ecolor='k',elinewidth=1,label='mean')
        plt.legend(loc='upper right')
        plt.xlabel(scan.x_label)
        plt.ylabel(scan.y_label)
        plt.title('Interferograms for %s'%(self.file_prefix))

    def plot_spectra(self,fig_num=2):
        plt.figure(fig_num)
        plt.errorbar(self.f, self.S_mean, self.S_std, color='k',linewidth=2,ecolor='k',elinewidth=2,label='mean')
        for ii in range(self.num_scans):
            plt.plot(self.f,self.S[:,ii],'.',linewidth=0.5,label=self.scan_list[ii].current_scan)

        plt.legend(loc='upper right')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Response (arb)')
        plt.title('Spectra for %s file numbers %s - %s'%(self.file_prefix,self.file_number_list[0],self.file_number_list[-1]))

    def plot_phase_corrected_spectrum(self,fig_num=3):
        fig, axs = plt.subplots(nrows=2, ncols=1, num=fig_num)
        for ii, scan in enumerate(self.scan_list):
            axs[0].plot(scan.x,self.remove_poly(scan.x,scan.y,deg=0),'b-',linewidth=0.25,alpha=0.3)
        axs[0].errorbar(self.x, self.y_mean, self.y_std, color='k',linewidth=1,ecolor='k',elinewidth=1,label='mean')
        #axs[0].set_xlim(0, 2)
        axs[0].set_xlabel(scan.x_label)
        axs[0].set_ylabel(scan.y_label)
        axs[0].grid(True)

        axs[1].plot(self.f_phase_corr,np.abs(self.S_phase_corr))
        axs[1].plot(self.f_phase_corr,np.real(self.S_phase_corr))
        axs[1].plot(self.f_phase_corr,np.imag(self.S_phase_corr))
        axs[1].set_xlabel('Frequency (GHz)')
        axs[1].set_ylabel('Response (arb)')
        axs[1].legend(('abs','real','imag'))
        fig.suptitle('Phase corrected spectrum')

    def get_ifg_mean_and_std(self,remove_poly=True,poly_deg=1):
        ys = np.zeros((self.num_samples,self.num_scans))
        for ii, scan in enumerate(self.scan_list):
            if remove_poly:
                ys[:,ii] = self.remove_poly(scan.x,scan.y,deg=poly_deg)
            else:
                ys[:,ii] = scan.y
        y_mean = np.mean(ys,axis=1)
        y_std = np.std(ys,axis=1)
        return y_mean, y_std

    def get_spectra_mean_and_std(self,poly_filter_deg=1,window=None):
        Ss = []
        for ii in range(self.num_scans):
            scan = self.scan_list[ii]
            f,S = self.to_spectrum_simple(scan.x,scan.y,self.speed,poly_filter_deg=1,plotfig=False)
            #f,S = IfgToSpectrum().to_spectrum_alt(scan.x,scan.y,poly_filter_deg=1,zpd_index=None,plotfig=False)
            Ss.append(S)

        N = len(Ss[0])
        S = np.zeros((N,self.num_scans))
        for ii in range(self.num_scans):
            S[:,ii] = Ss[ii]
        f = f
        S_mean = np.mean(S,axis=1)
        S_std = np.std(S,axis=1)
        return f, S, S_mean, S_std

    def print_measurement_metadata(self):
        attrs = vars(self)
        exclude_list = ['scan_list','x','y_mean','y_std','f','S','S_mean','S_std','f_phase_corr','S_phase_corr']
        for key in attrs:
            if key not in exclude_list:
                print(key, '::', attrs[key])

class FtsMeasurementSet():
    ''' class for analysis of an FTS measurement set.

        INPUT:
            path: <str> to folder which holds fts data csv files

            Data files are stored in a single directory with filename structure:
            <prefix>_YrMthDay_<4 digit scan number>.csv
            The prefix is typically the readout channel row select.
            Example: rs03_210318_0002.csv is row select 3, measurement taken
            on March 18, 2021 and is the 3nd scan (zero indexed)

        ATTRIBUTES:
            path: <str> path to folder which holds fts data csv files
            filename_list: <list> of all csv files
            all_scans: <list> of FtsData instances, one per file in filename_list
            prefix_set: <set> of prefixes in the Measurement Set (in standard practice these are the row selects)
            num_scans: <int> number of individual FTS scans
            measurements: <list> of "measurements", unique instances of FtsMeasurement
            measurement_filenames: nested <list> of filenames for each "measurement"
            measurement_scan_list: nested <list> of scans for each "measurement"

        NOMENCLATURE:
            scan: one sweep of the FTS
            Measurement: N repeat scans for a given detector
            Measurement Set: a collection of scans spanning detectors and multiple scans per configuration
            

    '''
    def __init__(self, path):
        self.path = path
        if path[-1] != '/':
            self.path = path+'/'
        self.filename_list = self._get_filenames(path)
        self.all_scans = self._get_all_scans()
        self.prefix_set = self._get_prefix_set() # normally the row select
        self.num_scans = len(self.filename_list)
        self.measurements, self.measurement_filenames, self.measurement_scan_list = self._get_fts_measurements() # nested list of FTS measurement filenames
        self.num_measurements = len(self.measurements)

    # helper magic methods -----------------------------------------------------
    def _get_filenames(self,path):
        files = self._gather_csv(path)
        files = self._sort_measurements_by_number(files)
        return files

    def _gather_csv(self,path):
        files=[]
        for file in os.listdir(path):
            if file.endswith(".csv"):
                #print file.split('_')
                if file.split('_')[-1]=='ifg.csv' and 'avg' not in file:
                    files.append(file)
        return files

    def _sort_measurements_by_number(self, filename_list):
        ''' filenames in files assumed to be of the form XX_YY_ZZ_..., where ZZ is the 4 digit measurement number.
            Sort in ascending order by measurement number.
            Assumes no duplicate measurement numbers.
        '''
        runs=[]
        orderedfiles=[]
        for file in filename_list:
            runs.append(file.split('_')[2])
        for run in sorted(runs):
            for file in filename_list:
                if file.split('_')[2]==run:
                    orderedfiles.append(file)
        return orderedfiles

    def _get_all_scans(self):
        scans = []
        for file in self.filename_list:
            scans.append(load_fts_scan_fromfile(self.path+file))
        return scans

    def _get_prefix_set(self):
        prefix_list = []
        for scan in self.all_scans:
            prefix_list.append(scan.file_prefix)
        return set(prefix_list)

    def _isSingleDate(self):
        ''' returns boolean if the folder has scans taken on more than
            one date
        '''
        result = True
        date = self.filename_list[0].split('_')[1]
        for fname in self.filename_list:
            if fname.split('_')[1] != date:
                result = False
                break
        return result

    def _get_fts_measurements(self):
        ''' Group individual scans into measurements '''
        ii = 0
        fts_meas_filenames=[]
        fts_meas_scans=[]
        fts_meas = []
        while ii < self.num_scans:
            scan = self.all_scans[ii]
            meas_filenames, meas_scans = self.get_scans_from_prefix_and_filenumber(scan.file_prefix,'%04d'%(scan.file_number))
            fts_meas.append(FtsMeasurement(meas_scans))
            fts_meas_filenames.append(meas_filenames)
            fts_meas_scans.append(meas_scans)
            num_scans = scan.num_scans
            ii = ii + scan.num_scans
        return fts_meas, fts_meas_filenames, fts_meas_scans

    # helper methods -----------------------------------------------------------
    def get_measurement_indices_for_prefix(self,prefix):
        assert prefix in self.prefix_set, print(prefix,' not in prefix_set')
        idx = []
        for ii in range(self.num_measurements):
            if self.measurements[ii].file_prefix == prefix:
                idx.append(ii)
        return idx

    def get_scans_from_prefix_and_filenumber(self,prefix,num):
        assert self._isSingleDate(), "The measurement set contains data from more than one day"
        filename = prefix+'_'+self.filename_list[0].split('_')[1]+'_'+num+'_ifg.csv'
        assert filename in self.filename_list, 'filename %s does not exist in the measurement set'%filename
        dex=self.filename_list.index(filename)
        scan = self.all_scans[dex] # get the scan for the file in question

        # now find all scans for this measurement
        file_number_list = list(np.arange(scan.num_scans) + scan.file_number - scan.current_scan)
        file_names = []
        scan_list = []
        for file_number in file_number_list:
            fname = prefix+'_'+self.filename_list[0].split('_')[1]+'_%04d'%file_number+'_ifg.csv'
            file_names.append(fname)
            scan_list.append(self.all_scans[self.filename_list.index(fname)])
        return file_names, scan_list

    def print_measurement_metadata(self):
        print('\n')
        for ii in range(len(self.measurements)):
            print('Measurement number: ',ii)
            self.measurements[ii].print_measurement_metadata()
            print('\n')

    # plotting methods ---------------------------------------------------------
    def plot_all_measurements(self,fig_num=1,showfig=True,savefig=False):
        ''' plot all measurement ifgs and spectra '''
        for ii, measurement in enumerate(self.measurements):
            measurement.print_measurement_metadata()
            measurement.plot_ifgs(fig_num=fig_num+2*ii)
            measurement.plot_spectra(fig_num=fig_num+2*ii+1)
            plt.show()

    def plot_all_measurements_for_prefix(self,prefix='rs03',fig_num=1,normalize=False):
        assert prefix in list(self.prefix_set), print(prefix + 'not in prefix_set: '%self.prefix_set)
        plt.title(prefix)
        for ii, measurement in enumerate(self.measurements):
            if prefix == measurement.file_prefix:
                plt.subplot(211)
                plt.plot(measurement.x,measurement.y_mean,label=measurement.file_number_list[0])
                plt.legend(loc='upper right')
                plt.subplot(212)
                if normalize: S = IfgToSpectrum().peak_normalize(measurement.f,measurement.S_mean,x_range=[10,10000])
                else: S = measurement.S_mean
                plt.plot(measurement.f,S)
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Response (arb)')

    def plot_measurements(self,measurement_indices=None,ylog=False,normalize=True,xlim=None,phase_corrected=True,fig=None,ax=None):
        if measurement_indices is None:
            measurement_indices = list(range(len(self.measurements)))
        if np.logical_and(fig is None, ax is None):
            fig,ax = plt.subplots(1,1)
        for ii in measurement_indices:
            m=self.measurements[ii]
            if phase_corrected: x=m.f_phase_corr ; y=m.S_phase_corr
            else: x=m.f ; y=m.S_mean
            if normalize: y = IfgToSpectrum().peak_normalize(x,y,x_range=[10,10000])
            if ylog: ax.semilogy(x,y)
            else: ax.plot(x,y)
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Response (arb)')
        return fig,ax

class PassbandMetrics():
    def __cull_range(self,x,y,x_range):
        indices = np.where((x >= x_range[0]) & (x <= x_range[1]))
        return x[indices], y[indices]

    def get_passband_metrics(self,f_ghz,S,f_range_ghz):
        fc = None ; bw = None
        if S is not None:
            fc = self.calc_center_frequency(f_ghz,S,f_range_ghz)
            bw = self.calc_bandwidth(f_ghz,S,f_range_ghz)
        return fc, bw

    def calc_center_frequency(self,f_ghz,S,f_range_ghz=None,source_index=0):
        ''' calculate the center frequency of the passband as

            fc = \int_f1^f2 f S(f) f^source_index / \int_f1^f2 S(f) f^source_index df

            where S(f) is the passband, f is the frequency, and the f^source_index term
            accounts for the spectral shape of the source.

            f^-0.7, 3.6, and 0 for synchrotron, dust, and thermal BB
        '''
        if f_range_ghz is not None:
            f_ghz, S = self.__cull_range(f_ghz,S,f_range_ghz)

        fc = simpson(y=S*f_ghz*f_ghz**source_index,x=f_ghz) / simpson(y=S*f_ghz**source_index,x=f_ghz)
        return fc

    def calc_bandwidth(self,f_ghz,S,f_range_ghz=None):
        ''' calculate bandwidth of passband from equation:
            bw = [ \int_f1^f2 S(f_ghz) df ] ^2 / [ \int_f1^f2 S(f_ghz)^2 df ]
        '''
        if f_range_ghz is not None:
            f_ghz, S = self.__cull_range(f_ghz,S,f_range_ghz)

        integral_numerator = simpson(y=S, x=f_ghz) #simpson(y, x=None, dx=1, axis=-1, even='avg')
        integral_denom = simpson(y=S**2,x=f_ghz)
        return integral_numerator**2 / integral_denom

    def integrate_passband(self,f_ghz,S,f_range_ghz=None):
        ''' integrate the passband over range f_range_ghz
        '''
        if f_range_ghz is not None:
            f_ghz, S = self.__cull_range(f_ghz,S,f_range_ghz)

        return simpson(y=S, x=f_ghz)

    def calc_fwhm(self,f_ghz,S,f_range_ghz=None,debug=False):
        if f_range_ghz is not None:
            f_ghz, S = self.__cull_range(f_ghz,S,f_range_ghz)

        f_res = 0.1 # ghz resolution
        f=np.arange(f_ghz[0],f_ghz[-1],f_res)
        S=np.interp(f,f_ghz,S)

        S = S/np.max(S)
        max_dex = np.argmax(S)

        f1 = f[np.argmin(abs(S[0:max_dex]-0.5))]
        f2 = f[np.argmin(abs(S[max_dex:]-0.5))+len(S[0:max_dex])]
        if debug:
            dex1 = np.argmin(abs(S[0:max_dex]-0.5))
            dex2 = np.argmin(abs(S[max_dex:]-0.5))+len(S[0:max_dex])
            plt.plot(f,S,'o-')
            plt.plot(f1,S[dex1],'g*')
            plt.plot(f2,S[dex2],'r*')
            plt.show()
        return f2-f1,f1,f2

    def get_passband_norm(self,f_ghz,S,f_range_ghz=None):
        if f_range_ghz is not None:
            f_ghz, S = self.__cull_range(f_ghz,S,f_range_ghz)

        num = simpson(S**2,x=f_ghz)
        denom = simpson(S,x=f_ghz)
        result = num/denom
        return result

class Passband(PassbandMetrics):
    def __init__(self,f_measure_ghz,S_measure_complex,f_model_ghz=None,S_model=None,f_range_ghz=None):
        ''' S_measure_complex assumed to be phase corrected.

            input:
            f_measure_ghz: frequency vector for measured spectrum
            S_measure_complex: spectrum (complex), assumed properly phase corrected
            f_model_ghz: frequency vector for modeled passband
            S_model: response vector for modeled passband
            f_range_ghz: limits of integration (f_min,f_max) for the measured spectrum

        '''
        # measurement
        self.f_ghz, self.S_complex, self.S_norm, self.fc_measured_ghz, self.bw_measured_ghz = self._handle_input(f_measure_ghz,S_measure_complex,f_range_ghz)

        # model/simulation
        self.f_model_ghz, self.S_model, self.S_model_norm, self.fc_model_ghz, self.bw_model_ghz = self._handle_input(f_model_ghz,S_model,f_range_ghz)

        self.f_range_ghz = f_range_ghz

    def _handle_input(self,f_ghz,S,f_range_ghz):
        if S is not None:
            S_real = np.real(S)
            S_norm = self.normalize_passband(f_ghz,S_real,f_range_ghz)
            fc, bw = self.get_passband_metrics(f_ghz,S_norm,f_range_ghz)
        else:
            f_ghz = S = S_norm = fc = bw = None
        return f_ghz, S, S_norm, fc, bw

    def print_passband_metrics(self):
        attrs = vars(self)
        for key in attrs:
            if key in ['fc_measured_ghz','fc_model_ghz','bw_measured_ghz','bw_model_ghz']:
                print(key, ':: %.1f GHz'%attrs[key])

    def peak_normalize(self,a):
        return a/np.max(a)

    def plot(self,fig_num=None,normalize = True):
        # in future, and self weighted normalization
        fig,ax = plt.subplots(num=fig_num)
        legend_labels=[]
        if self.S_complex is not None:
            if normalize:
                norm = np.max(self.S_complex.real)
                ax.plot(self.f_ghz,self.peak_normalize(np.abs(self.S_complex)),'-')
                ax.plot(self.f_ghz,self.S_complex.real/norm,'-')
                ax.plot(self.f_ghz,self.S_complex.imag/norm,'-')
            else:
                ax.plot(self.f_ghz,np.abs(self.S_complex),'-')
                ax.plot(self.f_ghz,self.S_complex.real,'-')
                ax.plot(self.f_ghz,self.S_complex.imag,'-')
            legend_labels.extend(['abs','real','imag'])
        if self.S_model is not None:
            if normalize:
                ax.plot(self.f_model_ghz,self.peak_normalize(self.S_model),'k--')
            else:
                ax.plot(self.f_model_ghz,self.S_model,'k--')
            legend_labels.append('model')
        ax.legend(legend_labels)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Response (arb)')
        return fig, ax

    def normalize_passband(self,freq_ghz,S,f_range_ghz=None):
        norm = self.get_passband_norm(freq_ghz,S,f_range_ghz)
        return S/norm

    # methods for calculating radiated power emitted from a black body ------------------------------
    def pnu_thermal(self,freq_hz,temp_k):
        ''' power spectral density (W/Hz) of single mode from thermal source at
            temp_k (in K) and freq_hz (Hz).
        '''
        x = scipy.constants.h*freq_hz/(scipy.constants.k*temp_k)
        result = scipy.constants.h*freq_hz * (np.exp(x)-1)**-1
        return result

    def calc_thermal_power_rj(self,temp_k,bw_hz):
        return scipy.constants.k*temp_k*bw_hz

    def calc_thermal_power_from_tophat(self,freq_min_hz,freq_max_hz,temp_k):
        ''' Calculate the single mode thermal power (in pW) emitted from a blackbody
            at temperature temp_k (in K) from freq_min_hz to freq_max_hz assuming
            a tophat bandpass
        '''
        P = quad(self.pnu_thermal,freq_min_hz,freq_max_hz,args=(temp_k))[0] # toss the error
        return P

    def calc_thermal_power_from_passband_metrics(self,fc_ghz,bw_ghz,temp_k):
        result = self.pnu_thermal(fc_ghz*1e9,temp_k)*bw_ghz*1e9
        return result

    def calc_thermal_power_over_spectrum(self,temp_k,freq_ghz,passband,f_min_mask_ghz=None,f_max_mask_ghz=None,peak_normalize=False,debug=False):
        ''' return single-mode power (in W) from thermal source at temp_k with user
            supplied vectors describing the spectral response

            input
            temp_k: source temperature in K
            freq_ghz: frequency vector of passband
            passband: vector of passband, assumed normalized
            f_min_mask_ghz: freq_ghz < f_min_mask_ghz excluded from calculation
            f_max_mask_ghz: freq_ghz > f_max_mask_ghz excludded from calculation
            peak_normalize: <bool>
            debug: <bool>, if True show a bunch of plots

            return: float(P) (in W)
        '''
        S = passband # shorthand
        if freq_ghz[0] == 0:
            freq_ghz = freq_ghz[1:]
            S=S[1:]
        if peak_normalize:
            S=S/np.max(S)

        pnu = self.pnu_thermal(freq_ghz*1e9,temp_k) # single mode power per unit bandwidth
        # build a mask
        if f_min_mask_ghz is not None:
            mask = np.where(freq_ghz < f_min_mask_ghz, 0, 1)
        else:
            mask = np.ones(len(freq_ghz))
        if f_max_mask_ghz is not None:
            mask = np.where(freq_ghz > f_max_mask_ghz, 0,mask)

        #print(len(pnu),len(B),len(mask))
        integrand = pnu*S*mask
        P = simpson(integrand,freq_ghz*1e9)

        if debug:
            plt.figure(1)
            plt.plot(freq_ghz,pnu)
            plt.plot(freq_ghz,B*np.max(pnu))
            plt.plot(freq_ghz,mask*np.max(pnu))
            plt.legend(('Pnu','passband','mask'))
            plt.xlabel('Frequency (GHz)')
            plt.ylabel('W/Hz')
            plt.title('T_bb = %.1f K'%temp_k)

            plt.figure(2)
            plt.xlabel('Frequency (GHz)')
            plt.ylabel('Integrand (W/Hz)')
            plt.plot(freq_ghz,pnu*B*mask)
            plt.show()

        return P

    def get_PvT(self,temp_k_list,freq_ghz,passband,f_mask=None,peak_normalize=False):
        if f_mask is not None:
            f1 = f_mask[0] ; f2 = f_mask[1]
        else:
            f1 = f2 = None
        Ps=np.zeros(len(temp_k_list))
        for ii,temp_k in enumerate(temp_k_list):
            Ps[ii] = self.calc_thermal_power_over_spectrum(temp_k,freq_ghz,passband,f_min_mask_ghz=f1,f_max_mask_ghz=f2,peak_normalize=peak_normalize,debug=False)
        return Ps

    def get_PvT_from_measured_passband(self,temp_k_list,f_mask):
        assert self.S_norm is not None, print('There is no measured passband.  self.S_norm = ',self.S_norm)
        Ps = self.get_PvT(temp_k_list,self.f_ghz, self.S_norm,f_mask=f_mask,peak_normalize=False)
        return Ps

    def get_PvT_from_model_passband(self,temp_k_list,f_mask):
        assert self.S_norm is not None, print('There is no measured passband.  self.S_norm = ',self.S_norm)
        Ps = self.get_PvT(temp_k_list,self.f_model_ghz, self.S_model_norm,f_mask=f_mask,peak_normalize=False)
        return Ps

    def get_PvT_from_tophat(self,temp_k_list,freq_edges_ghz):
        Ps=np.zeros(len(temp_k_list))
        for ii,temp_k in enumerate(temp_k_list):
            Ps[ii] = self.calc_thermal_power_from_tophat(freq_min_hz=freq_edges_ghz[0]*1e9,freq_max_hz =freq_edges_ghz[1]*1e9,temp_k=temp_k)
        return Ps

    def get_PvT_from_measured_fc_and_bw(self,temp_k_list):
        assert self.f_ghz is not None
        return self.get_PvT_from_tophat(temp_k_list,freq_edges_ghz=[self.fc_measured_ghz-self.bw_measured_ghz/2,self.fc_measured_ghz+self.bw_measured_ghz/2])

    def get_dT_and_dP(self,temp_k_list,P,zero_index=0):
        dT = np.array(temp_k_list) - temp_k_list[zero_index]
        dP = np.array(P) - P[zero_index]
        return dT, dP

    def get_dT_and_dP_from_measured_passband(self,temp_k_list,f_mask,zero_index=0):
        P = self.get_PvT_from_measured_passband(temp_k_list,f_mask)
        return self.get_dT_and_dP(temp_k_list,P,zero_index=0)

    def get_dT_and_dP_from_model_passband(self,temp_k_list,f_mask,zero_index=0):
        P = self.get_PvT_from_model_passband(temp_k_list,f_mask)
        return self.get_dT_and_dP(temp_k_list,P,zero_index=0)

    def get_dT_and_dP_from_tophat(self,temp_k_list,freq_edges_ghz,zero_index=0):
        P = self.get_PvT_from_tophat(temp_k_list,freq_edges_ghz)
        return self.get_dT_and_dP(temp_k_list,P,zero_index=0)

    def get_dT_and_dP_from_measured_fc_and_bw(self,temp_k_list,zero_index):
        P = self.get_PvT_from_measured_fc_and_bw(temp_k_list)
        return self.get_dT_and_dP(temp_k_list,P,zero_index=0)

    def plot_PvTs(self,temp_k_list,f_mask,freq_edges_ghz,fig_num=1):
        if f_mask is not None:
            mask_state = "True"
            f_mask_min = f_mask[0]
            f_mask_max = f_mask[1]
        else:
            mask_state = "False"
            f_mask_min = f_mask_max = None
        P1 = self.get_PvT_from_measured_passband(temp_k_list,f_mask)
        P2 = self.get_PvT_from_model_passband(temp_k_list,f_mask)
        P3 = self.get_PvT_from_tophat(temp_k_list,freq_edges_ghz)
        P4 = self.get_PvT_from_measured_fc_and_bw(temp_k_list)
        Ps = np.array([P1,P2,P3,P4]).transpose()
        fig,ax = plt.subplots(num=fig_num)
        ax.plot(temp_k_list,Ps,'o-')
        ax.set_xlabel('BB Temp (K)')
        ax.set_ylabel('Power (W)')
        ax.grid(1)
        ax.legend(tuple(['measure','model','tophat [%.1f,%.1f],'%(freq_edges_ghz[0],freq_edges_ghz[1]),'tophat measured fc and bw']),loc='upper left')
        ax.set_title('P vs T (mask applied: %s [%.1f,%.1f])'%(mask_state,f_mask_min,f_mask_max))
        return fig,ax

class PassbandModel():
    def __init__(self,txtfilename,delimiter=','):
        self.txtfilename = txtfilename
        self.delimiter=delimiter
        self.n_freqs, self.n_cols, self.header, self.model = self.from_file(self.txtfilename,delimiter)
        self.f_ghz = self.model[:,0]

    def from_file(self, txtfilename,delimiter=','):
        model = np.loadtxt(txtfilename,skiprows=1,delimiter=delimiter)
        f=open(txtfilename,'r')
        header_raw = f.readline()
        f.close()
        header = header_raw.rstrip().split(self.delimiter)
        n_freqs, n_cols = np.shape(model)
        return n_freqs, n_cols, header, model

    def get_bandwidth(self,B,f_range_ghz):
        return PassbandMetrics().calc_bandwidth(self.f_ghz,B,f_range_ghz)

    def get_center_frequency(self,B,f_range_ghz,source_index=0):
        return PassbandMetrics().calc_center_frequency(self.f_ghz,B,f_range_ghz,source_index)

    def plot_model(self,fig_num=1):
        plt.figure(fig_num)
        for ii in list(range(1,self.n_cols)):
            print(ii)
            plt.plot(self.f_ghz,self.model[:,ii],label=self.header[ii])
        plt.xlabel(self.header[0])
        plt.ylabel('Response')
        plt.legend()
        #plt.show()

if __name__ == "__main__":
    path = '/Users/hubmayr/projects/uber_omt/data/fts/20230623/' # on Hannes' machine
    fms = FtsMeasurementSet(path)
    fms.print_measurement_metadata()
    fig,ax=fms.plot_measurements([1,3,5,6],ylog=False,phase_corrected=False)
    plt.show()

    # filename='hftv0_hft2_diplexer_model.txt'
    # pbm = PassbandModel(path+filename)
    # #pbm.plot_model()
    # #plt.show()
    #
    # d=np.load('measured_spectrum_example.npz')
    # f_ghz = d['f']
    # S = d['B']
    # f_range_ghz = [175,300]
    # # plt.plot(f_ghz,np.real(S))
    # # plt.show()
    #
    # pb = Passband(f_measure_ghz=f_ghz,S_measure_complex=S,f_model_ghz=pbm.f_ghz,S_model=pbm.model[:,2],f_range_ghz=f_range_ghz)
    # print(type(pb))
    # pb.plot_PvTs(temp_k_list=list(range(4,12)),f_mask=f_range_ghz,freq_edges_ghz=[200,275],fig_num=1)
    # pb.print_passband_metrics()
    # plt.show()

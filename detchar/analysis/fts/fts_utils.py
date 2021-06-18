'''
fts_utils.py

everything one should need to analyze data taken with the NIST FTS system
@author JH, started 03/2021

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

4) np.arctan and np.arctan2 give same behavior
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
from scipy.integrate import quad, simps

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
        # del attrs['x']
        # del attrs['y']
        for key in attrs:
            if key not in ['x','y']:
                print(key, '::', attrs[key])
        #print(', '.join("%s: %s" % item for item in attrs.items()))
        # meta_list = [file,version,file_prefix,file_number,num_scans,current_scan,resolution,nyquist,num_smaples,speed,zpd,buffer_size,date,juldate,source,comment,x_label,y_label]
        # print('file: ', self.file)


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
        y_filt = self.notch_frequencies(x,y,v,filter_freqs_hz,filter_width_hz,plotfig=False)
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

class IfgToSpectrum():

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
        f=scipy.fftpack.fftfreq(len(x),samp_int)

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

    # medium level methods --------------------------------------------------------------
    def get_double_sided_ifg(self,x,y,zpd_index=None,x_to_opd=True,
                             fftpacking=True,plotfig=False):
        ''' return only the symmetric portion of an asymmetric IFG given the index of the
            zero path difference (zpd_index).  This algorithm always returns an even number of
            data points.  It returns -\delta_max -> +\delta_max - \delta_res.
        '''
        if zpd_index == None:
            zpd_index, zpd_val = self.get_zpd(x,y)
        else:
            zpd_val = x[zpd_index]
        N=zpd_index*2
        xsym=x[0:N]
        ysym=y[0:N] # length of xsym, ysym always odd by construction
        if fftpacking:
            #xsym,ysym = self.ascending_to_standard_fftpacking(xsym,ysym,zpd_index)
            xsym = fftshift(xsym) ; ysym = fftshift(ysym)
        if x_to_opd:
            xsym = xsym - zpd_val
            x=x-zpd_val

        if plotfig:
            plt.figure(1,figsize = (12,6))
            plt.plot(x,y,label = "Entire interferogram")
            plt.plot(x[zpd_index],y[zpd_index],'ro',label='ZPD')
            plt.plot(xsym,ysym,label = "Double-sided interferogram")
            plt.xlabel('OPD (cm)')
            plt.ylabel('Detector response (arb)')
            plt.title('IFG')
            plt.legend()
            plt.show()
        return xsym,ysym

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
        x_ds,y_ds = self.get_double_sided_ifg(x,y,zpd_index,x_to_opd=True,
                                             fftpacking=True,plotfig=False) # packing is "standard"
        f_ds, B_ds = self.get_fft(x_ds, y_ds*fftshift(np.hanning(len(y_ds))), plotfig=False) # apodize ifg before FFT
        phi_ds = np.arctan(np.imag(B_ds)/np.real(B_ds)) # output between -pi/2 and pi/2
        return f_ds, phi_ds

    # high-level methods -------------------------------------------------------------------
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

        # x_ds,y_ds = self.get_double_sided_ifg(x,y,zpd_index,x_to_opd=True,
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
        tddp = TimeDomainDataProcessing()

        # get phase from low res, double-sided IFG
        y_filt = tddp.remove_poly(x,y,poly_filter_deg)
        zpd_index, zpd = self.get_zpd(x,y_filt,plotfig=False)
        x_sym, y_sym = self.get_double_sided_ifg(x,y_filt,zpd_index,plotfig=False)
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
        tddp = TimeDomainDataProcessing()
        samp_int=x[1]-x[0]
        N=len(y)
        f=scipy.fftpack.fftfreq(N,samp_int)[0:N//2]*icm2ghz
        y_filt = tddp.standardProcessing(x,y,v,filter_freqs_hz=[60,180],filter_width_hz=0.5,poly_filter_deg=poly_filter_deg,plotfig=False)
        #y_filt = tddp.remove_poly(x,y,poly_filter_deg)
        B=np.abs(scipy.fftpack.fft(y_filt)[0:N//2])
        if plotfig:
            plt.plot(f,B,'b-',label='no window')
            plt.xlabel('Frequency (GHz)')
            plt.ylabel('Response (arb)')
            #plt.show()
        return f,B

    def to_spectrum_alt(self,x,y,poly_filter_deg=1,zpd_index=None,plotfig=False):

        tddp = TimeDomainDataProcessing()
        y_filt = tddp.remove_poly(x,y,poly_filter_deg)
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

class FtsMeasurement():
    def __init__(self,scan_list):
        self.file_prefix = scan_list[0].file_prefix
        self.first_file_number = scan_list[0].file_number
        self.last_file_number = scan_list[-1].file_number
        self.comment = scan_list[0].comment
        self.source = scan_list[0].source

        self.scan_list = scan_list
        self.x = self.scan_list[0].x
        self.speed = self.scan_list[0].speed
        self.num_scans = self.scan_list[0].num_scans
        self.num_samples = self.scan_list[0].num_samples
        self.y_mean, self.y_std = self.get_ifg_mean_and_std()
        self.f, self.S, self.S_mean, self.S_std = self.get_spectra_mean_and_std()

    def plot_ifgs(self,fig_num=1):
        plt.figure(fig_num)
        plt.errorbar(self.x, self.y_mean, self.y_std, color='k',linewidth=1,ecolor='k',elinewidth=1,label='mean')
        for scan in self.scan_list:
            plt.plot(self.x,scan.y,'.',linewidth=0.5,label=scan.current_scan)
        plt.legend()
        plt.xlabel(scan.x_label)
        plt.ylabel(scan.y_label)
        plt.title('Interferograms for %s'%(self.file_prefix))

    def plot_spectra(self,fig_num=2):
        plt.figure(fig_num)
        plt.errorbar(self.f, self.S_mean, self.S_std, color='k',linewidth=2,ecolor='k',elinewidth=2,label='mean')
        for ii in range(self.num_scans):
            plt.plot(self.f,self.S[:,ii],'.',linewidth=0.5,label=self.scan_list[ii].current_scan)

        #plt.axvline(165.75)
        #plt.axvline(224.25)
        plt.legend()
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Response (arb)')
        plt.title('Spectra for %s, file numbers %s - %s'%(self.file_prefix,self.first_file_number,self.last_file_number))

    def get_ifg_mean_and_std(self):
        ys = np.zeros((self.num_samples,self.num_scans))
        for ii in range(self.num_scans):
            ys[:,ii] = self.scan_list[ii].y
        y_mean = np.mean(ys,axis=1)
        y_std = np.std(ys,axis=1)
        return y_mean, y_std

    def get_spectra_mean_and_std(self,poly_filter_deg=1,window=None):
        Ss = []
        for ii in range(self.num_scans):
            scan = self.scan_list[ii]
            f,S = IfgToSpectrum().to_spectrum_simple(scan.x,scan.y,self.speed,poly_filter_deg=1,plotfig=False)
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


class FtsMeasurementSet():
    ''' class for analysis of an FTS measurement set.
        Data files are stored in a single directory with filename structure:
        <prefix>_YrMthDay_<4 digit scan number>
        The prefix is typically the readout channel row select.
        Example: rs03_210318_0002.csv is row select 3, measurement taken
        on March 18, 2021 and is the 3nd scan (zero indexed)

        Nomenclature:
        Measurement Set: a collection of scans spanning detectors and multiple scans per configuration
        Measurement: N repeat scans for a given detector
        scan: one sweep of the FTS
    '''
    def __init__(self, path):
        self.path = path
        if path[-1] != '/':
            self.path = path+'/'
        self.filename_list = self.get_filenames(path)
        self.all_scans = self.get_all_scans()

    def plot_all_measurements(self,showfig=True,savefig=False):
        ''' plot/save all measurement ifgs and spectra '''
        ii = 0
        print('\n## scan index; file_number; prefix; n_repeat_scan; source; speed; comment ##')
        while ii < len(self.filename_list):
            scan = self.all_scans[ii]
            #fm = FtsMeasurement(self.get_scans_from_prefix_and_filenumber(scan.file_prefix,'%04d'%(ii)))
            fm = FtsMeasurement(self.get_scans_from_prefix_and_filenumber(scan.file_prefix,'%04d'%(scan.file_number)))
            print(ii, ';', scan.file_number, ';', fm.file_prefix, ';',fm.num_scans, ';',fm.source, ';',fm.speed, ';',fm.comment)
            fm.plot_ifgs(fig_num=1)
            fm.plot_spectra(fig_num=2)
            plt.show()
            num_scans = scan.num_scans
            ii = ii + num_scans
            #ii = scan.file_number + num_scans

    def get_all_scans(self):
        scans = []
        for file in self.filename_list:
            scans.append(load_fts_scan_fromfile(self.path+file))
        return scans

    def isSingleDate(self):
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

    def get_scans_from_prefix_and_filenumber(self,prefix,num):
        assert self.isSingleDate(), "The measurement set contains data from more than one day"
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
        return scan_list

    def get_filenames(self,path):
        files = self.gather_csv(path)
        files = self.sort_measurements_by_number(files)
        return files

    def gather_csv(self,path):
        files=[]
        for file in os.listdir(path):
            if file.endswith(".csv"):
                #print file.split('_')
                if file.split('_')[-1]=='ifg.csv' and 'avg' not in file:
                    files.append(file)
        return files

    def cull_index_range(self, files):
        run_nums=[]
        new_files=[]
        for ii in range(dex_start,dex_end+1):
            run_nums.append('%04d'%ii)

        for file in files:
            if file.split('_')[2] in run_nums:
                new_files.append(file)
        return new_files

    def sort_measurements_by_number(self, filename_list):
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

    def average_ifgs(self, filenames,v=5.0,deg=3, notch_freqs=[60.0,120.0,180.0,240.0,300.0,420.0,480.0,540.0], plotfig=False):
        ''' remove drift and noise pickup from several IFG and then average together '''

        for ii in range(len(filenames)):
            df = load_fts_scan_fromfile(filenames[ii])
            y = NotchFrequencies(df.x,df.y,v=df.speed,freqs=notch_freqs,df=.2,plotfig=False) # remove noise pickup
            y=RemoveDrift(x,y,deg=deg,plotfig=False) # remove detector drift
            if plotfig:
                ax1 = plt.subplot(211)
                plt.plot(x,y,label=str(i))
                plt.title('Individual IFGs post processing')
            if i==0:
                y_all = y
            else:
                y_all=np.vstack((y_all,y))

        if len(filenames) == 1:
            m = y_all * 1.
            s = y_all * 0.
        else:
            m = np.mean(y_all,axis=0)
            s = np.std(y_all,axis=0)

        if plotfig:

            plt.figure(1,figsize = (12,6))
            plt.legend()
            plt.subplot(212, sharex=ax1, sharey=ax1) #get plots to zoom together
            plt.title('Averaged response')
            #plt.errorbar(x=x, y=m, yerr=s)
            plt.plot(x,m)
            plt.xlabel('OPD (cm)')
            plt.ylabel('Response (arb)')
            plt.show()
        return x,m,s

class PassbandModel():
    def __init__(self,txtfilename):
        self.txtfilename = txtfilename
        self.n_freqs, self.n_cols, self.header, self.model = self.from_file(self.txtfilename)
        self.f_ghz = self.model[:,0]

    def from_file(self, txtfilename):
        model = np.loadtxt(txtfilename,skiprows=1)
        f=open(txtfilename,'r')
        header_raw = f.readline()
        f.close()
        header = header_raw[0:-1].split('\t')
        n_freqs, n_cols = np.shape(model)
        return n_freqs, n_cols, header, model

    def get_bandwidth(self,B,f_range_ghz):
        return PassbandMetrics().calc_bandwidth(self.f_ghz,B,f_range_ghz)

    def get_center_frequency(self,B,f_range_ghz,source_index=0):
        return PassbandMetrics().calc_center_frequency(self.f_ghz,B,f_range_ghz,source_index)

    def plot_model(self,fig_num=1):
        plt.figure(fig_num)
        for ii in range(1,self.n_cols):
            plt.plot(self.model[:,0],self.model[:,ii],label=self.header[ii])
        plt.xlabel(self.header[0])
        plt.ylabel('Response')
        plt.legend()
        plt.show()

class PassbandMetrics():
    def __cull_range(self,x,y,x_range):
        indices = np.where((x >= x_range[0]) & (x <= x_range[1]))
        return x[indices], y[indices]

    def calc_bandwidth(self,f_ghz,B,f_range_ghz=None):
        ''' calculate bandwidth of passband from equation:
            bw = [ \int_f1^f2 B(f_ghz) df ] ^2 / [ \int_f1^f2 B(f_ghz)^2 df ]
        '''
        if f_range_ghz != None:
            f_ghz, B = self.__cull_range(f_ghz,B,f_range_ghz)

        integral_numerator = simps(y=B, x=f_ghz) #simps(y, x=None, dx=1, axis=-1, even='avg')
        integral_denom = simps(y=B**2,x=f_ghz)
        return integral_numerator**2 / integral_denom

    def calc_center_frequency(self,f_ghz,B,f_range_ghz=None,source_index=0):
        ''' calculate the center frequency of the passband as

            fc = \int_f1^f2 f B(f) S(f) f / \int_f1^f2 B(f) S(f) df

            where B(f) is the spectrum, f is the frequency, S(f) is the the frequency
            dependance of the source = f^-0.7, 3.6, and 1 for synchrotron, dust, and thermal BB
        '''
        if f_range_ghz != None:
            f_ghz, B = self.__cull_range(f_ghz,B,f_range_ghz)

        fc = simps(y=B*f_ghz*f_ghz**source_index,x=f_ghz) / simps(y=B*f_ghz**source_index,x=f_ghz)
        return fc

if __name__ == "__main__":
    path = '/Users/hubmayr/projects/lbird/HFTdesign/hft_v0/measurement/fts/20210607/20210617/'
    fms = FtsMeasurementSet(path)
    fms.plot_all_measurements()

    # path = '/home/pcuser/data/lbird/20210320/fts_raw/d3/'
    # fms = FtsMeasurementSet(path)
    # #fms.plot_all_measurements()
    # scan_num_list = []
    # # start_dex = 88
    # # n_repeat_scans = 4
    # # for ii in range(2):
    # #     scan_num_list.append(list(range(start_dex+n_repeat_scans*ii,(start_dex+n_repeat_scans)+n_repeat_scans*ii)))
    # A=52
    # scan_num_list = [[25+A,26+A,27+A,28+A],[92,93,94,95]]
    # f_lims = [[175,280],[260,400]]
    # for ii in range(len(scan_num_list)):
    #     scan_indices = scan_num_list[ii]
    #     scans = []
    #     for scan_ii in scan_indices:
    #         scans.append(fms.all_scans[scan_ii])
    #     fm = FtsMeasurement(scans)
    #     dex1 = np.argmin(abs(fm.f-f_lims[ii][0]))
    #     dex2 = np.argmin(abs(fm.f-f_lims[ii][1]))
    #     norm = np.max(fm.S_mean[dex1:dex2])
    #     plt.errorbar(fm.f, fm.S_mean/norm, fm.S_std,linewidth=1,elinewidth=1,label=ii)
    #     fc = PassbandMetrics().calc_center_frequency(fm.f,fm.S_mean,f_range_ghz=f_lims[ii],source_index=0)
    #     bw = PassbandMetrics().calc_bandwidth(fm.f,fm.S_mean,f_range_ghz=f_lims[ii])
    #     print(fc,bw)
    #     plt.xlabel('Frequency (GHz)')
    #     plt.ylabel('Response (arb)')
    #     plt.legend()
    # #plt.show()
    #
    # path = '/home/pcuser/data/lbird/20210320/fts_raw/modeled_response/'
    # filename='hftv0_hft2_diplexer_model.txt'
    # pb = PassbandModel(path+filename)
    # plt.plot(pb.model[:,0],pb.model[:,2],'k--')
    # plt.plot(pb.model[:,0],pb.model[:,3],'k--')
    # plt.show()
    #
    # bw1 = pb.get_bandwidth(B=pb.model[:,2],f_range_ghz=None)
    # fc1 = pb.get_center_frequency(B=pb.model[:,2],f_range_ghz=None)
    # bw2 = pb.get_bandwidth(B=pb.model[:,3],f_range_ghz=None)
    # fc2 = pb.get_center_frequency(B=pb.model[:,3],f_range_ghz=None)
    # print(fc1,bw1,fc2,bw2)

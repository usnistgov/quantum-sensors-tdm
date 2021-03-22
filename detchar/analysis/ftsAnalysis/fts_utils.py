'''
fts_utils.py

everything one should need to analyze data taken with the NIST FTS system
@author JH, 03/2021

Notes:
1) data format of x,y np.array or list?
'''
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import dataclasses
#from dataclasses_json import dataclass_json
from typing import Any, List
import scipy
import scipy.fftpack
from scipy import signal
import scipy.constants
import os

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

    def get_spectrum(self,poly_filter_deg=1,PLOT=False):
        f,B,B_apod = IfgToSpectrum().to_spectrum_simple(self.x,self.y,poly_filter_deg=1,window="hanning",PLOT=True)
        return f,B


class TimeDomainDataProcessing():
    def remove_poly(self,x,y,deg=1):
        ''' remove polynomial from detector response
        '''
        p = scipy.polyfit(x,y,deg)
        y=y-scipy.polyval(p,x)
        return y

    def notch_frequencies(self,x,y,v,filter_freqs_hz=[60.0,120.0,180.0,240.0,300.0,420.0,480.0,540.0],
                          filter_width_hz=.1,PLOT=False):
        ''' remove power at discrete frequencies '''
        t = x/v # timestream
        samp_int=t[1]-t[0]
        y=y-y.mean()
        f,ffty = IfgToSpectrum().get_fft(t,y,PLOT=False)
        if PLOT:
            plt.plot(f,abs(ffty))

        for i in freqs:
            js1 = f[(f>(i-df/2.0)) & (f<(i+df/2.0))] # positive frequencies
            js2 = f[(f<-1*(i-df/2.0)) & (f>-1*(i+df/2.0))] # positive frequencies
            js=np.concatenate((js1,js2))
            for j in js:
                ffty[list(f).index(j)]=0

        if PLOT:
            plt.plot(f,abs(ffty))
            plt.xlabel('Frequency (Hz)')
            plt.show()

        y = scipy.fftpack.ifft(ffty)
        return y

class IfgToSpectrum():
    def find_zero_path_difference(self,x,y,PLOT=False):
        ''' find the zero path difference from the maximum intensity of the IFG '''
        z=(y-y.mean())**2
        ZPD=x[z==z.max()]
        if len(ZPD)>1:
            print('warning, more than one maximum found.  Estimated ZPD not single valued!  Using first index')
            ZPD = ZPD[0]
        dex=list(x).index(ZPD)
        if PLOT:
            plt.figure(1,figsize = (12,6))
            plt.plot(x,y,'b.-')
            plt.plot(x[dex],y[dex],'ro')
            plt.xlabel('OPD (cm)')
            plt.ylabel('Detector response (V)')
            plt.title('Location of ZPD')
            plt.show()
        return dex,ZPD

    def get_symmetric_ifg(self,x,y,zpd_index,PLOT=False):
        ''' return only the symmetric portion of an asymmetric IFG given the index of the
            zero path difference (zpd_index)
        '''
        N=zpd_index*2
        xsym=x[0:N+1]
        ysym=y[0:N+1]
        if PLOT:
            plt.figure(1,figsize = (12,6))
            plt.plot(x,y,label = "Entire interferogram")
            plt.plot(xsym,ysym,label = "Symmetric interferogram")
            plt.xlabel('OPD (cm)')
            plt.ylabel('Detector response (arb)')
            plt.title('Symmetric portion of IFG')
            plt.legend()
            plt.show()
        return xsym,ysym

    def get_fft(self,x,y,PLOT=False):
        ''' return the FFT of y and the frequencies sampled assuming equally spaced samples in x '''
        samp_int = x[1]-x[0]
        ffty=scipy.fftpack.fft(y)
        f=scipy.fftpack.fftfreq(len(x),samp_int)

        if PLOT:
            plt.figure(1,figsize = (12,6))
            plt.subplot(211)
            plt.title('Time and FFT space')
            plt.plot(x,y)
            plt.subplot(212)
            plt.plot(f,np.abs(ffty))
            plt.plot(f,np.real(ffty))
            plt.plot(f,np.imag(ffty))
            plt.legend(('abs','real','imag'))
            plt.show()
        return f,ffty

    def make_high_res_symmetric_ifg(self,x,y,zpd_index=None,PLOT=False):
        ''' force a symmetric IFG from the single sided IFG by mirroring the IFG '''
        if zpd_index==None:
            zpd_index, ZPD = self.find_zero_path_difference(x,y,PLOT=False)
        else:
            ZPD = x[zpd_index]
        x_cat = np.concatenate((-x[2*zpd_index+1:][::-1]+ZPD,x-ZPD))
        y_cat = np.concatenate((y[2*zpd_index+1:][::-1],y)) # just mirror the -delta portion not measured

        if PLOT:
             plt.plot(x,y,color= "b",linewidth=0.5,label = "raw data")
             plt.plot(x_cat,y_cat,color= "k",linewidth=0.5,label = "concatenated")
             plt.legend()
             plt.show()

        return x_cat,y_cat

    def window_and_shift(self,x,y,window="hanning"):
        x = scipy.fftpack.ifftshift(x) # packing is now 0,1,2,...N/2,N/2-1,N/2-2,...1
        if window == "hanning":
            y = scipy.fftpack.ifftshift(y*np.hanning(len(y))) # packing is now 0,1,2,...N/2,N/2-1,N/2-2,...1
        else:
            y = scipy.fftpack.ifftshift(y)
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

    def phase_correction_mertz(self,x,y,theta,window="hanning",PLOT=False):
        f,S=self.get_fft(x,y)
        return np.real(S)*np.cos(theta) + np.imag(S)*np.sin(theta)

    def phase_correction_richards(self,x,y,theta,window="hanning",PLOT=False):
        samp_int = x[1]-x[0]

        # frequency filter here first?
        boxcar = np.ones(len(y)) # built in for future filtering
        phase_ifft = np.fft.ifft(np.exp(-1j*theta)*boxcar)
        N=len(phase_ifft)

        # convolve in interferogram space and trim extra stuff from convolution
        y_corr = np.real(signal.convolve(scipy.fftpack.fftshift(y),scipy.fftpack.fftshift(phase_ifft)))[N//2:3*N//2]
        if window == "hanning":
            y_corr = y_corr*np.hanning(len(y_corr)) #appodize should be made into option
        else:
            y_corr = y_corr

        if PLOT:
            plt.figure(-3,figsize = (12,6))
            plt.subplot(211)
            plt.plot(np.arange(-N//2+1,N//2+1)*samp_int,scipy.fftpack.fftshift(y),label = "raw interferogram")
            plt.plot(np.arange(-N//2+1,N//2+1)*samp_int,y_corr, label = "phase corrected/filtered interferogram")
            plt.xlabel("Path length (cm)")
            plt.legend()
            plt.subplot(212)
            plt.plot(np.arange(-N//2+1,N//2+1)*samp_int,scipy.fftpack.fftshift(y),label = "raw interferogram")
            plt.plot(np.arange(-N//2+1,N//2+1)*samp_int,y_corr, label = "phase corrected/filtered interferogram")
            plt.xlabel("Path length (cm)")
            plt.xlim(-samp_int*100,samp_int*100)
            plt.legend()
            plt.show()

        # Do the Fourier Transform
        f, B = self. get_fft(x,scipy.fftpack.ifftshift(y_corr),PLOT=False)
        return f,B

    def to_spectrum(self,x,y,poly_filter_deg=1,window="hanning"):
        tddp = TimeDomainDataProcessing()

        # get phase from low res, double-sided IFG
        y_filt = tddp.remove_poly(x,y,poly_filter_deg)
        zpd_index, zpd = self.find_zero_path_difference(x,y_filt,PLOT=False)
        x_sym, y_sym = self.get_symmetric_ifg(x,y_filt,zpd_index,PLOT=False)
        x_sym, y_sym = self.window_and_shift(x,y_filt,window)
        f_sym, S_sym = self.get_fft(x_sym, y_sym,PLOT=False)
        theta_lowres=np.angle(S_sym)

        # get high res spectrum
        x_highres, y_highres = self.make_high_res_symmetric_ifg(x,y_filt,PLOT=False)
        x_highres, y_highres = self.window_and_shift(x_highres,y_highres,window)
        f,S = self.get_fft(x_highres,y_highres,PLOT=False)

        # do phase correction
        theta = self.interp_angle(f,f_sym,theta_lowres,debug = False)
        f,B = self.phase_correction_richards(x_highres,y_highres,theta,window,True)

        return f,B

    def to_spectrum_simple(self,x,y,poly_filter_deg=1,PLOT=False):
        tddp = TimeDomainDataProcessing()
        samp_int=x[1]-x[0]
        N=len(y)
        f=scipy.fftpack.fftfreq(N,samp_int)[0:N//2]*icm2ghz
        y_filt = tddp.remove_poly(x,y,poly_filter_deg)
        B=np.abs(scipy.fftpack.fft(y_filt)[0:N//2])
        if PLOT:
            plt.plot(f,B,'b-',label='no window')
            plt.xlabel('Frequency (GHz)')
            plt.ylabel('Response (arb)')
            #plt.show()
        return f,B

    def to_spectrum_alt(self,x,y,poly_filter_deg=1,zpd_index=None,PLOT=False):
        tddp = TimeDomainDataProcessing()
        y_filt = tddp.remove_poly(x,y,poly_filter_deg)
        x_highres, y_highres = self.make_high_res_symmetric_ifg(x,y_filt,zpd_index,PLOT=False)
        samp_int=x[1]-x[0]
        N = len(x_highres)
        f=scipy.fftpack.fftfreq(N,samp_int)[0:N//2]*icm2ghz
        B=np.abs(scipy.fftpack.fft(y_highres*np.hanning(N))[0:N//2])
        if PLOT:
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

        self.scan_list = scan_list
        self.x = self.scan_list[0].x
        self.num_scans = self.scan_list[0].num_scans
        self.num_samples = self.scan_list[0].num_samples
        self.y_mean, self.y_std = self.get_ifg_mean_and_std()
        self.f, self.S, self.S_mean, self.S_std = self.get_spectra_mean_and_std()

    def plot_ifgs(self,fig_num=1):
        plt.figure(fig_num)
        plt.errorbar(self.x, self.y_mean, self.y_std, color='k',linewidth=1,ecolor='k',elinewidth=1,label='mean')
        for scan in self.scan_list:
            plt.plot(self.x,scan.y,'.-',linewidth=0.5,label=scan.current_scan)
        plt.legend()
        plt.xlabel(scan.x_label)
        plt.ylabel(scan.y_label)
        plt.title('Interferograms for %s'%(self.file_prefix))

    def plot_spectra(self,fig_num=2):
        plt.figure(fig_num)
        plt.errorbar(self.f, self.S_mean, self.S_std, color='k',linewidth=2,ecolor='k',elinewidth=2,label='mean')
        for ii in range(self.num_scans):
            plt.plot(self.f,self.S[:,ii],'.-',linewidth=0.5,label=self.scan_list[ii].current_scan)

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
            f,S = IfgToSpectrum().to_spectrum_simple(scan.x,scan.y,poly_filter_deg=1,PLOT=False)
            #f,S = IfgToSpectrum().to_spectrum_alt(scan.x,scan.y,poly_filter_deg=1,zpd_index=None,PLOT=False)
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
        while ii < len(self.filename_list):
            scan = self.all_scans[ii]
            print(scan.file_prefix,ii)
            fm = FtsMeasurement(self.get_scans_from_prefix_and_filenumber(scan.file_prefix,'%04d'%(ii)))
            fm.plot_ifgs(fig_num=1)
            fm.plot_spectra(fig_num=2)
            plt.show()
            num_scans = scan.num_scans
            ii = ii + num_scans

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
        scan = self.all_scans[dex]
        num_scans = scan.num_scans
        file_number_list = list(np.arange(scan.num_scans) + scan.file_number - scan.current_scan)
        scan_list=[]
        for ii in file_number_list:
            scan_list.append(self.all_scans[ii])
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

    def average_ifgs(self, filenames,v=5.0,deg=3, notch_freqs=[60.0,120.0,180.0,240.0,300.0,420.0,480.0,540.0], PLOT=False):
        ''' remove drift and noise pickup from several IFG and then average together '''

        for ii in range(len(filenames)):
            df = load_fts_scan_fromfile(filenames[ii])
            y = NotchFrequencies(df.x,df.y,v=df.speed,freqs=notch_freqs,df=.2,PLOT=False) # remove noise pickup
            y=RemoveDrift(x,y,deg=deg,PLOT=False) # remove detector drift
            if PLOT:
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

        if PLOT:

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

if __name__ == "__main__":
    path = '/Users/hubmayr/projects/lbird/HFTdesign/hft_v0/measurement/fts/d3/'
    fms = FtsMeasurementSet(path)
    fms.plot_all_measurements()
    # prefix = 'rs03'
    # scans = fms.get_scans_from_prefix_and_filenumber(prefix,'0004')
    # fm = FtsMeasurement(scans)
    # fm.plot_ifgs(fig_num=1)
    # fm.plot_spectra(fig_num=2)
    # plt.show()

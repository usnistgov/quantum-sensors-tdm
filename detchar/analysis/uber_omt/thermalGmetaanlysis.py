# load pkl files from ivanalyze.py that have k,n,Tc data and make histograms
# plot G vs bolo leg length and width with fits
# also do time constant and heat capacity analysis from complex responsivity measurements
# all references to 2018/2019 are for v0 analysis FYI, we are now on v2 in 2020, and v3 in 2022
# Using this for uber_omt analysis, this is completely 100% overkill as we are only using the first 10% of this monstrosity

import sys
import os
import pickle
import pylab as pl
import numpy as np
import matplotlib
import csv
from copy import copy
from IPython import embed
from scipy.signal import savgol_filter as savitzky_golay
#from CU_code.analysis.fmux_IV import IV_analysis as cu
#from CU_code.analysis.fmux_IV.IV_analysis import KnTc
from scipy.optimize import curve_fit as curve_fit
import matplotlib.gridspec as gridspec
from scipy.optimize import fmin
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection, HandlerLine2D, HandlerPolyCollection
import matplotlib.cm as cm
import mpl_toolkits.mplot3d as m3d
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches

#import turbo_colormap_mpl as turbo
turbo = cm.turbo
turbo_colors = cm.colors.LinearSegmentedColormap.from_list('turbo', turbo.colors)
#turbo_colors = cm.colors.LinearSegmentedColormap.from_list('turbo', turbo.turbo_colormap_data)
#import ivLib

matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amssymb}')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['figure.max_open_warning'] = 100
dpi=200 # 200 nice big plots
figsize=(4,3)
bluish='#377eb8'
replotThermal = True
k_b = 1.3806503e-23 #J/K
bath_temperature = 0.1 # use this set of IV curves when there are many to choose
Tc1 = 0.171 #  Tc we thought was optimal for v0-2
Tc2 = 0.1844 # Tc that might be best for v3

# Optical Powers of LiteBIRD for each band
# v28.20.1 https://docs.google.com/spreadsheets/d/1dFGiC9bydefKYmuOvPHMKjrcmzb1y1pE_7hYDOp_6N4
Popt_lft = np.array([0.29184686, 0.24187821, 0.26855768, 0.30598544, 0.27094372,
            0.2958127, 0.3278966, 0.31625242, 0.37653296, 0.32719336, 0.30607807,0.35566829])
Popt_mft = np.array([0.35588565,0.43860747,0.42063367,0.39077456,0.35717258])
Popt_hft = np.array([0.62886291,0.4708167,0.37702823,0.2996769,0.22026075])

# CSV format of sensitivity table:
lb = [[b'Band#', 'Freq',	'Npix', 'Popt',	'EtoE eff',	'NEPph',	'NEPg',	'NEPread',	'NEPint',	'NEPext',	'NEPdet',	'NETdet',	'NETarr'],
    [1, 40,	24,	0.29184686,	0.3226757672,	5.44441647,	3.94755931,	2.67399478,	7.23706732,	4.0939035,	8.31475732,	114.6333415,	18.49887594],
    [2, 60,	24,	0.24187821,	0.4886461586,	5.26100438,	3.59376372,	2.43434099,	6.82050737,	3.85826161,	7.83616637,	65.28457555,	10.53525308],
    [3, 78,	24,	0.26855768,	0.4919743332,	5.97783229,	3.7867788,	2.5650854,	7.52687423,	4.25784305,	8.64772011,	58.61062675,	9.458249221],
    [4, 50,	12,	0.30598544,	0.4341856444,	5.71954248,	4.04204858,	2.73799985,	7.51983817,	4.25386285,	8.63963629,	72.4753818,	16.54016728],
    [5,  68,	12,	0.27094372,	0.4559045974,	5.81005536,	3.80356373,	2.57645516,	7.40688609,	4.18996751,	8.50986423,	68.81362931,	15.70449043],
    [6 , 89,	12,	0.2958127,	0.4595306319,	6.58377923,	3.97429005,	2.69210163,	8.14791639,	4.60915755,	9.36124323,	62.32555158,	14.22379605],
    [7, 68,	72,	0.3278966,	0.3325348093,	6.57637918,	4.1842691,	2.83433708,	8.29399407,	4.69179156,	9.5290737,	105.6426718,	9.842674811],
    [8, 89,	72,	0.31625242,	0.4563831619,	6.85404996,	4.10930217,	2.78355604,	8.46241983,	4.78706756,	9.72258017,	65.17769352,	6.072573055],
    [9, 119,	72,	0.37653296,	0.5649488775,	8.18115868,	4.48386626,	3.03727798,	9.81129306,	5.55010548,	11.27231753,	40.78117714,	3.799561845],
    [10, 78,	72,	0.32719336,	0.3917112096,	6.75923232,	4.17977967,	2.83129603,	8.43646945,	4.77238781,	9.69276546,	82.50853771,	7.687279127],
    [11, 100,	72,	0.30607807,	0.510751992,	6.97036698,	4.04266035,	2.73841425,	8.51046598,	4.81424656,	9.77778099,	54.87639749,	5.112806464],
    [12, 140,	72,	0.35566829,	0.58823042,	8.45472357,	4.35786451,	2.95192701,	9.95927743,	5.63381808,	11.44233862,	38.44219747,	3.581640281],
    [13, 100,	183,	0.35588565,	0.425383725,	7.61933154,	4.35919591,	2.95282887,	9.26153338,	5.23911444,	10.64069174,	71.70393057,	4.190418773],
    [14, 119,	244,	0.43860747,	0.449931357,	8.9189256,	4.83937375,	3.27809138,	10.66361361,	6.0322508,	12.25155928,	55.65435682,	2.816723802],
    [15, 140,	183,	0.42063367,	0.4580819545,	9.26864051,	4.73917962,	3.21022195,	10.89371587,	6.16241629,	12.51592665,	53.99568759,	3.15553891],
    [16, 166,	244,	0.39077456,	0.4664105821,	9.55735219,	4.56787585,	3.09418433,	11.03550848,	6.2426263,	12.67883396,	54.37090235,	2.751766861],
    [17, 195,	183,	0.35717258,	0.4590536816,	9.78367735,	4.36707054,	2.95816298,	11.1149618,	6.28757189,	12.77011888,	59.61354251,	3.48384957],
    [18, 195,	127,	0.62886291,	0.4981439326,	13.22647082,	5.7946758,	3.92519316,	14.96412174,	8.46498556,	17.19246695,	73.95998567,	5.18841811],
    [19, 235,	127,	0.4708167,	0.5266164798,	12.31481045,	5.0139163,	3.39632288,	13.72329851,	7.76306995,	15.76686961,	76.0590279,	5.335669473],
    [20, 280,	127,	0.37702823,	0.5150159968,	11.91261558,	4.48681422,	3.03927487,	13.08736427,	7.40333122,	15.03623678,	97.26498238,	6.823303053],
    [21, 337,	127,	0.2996769,	0.4790056389,	11.58180607,	4.00016388,	2.70962803,	12.54916838,	7.09888165,	14.41789678,	154.6407187,	10.84830802],
    [22, 402,	169,	0.22026075,	0.4301514353,	10.84704975,	3.42941234,	2.32301277,	11.61101829,	6.56818382,	13.34004439,	385.6860398,	23.45473745],]
ld = {} # make a dictionary from the lb spreadsheet, for each band: name each quantity and record the value
for ll in lb[1:]:
	ld[ll[0]] = {}
	for mm,nn in zip(lb[0],ll):
		ld[ll[0]][mm] = nn

def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
    return p

def NEPg(Tc,Pstop,Tb,n,LG):
    '''
    Noise equivalent power contribution from phonons
    '''
    # Greg's
    #return np.sqrt(4.*k_b*Pstop*Tc * (n+1)**2 / (2.*n+3) * (1-(Tb/Tc)**(2.*n+3)) / (1-(Tb/Tc)**(n+1))**2)*(LG/(1.+LG))
    return np.sqrt(4.*k_b*Pstop*Tc * (n+1)**2 / (2.*n+3) * (1-(Tb/Tc)**(2.*n+3)) / (1-(Tb/Tc)**(n+1))**2)
    # Bolo Calc: (agrees numerically)
    #return np.sqrt(4*self.ph.kB*psat*Tb*(((np.power((n+1),2.)/((2*n)+3))*((np.power((Tc/Tb),((2*n)+3)) - 1)/np.power((np.power((Tc/Tb),(n+1)) - 1),2.)))))
    #return np.sqrt(4*k_b*Pstop*Tb*(((np.power((n+1),2.)/((2*n)+3))*((np.power((Tc/Tb),((2*n)+3)) - 1)/np.power((np.power((Tc/Tb),(n+1)) - 1),2.)))))

def linec(x,p1,p2):
    '''
    y=p1*x+p2
    '''
    return p1*x+p2

def logmc(x,p1,p2):
    '''
    y=p1*np.log(x)+p2
    '''
    return p1*np.log(x)+p2

def line(x,p1):
    '''
    y=p1*x
    '''
    return p1*x

def oneoverx(x,p1,p2):
    '''
    a/x + c with offset
    '''
    return p1/x+p2

def oneoverx2(x,p1):
    '''
    a/x with no offset
    '''
    return np.divide(p1,x)

def powerVStemp(Tb,k,Tc,n):
    '''
    model for power vs temperature for TES
    Tb can be an array and k,n,Tc are scalars
    '''
    return k*(Tc**n-Tb**n)

def calcG(k,Tc,n):
    '''
    calculate thermal conductance from best fit PvT model
    '''
    return n*k*Tc**(n-1)

def calcGstd(k,Tc,n,kstd,nstd):
    '''
    calculate standard deviation of thermal conductance from best fit PvT model
    using propagation of errors
    '''
    return np.sqrt(((n*Tc**(n-1)*kstd)**2) + (-k*Tc**(n-1) - n*k*np.log(Tc)*Tc**(n-1))**2 * nstd**2)

def calc_loopgain(ts,hwmap):
    '''
    Calculate Loop gain
    Vary the window size of a savitzky_golay filter to get a stable estimate
    '''
    # Combine many window lengths for meta-smoothing, averaging smoothed data and obtain errorbar
    # window size, for linear order +3, for quadratic +4, etc...
    N_windows = 12 # number of windows to use in savgol_filter, larger for more smoothing,
    N_window0 = 2 # starting N of savgol_filter, low N results in noisy non-smooth results
    for cc in sorted(ts): # bayname A,B, or C
        for rr in sorted(ts[cc]): # row # 0,2,4...
            ts[cc][rr]['loopgain'] = {}
            for nn,tb in enumerate(ts[cc][rr]['temp_list_k']):

                ites = np.array(ts[cc][rr]['i_clean'])[:,nn]
                rtes = np.array(ts[cc][rr]['r_clean'])[:,nn]
                vtes = np.array(ts[cc][rr]['v_clean'])[:,nn]
                # loop gain array
                lga = np.zeros((N_windows ,len(ites)))
                R_parasitic = hwmap[cc][rr]['Autobias R']+0.0003
                Rn = np.mean(rtes[:10])
                rfrac = rtes/Rn
                for jj in range(N_windows):
                    ws = 2*jj+3+N_window0 # window size, for linear order +3, for quadratic +4, etc...
                    if len(ites[~np.isnan(ites)])<ws:
                        #print('col{} row{} tb{} jj{} ites is too short to calculate loopgain'.format(cc,rr,tb,jj))
                        # it's just A.2 that is too short, all others are fine
                        continue
                    dI = savitzky_golay(x=ites, window_length=ws, polyorder=1, deriv=1)
                    dV = savitzky_golay(x=vtes, window_length=ws, polyorder=1, deriv=1)
                    # interpolate endpoints, they end up being tiny and throw off calculation later,
                    xx = np.arange(len(dI))
                    dI = pl.interp(xx,xx[1:-1],dI[1:-1])
                    dV = pl.interp(xx,xx[1:-1],dV[1:-1])
                    # Interpolate over zeros and if interpolation left any zeros, make them the median
                    if any(dI==0):
                    		dI = pl.interp(xx,xx[dI!=0],dI[dI!=0])
                    		if any(dI==0):
                    				dI[dI==0]=np.median(dI)
                    dV0 = np.isclose(dV,0,rtol=1E-17,atol=1E-22)
                    if any(dV0):
                    		dV = pl.interp(xx,xx[~dV0],dV[~dV0])
                    		if any(dV==0):
                    				dV[dV==0]=np.median(dV)
                    lga[jj] = (1-(rtes+R_parasitic)*dI/dV)/(1+(rtes-R_parasitic)*dI/dV)
                    #pl.plot(rtes,lga[jj],'.',label=ws)
                    #lga[jj] = (1-rtes*dI/dV)/(1+rtes*dI/dV)

                LGiv = np.mean(lga,axis=0)
                LGivstd = np.std(lga,axis=0)
                LGrange = (0.5,0.9)
                rngmsk = (rfrac>LGrange[0])*(rfrac<LGrange[1]) # LGrange mask


                msf = 10. # multiplicative sensitivity factor, 100 is high enough to turn this off ... effectively, setting to 5-10 for its intended purpose
                utv = msf*matplotlib.cbook.boxplot_stats(LGiv)[0]['whishi'] # upper_threshold value
                if utv<200:
                    utv=200
                try:
                    rtes_smooth = savitzky_golay(rtes,7,1)
                except:
                    rtes_smooth = copy(rtes)
                # mask = (ll>0) * (rtes>0.01) * np.hstack([True,(np.diff(rtes_smooth)<0.005)])
                #mask = (LGiv<utv) * (LGiv>0) * (rtes>0.01) * np.hstack([True,(np.diff(rtes)>0)]) # Take out negative loop gain, negative TES resistance, where TES resistance is falling, assuming rtes is in descedning order
                mask = (LGiv<utv) * (LGiv>0) * (rfrac>0.4) * np.hstack([True,(np.diff(rtes)>-0.025)]) # Take out negative loop gain, negative TES resistance, where TES resistance is falling, assuming rtes is in descedning order
                if any(mask*rngmsk):
                    LGmax = np.max(LGiv[rngmsk*mask])
                elif any(rngmsk):
                    LGmax = np.max(LGiv[rngmsk])
                elif any(mask):
                    LGmax = np.max(LGiv[mask])
                else:
                    LGmax = np.max(LGiv)

                try:
                    LGmidx = np.where(LGmax==LGiv)[0][0]
                except:
                    print('ivanalysis could not get index of LGmax, setting to -1 and continuing')
                    LGmidx = -1
                LGrng = np.mean(LGiv[mask*rngmsk])
                LGstd = np.std(LGiv[mask*rngmsk])
                #if rr==4: embed();sys.exit()
                ts[cc][rr]['loopgain'][tb] = {'LGiv':LGiv,
                                              'LGivstd':LGivstd,
                                              'lga':lga,
                                              'LGrange':LGrange,
                                              'rngmsk':rngmsk,
                                              'LGrng':LGrng,
                                              'LGstd':LGstd,
                                              'mask':mask,
                                              'LGmax':LGmax,
                                              'LGmidx':LGmidx,
                                               }


    return ts

def plot_loopgain(ts,outpath):
    # Plot LG from IV vs Rfrac for Tbaths, one detector per plot

    for cc in sorted(ts): # bayname A,B, or C
        for rr in sorted(ts[cc]): # row # 0,2,4...
            #print(cc,rr)
            fig = pl.figure(num='{}{}'.format(cc,rr),dpi=dpi,figsize=figsize)
            ax = fig.add_subplot(1, 1, 1)
            LGs = []
            LGstds = []
            rfs = []
            LGymax = []
            masks = []
            LGranges = []
            LGrngs = []
            LGrngstds = []
            LGmaxs = []
            LGmidxs = []

            Tbaths = list(ts[cc][rr]['loopgain'].keys())
            for tb in Tbaths:
                if tb in ts[cc][rr]['temp_list_k']:
                    idx = ts[cc][rr]['temp_list_k'].index(tb) # bath temp idx for tes arrays
                else:
                    continue
                ll = ts[cc][rr]['loopgain'][tb]['LGiv']
                llstd = ts[cc][rr]['loopgain'][tb]['LGivstd']
                rtes = np.array(ts[cc][rr]['r_clean'])[:,idx]
                Rn = np.mean(rtes[:10])
                LGranges.append(ts[cc][rr]['loopgain'][tb]['LGrange'])
                LGrngs.append(ts[cc][rr]['loopgain'][tb]['LGrng'])
                LGrngstds.append(ts[cc][rr]['loopgain'][tb]['LGstd'])
                LGmaxs.append(ts[cc][rr]['loopgain'][tb]['LGmax'])
                LGmidxs.append(ts[cc][rr]['loopgain'][tb]['LGmidx'])
                LGs.append(ll)
                LGstds.append(llstd)
                rfs.append(rtes/Rn)
                mask = ts[cc][rr]['loopgain'][tb]['mask']
                masks.append(mask)
                if any(ll[mask]):
                    LGymax.append(np.max(ll[mask]))
                else:
                    LGymax.append(0)
                #bp = matplotlib.axes.Axes.boxplot(ts[cc][rr]['loopgain']['LGiv'])
            turbo_colors = cm.colors.LinearSegmentedColormap.from_list('turbo', turbo.turbo_colormap_data)
            #colors = cm.jet(np.linspace(0, 1, len(ivRunNames)))
            colors = turbo_colors(np.linspace(0, 1, len(Tbaths)))
            for tt,ll,llstd,rrr,mm,ccc,dx,y,dy,lm,li in zip(Tbaths,LGs,LGstds,rfs,masks,colors,LGranges,LGrngs,LGrngstds,LGmaxs,LGmidxs):
                #ax.plot(rrr[mask],ll[mask],marker='.',ls='-',label='{:.0f}'.format(tt*1e3),alpha=0.7,color=ccc)
                ax.errorbar(x=rrr[mm][:-1],y=ll[mm][:-1],yerr=llstd[mm][:-1],marker='.',ls='-',label='{:.0f} {:.0f}'.format(tt*1e3,y),alpha=0.5 ,color=ccc)
                ax.errorbar(x=dx,y=[y,y],yerr=[dy,dy],marker='o',ls='-',alpha=0.5 ,color=ccc)
                ax.errorbar(x=rrr[li],y=lm,marker='*',ls='None',alpha=0.6,ms=6 ,color=ccc)
            ax.set_title('Loop Gain vs Bias Point for {} {}'.format(cc,rr))
            ax.set_xlabel('Fractional Resistance')
            ax.set_ylabel('Loop Gain')
            ax.set_xlim(-0.05,1.05)
            ymax = matplotlib.cbook.boxplot_stats(LGymax)[0]['whishi']
            #if np.isnan(ymax):

            ax.set_ylim(ymin=-0.05*ymax,ymax=1.05*ymax)
            ax.grid(True)
            #ax.legend(title=r'T$_\\textrm{{bath}}$',fontsize='x-small')
            ax.legend(title='Bath T, LG',fontsize='x-small')
            thfname = os.path.join(outpath,'LoopGain_vs_Rfrac_bay{}_row{}.png'.format(cc,str(rr).zfill(2)))
            fig.savefig(fname=thfname,bbox_inches='tight')

def main():

    # Define input/output paths and files
    # tsweep_analyzed_20220216_2 pmin cut of 1E-15 W for fitting
    # 20220228 has relabled hwamp for colA, pmin cut of 1E-15 W, and Tbath cut at 170 mK

    hwmappkl ='/home/pcuser/qsp/src/nistqsptdm/detchar/analysis/uber_omt/hwmap.pkl'
    tspkl0 = '/data/uber_omt/20230421/uber_omt_PvT_output_20230505/tsweep_analyzed_170mK.pkl' # for cmb-s4 ts = {'C':{0:copy(tsraw['2'][0])}}
    tspkl1 = '/data/uber_omt/20230421/uber_omt_PvT_output_20230505/tsweep_analyzed_200mK.pkl' # for litebird
    tspkl2 = '/data/uber_omt/20230517/uber_omt_PvT_output_20230524/tsweep_analyzed_colA.pkl' # col A

    outpath = '/home/pcuser/qsp/src/nistqsptdm/detchar/analysis/uber_omt/thermalGanalysis_plots/'


    with open(hwmappkl,'rb') as opf:
        hwmap = pickle.load(opf,encoding='bytes')
    with open(tspkl0,'rb') as opf:
        tsraw0 = pickle.load(opf,encoding='bytes') 
    with open(tspkl1,'rb') as opf:
        tsraw1 = pickle.load(opf,encoding='bytes') 
    with open(tspkl2,'rb') as opf:
        tsraw2 = pickle.load(opf,encoding='bytes')               
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # # Rename row numbers
    ts = {'A':{6:copy(tsraw2['0'][0]),
    	       7:copy(tsraw2['0'][1]),
    	       8:copy(tsraw2['0'][2]),
    	       10:copy(tsraw2['0'][3]),
    	       12:copy(tsraw2['0'][4]),
    	       14:copy(tsraw2['0'][5]),
    	       16:copy(tsraw2['0'][6]),
    	       18:copy(tsraw2['0'][7]),
    	       19:copy(tsraw2['0'][8]),
    	       20:copy(tsraw2['0'][9]),
    	       21:copy(tsraw2['0'][10]),
    	       23:copy(tsraw2['0'][11]),
    	       },
    	   'C':{0:copy(tsraw0['2'][0]),
                1:copy(tsraw1['2'][1]),
                2:copy(tsraw1['2'][2]),
                4:copy(tsraw1['2'][4]),
                },
            }
    # ts = {}
    # for cc in tsraw:
    #     ts[cc] = {}
    #     for rr in tsraw[cc]:
    #         ts[cc][int(rr/2)] = tsraw[cc][rr]
    
    # calculate loop gain
    #ts = calc_loopgain(ts,hwmap)
    #plot_loopgain(ts,outpath)
    # pop out non-working channels
    # A8, A11, C1, C12
    #ts['A'].pop(0)
    # nothing left on A 2
    # bogus Tc on A 3
    # bogus Tc on A 21
    # outlier in G
    #ts['B'].pop(18)

    # Select fractional resisitance results to plot
    rfrac_select = 0.8

    # select bath temperature to plot
    temp_select = 0.1 # K

    # select the minimum fractional resistance to include in IV_analysis
    # If there's no good data above this, just ignore the rfrac_min
    rfrac_min = 0.1

    # Some channels just have a few bad rfracs so just cut those specific points
    rfrac_cuts = {#('A',1):[0.95],
                   }

    # some channels have a bad iv at a certain temperatures and certain rfracs
    temp_cuts = {('A',6):[0.16,0.165,0.17],
                 ('A',7):[0.165,0.17], 
    		     ('A',8):[0.165,0.17],
    		     ('A',10):[0.165,0.17],
    		     ('A',12):[0.16,0.165,0.17],
    		     ('A',14):[0.16,0.165,0.17],
                 ('A',16):[0.16,0.165,0.17],
                 ('A',18):[0.16,0.165,0.17],
                 ('A',19):[0.165,0.17],
                 ('A',20):[0.170],
                 ('A',21):[0.16,0.165,0.17],
                 ('A',23):[0.165,0.17],
                 ('C',4):[0.185],

                }

    # minimum power cut, this cut is applied on velma,
    # here we're just repoducing it for plotting purposes right now
    pmin_cut = 1E-15

    # Make hist of Tc's, n's, Rn's, and Rs's,
    Tcs = [] # crtical temps
    Tcstd = [] # crtical temps std
    ns = [] # thermal conductance exponent n
    nerrs = [] # errors on thermal conductance exponent n
    rns = [] # normal resistance from last point of R array
    rns2 = [] # normal resisitance from 'rs' value
    rshs = [] # shunt resitiance
    rps = [] # parasitic resistance
    mrat = [] # ratio of sc/nm slopes
    cuts = [] # Cut these rows and cols
    nsplits = [] # split #
    nsquares = [] # TES squares
    PPs = [] # powers at various Tbaths from the rfrac selected above
    PPests = [] # estimated power from the hwmap
    Gs = []   # Conductances
    KALs = [] # K/A/L
    GALs = [] # G/A/L
    ALs = []  # A/L of leg
    Vs = []   # volume of PdAu
    rdis = [] # radial distance from wafer center
    cpos = [] # chip position
    bpos = [] # bolo position
    cids = [] # chipids
    Rabs = [] # designed autobias resistance
    cols = [] # column letter
    rows = [] # row number,
    for cc in sorted(ts): # bayname A,B, or C
        for rr in sorted(ts[cc]): # row # 0,2,4...
            # Let's try to take the average of the good fits since some detectors have good data at lower rfrac than others
            # there's no one single rfrac that's perfect.
            # if we don't have a fit the results come out as [2e-08, 0.2, 4.0]
            # so pop any of those results out!
            pfit_default = [2e-08, 0.2, 4.0]
            pfits_poplist = []
            rfmin_poplist = []

            pfits = copy(ts[cc][rr]['pfits']) # these fits include data that will be cut by things like temp_cuts, we should cut the data first, then refit then go on to averaging the data, this is admittedly all overkill
            rfracs = copy(ts[cc][rr]['rn_fracs'])
            p_at_rnfrac = copy(ts[cc][rr]['p_at_rnfrac'])
            #for nn,(ppff,rrff) in enumerate(zip(pfits,rfracs)):
            #    if ppff == pfit_default:
            #        pfits_poplist.append(nn)
            #    if rrff < rfrac_min:
            #        rfmin_poplist.append(nn)
            for nn,rrff in enumerate(rfracs):
                if rrff < rfrac_min:
                    rfmin_poplist.append(nn)

            # check if the rfrac cut would eliminate all data
            if len(pfits_poplist) < len(pfits): # we have data with good fits
                if len(set(pfits_poplist+rfmin_poplist)) == len(pfits): # we don't have any data
                    rfmin_poplist = []
                    print('Cant cut rfrac_min for {} {}'.format(cc,rr))

            if len(pfits_poplist) < len(pfits): # we have data with good fits
                if len(set(pfits_poplist+rfmin_poplist)) == len(pfits): # we don't have any data
                    rfmin_poplist = []
                    print('Cant cut rfrac_min for {} {}'.format(cc,rr))

            # check cut dictionary and select which rfracs to cut
            rfselect_poplist = []
            if (cc,rr) in rfrac_cuts:
                for nn,rrff in enumerate(rfracs):
                    if rrff in rfrac_cuts[(cc,rr)]:
                        rfselect_poplist.append(nn)

            # check temp_cuts
            temp_poplist = []
            if (cc,rr) in temp_cuts:
                temps = ts[cc][rr]['temp_list_k']
                for tt in temp_cuts[(cc,rr)]:
                    if tt in temps:
                        temp_poplist.append(temps.index(tt))
                for tt in sorted(temp_poplist,reverse=True):
                    ts[cc][rr]['temp_list_k'].pop(tt)
                for pp in p_at_rnfrac:
                    for tt in sorted(temp_poplist,reverse=True):
                        pp.pop(tt)


            # Now pop them out
            for ppll in list(set(pfits_poplist+rfmin_poplist+rfselect_poplist))[::-1]:
                pfits.pop(ppll)
                rfracs.pop(ppll)
                p_at_rnfrac.pop(ppll)
            embed();foo
            pguess = np.mean(pfits,axis=0)
            pfits=[]
            # Now we can refit and build a better pfits
            for nn,(pp,rrff) in enumerate(zip(p_at_rnfrac,rfracs)):
                # pop out nans in pp
                if sum(~np.isnan(pp))<3:
                    print('there are too many nans in p_at_rnfrac to make a fit, need atleast 3 good powers, col{}, row{}, rfrac {}'.format(cc,rr,rrff))
                    continue
                tt_nonan = np.array(temps)[~np.isnan(pp)]    
                pp_nonan = np.array(pp)[~np.isnan(pp)]
                popt,pcov = curve_fit(powerVStemp, xdata = tt_nonan,
                                                 ydata = pp_nonan, p0=pguess)
                pfits.append(popt)                                  

            if pfits:
                if rfrac_select in rfracs:
                    rfidx = rfracs.index(rfrac_select)
                else:
                    print('Could not find specified rfrac_select in results dict for {} {}'.format(cc,rr))
                    rfidx = -1

                pfitmean = np.mean(pfits,axis=0)
                pfitstd = np.std(pfits,axis=0)
                thisTc = pfitmean[1]*1e3

                if (thisTc!=200.0) and (thisTc<400) and (thisTc>10):
                    k,Tc,n = pfitmean #ts[cc][rr]['pfits'][0] # W/K, K, #
                    kstd,thisTcstd,nstd = pfitstd
                    Tbline = np.linspace(0.090,thisTc*1e-3,1000)
                    # Take the average power at each Tb and fit that to model
                    p_at_rnfrac_mean = np.mean(p_at_rnfrac,axis=0)
                    if any(p_at_rnfrac_mean<pmin_cut):
                        print('we got to pmin cut on {} {}'.format(cc,rr))
                    p_at_rnfrac_std = np.std(p_at_rnfrac,axis=0)
                    p_at_rnfrac_std = p_at_rnfrac_std[p_at_rnfrac_mean>pmin_cut]
                    temp_list_cut = np.array(ts[cc][rr]['temp_list_k'])[p_at_rnfrac_mean>pmin_cut]
                    p_at_rnfrac_mean = p_at_rnfrac_mean[p_at_rnfrac_mean>pmin_cut]
                    tbidx = list(temp_list_cut).index(temp_select)

                    # combine all the powers and fit the mean
                    popt,pcov = curve_fit(powerVStemp, xdata = temp_list_cut,
                                                 ydata = p_at_rnfrac_mean,
                                                 sigma = p_at_rnfrac_std, p0=pfitmean)                           
                    thisG = calcG(*pfitmean)*1e12
                    thisGstd = calcGstd(k,Tc,n,kstd,nstd)*1e12 # pW/K
                    athisG = calcG(*popt)*1e12
                    akstd,aTcstd,anstd = np.sqrt(np.diag(pcov))
                    athisGstd = calcGstd(*popt,akstd,anstd)*1e12
                    ak,athisTc,an = popt
                    avepowerpoints = powerVStemp(temp_list_cut,*popt)
                    avepowerline = powerVStemp(Tbline,*popt)

                    # calc the leg A/L
                    AL = 4.*hwmap[cc][rr]['bolo leg width'] / hwmap[cc][rr]['bolo leg length'] # in um # height is 1um # We got 4 legs!
                    '''
                    # Now calc the theoretical n based on the v2 model n(A/L) and ask if we fix n, how does G and Tc change?
                    ntheo = 0.3275*np.log(AL)+3.1629
                    epsilon = 1E-6
                    npopt,npcov = curve_fit(powerVStemp, xdata = temp_list_cut,
		                                     ydata = p_at_rnfrac_mean,
		                                     sigma = p_at_rnfrac_std, p0=(pfitmean[0],pfitmean[1],ntheo),
		                                    bounds=((-np.inf,-np.inf,ntheo-epsilon,),
		                                                (np.inf,np.inf, ntheo+epsilon))
		                                                )                               
                    nk,nTc,nntheo = npopt
                    nTbline = np.linspace(0.090,nTc,1000)
                    nG = calcG(*npopt)*1e12
                    nkstd,nTcstd,nnstd = np.sqrt(np.diag(npcov))
                    nGstd = calcGstd(*npopt,nkstd,nnstd)*1e12
                    npowerpoints = powerVStemp(temp_list_cut,*npopt)
                    npowerline = powerVStemp(nTbline,*npopt)
                    '''
                    if replotThermal:
                        # plot P vs T vs Rfrac for good data
                        fig, ax = pl.subplots(2, 1, num='PvT {}{}'.format(cc,rr), figsize=(4,4), gridspec_kw = {'height_ratios':[2, 1]},dpi=dpi,sharex=True)
                        ax[0].errorbar(x=temp_list_cut*1e3,y=p_at_rnfrac_mean*1e12,yerr=p_at_rnfrac_std*1e12,label='Average',marker='s',ls='None',color='k',ms=5,alpha=0.7)
                        ax[0].plot(Tbline*1e3,avepowerline*1e12,marker=None,color='k',alpha=0.7)
                        ax[1].errorbar(x=temp_list_cut*1e3,y=(p_at_rnfrac_mean-avepowerpoints)*1e12,yerr=p_at_rnfrac_std*1e12,marker='s',ls='None',color='k',ms=5,alpha=0.7)
                        #turbo_colors = cm.colors.LinearSegmentedColormap.from_list('turbo', turbo.turbo_colormap_data)
                        colors = turbo_colors(np.linspace(0, 1, len(pfits)))
                        for ff,rf,pp,color in zip(pfits,rfracs,p_at_rnfrac,colors):
                            pmask = np.array(pp)>pmin_cut
                            Tbpoints = np.array(ts[cc][rr]['temp_list_k'])[pmask]*1e3
                            measured_powers = np.array(pp)[pmask]
                            rfTbline = np.linspace(0.090,ff[1],1000)
                            powerline = powerVStemp(rfTbline,*ff)
                            powerpoints = powerVStemp(Tbpoints*1e-3,*ff)
                            powerpoints = np.array([x for _, x in sorted(zip(Tbpoints, powerpoints))])
                            measured_powers = np.array([x for _, x in sorted(zip(Tbpoints, measured_powers))])
                            Tbpoints = np.sort(Tbpoints)
                            ax[0].plot(Tbpoints,measured_powers*1e12,label=rf,marker='o',ls='None',color=color,ms=5,alpha=0.7)
                            ax[0].plot(rfTbline*1e3,powerline*1e12,color=color,lw=0.8,alpha=0.7)
                            ax[1].plot(Tbpoints,(measured_powers-powerpoints)*1e12,color=color,marker='o',alpha=0.7,ms=5)

                        fig.suptitle('Power vs Temperature for Col{}, Row{}'.format(cc,rr))
                        ax[0].set_ylabel('Power (pW)')
                        ax[1].set_ylabel('Residual (pW)')
                        ax[1].set_xlabel('Temperature (mK)')
                        ax[0].grid(True)
                        ax[1].grid(True)
                        thisLegend = ax[0].legend(fontsize=7,loc='upper right')
                        thisLegend.set_title(r'R$_{\textrm{frac}}$')
                        pl.setp(thisLegend.get_title(),fontsize=7)

                        annotext = 'G(ave(P),T))\nG: {:.2f}$\pm${:.2} pW/K\nTc: {:.1f}$\pm${:.2} mK\nn: {:.2f}$\pm${:.2}'.format(athisG,athisGstd,athisTc*1e3,aTcstd*1e3,an,anstd)
                        annotext += '\nave(G(P,T))\nG: {:.2f}$\pm${:.2} pW/K\nTc: {:.1f}$\pm${:.2} mK\nn: {:.2f}$\pm${:.2}'.format(thisG,thisGstd,thisTc,thisTcstd*1e3,n,nstd)
                        ax[0].annotate(text=annotext,
                                    xy=(0.01,0.01),
                                    xycoords = 'axes fraction',
                                    fontsize=9)

                        thfname = os.path.join(outpath,'pt_plot_analyzed_bay{}_row{}.png'.format(cc,str(rr).zfill(2)))
                        pl.savefig(fname=thfname,bbox_inches='tight')
                        
                        ###########################################################################
                        '''
                        # Test if the theoretical n is a better fit than the average
                        fig2, ax2 = pl.subplots(2, 1, num='PvT ntheo {}{}'.format(cc,rr), figsize=(4,4), gridspec_kw = {'height_ratios':[2, 1]},dpi=dpi,sharex=True)
                        ax2[0].errorbar(x=temp_list_cut*1e3,y=p_at_rnfrac_mean*1e12,yerr=p_at_rnfrac_std*1e12,marker='s',ls='None',color='k',ms=5,alpha=0.7)
                        ax2[0].plot(Tbline*1e3,avepowerline*1e12,marker=None,color='k',alpha=0.7,label='n float')
                        ax2[1].errorbar(x=temp_list_cut*1e3,y=(p_at_rnfrac_mean-avepowerpoints)*1e12,yerr=p_at_rnfrac_std*1e12,marker='s',ls='None',color='k',ms=5,alpha=0.7)
                        ax2[0].plot(nTbline*1e3,npowerline*1e12,marker=None,color='r',alpha=0.7,label='n fixed')
                        ax2[1].errorbar(x=temp_list_cut*1e3,y=(p_at_rnfrac_mean-npowerpoints)*1e12,yerr=p_at_rnfrac_std*1e12,marker='s',ls='None',color='r',ms=5,alpha=0.7)

                        fig2.suptitle('Power vs Temperature for Col{}, Row{}'.format(cc,rr))
                        ax2[0].set_ylabel('Power (pW)')
                        ax2[1].set_ylabel('Residual (pW)')
                        ax2[1].set_xlabel('Temperature (mK)')
                        ax2[0].grid(True)
                        ax2[1].grid(True)
                        thisLegend2 = ax2[0].legend(fontsize=7,loc='upper right')
                        pl.setp(thisLegend.get_title(),fontsize=7)

                        annotext2 = 'G(ave(P),T))\nG: {:.2f}$\pm${:.2} pW/K\nTc: {:.1f}$\pm${:.2} mK\nn: {:.2f}$\pm${:.2}'.format(athisG,athisGstd,athisTc*1e3,aTcstd*1e3,an,anstd)
                        annotext2 += '\nn(A/L)v2\nG: {:.2f}$\pm${:.2} pW/K\nTc: {:.1f}$\pm${:.2} mK\nn: {:.2f}$\pm${:.2}'.format(nG,nGstd,nTc*1e3,nTcstd*1e3,nntheo,nnstd)
                        ax2[0].annotate(text=annotext2,
                                    xy=(0.01,0.01),
                                    xycoords = 'axes fraction',
                                    fontsize=9)

                        thfname = os.path.join(outpath,'ntheo_pt_plot_analyzed_bay{}_row{}.png'.format(cc,str(rr).zfill(2)))
                        pl.savefig(fname=thfname,bbox_inches='tight')
                        '''


                    # Append all the things
                    histtype = 'averagePower' # use the average power at different
                                        # Rfracs to calcuate G instead of taking
                                        # the average of G from the different Rfracs with individual fits
                                        # or use the theoretical n from v2 data

                    if histtype == 'averagePower':
                        k = popt[0]*1e12 # pW/K^n
                        GG = athisG #pW/K
                        Tcs.append(athisTc*1000.) # convert K to mK
                        Tcstd.append(aTcstd*1000.) # convert K to mK
                        Gs.append(athisG)
                        KALs.append(k/AL)
                        GALs.append(athisG/AL)
                        ns.append(an)
                        nerrs.append(anstd)
                    elif histtype == 'theon':
                        k = npopt[0]*1e12 # pW/K^n
                        GG = nG #pW/K
                        Tcs.append(nTc*1000.) # convert K to mK
                        Tcstd.append(nTcstd*1000.) # convert K to mK
                        Gs.append(nG)
                        KALs.append(k/AL)
                        GALs.append(GG/AL)
                        ns.append(nntheo)
                        nerrs.append(nnstd)
                    else:
                        k*=1e12 # pW/K^n
                        GG = calcG(*pfitmean)*1e12 #pW/K
                        Tcs.append(pfitmean[1]*1000.) # convert K to mK
                        Tcstd.append(pfitstd[1]*1000.) # convert K to mK
                        Gs.append(GG)
                        KALs.append(k/AL)
                        GALs.append(GG/AL)
                        ns.append(n)
                        nerrs.append(pfitstd[2])

                    rns.append(np.nanmean(np.array(ts[cc][rr]['r_clean'])[:][:10]))
                    nsplits.append(hwmap[cc][rr]['#'])
                    nsquares.append(float(hwmap[cc][rr]['Science TES width'])/float(hwmap[cc][rr]['Science TES height']))
                    PPs.append(ts[cc][rr]['p_at_rnfrac'][rfidx][tbidx])
                    ALs.append(AL)
                    #Vs.append(hwmap[cc][rr]['C']*0.5) # half micron thick
                    #rdis.append(hwmap[cc][rr]['chip distance'])
                    #cpos.append(hwmap[cc][rr]['chip position'])
                    bpos.append(hwmap[cc][rr]['bolo position'])
                    cids.append(hwmap[cc][rr]['chip id'])
                    PPests.append(hwmap[cc][rr]['Psat target (pW)'])
                    #Rabs.append(hwmap[cc][rr]['Autobias R'])
                    cols.append(cc)
                    rows.append(rr)

                    ts[cc][rr]['results'] = {'Column':cc,
                                            'Row':rr,
                                            'k (pW/K^n)': k,
                                            'G (pW/K)': GG,
                                            'Tc (mK)': Tcs[-1],
                                            'Tc_std (mK)': Tcstd[-1],
                                            'KAL (pW/K^n/um)': KALs[-1],
                                            'GAL (pW/K/um)': GALs[-1],
                                            'n': ns[-1],
                                            'n_std': nerrs[-1],
                                            'P at 100mK (pW)': PPs[-1]*1e12,
                                            'Rn (Ohms)': rns[-1],
                                            'Chip ID': cids[-1],
                                            #'Designed Autobias Resistance (Ohms)': Rabs[-1],
                                            'Bolo Split #': nsplits[-1],
                                            'TES # of squares': nsquares[-1],
                                            'Total A/L (um)': ALs[-1],
                                            #'PdAu Volume (um^3)': Vs[-1],
                                            #'chip distance (mm)': rdis[-1],
                                            #'chip position (mm)': cpos[-1],
                                            #'bolo position (mm)': bpos[-1],
                                            }
                    for key,value in zip(hwmap[cc][rr].keys(),hwmap[cc][rr].values()):
                        ts[cc][rr]['results'][key] = value

                else:
                    print('bogus Tc on {} {}'.format(cc,rr))
                    cuts.append((cc,rr))

            else:
                print('nothing left on {} {}'.format(cc,rr))
                cuts.append((cc,rr))

    rns = np.array(rns)
    nsplits = np.array(nsplits)
    nsquares = np.array(nsquares)
    Gs = np.array(Gs)
    KALs = np.array(KALs)
    GALs = np.array(GALs)
    ALs = np.array(ALs)
    #Vs = np.array(Vs)
    nsplits = np.array(nsplits)
    ns = np.array(ns)
    nerrs = np.array(nerrs)
    #rdis = np.array(rdis)
    #cpos = np.array(cpos)
    #bpos = np.array(bpos)
    cids = np.array(cids)
    Tcs = np.array(Tcs)
    PPs = np.array(PPs)
    PPests = np.array(PPests)
    #Rabs = np.array(Rabs)

    embed();sys.exit()
    ############################################################################


    ######################  HISTOGRAMS #######################################

    pl.figure('Tc',dpi=dpi)
    hhh = pl.hist(Tcs,bins=25)
    pl.title('Critical Temperature')
    pl.xlabel('T$_\mathrm{{c}}$ (mK)')
    pl.ylabel('Counts')
    #pl.xlim(195,215)
    tcstr = 'N: {}\nT$_\mathrm{{c}}$: {:.1f}$\pm${:.1f}'.format(len(Tcs),np.mean(Tcs),np.std(Tcs))
    pl.figtext(x=0.15,y=0.85,s=tcstr,ha='left',va='top',bbox=dict(facecolor='white',edgecolor='gray'))
    thfname = os.path.join(outpath,'lbirdv3_Tc_hist.png')
    pl.savefig(fname=thfname,bbox_inches='tight')


    # # Make hist of Tc's by column and chip id
    pl.figure('TcCol',dpi=dpi,figsize=(4,3))
    # make a dict with keys as column.chipid
    tcdict = {}
    for cc,id,tt in zip(cols,cids,Tcs):
        key = cc+str(id)
        if key in tcdict:
            tcdict[key].append(tt)
        else:
            tcdict[key]= [tt]
    # hhh = pl.hist([TcA,TcB,TcC],histtype='barstacked',bins=25,label=[b'Col A','Col B','Col C'])
    tclist = [tts for lls,tts in sorted(zip(tcdict.keys(),tcdict.values()))]
    labellist = sorted(tcdict.keys())
    colors = turbo_colors(np.linspace(0, 1, len(labellist)))
    hhh = pl.hist(tclist,histtype='barstacked',bins=25,label=labellist,color=colors)
    pl.title('Critical Temperature')
    pl.xlabel('T$_\mathrm{{c}}$ (mK)')
    pl.ylabel('Counts')
    #pl.xlim(195,215)
    tcstr = 'N: {}\nT$_\mathrm{{c}}$: {:.1f}$\pm${:.1f} mK'.format(len(Tcs),np.mean(Tcs),np.std(Tcs))
    pl.figtext(x=0.15,y=0.85,s=tcstr,ha='left',va='top')
    #tcleg = pl.legend(loc='lower left',fontsize=7)
    tcleg = pl.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',fontsize=7)
    tcleg.set_title('Column ChipID',prop={'size':7})
    pl.grid(True)
    tchfname = os.path.join(outpath,'lbirdv3_TcCol_hist.png')
    pl.savefig(fname=tchfname,bbox_inches='tight')

    # plot Tcs on the wafer to see Tc as fn of position
    # Do the 2D plot with colorbar
    # try to fit a surface to the data
    # regular grid covering the domain of the data
    x0,y0 = np.min(bpos,axis=0)
    x1,y1 = np.max(bpos,axis=0)
    X,Y = np.meshgrid(np.linspace(x0, x1, 20), np.linspace(y0, y1, 20))
    XX = X.flatten()
    YY = Y.flatten()
    data = np.c_[bpos[:,0],bpos[:,1],Tcs]
    order = 1    # 1: linear, 2: quadratic
    if order == 1:
        # best-fit linear plane
        A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
        C,_,_,_ = np.linalg.lstsq(A, data[:,2])    # coefficients
        # evaluate it on grid
        Z = C[0]*X + C[1]*Y + C[2]
        # or expressed using matrix/vector product
        #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)
    elif order == 2:
        # best-fit quadratic curve
        A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
        C,_,_,_ = np.linalg.lstsq(A, data[:,2])
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)

    colors = turbo_colors(sorted((Tcs-np.min(Tcs))/np.max(Tcs-np.min(Tcs)),reverse=True)) # reverse these so the colors line up with the values
    fig = pl.figure('Tc vs Wafer Position',dpi=dpi,figsize=[4.75, 3.78])
    pl.xlabel('X Distance from Wafer Center (mm)')
    pl.ylabel('Y Distance from Wafer Center (mm)')
    pl.title('T$_\mathrm{{c}}$ Uniformity')
    #pl.grid(True)
    pitch = 6.15 # mm
    dcp = np.array([pitch/2.,pitch/2.])
    for gg,rrr,sss,nn,tc in sorted(zip(Tcs,bpos,cpos,nsplits,colors)):
        pl.plot(rrr[0],sss[1],ls='None',marker='|',ms=7,color=tc,alpha=1) # use the bolo pos for x and chip pos for y
        chiprec = matplotlib.patches.Rectangle(xy=sss-dcp, width=pitch, height=pitch, lw=0.7, fill=False, ls='-' if nn<12 else '--')
        pl.gca().add_patch(chiprec)
    wafercircle = matplotlib.patches.Circle((0,0), radius=75,fill=False)
    pl.gca().add_patch(wafercircle)
    pl.axis('equal')
    pl.ylim(-85,85)
    pl.xlim(-85,85)
    pl.axvline(0,color='k',zorder=0,lw=0.4)
    pl.axhline(0,color='k',zorder=0,lw=0.4)
    # Make two fake lines for the legend of chips A vs B
    pl.plot(0,0,ls='-',lw=0.7,color='k',label='Baseline')
    pl.plot(0,0,ls='--',lw=0.7,color='k',label='Autobias')
    sm = pl.cm.ScalarMappable(cmap=turbo_colors, norm=pl.Normalize(vmin=0, vmax=1))
    sm._A = []
    thebar = fig.colorbar(sm)
    float_ticks = np.linspace(np.min(Tcs),np.max(Tcs),len(thebar.get_ticks()))
    str_ticks = ['{:.0f}'.format(tt) for tt in float_ticks]
    thebar.ax.set_yticklabels(str_ticks)
    thebar.set_label('T$_\mathrm{{c}}$ (mK)')
    # create a unit vector along the gradient of change
    ux,uy = C[:2]
    mxy = np.sqrt(ux**2+uy**2) # vector length
    ux*=-20./mxy
    uy*=-20./mxy
    arlab = 'Gradient {:.0f}$^o$'.format(np.rad2deg(np.arctan(uy/ux))) # arrow label
    arrow = pl.arrow(x=0, y=0, dx=ux, dy=uy, head_width=3, head_length=5,label=arlab,color='k')
    pl.legend(fontsize='small',loc='lower left')
    wfname = os.path.join(outpath,'lbirdv3_Tc_vs_wafer_position.png')
    pl.savefig(fname=wfname,bbox_inches='tight')


    pl.figure('n',dpi=dpi)
    hhh = pl.hist(ns,bins=25)
    pl.title('Carrier Index')
    pl.xlabel('n')
    pl.ylabel('Counts')
    tcstr = 'N: {}\nn: {:.2f}$\pm${:.2f}'.format(len(ns),np.mean(ns),np.std(ns))
    pl.figtext(x=0.15,y=0.85,s=tcstr,ha='left',va='top',bbox=dict(facecolor='white',edgecolor='gray'))
    nfname = os.path.join(outpath,'lbirdv3_n_hist.png')
    pl.savefig(fname=nfname,bbox_inches='tight')

    pl.figure('rns',dpi=dpi)
    hhh = pl.hist(rns,bins=25)
    pl.title('Normal Resistance from IV')
    pl.xlabel('Resistance ($\Omega$)')
    pl.ylabel('Counts')
    tcstr = 'N: {}\nRn: {:.2f}$\pm${:.2f}$\Omega$'.format(len(rns),np.mean(rns),np.std(rns))
    pl.figtext(x=0.15,y=0.85,s=tcstr,ha='left',va='top',bbox=dict(facecolor='white',edgecolor='gray'))
    rnfname = os.path.join(outpath,'lbirdv3_rn_hist.png')
    pl.savefig(fname=rnfname,bbox_inches='tight')

    pl.figure('Rn vs split',dpi=dpi,figsize=(4,3))
    pl.xlabel('Split #')
    pl.ylabel('Normal Resistance ($\Omega$)')
    pl.title('Normal Resistance of Splits')
    pl.grid(True)
    pl.plot(nsplits,rns,marker='o',ls='None')
    wfname = os.path.join(outpath,'lbirdv3_rnormal_v_splits.png')
    pl.savefig(fname=wfname,bbox_inches='tight')

    # Plot Rn against number of squares
    pl.figure('Rn vs squares',dpi=dpi,figsize=(4,3))
    pl.xlabel('Number of Squares')
    pl.ylabel('Total Normal Resistance ($\Omega$)')
    pl.title('Total Resistance')
    pl.grid(True)
    popt,pcov = curve_fit(line, nsquares[(nsplits<12)],rns[(nsplits<12)], p0=[1])
    #poptc,pcovc = curve_fit(linec, nsquares[(nsplits!=5)],rns[(nsplits!=5)], p0=[1,0])
    sqline = np.linspace(5.5,10,100)
    rnline = line(sqline,*popt)
    #rnlinec = linec(sqline,*poptc)
    # exclude autobias
    pl.plot(nsquares[(nsplits<12)],rns[(nsplits<12)],marker='o',ls='None',alpha=0.7,label='Baseline Autobias') #,ms=10)
    # plot autobias
    pl.plot(nsquares[(nsplits>=12)],rns[(nsplits>=12)],marker='x',ls='None',alpha=0.7,label='Autobias Splits') #,ms=10)
    #pl.plot(nsquares[(nsplits!=5) & (nsplits<12)],rns[(nsplits!=5) & (nsplits<12)],marker='o',ls='None',alpha=0.7,label='Standard') #,ms=10)
    #pl.plot(nsquares[(nsplits==5)],rns[(nsplits==5)],marker='x',ls='None',alpha=0.7,label='No e-couple')
    #pl.plot(nsquares[nsplits>11],rns[nsplits>11],marker='^',ls='None',alpha=0.7,label='Autobias')
    pl.plot(sqline,rnline,label='$\Omega/\square$={:.3f}'.format(popt[0]))
    pl.legend()
    wfname = os.path.join(outpath,'lbirdv3_rnormal_v_sqaures.png')
    pl.savefig(fname=wfname,bbox_inches='tight')

    # Plot Rn-Rabs against number of squares
    pl.figure('Rn-Rabs vs squares',dpi=dpi,figsize=(4,3))
    pl.xlabel('Number of Squares')
    pl.ylabel('Measured Normal Resistance - Designed Autobias Resistance ($\Omega$)')
    pl.title('TES Sheet Resistance')
    pl.grid(True)
    #popt,pcov = curve_fit(line, nsquares[(nsplits<12)],rns[(nsplits<12)]-Rabs[(nsplits<12)], p0=[1])
    popt,pcov = curve_fit(line, nsquares,rns-Rabs, p0=[1])
    #poptc,pcovc = curve_fit(linec, nsquares[(nsplits!=5)],rns[(nsplits!=5)], p0=[1,0])
    sqline = np.linspace(5.5,10,100)
    rnline = line(sqline,*popt)
    #rnlinec = linec(sqline,*poptc)
    # exclude autobias
    pl.plot(nsquares[(nsplits<12)],rns[(nsplits<12)],marker='o',ls='None',alpha=0.7,label='Baseline Autobias') #,ms=10)
    # plot autobias
    pl.plot(nsquares[(nsplits>=12)],rns[(nsplits>=12)],marker='x',ls='None',alpha=0.7,label='Autobias Splits') #,ms=10)
    #pl.plot(nsquares[(nsplits!=5) & (nsplits<12)],rns[(nsplits!=5) & (nsplits<12)],marker='o',ls='None',alpha=0.7,label='Standard') #,ms=10)
    #pl.plot(nsquares[(nsplits==5)],rns[(nsplits==5)],marker='x',ls='None',alpha=0.7,label='No e-couple')
    #pl.plot(nsquares[nsplits>11],rns[nsplits>11],marker='^',ls='None',alpha=0.7,label='Autobias')
    pl.plot(sqline,rnline,label='$\Omega/\square$={:.3f}'.format(popt[0]))
    pl.legend()
    wfname = os.path.join(outpath,'lbirdv3_rnormal-rabs_v_sqaures.png')
    pl.savefig(fname=wfname,bbox_inches='tight')

    # Plot Pturn vs the estimated P from the hwmap vs index, is the hwmap wrong?
    pl.figure('Psat Pestimate vs index',dpi=dpi,figsize=(4,3))
    pl.xlabel('Array Index')
    pl.ylabel('Power (pw)')
    pl.title('Comparing Measured Power to Design Estimate')
    pl.grid(True)
    pl.plot(PPests,marker='o',ls='-',alpha=0.7,label='Design Estimates') #,ms=10)
    #pl.plot(PPs[:,0]*1e12,marker='o',ls='-',alpha=0.7,label='Measured Power')
    pl.plot(PPs*1e12,marker='o',ls='-',alpha=0.7,label='Measured Power')
    pl.legend()
    pl.ylim([-0.2,4.5])
    wfname = os.path.join(outpath,'lbirdv3_psat_pest_compare.png')
    pl.savefig(fname=wfname,bbox_inches='tight')

    # Plot Pturn vs the estimated P from the hwmap vs each other, mark out psat=2.5popt
    pl.figure('Psat vs Pestimate',dpi=dpi,figsize=(4,4))
    pl.xlabel('Designed Saturation Power (pW)')
    pl.ylabel('Measured Saturation Power (pw)')
    pl.title('Comparing Measured Power to Design Estimate')
    pl.grid(True)
    pl.plot(PPests,PPs*1e12,marker='o',ls='None',alpha=0.7)#,label='Design Estimates') #,ms=10)
    pl.plot(np.zeros(2),np.ones(2),color='k')
    #pl.plot(PPs[:,0]*1e12,marker='o',ls='-',alpha=0.7,label='Measured Power')
    #pl.plot(PPs*1e12,marker='o',ls='-',alpha=0.7,label='Measured Power')
    #pl.legend()
    xymax = 5
    pl.ylim([-0,xymax])
    pl.xlim([-0,xymax])
    pl.plot([0,xymax],[0,xymax],color='k',ls='-')
    pl.plot([0,xymax],[0,xymax/2.5],color='k',ls='-')
    pl.plot([0,xymax],[0,xymax/2.5*3],color='k',ls='-')
    pl.axis('equal')
    wfname = os.path.join(outpath,'lbirdv3_psat_vs_pest_compare.png')
    pl.savefig(fname=wfname,bbox_inches='tight')


    # The thin and thick legs have different n, so this has more scatter than G where n is factored out
    pl.figure('K/A/L vs split',dpi=dpi,figsize=(4,3))
    pl.xlabel('Bolo Split (\#)')
    pl.ylabel('K/A/L (pW/K$^n$/$\mu$m)')
    pl.title('SiN Leg Conductivity')
    pl.grid(True)
    pl.plot(nsplits,KALs,marker='o',ls='None',alpha=0.7)
    pl.axhline(np.mean(KALs),color='r',label='{:.1f}$\pm${:.1f} pW/K$^n$/$\mu$m'.format(np.mean(KALs),np.std(KALs)))
    pl.ylim(0,4000)
    pl.legend()
    wfname = os.path.join(outpath,'lbirdv3_SiN_k_conductivity_vs_split.png')
    pl.savefig(fname=wfname,bbox_inches='tight')

    pl.figure('G/A/L vs split',dpi=dpi,figsize=(4,3))
    pl.xlabel('Bolo Split (\#)')
    pl.ylabel('G/A/L (pW/K/$\mu$m)')
    pl.title('SiN Leg Conductivity')
    pl.grid(True)
    pl.plot(nsplits,GALs,marker='o',ls='None',alpha=0.7)
    pl.axhline(np.mean(GALs),color='r',label='{:.1f}$\pm${:.1f} pW/K/$\mu$m'.format(np.mean(GALs),np.std(GALs)))
    #pl.ylim(0,800)
    pl.legend()
    wfname = os.path.join(outpath,'lbirdv3_SiN_conductivity_vs_split.png')
    pl.savefig(fname=wfname,bbox_inches='tight')

    pl.figure('G/A/L vs A/L',dpi=dpi,figsize=(4,3))
    pl.xlabel('A/L ($\mu$m)')
    pl.ylabel('G/A/L (pW/K/$\mu$m)')
    pl.title('SiN Leg Conductivity')
    pl.grid(True)
    pl.plot(ALs,GALs,marker='o',ls='None',alpha=0.7)
    pl.axhline(np.mean(GALs),color='r',label='{:.1f}$\pm${:.1f} pW/K/$\mu$m'.format(np.mean(GALs),np.std(GALs)))
    #pl.ylim(0,800)
    # Holup just fit points with A/L<0.16
    # ALlo = 0.16
    # ALlinelo = np.linspace(0,ALlo,100)
    # cpopt,cpcov = curve_fit(linec, ALs[ALs<ALlo],GALs[ALs<ALlo]*1e12, p0=[1,0])
    # GALlinelo = linec(ALlinelo,*cpopt)
    # pl.plot(ALlinelo,GALlinelo)
    pl.legend(fontsize=9)
    wfname = os.path.join(outpath,'lbirdv3_SiN_conductivity_vs_AL.png')
    pl.savefig(fname=wfname,bbox_inches='tight')



    # pl.figure('rns2',dpi=dpi)
    # hhh = pl.hist(rns2,bins=25)
    # pl.title('Normal Resistance from converted data')
    # pl.xlabel('Resistance ($\Omega$)')
    # pl.ylabel('Counts')
    # rssstr = 'N: {}\nRn: {:.2f}$\pm${:.2f}$\Omega$'.format(len(rns2),np.mean(rns2),np.std(rns2))
    # pl.figtext(x=0.15,y=0.85,s=rssstr,ha='left',va='top',bbox=dict(facecolor='white',edgecolor='gray'))
    # #pl.annotate(text=rssstr,xytext=(0.15,0.85),bbox=dict(facecolor='white',edgecolor='gray'))
    # rnfname = os.path.join(outpath,'lbirdv3_rn2_hist.png')
    # pl.savefig(fname=rnfname,bbox_inches='tight')
    #
    # pl.figure('rshs',dpi=dpi)
    # hhh = pl.hist(rshs,bins=25)
    # pl.title('Shunt Resistance')
    # pl.xlabel('Resistance ($\Omega$)')
    # pl.ylabel('Counts')
    # rssstr = 'N: {}\nRsh: {:.2f}$\pm${:.2f}m$\Omega$'.format(len(rshs),np.mean(rshs)*1e3,np.std(rshs)*1e3)
    # pl.figtext(x=0.15,y=0.85,s=rssstr,ha='left',va='top',bbox=dict(facecolor='white',edgecolor='gray'))
    # #pl.annotate(text=rssstr,xytext=(0.15,0.85),bbox=dict(facecolor='white',edgecolor='gray'))
    # rnfname = os.path.join(outpath,'lbirdv3_rsh_hist.png')
    # pl.savefig(fname=rnfname,bbox_inches='tight')
    #
    # pl.figure('rps',dpi=dpi)
    # hhh = pl.hist(rps,bins=25)
    # pl.title('Parasitic Resistance')
    # pl.xlabel('Resistance ($\Omega$)')
    # pl.ylabel('Counts')
    # rssstr = 'N: {}\nRp: {:.2f}$\pm${:.2f}m$\Omega$'.format(len(rshs),np.mean(rps)*1e3,np.std(rps)*1e3)
    # pl.figtext(x=0.15,y=0.85,s=rssstr,ha='left',va='top',bbox=dict(facecolor='white',edgecolor='gray'))
    # #pl.annotate(text=rssstr,xytext=(0.15,0.85),bbox=dict(facecolor='white',edgecolor='gray'))
    # rnfname = os.path.join(outpath,'lbirdv3_rp_hist.png')
    # pl.savefig(fname=rnfname,bbox_inches='tight')
    #
    # pl.figure('mrat',dpi=dpi)
    # hhh = pl.hist(mrat,bins=25)
    # pl.title('Ratio of Superconducting to Normal Branch Slopes')
    # pl.xlabel('SC/NM Ratio')
    # pl.ylabel('Counts')
    # rssstr = 'N: {}\nSC/NM: {:.2f}$\pm${:.2f}'.format(len(mrat),np.mean(mrat),np.std(mrat))
    # pl.figtext(x=0.15,y=0.85,s=rssstr,ha='left',va='top',bbox=dict(facecolor='white',edgecolor='gray'))
    # #pl.annotate(text=rssstr,xytext=(0.15,0.85),bbox=dict(facecolor='white',edgecolor='gray'))
    # rnfname = os.path.join(outpath,'lbirdv3_mrat_hist.png')
    # pl.savefig(fname=rnfname,bbox_inches='tight')

    # pl.figure('Rp from mrat',dpi=dpi)
    # rpmrat = np.array(rns)/np.array(mrat)*1000.
    # hhh = pl.hist(rpmrat,bins=25)
    # pl.title('Parasitic Resistance from Ratio of Slopes * Rnormal')
    # pl.xlabel('Resistance (m$\Omega$)')
    # pl.ylabel('Counts')
    # rssstr = 'N: {}\nR: {:.1f}$\pm${:.1f} m$\Omega$'.format(len(rpmrat),np.mean(rpmrat),np.std(rpmrat))
    # pl.figtext(x=0.15,y=0.85,s=rssstr,ha='left',va='top',bbox=dict(facecolor='white',edgecolor='gray'))
    # #pl.annotate(text=rssstr,xytext=(0.15,0.85),bbox=dict(facecolor='white',edgecolor='gray'))
    # rnfname = os.path.join(outpath,'lbirdv3_rpmrat_hist.png')
    # pl.savefig(fname=rnfname,bbox_inches='tight')

    # # Make hist of Tc's by column
    # TcA=[]
    # TcB=[]
    # TcC=[]
    # for cc,tt in zip([colA,colB,colC],[TcA,TcB,TcC]):
    #     kk = list(cc.keys())[0] # get bay name
    #     for rr in cc[kk]:
    #         tt.append(cc[kk][rr][b'iv'][b'conductance'][b'fit parameters'][b'T'][rfidx]*1000.)
    #
    # pl.figure('TcCol',dpi=dpi,figsize=(4,3))
    # hhh = pl.hist([TcA,TcB,TcC],histtype='barstacked',bins=25,label=[b'Col A','Col B','Col C'])
    # pl.title('Critical Temperature')
    # pl.xlabel('T$_\mathrm{{c}}$ (mK)')
    # pl.ylabel('Counts')
    # pl.xlim(195,215)
    # tcstr = 'N: {}\nT$_\mathrm{{c}}$: {:.1f}$\pm${:.1f}'.format(len(Tcs),np.mean(Tcs),np.std(Tcs))
    # pl.figtext(x=0.15,y=0.85,s=tcstr,ha='left',va='top')
    # pl.legend(loc=6)
    # pl.grid(True)
    #
    # tchfname = os.path.join(outpath,'lbirdv3_TcCol_hist.png')
    # pl.savefig(fname=tchfname,bbox_inches='tight')

    ###########################################################################
    # Exit replotThermal option

    # # Plot Pturn vs Leg length for leg splits
    # legl = [] # leg length
    # legp = [] # leg-split powers
    # legpest = [] # psat estimates
    # legcorow = [] # get column row of the bolo
    # legbolos = [6,9,11,12,18]
    # widw = [] # widths
    # widp = [] # psat of width splits
    # widpest = [] # psat estimates
    # widcorow = [] # get column row of the bolo
    # widbolos = [7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23]
    # rns=[] # normal_resistance
    # for cc in [colA,colB,colC]:
    #     kk = list(cc.keys())[0] # get bay name
    #     for rr in cc[kk]:
    #         col = kk.split(b'Bay')[1]
    #         row = int(rr.split(b'Row')[1])
    #         #print 'col,row,split',col,row,hwmap[col][row][b'Split']
    #         if hwmap[col][row][b'Split'] in legbolos:
    #             psats = cc[kk][rr][b'iv'][b'conductance'][b'Psats']
    #             tbs = cc[kk][rr][b'iv'][b'conductance'][b'Tbaths']
    #             for tt,pp in zip(tbs,psats):
    #                 if np.isclose(0.1,tt,rtol=1E-2,atol=1E-3):
    #                     legp.append(pp[0]*1e12)
    #                     legl.append(hwmap[col][row][b'bolo leg length'])
    #                     legpest.append(hwmap[col][row][b'Psat est'])
    #                     legcorow.append((col,row))
    #         if hwmap[col][row][b'Split'] in widbolos:
    #             psats = cc[kk][rr][b'iv'][b'conductance'][b'Psats']
    #             tbs = cc[kk][rr][b'iv'][b'conductance'][b'Tbaths']
    #             for tt,pp in zip(tbs,psats):
    #                 if np.isclose(0.1,tt,rtol=1E-2,atol=1E-3):
    #                     widp.append(pp[0]*1e12)
    #                     widw.append(hwmap[col][row][b'bolo leg width']+hwmap[col][row][b'w_leg4_bias']/4.)
    #                     widpest.append(hwmap[col][row][b'Psat est'])
    #                     widcorow.append((col,row))
    #
    # ###########################################################################
    # # Rescale Psat if Tc were correct
    # nth = 2.6
    # Tcr = 208
    # Tcrescale = (Tcr**nth-100**nth)/(171**nth-100**nth)
    #
    # xmaxlegl = 1200.
    # xmaxlegw = 60.
    # if replotThermal:
    #     # Do a length fit, plot it with data and estimates
    #     apopt,apcov = curve_fit(oneoverx2, legl, legp, p0=[1])
    #     #apoptcu,apcovcu = curve_fit(oneoverx2, culegl, culegp, p0=[1])
    #     #apopt,apcov = iv.curve_fit(iv.oneoverx, ax, ay, p0=[1,0],sigma=ayerr,absolute_sigma=False)
    #     lengths = np.arange(0,xmaxlegl,1)
    #     apturns = oneoverx2(lengths,*apopt)
    #     apturnsrs = oneoverx2(lengths,apopt[0]/Tcrescale)
    #     #apturnscu = oneoverx2(lengths,*apoptcu)
    #     pl.figure('length split',dpi=dpi,figsize=(4,3))
    #     pl.plot(legl,legp,ls='',marker='o',color=bluish,alpha=0.7) #,label='Data')
    #     #pl.plot(lesmlegl,lesmlegp,ls='',marker='>',color='lightgreen',alpha=0.7,label='  Less Metal on Legs')
    #     pl.plot(lengths,apturns,color='r',alpha=0.7 ,label='Fit $y=C/x$')
    #     pl.plot(lengths,apturnsrs,color='g',alpha=0.7 ,label='Fit Tc Rescaled')
    #     # for debuggin purposes, you can label each point with its readout ID
    #     # for corow,ll,pp in zip(legcorow,legl,legp):
    #     #    pl.text(x=ll, y=pp, s=corow[0]+str(corow[1]))
    #
    #     #pl.plot(culegl,culegp,ls='',marker='o',color=bluish,alpha=0.7,label='CU FDM Data')
    #     #pl.plot(lengths,apturnscu,color=bluish,alpha=0.7)#,label='CU FDM Fit')
    #     #pl.plot(legl,legpest,ls='',marker='s',color='r',alpha=0.7,label='Designed T$_\mathrm{{c}}$ 171mK')
    #     #pl.plot(legl,np.array(legpest)*Tcrescale2,ls='',marker='s',color='orange',alpha=0.7,label='Scaled DE if T$_\mathrm{{c}}$ {}mK, n {}'.format(Tcr2,mth))
    #     #pl.plot(legl,np.array(legpest)*Tcrescale,ls='',marker='s',color='y',alpha=0.7,label='Scaled DE if T$_\mathrm{{c}}$ {}mK, n {}'.format(Tcr,nth))
    #
    #     pstring = '$C=${:.0f}$\pm${:.0f} pW$\cdot \mu$m'.format(apopt[0],np.sqrt(apcov[0][0]))
    #     pl.annotate(text=pstring,xy=(650,10),va='bottom',ha='left',bbox=dict(facecolor='white',edgecolor='gray',alpha=0.7)) # (0.05*max(legl),0.07*max(legp))
    #     pl.title('Bolo Leg Length Split')
    #     pl.xlabel('Leg Length ($\mu$m)')
    #     pl.ylabel('Power (pW)')
    #     pl.ylim(0,17.5) #1.1*max(legp+lesmlegp+culegp+legpest))
    #     pl.xlim(0,xmaxlegl)
    #     pl.grid(True)
    #     pl.legend()
    #     lfname = os.path.join(outpath,'lbirdv3_psat_v_leglength.png')
    #     pl.savefig(fname=lfname,bbox_inches='tight')
    #
    #     # Do a width fit, plot it with data and estimates
    #     bpopt,bpcov = curve_fit(line, widw, widp, p0=[1])
    #     #bpoptcu,bpcovcu = curve_fit(line, cuwidw, cuwidp, p0=[1])
    #     #bpopt,bpcov = iv.curve_fit(iv.oneoverx, bx, by, p0=[1,0],sigma=byerr,absolute_sigma=False)
    #     widths = np.arange(0,xmaxlegw,1)
    #     bpturns = line(widths,*bpopt)
    #     #bpturnscu = line(widths,*bpoptcu)
    #     pl.figure('width split',dpi=dpi,figsize=(4,3))
    #     pl.plot(widw,widp,ls='',marker='o',color=bluish,alpha=0.7)#,label='Data')
    #     pl.plot(widths,bpturns,color='r',alpha=0.7, label='Fit $y=mx$')
    #     pl.plot(widths,bpturns/Tcrescale,color='g',alpha=0.7, label='Fit Tc Rescaled')
    #     # for debuggin purposes, you can label each point with its readout ID
    #     #for corow,ww,pp in zip(widcorow,widw,widp):
    #     #    pl.text(x=ww, y=pp, s=corow[0]+str(corow[1]))
    #
    #     #pl.plot(cuwidw,cuwidp,ls='',marker='o',color=bluish,alpha=0.7,label='CU FDM Data')
    #     #pl.plot(widths,bpturnscu,color=bluish,alpha=0.7)# CU Fit
    #     #pl.plot(widw,widpest,ls='',marker='s',color='r',alpha=0.7,label='Designed Estimate (DE) T$_\mathrm{{c}}$ 171mK')
    #     #pl.plot(widw,np.array(widpest)*Tcrescale2,ls='',marker='s',color='orange',alpha=0.7,label='Scaled DE if T$_\mathrm{{c}}$ {}mK, n {}'.format(Tcr2,mth))
    #     #pl.plot(widw,np.array(widpest)*Tcrescale,ls='',marker='s',color='y',alpha=0.7,label='Scaled DE if T$_\mathrm{{c}}$ {}mK, n {}'.format(Tcr,nth))
    #
    #     bstring = 'm={:.1f}$\pm${:.1f} fW/$\mu$m'.format(bpopt[0]*1e3,np.sqrt(bpcov[0][0])*1e3)
    #     pl.annotate(text=bstring,xy=(2,6),bbox=dict(facecolor='white',edgecolor='gray',alpha=0.7))
    #     pl.title('Bolo Leg Width Split')
    #     pl.xlabel('Leg Width ($\mu$m)')
    #     pl.ylabel('Power (pW)')
    #     pl.ylim(0,10)
    #     pl.xlim(0,xmaxlegw)
    #     pl.grid(True)
    #     pl.legend()
    #     wfname = os.path.join(outpath,'lbirdv3_psat_v_legwidth.png')
    #     pl.savefig(fname=wfname,bbox_inches='tight')
    #
    # ####################################################################
    # # Compare the paraistic to expected autobias resistance
    # abbolo = {'BayA':[b'Row12', b'Row13', b'Row14', b'Row15', b'Row16',
    #                   b'Row17', b'Row18', b'Row19', b'Row20', b'Row21',
    #                   b'Row22', b'Row23'],
    #            'BayC':[b'Row13', b'Row14', b'Row15', b'Row16', b'Row17',
    #                    b'Row18', b'Row19', b'Row20', b'Row21',
    #                    b'Row22', b'Row23'],
    #           }
    # # Gather Rx and expected Rab
    # Rxs = [] # paraistic
    # Rabs = [] # expected autobias resistance
    # bolon = [] # bolo number
    # sqrs = [] # number of squares used in autobias
    # for bay in abbolo:
    #     for bb in abbolo[bay]:
    #         bytebay = bytes(bay[-1],'ascii')
    #         iirr = int(bb.split(b'Row')[1])
    #         Rxs.append(ts[bay][bb][b'iv'][b'converted_data'][b'iv000'][4][2]*1000.)
    #         Rabs.append(hwmap[bytebay][iirr][b'Autobias R']*1000.)
    #         bolon.append(hwmap[bytebay][iirr][b'Split']-12)
    #         nsq = float(hwmap[bytebay][iirr][b'Autobias y'])/float(hwmap[bytebay][iirr][b'Autobias x'])
    #         if hwmap[bytebay][iirr][b'Symmetric autobias']:
    #             nsq*=2
    #         sqrs.append(nsq)
    #         print(bay,bb,bolon[-1],Rxs[-1],Rabs[-1])
    # Rxs = np.array(Rxs)
    # Rabs = np.array(Rabs)
    # bolon = np.array(bolon)
    # sqrs = np.array(sqrs)
    # pl.figure('autobias split',dpi=dpi,figsize=(4,3))
    # pl.xlabel('Designed Resistance (m$\Omega$)')
    # pl.ylabel('Measured Parasitic Resistance (m$\Omega$)')
    # pl.title('Autobias Resistance Splits')
    # pl.grid(True)
    # cpopt,cpcov = curve_fit(line, Rabs, Rxs, p0=[1])
    # dpopt,dpcov = curve_fit(linec, Rabs, Rxs, p0=[1,0])
    # abrs = np.linspace(0,250,100)
    # cprs = line(abrs,*cpopt)
    # dprs = linec(abrs,*dpopt)
    # pl.plot(abrs,cprs,color='r',label = 'Fit $y=m_0x$',alpha=0.7)
    # pl.plot(abrs,dprs,color='orange',label = 'Fit $y=m_1x+c$',alpha=0.7)
    # #pl.plot(Rabs,Rxs,marker='.',ls='None')
    # pl.plot(Rabs[bolon<=5],Rxs[bolon<=5],marker='o',ls='None',label='Symmetric',color=bluish,alpha=0.7)
    # pl.plot(Rabs[bolon>5],Rxs[bolon>5],marker='o',ls='None',label='Asymmetric',color='g',alpha=0.7)
    # pl.legend(loc=2)
    # bstring = '$m_0$ = {:.3f}$\pm${:.3f} \n$m_1$ = {:.3f}$\pm${:.3f} \n$c$ = {:.2f}$\pm${:.2f} m$\Omega$'.format(cpopt[0],np.sqrt(cpcov[0][0]),dpopt[0],np.sqrt(dpcov[0][0]),dpopt[1],np.sqrt(dpcov[1][1]))
    # pl.annotate(text=bstring,xy=(155,6),bbox=dict(facecolor='white',edgecolor='gray',alpha=0.7))
    # wfname = os.path.join(outpath,'lbirdv3_autobias_splits.png')
    # pl.savefig(fname=wfname,bbox_inches='tight')
    #
    # #########################################################################
    # # Redo the autobias plot wrt to # of squares
    # # Gather Rx and expected Rab
    #
    # Rabmxs = Rxs-0.3 #mOhms
    # cpopt,cpcov = curve_fit(line, sqrs, Rabmxs, p0=[1])
    # dpopt,dpcov = curve_fit(linec, sqrs, Rabmxs, p0=[1,0])
    # abrs = np.linspace(0,0.33,100)
    # cprs = line(abrs,*cpopt)
    # dprs = linec(abrs,*dpopt)
    # pl.figure('autobias split Resistivity',dpi=dpi,figsize=(4,3))
    # pl.xlabel('Autobias Squares ($\square$)')
    # pl.ylabel('Resistance (m$\Omega$)')
    # pl.title('Autobias Resistivity')
    # pl.grid(True)
    # pl.plot(abrs,cprs,color='r',label = 'Fit $y=m_0x$',alpha=0.7)
    # pl.plot(abrs,dprs,color='orange',label = 'Fit $y=m_1x+c$',alpha=0.7)
    # #pl.plot(Rabs,Rxs,marker='.',ls='None')
    # pl.plot(sqrs[bolon<=5],Rabmxs[bolon<=5],marker='o',ls='None',label='Symmetric',color=bluish,alpha=0.7)
    # pl.plot(sqrs[bolon>5],Rabmxs[bolon>5],marker='o',ls='None',label='Asymmetric',color='g',alpha=0.7)
    # pl.legend(loc=2)
    # bstring = '$m_0$ = {:.3f}$\pm${:.3f} \n$m_1$ = {:.3f}$\pm${:.3f} \n$c$ = {:.2f}$\pm${:.2f} m$\Omega$'.format(cpopt[0],np.sqrt(cpcov[0][0]),dpopt[0],np.sqrt(dpcov[0][0]),dpopt[1],np.sqrt(dpcov[1][1]))
    # pl.annotate(text=bstring,xy=(0,6),bbox=dict(facecolor='white',edgecolor='gray',alpha=0.7))
    # wfname = os.path.join(outpath,'lbirdv3_autobias_splits_ohmspsq.png')
    # pl.savefig(fname=wfname,bbox_inches='tight')

    ####################################################################
    # Plot Rn against bolo split type, is Rn lower for autobias bolo? yes by design!
    # rns = []
    # nsplits = []
    # nsquares = []
    # for col in ts:
    #     for rr in ts[col]:
    #         rfs = ts[col][rr][b'iv'][b'config_file'][b'runconfig'][b'rnFractions']
    #         if rfrac in rfs:
    #             rfidx = rfs.index(rfrac)
    #         else:
    #             print('Could not find specified rfrac in results dict')
    #             rfidx = -1
    #         ivkey = np.sort(list(ts[col][rr][b'iv'][b'converted_data'].keys()))[0]
    #         rtesidx = ts[col][rr][b'iv'][b'converted_data_indices'].index(b'rtes')
    #         rns.append(np.mean(ts[col][rr][b'iv'][b'converted_data'][ivkey][rtesidx][-5:]))
    #         cc = col.split('Bay')[1]
    #         rint = int(rr.split(b'Row')[1])
    #         bytebay = bytes(cc[-1],'ascii')
    #         nsplits.append(hwmap[bytebay][rint][b'Split'])
    #         nsquares.append(float(hwmap[bytebay][rint][b'R TES x'])/float(hwmap[bytebay][rint][b'R TES y']))
    # nsplits = np.array(nsplits)
    # nsquares = np.array(nsquares)
    # rns = np.array(rns)
    #
    # pl.figure('Rn vs split',dpi=dpi,figsize=(4,3))
    # pl.xlabel('Split #')
    # pl.ylabel('Normal Resistance ($\Omega$)')
    # pl.title('Normal Resistance of Splits')
    # pl.grid(True)
    # pl.plot(nsplits,rns,marker='o',ls='None')
    # wfname = os.path.join(outpath,'lbirdv3_rnormal_v_splits.png')
    # pl.savefig(fname=wfname,bbox_inches='tight')
    #
    # ####################################################################


    ####################################################################
    # Plot k/A/L vs bolo splits
    # Gs = []      # thermal conductances
    # KALs = []    # thermal k per area/length
    # GALs = []    # thermal G per area/length
    # GALd = {}    # thermal G per area/length stored by each chip
    # ALs = []     # area/length
    # Vs = []      # volumes of PdAu
    # PPs = []     # Power at bath_temperature and rfrac
    # Tcs = []
    # nsplits = []
    # ns = []
    # nerrs = []
    # rdis = [] # radial distance from wafer center to chip center
    # cpos = [] # (x,y) coordinates of chip position
    # bpos = [] # (x,y) coordinates of bolometer position
    # cids = [] # chip ids
    # for cc in sorted(ts):
    #     for rr in sorted(ts[cc]):
    #         #cc = col.split('Bay')[1]
    #         #cc = bytes(cc[-1],'ascii')
    #         #rint = int(rr.split(b'Row')[1])
    #         rfs = ts[cc][rr]['rn_fracs']
    #         if rfrac in rfs:
    #             rfidx = rfs.index(rfrac)
    #         else:
    #             print('Could not find specified rfrac in results dict')
    #             rfidx = -1
    #         k,Tc,n = ts[cc][rr]['pfits'][0] # W/K, K, #
    #         GG = n*k*Tc**(n-1)*1e12 #pW/K
    #         #KK = ts[cc][rr][b'iv'][b'conductance'][b'fit parameters'][b'K'][0]*1e12 #pW/K
    #         # This AL was previously messed up with 4x w_leg4_bias
    #         #AL = (4.(hwmap[cc][rint][b'bolo leg width']+1.*hwmap[cc][rr][b'w_leg4_bias'])) / float(hwmap[cc][rr][b'bolo leg length'])) # in um # height is 1um # We got 4 legs!
    #         AL = 4.*hwmap[cc][rr]['bolo leg width'] / hwmap[cc][rr]['bolo leg length'] # in um # height is 1um # We got 4 legs!
    #         #AL = (4.*hwmap[cc][rr][b'bolo leg width']) / float(hwmap[cc][rr][b'bolo leg length']) # get rid of w_leg4_bias
    #         ns.append(n)
    #         #nerrs.append(ts[cc][rr][b'iv'][b'conductance'][b'fit parameters'][b'sigma_n'][rfidx])
    #         #ptesidx = ts[cc][rr][b'iv'][b'converted_data_indices'].index(b'ptes')
    #         #ivkeys = np.sort(list(ts[cc][rr][b'iv'][b'converted_data'].keys()))
    #         # Get ivXXX that matches bath_temperature within 1% relative tolerance
    #         # for ivkey in ivkeys:
    #         #     if np.isclose(bath_temperature, ts[cc][rr][b'iv'][b'raw_data'][ivkey][b'measured_temperature'],rtol=0.01):
    #         #         break
    #         # ivint = list(ivkeys).index(ivkey)
    #
    #         PPs.append(ts[cc][rr]['p_at_rnfrac'][rfidx])
    #         Gs.append(GG)
    #         KALs.append(k/AL)
    #         GALs.append(GG/AL)
    #         ALs.append(AL)
    #         Vs.append(hwmap[cc][rr]['C']*0.5)
    #         nsplits.append(hwmap[cc][rr]['Split'])
    #         rdis.append(hwmap[cc][rr]['chip distance'])
    #         cpos.append(hwmap[cc][rr]['chip position'])
    #         bpos.append(hwmap[cc][rr]['bolo position'])
    #         cids.append(hwmap[cc][rr]['chip id'])
    #         Tcs.append(Tc)
    #



    pl.figure('G/A/L vs distance',dpi=dpi,figsize=(4,3))
    pl.xlabel('Chip Distance from Wafer Center (mm)')
    pl.ylabel('G/A/L (pW/K/$\mu$m)')
    pl.title('SiN Leg Conductivity Uniformity')
    pl.grid(True)
    pl.plot(rdis,GALs,marker='o',ls='None',alpha=0.7)
    for rrr,gg,nn in zip(rdis,GALs,nsplits):
        pl.text(rrr,gg,str(nn))
    #pl.axhline(np.mean(GALs),color='r',label='{:.1f}$\pm${:.1f} pW/K/$\mu$m'.format(np.mean(GALs),np.std(GALs)))
    pl.ylim(0,800)
    #pl.legend()
    wfname = os.path.join(outpath,'lbirdv3_SiN_conductivity_vs_distance.png')
    pl.savefig(fname=wfname,bbox_inches='tight')

    #cpos = np.array(cpos).transpose()
    pl.figure('G/A/L vs X distance',dpi=dpi,figsize=(4,3))
    pl.xlabel('Chip X Distance from Wafer Center (mm)')
    pl.ylabel('G/A/L (pW/K/$\mu$m)')
    pl.title('SiN Leg Conductivity Uniformity')
    pl.grid(True)
    pl.plot(cpos[:,0],GALs,marker='o',ls='None',alpha=0.7)
    for rrr,gg,nn in zip(cpos[:,0],GALs,nsplits):
        pl.text(rrr,gg,str(nn))
    #pl.axhline(np.mean(GALs),color='r',label='{:.1f}$\pm${:.1f} pW/K/$\mu$m'.format(np.mean(GALs),np.std(GALs)))
    pl.ylim(0,800)
    #pl.legend()
    wfname = os.path.join(outpath,'lbirdv3_SiN_conductivity_vs_x_distance.png')
    pl.savefig(fname=wfname,bbox_inches='tight')

    pl.figure('G/A/L vs Y distance',dpi=dpi,figsize=(4,3))
    pl.xlabel('Chip Y Distance from Wafer Center (mm)')
    pl.ylabel('G/A/L (pW/K/$\mu$m)')
    pl.title('SiN Leg Conductivity Uniformity')
    pl.grid(True)
    pl.plot(cpos[:,1],GALs,marker='o',ls='None',alpha=0.7)
    for rrr,gg,nn in zip(cpos[:,1],GALs,nsplits):
        pl.text(rrr,gg,str(nn))
    #pl.axhline(np.mean(GALs),color='r',label='{:.1f}$\pm${:.1f} pW/K/$\mu$m'.format(np.mean(GALs),np.std(GALs)))
    pl.ylim(0,800)
    #pl.legend()
    wfname = os.path.join(outpath,'lbirdv3_SiN_conductivity_vs_y_distance.png')
    pl.savefig(fname=wfname,bbox_inches='tight')

    # Do the 2D plot with colorbar
    # try to fit a surface to the data
    # regular grid covering the domain of the data
    x0,y0 = np.min(bpos,axis=0)
    x1,y1 = np.max(bpos,axis=0)
    X,Y = np.meshgrid(np.linspace(x0, x1, 20), np.linspace(y0, y1, 20))
    XX = X.flatten()
    YY = Y.flatten()
    data = np.c_[bpos[:,0],bpos[:,1],GALs]
    order = 1    # 1: linear, 2: quadratic
    if order == 1:
        # best-fit linear plane
        A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
        C,_,_,_ = np.linalg.lstsq(A, data[:,2])    # coefficients
        # evaluate it on grid
        Z = C[0]*X + C[1]*Y + C[2]
        # or expressed using matrix/vector product
        #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)
    elif order == 2:
        # best-fit quadratic curve
        A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
        C,_,_,_ = np.linalg.lstsq(A, data[:,2])
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)

    fig = pl.figure('3D points')
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(data[:,0], data[:,1], data[:,2], c='r', s=50)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.axis('equal')
    ax.axis('tight')
    wfname = os.path.join(outpath,'lbirdv3_SiN_conductivity_3D_position.png')
    pl.savefig(fname=wfname,bbox_inches='tight')


    #colors = cm.jet(np.linspace(0, 1, len(ivRunNames)))
    #colors = turbo_colors(np.linspace(0, 1, len(GALs)))
    colors = turbo_colors(sorted((GALs-np.min(GALs))/np.max(GALs-np.min(GALs)),reverse=True)) # reverse these so the colors line up with the values
    fig = pl.figure('G/A/L vs Wafer Position 2',dpi=dpi,figsize=[4.75, 3.78])
    pl.xlabel('X Distance from Wafer Center (mm)')
    pl.ylabel('Y Distance from Wafer Center (mm)')
    pl.title('SiN Leg Conductivity Uniformity')
    #pl.grid(True)
    pitch = 6.15 # mm
    dcp = np.array([pitch/2.,pitch/2.])
    for gg,rrr,sss,nn,tc in sorted(zip(GALs,bpos,cpos,nsplits,colors)):
        #dx = 2.*(rrr[0]-sss[0]) # spread out the detectors beyond the chip to see trends if marker is 's' square
        pl.plot(rrr[0],sss[1],ls='None',marker='|',ms=7,color=tc,alpha=1) # use the bolo pos for x and chip pos for y
        chiprec = matplotlib.patches.Rectangle(xy=sss-dcp, width=pitch, height=pitch, lw=0.7, fill=False, ls='-' if nn<12 else '--')
        pl.gca().add_patch(chiprec)
        #pl.text(rrr[0],rrr[1],'{:.0f}'.format(gg)) # check those colors are correct
    wafercircle = matplotlib.patches.Circle((0,0), radius=75,fill=False)
    pl.gca().add_patch(wafercircle)
    pl.axis('equal')
    pl.ylim(-85,85)
    pl.xlim(-85,85)

    pl.axvline(0,color='k',zorder=0,lw=0.4)
    pl.axhline(0,color='k',zorder=0,lw=0.4)
    # Make two fake lines for the legend of chips A vs B
    pl.plot(0,0,ls='-',lw=0.7,color='k',label='Baseline')
    pl.plot(0,0,ls='--',lw=0.7,color='k',label='Autobias')

    sm = pl.cm.ScalarMappable(cmap=turbo_colors, norm=pl.Normalize(vmin=0, vmax=1))
    sm._A = []
    thebar = fig.colorbar(sm)
    float_ticks = np.linspace(np.min(GALs),np.max(GALs),len(thebar.get_ticks()))
    str_ticks = ['{:.0f}'.format(tt) for tt in float_ticks]
    thebar.ax.set_yticklabels(str_ticks)
    thebar.set_label('G/A/L pW/K/$\mu$m')
    # create a unit vector along the gradient of change
    ux,uy = C[:2]
    mxy = np.sqrt(ux**2+uy**2) # vector length
    ux*=-20./mxy
    uy*=-20./mxy
    #pl.plot([0,ux],[0,uy],color='k',lw=2)
    arlab = 'Gradient {:.0f}$^o$'.format(np.rad2deg(np.arctan(uy/ux))) # arrow label
    arrow = pl.arrow(x=0, y=0, dx=ux, dy=uy, head_width=3, head_length=5,label=arlab,color='k')

    #f = pl.figure(figsize=(10,6))
    #arrow = pl.arrow(0,0, 0.5, 0.6, 'dummy', label='My label', )
    #pl.legend([arrow], [arlab], handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow),
    #                    })

    pl.legend(fontsize='small',loc='lower left')
    wfname = os.path.join(outpath,'lbirdv3_SiN_conductivity_vs_wafer_position.png')
    pl.savefig(fname=wfname,bbox_inches='tight')



    ###########################################################################
    # Try making a 3D plot so you can draw a line through it
    # xx = []
    # yy = []
    # zz = []
    # for gg,rrr in sorted(zip(GALs,bpos)):
    #         xx.append(rrr[0])
    #         yy.append(rrr[1])
    #         zz.append(gg)
    # ax = m3d.Axes3D(pl.figure('3d scatter'))
    # ax.scatter3D(xx,yy,zz)
    # save?

    ####################################################################
    # Plot n vs bolo splits

    pl.figure('n vs split',dpi=dpi,figsize=(4,3))
    pl.xlabel('Bolo Split')
    pl.ylabel('Carrier Index $n$')
    pl.title('Thermal Carrier Index')
    pl.grid(True)
    #pl.axhline(np.mean(ns),color='r',label=r'$\n$={:.1f}$\pm${:.1f}'.format(np.mean(ns),np.std(ns)),alpha=0.7)
    pl.plot(nsplits,ns,marker='o',ls='None')


    pl.legend()
    wfname = os.path.join(outpath,'lbirdv3_thermal_exponent_vs_split.png')
    pl.savefig(fname=wfname,bbox_inches='tight')

    ####################################################################
    # Plot n vs A/L

    pl.figure('n vs A/L',dpi=dpi,figsize=(4,3))
    pl.xlabel('Bolometer Leg A/L ($\mu$m)')
    pl.ylabel('Carrier Index $n$')
    pl.title('Thermal Carrier Index vs A/L')
    pl.grid(True)
    #pl.plot(ALs,ns,marker='o',ls='None',alpha=0.7)
    pl.errorbar(x=ALs,y=ns,yerr=nerrs,marker='o',ls='None',alpha=0.5)
    #pl.axhline(np.mean(ns),color='r',label='{:.1f}$\pm${:.1f} pW/K $\mu$m'.format(np.mean(GALs),np.std(GALs)))

    #popt,pcov = curve_fit(line, ALs, ns, p0=[1])
    popt,pcov = curve_fit(linec, ALs, ns, p0=[1,0])
    poptl,pcovl = curve_fit(linec, np.log(ALs), ns, p0=[1,0])
    poptl2,pcovl2 = curve_fit(logmc, ALs, ns, p0=[1,0]) # is this any different than the line above? nope, same!
    pstd = np.sqrt(np.diagonal(pcov))
    pstdl = np.sqrt(np.diagonal(pcovl))
    Ntot = 5000 # points in fit arrays aka "lines" in this spaghetti
    ALline = np.linspace(0.001,1.0,Ntot)
    yline = linec(ALline,*popt)
    nlinel = linec(np.log(ALline),*poptl)
    nlinel2 = logmc(ALline,*poptl)
    #rnlinec = linec(sqline,*poptc)

    #pl.plot(ALline,yline,label='m $n$/A/L+$n_0$=        {:.2f}$\pm${:.2f} + {:.2f}$\pm${:.2f}'.format(popt[0],pstd[0],popt[1],pstd[1]))
    pl.plot(ALline,nlinel,label=r'Fit $n$ = m $ln$(A/L)+$n_0$',color='r',alpha=0.7)
    #pstring = '$P=k(T_c^n-T_b^n)$\nm={:.2f}$\pm${:.2f}\n$n_0$={:.2f}$\pm${:.2f}'.format(poptl[0],pstdl[0],poptl[1],pstdl[1])
    #pl.annotate(text=pstring,xy=(0.4,1.1),va='bottom',ha='left',bbox=dict(facecolor='white',edgecolor='gray',alpha=0.7)) # (0.05*max(legl),0.07*max(legp))
    pl.legend()
    pl.ylim(1,3.25)
    pl.xlim(0.003,0.6)
    wfname = os.path.join(outpath,'lbirdv3_thermal_exponent_vs_AL.png')
    pl.savefig(fname=wfname,bbox_inches='tight')

    ####################################################################
    # Now calculate a normalized NEP for different n values and Tc values
    Pstop = 1e-12 #W
    n=3.
    Tb=0.1 #K
    LG=10.
    Tc = np.linspace(0.11,0.27,2000)
    NEPg3p0 = NEPg(Tc, Pstop, Tb, n=3.0, LG=10)
    NEPg2p7 = NEPg(Tc, Pstop, Tb, n=2.7, LG=10)
    NEPg2p4 = NEPg(Tc, Pstop, Tb, n=2.4, LG=10)
    NEPg2p0 = NEPg(Tc, Pstop, Tb, n=2.0, LG=10)
    Tcopt3p0 = fmin(NEPg,0.17,args=(Pstop,Tb,3.0,LG))[0]
    Tcopt2p7 = fmin(NEPg,0.17,args=(Pstop,Tb,2.7,LG))[0]
    Tcopt2p4 = fmin(NEPg,0.17,args=(Pstop,Tb,2.4,LG))[0]
    Tcopt2p0 = fmin(NEPg,0.17,args=(Pstop,Tb,2.0,LG))[0]
    idx3p0 = np.argmin(np.abs(Tc-Tcopt3p0))
    idx2p7 = np.argmin(np.abs(Tc-Tcopt2p7))
    idx2p4 = np.argmin(np.abs(Tc-Tcopt2p4))
    idx2p0 = np.argmin(np.abs(Tc-Tcopt2p0))

    pl.figure('NEPs vs Tc for ns',dpi=dpi,figsize=(4,3))
    # Draw NEP curves
    pl.plot(Tc*1e3, NEPg3p0*1e18,label='3.0',color=bluish)
    pl.plot(Tc*1e3, NEPg2p7*1e18,label='2.7',color='green')
    pl.plot(Tc*1e3, NEPg2p4*1e18,label='2.4',color='orange')
    pl.plot(Tc*1e3, NEPg2p0*1e18,label='2.0',color='red')
    # Mark the minima
    pl.plot(Tc[idx3p0]*1e3, NEPg3p0[idx3p0]*1e18,color=bluish,marker='o')
    pl.plot(Tc[idx2p7]*1e3, NEPg2p7[idx2p7]*1e18,color='green',marker='o')
    pl.plot(Tc[idx2p4]*1e3, NEPg2p4[idx2p4]*1e18,color='orange',marker='o')
    pl.plot(Tc[idx2p0]*1e3, NEPg2p0[idx2p0]*1e18,color='red',marker='o')
    pl.xlim(115,265)
    pl.ylim(3.6,5.1)
    pl.title('NEPg vs T$_\\textrm{c}$')
    pl.xlabel('Critical Temperature (mK)')
    pl.ylabel('NEPg aW/$\sqrt{\\textrm{Hz}}$')
    pl.legend(title='Index $n$',loc=1,framealpha=0.7)
    pl.grid(True)
    wfname = os.path.join(outpath,'NEPvsTc_for_ns.png')
    pl.savefig(fname=wfname,bbox_inches='tight')

    ###################################################################
    # Now plot optimal Tc vs index

    Tcopts = []
    for nn in nlinel:
        Tcopts.append(fmin(NEPg,0.17,args=(Pstop,Tb,nn,LG),xtol=1E-5, ftol=1E-5,disp=False)[0])
    Tcopts = np.array(Tcopts)
    pl.figure('Optimal Tc vs indices',dpi=dpi,figsize=(4,3))
    # Draw NEP curves
    pl.plot(nlinel, Tcopts*1e3,color=bluish)
    pl.title('Optimal T$_\\textrm{c}$ vs Index')
    pl.xlabel('Carrier Index $n$')
    pl.ylabel('T$_\\textrm{c}$ (mK)')
    pl.grid(True)
    wfname = os.path.join(outpath,'OptimalTc_vs_ns.png')
    pl.savefig(fname=wfname,bbox_inches='tight')

    ###################################################################
    # Now match the A/L to optimal Tc, n and Psat in 4-axis plot
    # These arrays are paired with one another
    # nlinel, Tcopts*1e3
    # ALs, ns #unordered,
    # ALline,nlinel this is the fit to n vs A/L

    # Now map Tc to n and P to A/L on fine scale for parameter determination
    Psats = np.mean(GALs)*ALline/(nlinel*Tcopts**(nlinel-1))*(Tcopts**nlinel-Tb**nlinel)
    #ALt2nl = poptl[0]*np.log(ALline)+poptl[1] # convert the ALline to n values
    # for aall,ttcc,nnnn in zip(ALline,Tcopts,nlinel): # ALline spans from ~0 to 1, but nlinel is only spanning a smaller subset??????
    #     #Psats.append(np.mean(GALs)*aall/(nnnn*ttcc**(nnnn-1))*aall*(ttcc**nnnn-Tb**nnnn)) #extra aall?
    #     Psats.append(np.mean(GALs)*aall/(nnnn*ttcc**(nnnn-1))*(ttcc**nnnn-Tb**nnnn))

    #AL2n = poptl[0]*np.log(ALline)+poptl[1] # this is just nlinel
    #Tcopti = pl.interp(x=AL2n, xp=nlinel, fp=Tcopts)
    #Tcopti = pl.interp(x=nlinel, xp=nlinel, fp=Tcopts)
    fig = pl.figure('Optimal Tc vs Als',dpi=dpi,figsize=(4.53,4))
    ax = fig.add_subplot(1, 1, 1)
    #ax.plot(ALline[ALline>0.02],Tcopti[ALline>0.02]*1e3,color='k',lw=3,zorder=10)
    #ax.plot(ALline[ALline>1E-5],Tcopti[ALline>1E-5]*1e3,color='k',lw=3,zorder=30)
    ax.plot(ALline[ALline>1E-5],Tcopts[ALline>1E-5]*1e3,color='k',lw=3,zorder=30)

    axy = ax.twinx() # second y-axis
    axx = ax.twiny() # second x-axis
    matplotlib.axes.Axes.set_xscale(ax,'log')
    matplotlib.axes.Axes.set_xscale(axx,'log')
    ax.set_title('TES Optimization',y=1.25)
    ax.set_xlabel('Bolometer Leg A/L ($\mu$m)')
    ax.set_ylabel('T$_\\textrm{c}$ (mK)')
    axy.set_ylabel('Carrier Index $n$')
    axx.set_xlabel('Saturation Power (pW)')
    ax.grid(True)
    ax.set_xlim(3E-3,0.6)
    ax.set_xticks([0.003,0.01,0.03,0.06,0.1,0.3,0.6])
    ax.set_xticklabels(labels=['3E-3','0.01','0.03','0.06','0.1','0.3','0.6'], fontdict=None, minor=False)
    #ax.text(0.1,0.1,'T$_b%=100 mK')

    # Now map Tc to n and P to A/L for the tick markers
    ax.set_ylim(168,207)
    ALticks = ax.get_xticks()
    Tcticks = ax.get_yticks()
    nticks = pl.interp(Tcticks,Tcopts[::-1]*1e3, nlinel[::-1])
    ALt2n = poptl[0]*np.log(ALticks)+poptl[1] # convert the ALticks to n values
    ALt2Tc = pl.interp(x=ALticks, xp=ALline, fp=Tcopts) # convert the ALticks to Tc values
    Pticks = []
    Tb = 0.1 #K
    varfac = [] # this variable differentiates k from G trends in Pitcks and Pticks2
    for aall,ttcc,nnnn in zip(ALticks,ALt2Tc,ALt2n):
        #print,'Pticks {:.2f},{:.3f},{:.2f},{:.2f}'.format(aall,ttcc,nnnn,np.mean(KALs)*aall*(ttcc**nnnn-Tb**nnnn)) # This gives reasonable, but
        Pticks.append(np.mean(KALs)*aall*(ttcc**nnnn-Tb**nnnn))
    Pticks2 = []
    for aall,ttcc,nnnn in zip(ALticks,ALt2Tc,ALt2n):
        #Pticks2.append(np.mean(GALs)*aall/(nnnn*ttcc**(nnnn-1.))*aall*(ttcc**nnnn-Tb**nnnn))
        Pticks2.append(np.mean(GALs)*aall/(nnnn*ttcc**(nnnn-1.))*(ttcc**nnnn-Tb**nnnn))
        varfac.append(np.mean(KALs)/np.mean(GALs)*(nnnn*ttcc**(nnnn-1.)))
    # Some Pticks will be nan due to the log in ALt2n
    Pticks = np.array(Pticks)
    Pticks[~np.isfinite(Pticks)] = 0
    Pticks2 = np.array(Pticks2)
    Pticks2[~np.isfinite(Pticks2)] = 0
    # Pticklabels = []
    # for pt in Pticks2:
    #     if pt<1:
    #         Pticklabels.append('{:.2f}'.format(pt))
    #     else:
    #         Pticklabels.append('{:.1f}'.format(pt))
    Pticklabels = ['{:.1f}'.format(pt) if pt>1 else '{:.2f}'.format(pt) for pt in Pticks2]

    # Add data points in
    Tcdats = pl.interp(ns,nlinel,Tcopts)
    dataplt = ax.plot(ALs,Tcdats*1e3,ls='None',marker='o',color='purple',alpha=0.5)#,label='Data')
    #dleg = ax.legend(loc=3)

    axx.set_xticks(ALticks)
    axx.set_xticklabels(Pticklabels)
    axy.set_yticks(Tcticks)
    axy.set_yticklabels(['{:.2f}'.format(nt) for nt in nticks])
    axy.set_ylim(ax.get_ylim())
    axx.set_xlim(ax.get_xlim())

    # Shade regions of Psat = 2.5Popt
    vspans = [(Popt_lft,'r','LFT'),
              (Popt_mft,'g','MFT'),
              (Popt_hft,'b','HFT'),
              ]
    for vv in vspans:
        almin = np.interp(2.5*np.min(vv[0]),Psats,ALline)
        almax = np.interp(2.5*np.max(vv[0]),Psats,ALline)
        axx.axvspan(almin,almax,color=vv[1],label=vv[2],alpha=.8)
        ax.axvline(almin,color=vv[1],alpha=.7)
        ax.axvline(almax,color=vv[1],alpha=.7)

        tcmin = np.interp(2.5*np.min(vv[0]),Psats,Tcopts)*1e3
        tcmax = np.interp(2.5*np.max(vv[0]),Psats,Tcopts)*1e3
        axy.axhspan(tcmin,tcmax,color=vv[1],alpha=.2)
        ax.axhline(tcmin,color=vv[1],alpha=.7)
        ax.axhline(tcmax,color=vv[1],alpha=.7)


    Psat_mean = 2.5*np.mean(np.hstack([Popt_lft,Popt_mft,Popt_hft]))
    Tc_mean = np.interp(Psat_mean,Psats,Tcopts)
    AL_mean = np.interp(Psat_mean,Psats,ALline)
    n_mean =  pl.interp(AL_mean,ALline,nlinel)
    ax.axhline(Tc_mean*1e3,color='k')
    ax.axvline(AL_mean,color='k')

    Psat_mean_MH = 2.5*np.mean(np.hstack([Popt_mft,Popt_hft]))
    Tc_mean_MH = np.interp(Psat_mean_MH,Psats,Tcopts)
    AL_mean_MH = np.interp(Psat_mean_MH,Psats,ALline)
    n_mean_MH =  pl.interp(AL_mean_MH,ALline,nlinel)

    Psat_mean_L = 2.5*np.mean(np.hstack([Popt_lft]))
    Tc_mean_L = np.interp(Psat_mean_L,Psats,Tcopts)
    AL_mean_L = np.interp(Psat_mean_L,Psats,ALline)
    n_mean_L =  pl.interp(AL_mean_L,ALline,nlinel)

    meanstr = '\ \ \ Mean Values\n'
    meanstr+= 'P$_\\textrm{{sat}}$ \  {:.2f} pW\n'.format(Psat_mean)
    meanstr+= 'T$_\\textrm{{c}}$ \ \ \  {:.1f} mK\n'.format(Tc_mean*1e3)
    meanstr+= '$n$ \ \ \ \ \  {:.2f}\n'.format(n_mean)
    meanstr+= 'A/L \ {:.3f} $\mu$m'.format(AL_mean)
    ax.text(0.1,193,meanstr,bbox=dict(facecolor='white', edgecolor='black',alpha=0.5 ),fontdict={'family' : 'monospace','size':'small'})#, boxstyle='round,pad=1'))
    leg = fig.legend()#loc=0)
    for hh in axx.get_legend_handles_labels()[0]:
        hh.set_alpha(0.2)
    fig.tight_layout()

    wfname = os.path.join(outpath,'OptimalTc_vs_AL_vs_n_vs_Psat.png')
    fig.savefig(fname=wfname,bbox_inches='tight')

    ###########################################################################
    ###################################################################
    # Make a variation on the previous plots
    # plot n vs A/L on the x and y as designed vs measured values
    # plot optimal Tc/Tb ratio on a second y-axis
    # plot design Psat on second x-axis
    # Try to add UCB BT5-02 and NIST v0 bolo data
    # These arrays are paired with one another
    # nlinel, Tcopts*1e3
    # ALs, ns #unordered,
    # ALline,nlinel this is the fit to n vs A/L

    # Now map Tc to n and P to A/L on fine scale for parameter determination
    Psats = np.mean(GALs)*ALline/(nlinel*Tcopts**(nlinel-1))*(Tcopts**nlinel-Tb**nlinel)
    #ALt2nl = poptl[0]*np.log(ALline)+poptl[1] # convert the ALline to n values
    # for aall,ttcc,nnnn in zip(ALline,Tcopts,nlinel): # ALline spans from ~0 to 1, but nlinel is only spanning a smaller subset??????
    #     #Psats.append(np.mean(GALs)*aall/(nnnn*ttcc**(nnnn-1))*aall*(ttcc**nnnn-Tb**nnnn)) #extra aall?
    #     Psats.append(np.mean(GALs)*aall/(nnnn*ttcc**(nnnn-1))*(ttcc**nnnn-Tb**nnnn))


    #AL2n = poptl[0]*np.log(ALline)+poptl[1] # this is just nlinel
    #Tcopti = pl.interp(x=AL2n, xp=nlinel, fp=Tcopts)
    #Tcopti = pl.interp(x=nlinel, xp=nlinel, fp=Tcopts)
    fig = pl.figure('Optimal Tc vs Als 2',dpi=dpi,figsize=(4.53,4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ALline[ALline>1E-5],nlinel[ALline>1E-5],color='k',lw=3,zorder=20,label='Model')
    axy = ax.twinx() # second y-axis
    axx = ax.twiny() # second x-axis
    matplotlib.axes.Axes.set_xscale(ax,'log')
    matplotlib.axes.Axes.set_xscale(axx,'log')
    ax.set_title('TES Data and Designs',y=1.25)
    ax.set_xlabel('Designed Bolometer Leg A/L ($\mu$m)')
    axy.set_ylabel('Optimal T$_\\textrm{c}$/T$_\\textrm{b}$')
    ax.set_ylabel('Measured Carrier Index $n$')
    axx.set_xlabel('Derived Saturation Power (pW)')
    ax.grid(True)
    #ax.set_xlim(0.01,0.6)
    ax.set_xlim(3E-3,0.6)
    ax.set_xticks([0.003,0.01,0.03,0.06,0.1,0.3,0.6])
    ax.set_xticklabels(labels=['3E-3','0.01','0.03','0.06','0.1','0.3','0.6'], fontdict=None, minor=False)
    #ax.text(0.1,0.1,'T$_b%=100 mK')

    # Now map Tc to n and P to A/L for the tick markers
    ax.set_ylim(1,3.2)
    ALticks = ax.get_xticks()
    nticks = ax.get_yticks()
    Tcticks = pl.interp(nticks,nlinel,Tcopts/Tb)
    ALt2n = poptl[0]*np.log(ALticks)+poptl[1] # convert the ALticks to n values
    ALt2Tc = pl.interp(x=ALticks, xp=ALline, fp=Tcopts) # convert the ALticks to Tc values
    Pticks = []
    Tb = 0.1 #K
    for aall,ttcc,nnnn in zip(ALticks,ALt2Tc,ALt2n):
        Pticks.append(np.mean(GALs)*aall/(nnnn*ttcc**(nnnn-1.))*(ttcc**nnnn-Tb**nnnn))
    # Some Pticks will be nan due to the log in ALt2n
    Pticks = np.array(Pticks)
    Pticks[~np.isfinite(Pticks)] = 0
    Pticklabels = ['{:.1f}'.format(pt) if pt>1 else '{:.2f}'.format(pt) for pt in Pticks2]

    # Add data points in
    dataplt = ax.errorbar(x=ALs,y=ns,yerr=nerrs,ls='None',marker='o',color='purple',alpha=0.5,zorder=101,label='Data')

    # Add in v0 data
    datav0pkl = '/Users/gregjaehnig/Google Drive/LiteBIRD/NIST Python/analysis/lbird/lb_bolo_v0/thermalGmetaanalysis.pkl'
    hwmapv0pkl ='/Users/gregjaehnig/Google Drive/LiteBIRD/NIST Python/analysis/lbird/lb_bolo_v0/hwmap.pkl'
    with open(datav0pkl,'rb') as opf:
        colv0 = pickle.load(opf,encoding='bytes')
    with open(hwmapv0pkl,'rb') as opf:
        hwmapv0 = pickle.load(opf,encoding='bytes')
    # gather ns and A/L for v0
    nsv0 = []
    nserrv0 = []
    ALv0 = []
    ALv02 = []
    for ccc in colv0:
        #kk = cc.keys()[0] # get bay name
        bayalpha = bytes(ccc.decode()[-1],'ascii')
        for rr in colv0[ccc]:
            rowint = int(rr[-2:])
            nsv0.append(colv0[ccc][rr][b'iv'][b'conductance'][b'fit parameters'][b'n'][-1])
            nserrv0.append(colv0[ccc][rr][b'iv'][b'conductance'][b'fit parameters'][b'sigma_n'][-1])
            ALv0.append(4.*float(hwmapv0[bayalpha][rowint][b'total w/l']))
            ALv02.append((4.*hwmapv0[bayalpha][rowint][b'bolo leg width']) / float(hwmapv0[bayalpha][rowint][b'bolo leg length'])) # in um # height is 1um # We got 4 legs!
    # datapltv0 = ax.plot(ALv0,nsv0,ls='None',marker='o',color='orange',alpha=0.5)#,label='Data')

    # gather ns and A/L for BT5-02
    dataBT5pkl = '/Users/gregjaehnig/Google Drive/LiteBIRD/Detector Testing/BT5-02/IVvsT/G_data.pkl'
    with open(dataBT5pkl,'rb') as opf:
        dataBT5 = pickle.load(opf,encoding='bytes')
    hwmapBT5 = {b'Ca1Sq1Ch01': 0.022, #2000	8 & 14
                b'Ca1Sq2Ch01': 0.0293, #1500	8 & 14
                b'Ca1Sq4Ch01': 0.088, #500	8 & 14
                b'Ca1Sq5Ch01': 0.088,  #500	8 & 14
                b'Ca1Sq7Ch01': 0.44,  #100	8 & 14
                b'Ca3Sq2Ch01': 0.044, #1500	10 & 23
                b'Ca3Sq4Ch01': 0.132, #500	10 & 23
                b'Ca3Sq6Ch01': 0.132, #500	10 & 23???
                }
    nsBT5 = []
    nerrsBT5 = []
    ALBT5 = []
    for ccc in dataBT5:
        nsBT5.append(dataBT5[ccc][b'thermal model'][b'n'])
        nerrsBT5.append(dataBT5[ccc][b'thermal model'][b'nstd'])
        ALBT5.append(hwmapBT5[ccc])
    # datapltBT5 = ax.errorbar(x=ALBT5,y=nsBT5,yerr=nerrsBT5,ls='None',marker='o',color='orange',alpha=0.5)#,label='Data')

    # Put in designs for v3
    ALsv3 = []
    nsv3 = []
    for pp in np.hstack([Popt_lft,Popt_mft,Popt_hft]):
        ALsv3.append(np.interp(2.5*pp,Psats,ALline))
        nsv3.append(np.interp(2.5*pp,Psats,nlinel))
    ax.plot(ALsv3,nsv3,marker='s',color='orange',alpha=0.7,ls='None',zorder=102,label='Next Designs')

    dleg = ax.legend(loc=4)
    ax.add_artist(dleg)

    axx.set_xticks(ALticks)
    axx.set_xticklabels(Pticklabels)
    axy.set_yticks(nticks)
    axy.set_yticklabels(['{:.2f}'.format(nt) for nt in Tcticks])
    axy.set_ylim(ax.get_ylim())
    axx.set_xlim(ax.get_xlim())

    # Shade regions of Psat = 2.5Popt
    vspans = [(Popt_lft,'r','LFT'),
              (Popt_mft,'g','MFT'),
              (Popt_hft,'b','HFT'),
              ]
    savedvspans = []
    for vv in vspans:
        almin = np.interp(2.5*np.min(vv[0]),Psats,ALline)
        almax = np.interp(2.5*np.max(vv[0]),Psats,ALline)
        thisvspan = axx.axvspan(almin,almax,color=vv[1],label=vv[2],alpha=.8)
        savedvspans.append(thisvspan)
        ax.axvline(almin,color=vv[1],alpha=.7)
        ax.axvline(almax,color=vv[1],alpha=.7)
        nmin = np.interp(2.5*np.min(vv[0]),Psats,nlinel)
        nmax = np.interp(2.5*np.max(vv[0]),Psats,nlinel)
        axy.axhspan(nmin,nmax,color=vv[1],alpha=.2)
        ax.axhline(nmin,color=vv[1],alpha=.7)
        ax.axhline(nmax,color=vv[1],alpha=.7)

    leg = fig.legend(savedvspans,['LFT','MFT','HFT'])#,loc=0)
    for hh in axx.get_legend_handles_labels()[0]:
        hh.set_alpha(0.2)
    fig.tight_layout()

    wfname = os.path.join(outpath,'OptimalTc_vs_AL_vs_n_vs_Psat2.png')
    fig.savefig(fname=wfname,bbox_inches='tight')


    ############################################################################
    # Export our beautiful data in csv format
    # Make header for report
    header=['Column','Row', 'Split', 'P at 100mK (pW)', 'k (pW/K^n)', 'G (pW/K)',
     'Tc (mK)', 'Tc_std (mK)', 'KAL (pW/K^n/um)',
     'GAL (pW/K/um)', 'n', 'n_std',  'Rn (Ohms)', 'Chip ID',
     'Designed Autobias Resistance (Ohms)', 'TES # of squares',
     'Total A/L (um)', 'PdAu Volume (um^3)', 'chip distance (mm)', 'chip position (mm)',
     'bolo position (mm)', 'bolo leg width', 'bolo leg length', 'total w/l',
      'R TES x', 'R TES y', 'Add autobias', 'Autobias x', 'Autobias y', '',
      'Symmetric autobias', 'C', 'e-coupling', 'w_leg4_bias',
      'carrier index', 'Psat est', 'autobias_TES_lead_gap',
      'autobias_y_offset', 'autobias_orientation', 'x_island', 'split type',]

    report = [header]
    # Loop through bolos with results and add them to the report
    for cc in sorted(ts): # bayname A,B, or C
        for rr in sorted(ts[cc]): # row # 0,2,4...
            if 'results' in ts[cc][rr]:
                row = []
                for hh in header:
                    row.append(ts[cc][rr]['results'][hh])
                report.append(row)

    # Write the report row by row
    with open(os.path.join(outpath,'results_report.csv'),'w') as f:
        wr = csv.writer(f)
        for row in report:
            wr.writerow(row)

    # ############################################################################
    # # compare different methods of calculating Psat
    # n0=2.6
    # n1=2.115
    # Psats2 = np.mean(GALs)*ALline/(nlinel*Tcopts**(nlinel-1))*(Tcopts**nlinel-Tb**nlinel) # same result as the for loop
    # Psats3 = np.mean(GALs)*ALline/(nlinel*Tc1**(nlinel-1))*(Tc1**nlinel-Tb**nlinel) # set Tc to constant
    # Psats3p5 = np.mean(GALs)*ALline/(n0*Tc1**(n0-1))*(Tc1**n0-Tb**n0) # set and n Tc to constant
    # Psats4 = np.mean(KALs)*ALline*(Tc1**nlinel-Tb**nlinel) # set Tc and G to constants
    # Psats5 = np.mean(KALs)*ALline*(Tc1**n0-Tb**n0) # set n Tc and G to constants
    # PAL = 52.76536/4. # pW/um: power per A/L from calculator https://docs.google.com/spreadsheets/d/1qL1gs0-3Io89dnlYP32mtDEaESh5SjIp4Z2sBDJ4Z7s/edit?usp=sharing
    # Psats6 = PAL*ALline
    # #PAL2 = # this is P/A/L from v2 measurements on baseline bolometer
    # Psats7 = np.mean(GALs)*ALline/(n1*Tc1**(n1-1))*(Tc1**n1-Tb**n1) # set and n Tc to constant
    # # Pick one Tc for MHFT focal planes and caluclate A/L per band using variable carrier index
    # Psats8 = np.mean(GALs)*ALline/(nlinel*Tc_mean_MH**(nlinel-1))*(Tc_mean_MH**nlinel-Tb**nlinel)
    #
    # # Lets put measurements on this plot
    # # We have ALs, PPs, ns, and Tcs
    # #PPs_rescale = PPs*(184.48**ns-100.**ns)/(Tcs**ns-100**ns) # mK
    # #PPs_rescale = PPs*(Tc_mean_MH**ns-Tb**ns)/((Tcs/1000.)**ns-Tb**ns) # K # this assumes constant k
    # #PPs_rescale = PPs*(((Tcs/1E3)**(ns-1))/(Tc_mean_MH**(ns-1)))*(Tc_mean_MH**ns-Tb**ns)/((Tcs/1E3)**ns-Tb**ns)
    #
    #
    # # plot a quick comparison of all these methods
    # fig = pl.figure('Psat calc compare',dpi=dpi,figsize=figsize)
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(ALline,Psats2,label='All Vars')
    # ax.plot(ALline,Psats6,label='Linear P/A/L v0')
    # ax.plot(ALline,Psats3,label='Constant Tc1=171')
    # ax.plot(ALline,Psats3p5,label='Constant Tc1 n1=2.6')
    # ax.plot(ALline,Psats4,label='Constant k Tc1')
    # ax.plot(ALline,Psats5,label='Constant k Tc1 n1')
    # ax.plot(ALline,Psats7,label='Constant Tc1 n2=2.0')
    # ax.plot(ALline,Psats8,label='Constant Tc2=184')
    # ax.set_xlabel('Bolometer Leg A/L ($\mu$m)')
    # ax.set_ylabel('Saturation Power (pW)')
    # ax.set_title('Psat Calculation Comparison')
    # ax.legend(fontsize='small')
    # ax.grid(True)
    # ax.set_xlim(0,0.1)
    # ax.set_ylim(0,2)
    # #ax.set_xscale('log')
    # wfname = os.path.join(outpath,'Psat_calc_comparison.png')
    # fig.savefig(fname=wfname,bbox_inches='tight')
    #
    # # add data and zoom out a bit
    # ax.set_xlim(0,1)
    # ax.set_ylim(0,10)
    # ax.plot(ALs,PPs_rescale*1e12,ls='None',marker='o',alpha=0.5,color='k')
    # wfname = os.path.join(outpath,'Psat_calc_comparison+data_zoomout.png')
    # fig.savefig(fname=wfname,bbox_inches='tight')
    #
    # Psat_MH = 2.5*np.hstack([Popt_mft,Popt_hft]) # in pW
    # AL_MH = pl.interp(Psat_MH,Psats8,ALline)
    #
    # # now just plot the Psats8 and compare to data
    # fig = pl.figure('Psat calc compare to data',dpi=dpi,figsize=figsize)
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(ALline,Psats8,color='grey',label='Model')
    # ax.plot(ALs,PPs_rescale*1e12,ls='None',marker='o',alpha=0.5,color='k',label='Bolo v2 Data @ Tc=185 mK')
    # ax.plot(AL_MH,Psat_MH,ls='None',marker='o',alpha=0.5,color='b',label='Bolo v3 MHFT Designs')
    # #ax.plot(ALs,np.array(PPs)*1e12,ls='None',marker='o',alpha=0.5,color='g',label='Data')
    # ax.set_xlabel('Bolometer Leg A/L ($\mu$m)')
    # ax.set_ylabel('Saturation Power (pW)')
    # ax.set_title('Psat Calculation Comparison to Data')
    # ax.legend(fontsize='small')
    # ax.grid(True)
    # ax.set_xlim(0,0.8)
    # ax.set_ylim(0,15)
    # # Zoom in on target region
    # axins = ax.inset_axes(ax, width="30%", height="40%", loc=4)
    # axins.plot(ALline,Psats8,color='grey')#,label='Model')
    # axins.plot(ALs,PPs_rescale*1e12,ls='None',marker='o',alpha=0.5,color='k')#,label='Bolo v2 Data')
    # axins.plot(AL_MH,Psat_MH,ls='None',marker='o',alpha=0.5,color='b')#,label='Bolo v3 Design')
    # axins.set_xlim(0,0.1)
    # axins.set_ylim(0,2)
    # axins.grid(True)
    #
    # #ax.set_xscale('log')
    # wfname = os.path.join(outpath,'Psat_calc_comparison_to_data.png')
    # fig.savefig(fname=wfname,bbox_inches='tight')
    #
    # # Try turning the ALs back into Psats, does it follow the model?
    # AL2ns = linec(ALs,*popt)
    # AL2logns = linec(np.log(ALs),*poptl)
    # Psats9 = np.mean(GALs)*ALs/(ns*(Tcs*1E-3)**(ns-1))*((Tcs*1E-3)**ns-Tb**ns)
    #
    # pl.plot(ALs,np.array(PPs)*1e12,ls='None',marker='o',label='Data')
    # pl.plot(ALs,Psats9,ls='None',marker='o',label='Model')

    # ############################################################################
    # # Calculate the A/L needed for the LB Psats for MHFT
    # # Put this into the v3 calculator
    # Psat_MH = 2.5*np.hstack([Popt_mft,Popt_hft]) # in pW
    # AL_MH = pl.interp(Psat_MH,Psats8,ALline)
    #
    # # Get the n and G for MHFT
    # n_MH = pl.interp(Psat_MH,Psats8,nlinel)
    # G_MH = n_MH * Psat_MH * Tc_mean_MH**(n_MH-1) / (Tc_mean_MH**n_MH - Tb**n_MH) # pW/K # =C7*G7*H$4^(G7-1)/(H$4^G7-I$4^G7)
    #
    # # Get the C for MHFT
    # tau0_MH = 0.01 # sec
    # C_MH = G_MH*tau0_MH*1E3 # fJ/K
    #
    # # Get the PdAu area for MHFT
    # # time_constant_metaanalysis.py: lbirdv2_CaJpK_v_bolon_+cZ_II.png has 74.7 aJ/K/um3
    # # subract off bolo island and TES heat capacities from v0 measurements
    # C0 = 33 # fJ/K, see Bolo Heat Capacity v0+2.xslx and 20200125 LBv2 TES Heat Capacity.pptx
    # C_PdAu = 74.7E-3 # fJ/K/um3
    #
    # V_PdAu = (C_MH-C0)/C_PdAu
    # A_PdAu = V_PdAu / 0.5 # half micron thick PdAu
    #
    # # These seem large compared to my original estimate
    #
    #
    # #########################################################################
    # # I'm lost with what to do about the discrepancy, just start over, maybe things got switched up between here and there and popt ain the popt youre looking for
    # poptl,pcovl = curve_fit(linec, np.log(ALs), ns, p0=[1,0])
    # Ntot = 5000 # points in fit arrays aka "lines" in this spaghetti
    # ALline = np.linspace(0.001,1.0,Ntot)
    # nlinel = linec(np.log(ALline),*poptl)
    # Psats8 = np.mean(GALs)*ALline/(nlinel*Tc_mean_MH**(nlinel-1))*(Tc_mean_MH**nlinel-Tb**nlinel)
    #
    #
    # ############################################################################
    # # Given a measured Psat, predict the A/L one would expect given our best model
    # # This section is WIP and not correct just yet....
    # ALp = pl.interp(np.array(PPs)*1e12,Psats8,ALline) # This assumes that Tc was correct?
    # ALp2 = pl.interp(np.array(PPs)*1e12,Psats2,ALline) # this would be if we could tune every Tc in each band
    #
    # # Do the same for n
    # nnp = pl.interp(np.array(PPs)*1e12,Psats8,nlinel)
    #
    #
    # ############################################################################
    # # Calculate the NEPg improvement of Tc=171 to 185 and n=3 to n_mean
    # Nrat = NEPg(Tc_mean,Pstop=1E-12,Tb=0.1,n=n_mean,LG=50)/NEPg(Tc1,Pstop=1E-12,Tb=0.1,n=3,LG=50)
    # print(Nrat)
    # for ii in ld:
    # 	ld[ii][b'NEPdet2'] = np.sqrt(Nrat*ld[ii][b'NEPg']**2 + ld[ii][b'NEPph']**2 + ld[ii][b'NEPread']**2 + ld[ii][b'NEPext']**2)
    # ip = [ld[ii][b'NEPdet2']/ld[ii][b'NEPdet'] for ii in ld]
    # print(np.mean(ip))
    # print((1-np.mean(ip))*100.,'% ')

    #######################################################################################


    ############################################################################
    # Plot Loop gain vs rfrac at 100 mK for all bolos, indicate which have autobias
    # Tb = 0.1 #K
    # LGs = []
    # rfracs = []
    # powers = []
    # abs = []
    # for col in ts:
    #     for rr in ts[col]:
    #         Tbs = ts[col][rr][b'iv'][b'conductance'][b'Tbaths']
    #         found=False
    #         for nn,(tt,ii) in enumerate(zip(Tbs,np.sort(ts[col][rr][b'iv'][b'converted_data'].keys()))):
    #             if np.isclose(Tb,tt,rtol=1E-2,atol=1E-3):
    #                 #print 'col,row,Tb:',col,row,tt,'found close to',Tb
    #                 found=True
    #                 break
    #         if found:
    #             ivkeys = ts[col][rr][b'iv'][b'converted_data'].keys()
    #             rtesidx = ts[col][rr][b'iv'][b'converted_data_indices'].index(b'rtes')
    #             ptesidx = ts[col][rr][b'iv'][b'converted_data_indices'].index(b'ptes')
    #             Rn = np.mean(ts[col][rr][b'iv'][b'converted_data'][ivkey][rtesidx][-5:])
    #             rfracs.append(ts[col][rr][b'iv'][b'converted_data'][ivkey][rtesidx]/Rn)
    #             powers.append(ts[col][rr][b'iv'][b'converted_data'][ivkey][ptesidx])
    #             LGs.append(ts[col][rr][b'iv'][b'loopgain'][ivkey][b'LGiv'])
    #             cc = col.split(b'Bay')[1]
    #             rint = int(rr.split(b'Row')[1])
    #             abs.append(hwmap[cc][rr][b'Add autobias'])
    #
    # LGs = np.array(LGs)
    # rfracs = np.array(rfracs)
    # #rps = np.array(rps)
    #
    # pl.figure('LG vs rfrac',dpi=dpi,figsize=(4,3))
    # pl.xlabel('Fractional Resistance')
    # pl.ylabel('Loop Gain')
    # pl.title('Loop Gain in Transition')
    # pl.grid(True)
    # for rr,pp,ll,aa in zip(rfracs,powers,LGs,abs):
    #     if aa == 1: # autobias dotted lines
    #         ls=':'
    #     else:
    #         ls='-'
    #     # only plot until latching point
    #     mask = (ll>0) * (rr>0.025) * np.hstack([True,(np.diff(rr)>0)])
    #     #didx = ivLib.FindDiscontinuity(x=rr[mask], y=pp[mask], thresh=5.0)
    #     #if any(didx):
    #     #     print didx,len(rr[mask])
    #     #     didx = didx[0]
    #     # else:
    #     #     didx=0
    #     #pl.plot(rr[mask][didx:],ll[mask][didx:],ls=ls,alpha=0.7) #,marker='.'
    #     try:
    #         pl.plot(rr[mask],ll[mask],ls=ls,alpha=0.7) #,marker='.'
    #     except:
    #         embed();sys.exit()
    #
    # pl.xlim(-0.02,1.02)
    # pl.ylim(-1,50)
    #
    # wfname = os.path.join(outpath,'lbird_LG_vs_rfrac_{:.0f}mk.png'.format(Tb*1e3))
    # pl.savefig(fname=wfname,bbox_inches='tight')

    # ############################################################################
    # # Plot Loop gain vs rfrac at 100 mK for all bolos, indicate which have autobias
    # Tb = 0.1 #K
    # LGs = []
    # rfracs = []
    # powers = []
    # abs = []
    # found=False
    # for col in ts:
    #     for rr in ts[col]:
    #         Tbs = ts[col][rr][b'iv'][b'conductance'][b'Tbaths']
    #         for nn,(tt,ii) in enumerate(zip(Tbs,np.sort(ts[col][rr][b'iv'][b'converted_data'].keys()))):
    #             if np.isclose(Tb,tt,rtol=1E-2,atol=1E-3):
    #                 print 'col,row,Tb:',col,row,tt,'found close to',Tb
    #                 found=True
    #                 break
    #         if found:
    #             ivkeys = ts[col][rr][b'iv'][b'converted_data'].keys()
    #             rtesidx = ts[col][rr][b'iv'][b'converted_data_indices'].index(b'rtes')
    #             ptesidx = ts[col][rr][b'iv'][b'converted_data_indices'].index(b'ptes')
    #             Rn = np.mean(ts[col][rr][b'iv'][b'converted_data'][ivkey][rtesidx][-5:])
    #             rfracs.append(ts[col][rr][b'iv'][b'converted_data'][ivkey][rtesidx]/Rn)
    #             powers.append(ts[col][rr][b'iv'][b'converted_data'][ivkey][ptesidx])
    #             LGs.append(ts[col][rr][b'iv'][b'loopgain'][ivkey])
    #             cc = col.split(b'Bay')[1]
    #             rint = int(rr.split(b'Row')[1])
    #             abs.append(hwmap[cc][rint][b'Add autobias'])
    #
    # LGs = np.array(LGs)
    # rfracs = np.array(rfracs)
    # #rps = np.array(rps)
    #
    # pl.figure('LG vs rfrac',dpi=dpi,figsize=(4,3))
    # pl.xlabel('Fractional Resistance')
    # pl.ylabel('Loop Gain')
    # pl.title('Loop Gain in Transition')
    # pl.grid(True)
    # for rrr,pp,ll,aa in zip(rfracs,powers,LGs,abs):
    #     if aa == 1: # autobias dotted lines
    #         ls=':'
    #     else:
    #         ls='-'
    #     # only plot until latching point
    #     mask = (ll>0) * (rrr>0.025)
    #     didx = ivLib.FindDiscontinuity(x=rrr[mask], y=pp[mask], thresh=5.0)
    #     if any(didx):
    #         print didx,len(rrr[mask])
    #         didx = didx[0]
    #     else:
    #         didx=0
    #     #pl.plot(rrr[mask][didx:],ll[mask][didx:],ls=ls,alpha=0.7) #,marker='.'
    #     pl.plot(rrr[mask],ll[mask],ls=ls,alpha=0.7) #,marker='.'
    #
    # pl.xlim(-0.02,1.02)
    # pl.ylim(-1,50)
    #

    ############################################################################
    # # Plot Loop gain vs rfrac for each bath temperatures for all bolos, indicate which have autobias
    # plot_LG_all_Tbs = False
    # Trng = (0.120,0.180) # K
    # LGs = []
    # rfracs = []
    # powers = []
    # abs = [] # autobias yes or no
    # LGrng = [] # average the loopgain curve into one number over the LGrange
    # LGstd = [] # std of LG over LGrange
    #
    # LGd = {} # average the average loop gain across bath_temperatures in Trng
    # for col in ts:
    #     if not col in LGd:
    #         LGd[col] = {}
    #     for rr in ts[col]:
    #         if not rr in LGd[col]:
    #             LGd[col][rr] = {'LGrng':[],
    #                                'LGstd':[],
    #                                'Tbs':[],
    #                                'Rps':[],
    #                                'LGmaxs':[],
    #                                'LGmidxs':[]}
    #         Tbs = ts[col][rr][b'iv'][b'conductance'][b'Tbaths']
    #         ivkeys = list(ts[col][rr][b'iv'][b'converted_data'].keys())
    #         rtesidx = ts[col][rr][b'iv'][b'converted_data_indices'].index(b'rtes')
    #         ptesidx = ts[col][rr][b'iv'][b'converted_data_indices'].index(b'ptes')
    #         rsidx = ts[col][rr][b'iv'][b'converted_data_indices'].index(b'rs')
    #         for tt,ivkey in zip(Tbs,ivkeys):
    #             Rn = np.mean(ts[col][rr][b'iv'][b'converted_data'][ivkey][rtesidx][-5:])
    #             rfraca = ts[col][rr][b'iv'][b'converted_data'][ivkey][rtesidx]/Rn
    #             power = ts[col][rr][b'iv'][b'converted_data'][ivkey][ptesidx]
    #             Rp = ts[col][rr][b'iv'][b'converted_data'][ivkey][rsidx][2]
    #             LG = ts[col][rr][b'iv'][b'loopgain'][ivkey][b'LGiv']
    #             LGrng = ts[col][rr][b'iv'][b'loopgain'][ivkey][b'LGrng']
    #             LGstd = ts[col][rr][b'iv'][b'loopgain'][ivkey][b'LGstd']
    #             LGrange = ts[col][rr][b'iv'][b'loopgain'][ivkey][b'LGrange']
    #             LGmax = ts[col][rr][b'iv'][b'loopgain'][ivkey][b'LGmax']
    #             LGmidx = ts[col][rr][b'iv'][b'loopgain'][ivkey][b'LGmidx']
    #             cc = col.split(b'Bay')[1]
    #             rint = int(rr.split(b'Row')[1])
    #             ab = hwmap[cc][rint][b'Add autobias']
    #             CC = hwmap[cc][rint][b'C']
    #             AL = hwmap[cc][rint][b'total w/l']
    #             split = hwmap[cc][rint][b'Split']
    #
    #             ttr = np.round(tt,3)
    #             if ttr>=Trng[0] and ttr<=Trng[1]:
    #                 LGd[col][rr][b'LGrng'].append(LGrng)
    #                 LGd[col][rr][b'LGstd'].append(LGstd)
    #                 LGd[col][rr][b'Tbs'].append(tt)
    #                 LGd[col][rr][b'Rps'].append(Rp)
    #                 LGd[col][rr][b'LGmaxs'].append(LGmax)
    #                 LGd[col][rr][b'LGmidxs'].append(LGmidx)
    #
    #             if ab == 1: # autobias dotted lines
    #                 ls=':'
    #             else:
    #                 ls='-'
    #
    #             # if CC == 8106:
    #             #     color='b'
    #             # elif CC >= 8106:
    #             #     color='g'
    #             # else:
    #             #     color='r'
    #
    #             if AL == 0.19:
    #                 marker='o'
    #             elif AL >= 0.19:
    #                 marker='x'
    #             else:
    #                 marker='s'
    #
    #             pl.figure('LG vs rfrac {:.0f}'.format(tt*1e3),dpi=dpi,figsize=(4,3))
    #             pl.xlabel('Fractional Resistance')
    #             pl.ylabel('Loop Gain')
    #             pl.title('Loop Gain in Transition at {:.0f} mK'.format(tt*1e3))
    #             pl.grid(True)
    #             rlp = pl.plot(rfraca,LG,ls=ls,alpha=0.7)
    #             pl.errorbar(x=LGrange,y=[LGrng,LGrng],yerr=[LGstd,LGstd],color=rlp[-1].get_color())
    #             pl.xlim(-0.02,1.02)
    #             pl.ylim(-1,50)
    #
    #             # fig = pl.figure('rfvstau_%s%d'%(cc,bb),dpi=dpi,figsize=1.8*figsize) #,gridspec_kw={'width_ratios': [2, 1]})
    #             # #fig = pl.figure('rfvstau_%d'%bb,dpi=dpi,figsize=figsize)
    #             # fig.suptitle('Time Constant from Complex Responsivity for {} {}'.format(ColName,RowName))
    #             # ax0 = fig.add_subplot(2, 2, 1)
    #
    #             fig = pl.figure('LGrng vs Rp {:.0f}'.format(tt*1e3),dpi=dpi,figsize=(4*2,3))
    #             fig.suptitle('Loop Gain vs Autobias and Split {:.0f} mK'.format(tt*1e3))
    #             ax0 = fig.add_subplot(1, 2, 1)
    #             ax0.set_xlabel('Autobias Resistance ($\Omega$)')
    #             ax0.set_ylabel('Loop Gain from {} to {} Rn'.format(LGrange[0],LGrange[1]))
    #             ax1 = fig.add_subplot(1, 2, 2)
    #             ax1.set_xlabel('Bolo Split #')
    #             ax0.set_ylabel('Loop Gain from {} to {} Rn'.format(LGrange[0],LGrange[1]))
    #             if ab:
    #                 ax0.errorbar(x=Rp,y=LGrng,yerr=LGstd,ls='None',marker=marker,color=rlp[-1].get_color())
    #             else:
    #                 ax1.errorbar(x=split,y=LGrng,yerr=LGstd,ls='None',marker=marker,color=rlp[-1].get_color())
    #             ax0.grid(True,zorder=-1)
    #             ax1.grid(True,zorder=-1)
    #             ax0.set_ylim(-1,20)
    #             ax1.set_ylim(-1,50)
    #
    # all_Tbs = np.array([0.090, 0.100, 0.110, 0.120, 0.130 , 0.140, 0.150, 0.160, 0.170, 0.180, 0.190, 0.200, 0.205])
    # for tt in all_Tbs:
    #     fig = pl.figure('LGrng vs Rp {:.0f}'.format(tt*1e3),dpi=dpi,figsize=(4*2,3))
    #     wfname = os.path.join(outpath,'lbird_LG_vs_rp_split_{:.0f}mk.png'.format(tt*1e3))
    #     pl.savefig(fname=wfname,bbox_inches='tight')
    #
    #
    # # Now plot the bath temp averaged values
    # # Save the LGave of the non-autobias bolos, baseline=bl
    # LGrange = ts[col][rr][b'iv'][b'loopgain'][ivkey][b'LGrange']
    # LGavesbl=[]
    # LGstdsbl=[]
    # fig = pl.figure('LGrng vs Rp over {:.0f} to {:.0f} mK'.format(Trng[0]*1e3,Trng[1]*1e3),dpi=dpi,figsize=(4*2,3))
    # fig.suptitle('Loop Gain vs Autobias and Split from {:.0f} to {:.0f} mK'.format(Trng[0]*1e3,Trng[1]*1e3))
    # ax0 = fig.add_subplot(1, 2, 1)
    # ax1 = fig.add_subplot(1, 2, 2)
    # ax0.set_xlabel('Bolo Split #')
    # ax0.set_ylabel('Loop Gain from {} to {} Rn'.format(LGrange[0],LGrange[1]))
    # ax1.set_xlabel('Autobias Resistance ($\Omega$)')
    # for col in LGd:
    #     for rr in LGd[col]:
    #         cc = col.split(b'Bay')[1]
    #         rint = int(rr.split(b'Row')[1])
    #         ab = hwmap[cc][rint][b'Add autobias']
    #         CC = hwmap[cc][rint][b'C']
    #         AL = hwmap[cc][rint][b'total w/l']
    #         split = hwmap[cc][rint][b'Split']
    #         LGrngs = np.array(LGd[col][rr][b'LGrng'])
    #         LGstds = np.array(LGd[col][rr][b'LGstd'])
    #         LGmax = np.mean(LGd[col][rr][b'LGmaxs'])
    #         LGave = np.average(LGrngs,weights=1./LGstds**2)
    #         LGstd = np.average(LGstds,weights=1./LGstds**2)
    #         Rp = np.average(LGd[col][rr][b'Rps'])
    #         if ab:
    #             ax1.errorbar(x=Rp,y=LGave,yerr=LGstd,marker='o',ls='None',alpha=0.7) #,marker=marker,color=rlp[-1].get_color())
    #             ax1.errorbar(x=Rp,y=LGmax,marker='.',ls='None',alpha=0.7)
    #         else:
    #             LGavesbl.append(LGave)
    #             LGstdsbl.append(LGstd)
    #             if LGave<9: marker='x'
    #             else: marker='o'
    #             splt = ax0.errorbar(x=split,y=LGave,yerr=LGstd,marker=marker,ls='None',alpha=0.7) #,marker=marker,color=rlp[-1].get_color())
    #             ax0.errorbar(x=split,y=LGmax,marker='.',ls='None',alpha=0.7)#,color=splt[-1][-1].get_color())
    # ax0.grid(True,zorder=-1)
    # ax1.grid(True,zorder=-1)
    # ax0.set_ylim(-1,30)
    # ax1.set_xlim(xmin=-0.01)
    # ax1.set_ylim(-1,30)
    #
    # LGavesbl = np.array(LGavesbl)
    # LGstdsbl = np.array(LGstdsbl)
    # LGcut = 9.
    # LGavebl = np.average(LGavesbl[LGavesbl>LGcut],weights=1./LGstdsbl[LGavesbl>LGcut]**2)
    # LGstdbl = np.average(LGstdsbl[LGavesbl>LGcut],weights=1./LGstdsbl[LGavesbl>LGcut]**2)
    # ax1.errorbar(x=0.0003,y=LGavebl,yerr=LGstdbl,marker='o',ms=10,ls='None',alpha=0.7)
    # ax0.axhspan(ymin=LGavebl-LGstdbl, ymax=LGavebl+LGstdbl, alpha=0.4,color='b')
    #
    # wfname = os.path.join(outpath,'lbird_LG_vs_rp_split_{:.0f}_to_{:.0f}_mK'.format(Trng[0]*1e3,Trng[1]*1e3))
    # pl.savefig(fname=wfname,bbox_inches='tight')
    #
    # ##################################################################################
    # # Now plot just what's on ax1 on its own
    # fig = pl.figure('LGrng vs Rp over {:.0f} to {:.0f} mK II'.format(Trng[0]*1e3,Trng[1]*1e3),dpi=dpi,figsize=(4,3))
    # fig.suptitle('Loop Gain vs Autobias from {:.0f} to {:.0f} mK'.format(Trng[0]*1e3,Trng[1]*1e3))
    # ax1 = fig.add_subplot(1, 1, 1)
    # #ax0.set_xlabel('Bolo Split #')
    # ax1.set_ylabel('Loop Gain from {} to {} Rn'.format(LGrange[0],LGrange[1]))
    # ax1.set_xlabel('Autobias Resistance ($\Omega$)')
    # for col in LGd:
    #     for rr in LGd[col]:
    #         cc = col.split(b'Bay')[1]
    #         rint = int(rr.split(b'Row')[1])
    #         ab = hwmap[cc][rint][b'Add autobias']
    #         CC = hwmap[cc][rint][b'C']
    #         AL = hwmap[cc][rint][b'total w/l']
    #         split = hwmap[cc][rint][b'Split']
    #         LGrngs = np.array(LGd[col][rr][b'LGrng'])
    #         LGstds = np.array(LGd[col][rr][b'LGstd'])
    #         LGmax = np.mean(LGd[col][rr][b'LGmaxs'])
    #         LGave = np.average(LGrngs,weights=1./LGstds**2)
    #         LGstd = np.average(LGstds,weights=1./LGstds**2)
    #         Rp = np.average(LGd[col][rr][b'Rps'])
    #         if ab:
    #             ax1.errorbar(x=Rp,y=LGave,yerr=LGstd,marker='o',ls='None',alpha=0.7,color=bluish) #,marker=marker,color=rlp[-1].get_color())
    #             #ax1.errorbar(x=Rp,y=LGmax,marker='.',ls='None',alpha=0.7)
    # ax1.grid(True,zorder=-1)
    # ax1.set_xlim(xmin=-0.01)
    # ax1.set_ylim(-1,20)
    #
    # LGavesbl = np.array(LGavesbl)
    # LGstdsbl = np.array(LGstdsbl)
    # LGcut = 9.
    # LGavebl = np.average(LGavesbl[LGavesbl>LGcut],weights=1./LGstdsbl[LGavesbl>LGcut]**2)
    # LGstdbl = np.average(LGstdsbl[LGavesbl>LGcut],weights=1./LGstdsbl[LGavesbl>LGcut]**2)
    # bigp = ax1.errorbar(x=0.0003,y=LGavebl,yerr=LGstdbl,marker='o',ms=10,ls='None',alpha=0.7,color='orange',label='$LG_0$')
    #
    # aline = np.linspace(0,0.25,1000)
    # #LGofAb0 = (LGavebl-LGstdbl)/((1+aline/0.2)**2) # Hannes' formula
    # LGofAb0 = (LGavebl)/((1+aline/0.1)**2) # Hannes' formula
    # LGofAb1 = (LGavebl)/((1+aline/0.9)**2) # Hannes' formula
    # LGofAb2 = (LGavebl)/((1+aline/0.7)**2)
    # ax1.plot(aline,LGofAb1,label='R(T)=0.9Rn',color='g',alpha=0.7)
    # ax1.plot(aline,LGofAb2,label='R(T)=0.7Rn',color='purple',alpha=0.7)
    # ax1.plot(aline,LGofAb0,label='R(T)=0.1Rn',color='r',alpha=0.7)
    # #ax1.plot(aline,LGofAb,label='LG=\frac{LG$_0$}{(1+R_x/R_{TES})^2}')
    # ax1.legend(title='$\\frac{{LG_0}}{{(1+R_{A}/R(T))^2}}$',fontsize='small')
    # wfname = os.path.join(outpath,'lbird_LG_vs_rp_{:.0f}_to_{:.0f}_mK'.format(Trng[0]*1e3,Trng[1]*1e3))
    # pl.savefig(fname=wfname,bbox_inches='tight')
    #
    # ############################################################################
    # # How does Rp affect the LG, take baseline bolo LG and increase Rp
    # col='BayC'
    # rr='Row08'
    # ivkey='iv005'
    # rtesidx = ts[col][rr][b'iv'][b'converted_data_indices'].index(b'rtes')
    # itesidx = ts[col][rr][b'iv'][b'converted_data_indices'].index(b'ites')
    # vtesidx = ts[col][rr][b'iv'][b'converted_data_indices'].index(b'vtes')
    # rsidx   =  ts[col][rr][b'iv'][b'converted_data_indices'].index(b'rs')
    # LG = ts[col][rr][b'iv'][b'loopgain'][ivkey][b'LGiv'] # baseline loopgain
    # rtes = ts[col][rr][b'iv'][b'converted_data'][ivkey][rtesidx]
    # ites = ts[col][rr][b'iv'][b'converted_data'][ivkey][itesidx]
    # vtes = ts[col][rr][b'iv'][b'converted_data'][ivkey][vtesidx]
    # rs = ts[col][rr][b'iv'][b'converted_data'][ivkey][rsidx]
    # R_p = rs[2]
    # #R_p=0.1
    #
    # Rn = np.mean(rtes[-50:])
    # rfraca = rtes/Rn
    #
    # # Invert LG to get dI/dV
    # dIdV = (1-LG)/(LG*(rtes-R_p)+rtes+R_p)
    #
    # Rps = np.linspace(0,0.25,20)
    # LGRps = np.array([(1-(rtes+rrpp)*dIdV)/(1+(rtes-rrpp)*dIdV) for rrpp in Rps])
    # # Now average the LG over LGrange
    # LGRpRngs = np.array([np.mean(ll[(rfraca>LGrange[0]) * (rfraca<LGrange[1])]) for ll in LGRps])
    # #ax1.plot(Rps,LGRpRngs)
    #
    # N_windows = 12
    # order=1
    # lga = np.zeros((N_windows ,len(ites)))
    # for jj in range(N_windows):
    #     ws = 2*jj+2+order # window size, for linear order +3, for quadratic +4, etc...
    #     dI = savitzky_golay(y=ites, window_size=ws, order=1, deriv=1, rate=1)
    #     dV = savitzky_golay(y=vtes, window_size=ws, order=1, deriv=1, rate=1)
    #     # interpolate endpoints, they end up being tiny and throw off calculation later,
    #     xx = np.arange(len(dI))
    #     dI = pl.interp(xx,xx[1:-1],dI[1:-1])
    #     dV = pl.interp(xx,xx[1:-1],dV[1:-1])
    #     # Interpolate over zeros and if interpolation left any zeros, make them the median
    #     if any(dI==0):
    #     		dI = pl.interp(xx,xx[dI!=0],dI[dI!=0])
    #     		if any(dI==0):
    #     				dI[dI==0]=np.median(dI)
    #     dV0 = np.isclose(dV,0,rtol=1E-17,atol=1E-22)
    #     if any(dV0):
    #     		dV = pl.interp(xx,xx[~dV0],dV[~dV0])
    #     		if any(dV==0):
    #     				dV[dV==0]=np.median(dV)
    #     lga[jj] = (1-(rtes+R_p)*dI/dV)/(1+(rtes-R_p)*dI/dV)
    #     #pl.plot(rtes,LGiv,'.',label=ws)
    #     #lga[jj] = (1-rtes*dI/dV)/(1+rtes*dI/dV)
    # LGiv = np.mean(lga,axis=0)
    # LGivstd = np.std(lga,axis=0)
    # LGrange = (0.7,0.9)
    # rngmsk = (rfraca>LGrange[0])*(rfraca<LGrange[1]) # LGrange mask
    # LGrng = np.mean(LGiv[rngmsk])
    # LGstd = np.std(LGiv[rngmsk])
    #
    # pl.figure('Rp variations')
    # rfcut = 0.34
    # pl.errorbar(x=rfraca[rfraca>rfcut],y=LGiv[rfraca>rfcut],yerr=LGivstd[rfraca>rfcut],marker='.',ls='-',alpha=0.5)
    # for ll in LGRps:
    #     pl.plot(rfraca[rfraca>rfcut],ll[rfraca>rfcut])
    # pl.plot(rfraca,np.mean(lga,axis=0))
    #
    # #*****************************************************************************
    # # does the PdAu affect the G/A/L?
    # bmsk = (nsplits>=3)*(nsplits<=5)
    # Vs = np.array(Vs)
    # GALs = np.array(GALs)
    # popt,pcov = curve_fit(linec, 2.*Vs[bmsk], GALs[bmsk], p0=[1,0])
    # vline = np.linspace(1,18E3,100)
    # gline = linec(vline,*popt)
    # mstr='m: {:.2f} fW/K/$\mu$m$^3$\nc: {:.0f} pW/K/$\mu$m'.format(popt[0]*1e3,popt[1])
    #
    # fig = pl.figure('PdAu GAL',figsize=figsize,dpi=dpi)
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_xlabel('PdAu Area ($\mu$m$^2$)')
    # ax.set_ylabel('G/A/L (pW/K/$\mu$m)')
    # ax.set_title('PdAu Area Affects Conductivity')
    # ax.plot(vline,gline,color='r',label=mstr,alpha=0.7)
    # ax.plot(2.*Vs[bmsk],GALs[bmsk],marker='o',ls='None',alpha=0.7,color=bluish)
    # ax.grid(True)
    # ax.legend(loc='lower right')
    # #ax.text(x=0.99, y=0.01, s=mstr, fontsize=10, ha='right', va='bottom', transform=ax.transAxes)
    # wfname = os.path.join(outpath,'lbird_PdAu_Area_vs_Conductivity.png')
    # pl.savefig(fname=wfname,bbox_inches='tight')
    #
    #
    # #*****************************************************************************
    # # Did the 4th leg bias work as expected?
    # bmsk = (nsplits>=12)
    # # convert split# to leg bias,
    # lgb = nsplits[bmsk]-12
    # lgb[lgb>=6]-=6
    # lgb*=2 # 2 microns of bias per bolo number
    #
    # # Group bolometers into chip groups
    # # initialize dict
    # lbd = {}
    # for cc in np.unique(cids[bmsk],axis=0):
    #     lbd[str(cc)] = []
    # # fill dict with (leg4 bias, GAL) for each chip
    # for cc,nn,gg in zip(cids[bmsk],lgb,GALs[bmsk]):
    #     lbd[str(cc)].append([nn,gg])
    # # convert to arrays
    # for cc in lbd:
    #     lbd[cc] = np.array(lbd[cc])
    # # subtract mean
    # for cc in lbd:
    #     lbd[cc][:,1]-= np.mean(lbd[cc][:,1])
    #
    # colors = turbo_colors(np.linspace(0,1,len(lbd))) # reverse these so the colors line up with the values
    #
    # fig = pl.figure('Leg 4 GAL bias2',figsize=figsize,dpi=dpi)
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_xlabel('Leg Bias ($\mu$m)')
    # ax.set_ylabel('$\Delta$G/A/L from Chip Average (pW/K/$\mu$m)')
    # ax.set_title('Conductivity vs Leg 4 Bias')
    # #ax.plot(vline,gline,color='r',label=mstr,alpha=0.7)
    # #ax.plot(lgb,GALs[bmsk],marker='o',ls='None',alpha=0.7,color=bluish)
    # pitch=6.15 # mm
    # lgbs = []
    # dGALs = []
    # for chip,color in zip(lbd,colors):
    #     for nn,(ll,gg) in enumerate(lbd[chip]):
    #         yy,xx = chip.strip('[b').strip(']').split(b' ')[:2]
    #         chipstr = '({:.0f},{:.0f})'.format(pitch*(int(xx)-9),pitch*(int(yy)-9))
    #         ax.plot(ll,gg,color=color,marker='o',ls='None',alpha=0.7,label=chipstr if nn==0 else '')
    #         lgbs.append(ll)
    #         dGALs.append(gg)
    # popt,pcov = curve_fit(linec, lgbs, dGALs, p0=[1,0])
    # lgline = np.linspace(0,10,100)
    # dgline = linec(lgline,*popt)
    # #ax.plot(lgline,dgline,color='k')
    #
    # ax.grid(True)
    # legendary = ax.legend(title='Chip Coords (mm)',fontsize='small',loc='lower right')
    # pl.setp(legendary.get_title(),fontsize='xx-small')
    # #ax.legend(loc='lower right')
    # #ax.text(x=0.99, y=0.01, s=mstr, fontsize=10, ha='right', va='bottom', transform=ax.transAxes)
    # wfname = os.path.join(outpath,'lbird_Leg4Bias_vs_Conductivity.png')
    # pl.savefig(fname=wfname,bbox_inches='tight')
    #
    # ###############################################################
    # # Do the same w/o A/L, just conductance
    #
    # # Group bolometers into chip groups
    # # initialize dict
    # lbd = {}
    # for cc in np.unique(cids[bmsk],axis=0):
    #     lbd[str(cc)] = []
    # # fill dict with (leg4 bias, G) for each chip
    # for cc,nn,gg in zip(cids[bmsk],lgb,Gs[bmsk]):
    #     lbd[str(cc)].append([nn,gg])
    # # convert to arrays
    # for cc in lbd:
    #     lbd[cc] = np.array(lbd[cc])
    # # subtract mean
    # for cc in lbd:
    #     lbd[cc][:,1]-= np.mean(lbd[cc][:,1])
    #
    # fig = pl.figure('Leg 4 G bias2',figsize=figsize,dpi=dpi)
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_xlabel('Leg Bias ($\mu$m)')
    # ax.set_ylabel('$\Delta$G from Chip Average (pW/K)')
    # ax.set_title('Conductance vs Leg 4 Bias')
    # #ax.plot(vline,gline,color='r',label=mstr,alpha=0.7)
    # #ax.plot(lgb,GALs[bmsk],marker='o',ls='None',alpha=0.7,color=bluish)
    # pitch=6.15 # mm
    # lgbs = []
    # dGALs = []
    # for chip,color in zip(lbd,colors):
    #     for nn,(ll,gg) in enumerate(lbd[chip]):
    #         yy,xx = chip.strip('[b').strip(']').split(b' ')[:2]
    #         chipstr = '({:.0f},{:.0f})'.format(pitch*(int(xx)-9),pitch*(int(yy)-9))
    #         ax.plot(ll,gg,color=color,marker='o',ls='None',alpha=0.7,label=chipstr if nn==0 else '')
    #         lgbs.append(ll)
    #         dGALs.append(gg)
    # popt,pcov = curve_fit(linec, lgbs, dGALs, p0=[1,0])
    # lgline = np.linspace(0,10,100)
    # dgline = linec(lgline,*popt)
    # #ax.plot(lgline,dgline,color='k')
    #
    # ax.grid(True)
    # legendary = ax.legend(title='Chip Coords (mm)',fontsize='small',loc='lower right')
    # pl.setp(legendary.get_title(),fontsize='xx-small')
    # #ax.legend(loc='lower right')
    # #ax.text(x=0.99, y=0.01, s=mstr, fontsize=10, ha='right', va='bottom', transform=ax.transAxes)
    # wfname = os.path.join(outpath,'lbird_Leg4Bias_vs_Conductance.png')
    # pl.savefig(fname=wfname,bbox_inches='tight')
    #
    # # Make a nice PvT plot for LTD
    # # fit to thermal conductance power law.
    # BayName = 'BayB'
    # RowName = 'Row10'
    # rsf = 1.3
    # pl.figure('pt plane',figsize=(3.6/rsf,3/rsf),dpi=200.*rsf)
    # Tbaths = ts[BayName][RowName][b'iv'][b'conductance'][b'Tbaths']
    # Psats = ts[BayName][RowName][b'iv'][b'conductance'][b'Psats'][:,0]
    # t_array = np.linspace(.05,0.2,100)
    # KK = ts[BayName][RowName][b'iv'][b'conductance'][b'fit parameters'][b'K'][0]
    # Kstd = ts[BayName][RowName][b'iv'][b'conductance'][b'fit parameters'][b'sigma_K'][0]
    # TT = ts[BayName][RowName][b'iv'][b'conductance'][b'fit parameters'][b'T'][0]
    # Tstd = ts[BayName][RowName][b'iv'][b'conductance'][b'fit parameters'][b'sigma_T'][0]
    # nn = ts[BayName][RowName][b'iv'][b'conductance'][b'fit parameters'][b'n'][0]
    # nstd = ts[BayName][RowName][b'iv'][b'conductance'][b'fit parameters'][b'sigma_n'][0]
    # GG = ts[BayName][RowName][b'iv'][b'conductance'][b'derived parameters'][b'G'][0]
    # Gstd = np.sqrt(((nn*TT**(nn-1)*Kstd)**2) + (-KK*TT**(nn-1) - nn*KK*np.log(TT)*TT**(nn-1))**2 * nstd**2)
    # pfit = np.array([KK,TT,nn])
    # p_array = ivLib.kTnFitFunction(pfit,t_array)
    # pl.plot(t_array*1e3, p_array*1e12,color='k',ls='-',label='Model')
    # pl.plot(Tbaths*1e3,Psats*1e12,'o',label='Data',color=bluish)
    # fitstr = 'G: {:.2f}$\pm${:.2f} pW/K\nTc: {:.1f}$\pm${:.1f} mK\nn: {:.2f}$\pm${:.2f}'.format(GG*1e12,Gstd*1e12,
    #                                                                                     TT*1e3,Tstd*1e3,
    #                                                                                     nn,nstd)
    # pl.annotate(xy=(0.05,0.05),s=fitstr,xycoords='axes fraction')
    # pl.xlabel('Bath Temperature [mK]')
    # pl.ylabel('Power [pW]')
    # pl.title('Electrothermal Response')
    # pl.grid(True)
    # pl.ylim(0,1.2)
    # pl.xlim(75,200)
    # pl.legend(loc=1)
    # wfname = os.path.join(outpath,'lbird_LTD19_PvT.png')
    # pl.savefig(fname=wfname,bbox_inches='tight')



    return


if __name__=='__main__':
    main()

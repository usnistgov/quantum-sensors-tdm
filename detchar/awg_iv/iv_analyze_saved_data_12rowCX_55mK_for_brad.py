import pickle
from dataclasses import dataclass
from typing import Any
import mass
from mass.mathstat.fitting import fit_kink_model
import numpy as np
import pylab as plt
plt.ion()
plt.close("all")

@dataclass
class IVvsField:
    ivs_pos: Any
    ivs_neg: Any
    v_fcs: Any
    
    def get_ivdata(self, ind_fc, row, polarity):
        if polarity == 1:
            return self.ivs_pos[ind_fc][row]
        elif polarity == -1:
            return self.ivs_neg[ind_fc][row]
        else:
            raise Exception("polarity must be 1 or -1") 
    


@dataclass
class RawAWGRowIVData:
    row: int
    ic_fb_units: float
    vbias: Any
    phi0_fb: float
    sampled_period_s: float
    # fb_ic_fixed: Any
    fb: Any
    err: Any
    
    def polarity(self):
        return np.sign(self.vbias[len(self.vbias)//2])
    
    def get_fb_ic_fixed(self):
        fb_fixed = add_flux_jumps(self.fb, phi0_fb = self.phi0_fb, 
                                  fb_step_threshold=self.phi0_fb*0.8)
        ic_fix_ind = np.where(np.abs(self.err)>500)[0][0]
        fb_ic_fixed = fb_fixed[:]-fb_fixed[0]
        fb_ic_fixed[ic_fix_ind:]-=fb_ic_fixed[-1]
        ic_ind = np.argmax(np.abs(fb_ic_fixed[:ic_fix_ind])) # limit the range we look over a bi
        ic_fb_units = fb_ic_fixed[ic_ind]
            
        return fb_ic_fixed, ic_fb_units, ic_ind
    
data = pickle.load(open("20211209_CX_55mK_for_brad.pkl","rb"))


def add_flux_jumps(fb, phi0_fb, fb_step_threshold):
    """return an array like fb, but with flux jumps resolved and one value removed from the end
    method: find indicies with np.diff(fb) greater ro less than fb_step_threshold
    and add phi0 units with the correct sign"""
    out = fb[1:]+np.cumsum(np.diff(fb)<-fb_step_threshold)*phi0_fb
    out -= np.cumsum(np.diff(fb)>fb_step_threshold)*phi0_fb
    return out

def find_kink_approx(fb, delta_threshold = 1000, pointstep = 10):
    for i in range(10, len(fb)):
        a = fb[i]-fb[i-pointstep]
        b = fb[pointstep] - fb[i]
        if np.abs(b)-np.abs(a) > delta_threshold:
            break
    return i

def find_kink(fb_ic_fixed, polarity):
    kink_ind_approx = find_kink_approx(fb_ic_fixed)
    a = kink_ind_approx-300
    b = kink_ind_approx+300
    model, (kbest,a,b,kink_c), X2 = fit_kink_model(np.arange(a,b),
                                fb_ic_fixed[a:b])
    kink_ind = int(kbest)
    # assert np.sign(b) == -np.sign(polarity), "kink model found wrong polarity"
    return kink_ind

def count_zeros(x):
    for i in range(len(x)):
        if x[i] != 0:
            break
    return i


@dataclass
class SampledIV:
    vbias: Any
    fb_mean: Any
    fb_std: Any
    polarity: int
    fb_ic_fixed: Any
    sample_inds: Any
    kink_ind: int
    
def get_sampled_iv(ind_fc, row, polarity, debug=False):

    iv = data.get_ivdata(ind_fc=ind_fc, row=row, polarity=polarity)
    fb_ic_fixed, ic_fb_units, ic_ind = iv.get_fb_ic_fixed()
    kink_ind = find_kink(fb_ic_fixed, polarity=iv.polarity())
    # hard coded kink shift to make this work
    kink_ind = 147204
    srate = 100

    n_leading_zeros = count_zeros(iv.vbias)-1
    vbias = iv.vbias[n_leading_zeros:-n_leading_zeros]
    step_period_ind = int(1/(iv.sampled_period_s*srate))
    n_for_std = 100
    sample_end_offset = kink_ind
    sample_start_offset = sample_end_offset-n_for_std
    r = np.arange(0,len(vbias))*step_period_ind
    sample_inds = sample_end_offset+r
    fb_sampled = fb_ic_fixed[sample_end_offset+r]
    fb_sampled_mean = np.zeros_like(fb_sampled)
    fb_sampled_std = np.zeros_like(fb_sampled)
    for i,R in enumerate(r):
        a = sample_start_offset+R
        b = sample_end_offset+R
        fb_sampled_mean[i] = np.mean(fb_ic_fixed[a:b])
        fb_sampled_std[i] = np.std(fb_ic_fixed[a:b])
        
    t_ms = np.arange(len(fb_ic_fixed))*iv.sampled_period_s
    if debug:
        plt.figure(figsize=(6,4))
        plt.plot(t_ms,fb_ic_fixed)
        plt.plot(t_ms[ic_ind], ic_fb_units, "o", label="ic")
        # plt.axvline(t_ms[ic_fix_ind], color="red", label="ic fix ind")
        plt.plot(t_ms[kink_ind], fb_ic_fixed[kink_ind], "s", label="kink")
        plt.plot(t_ms[sample_inds], fb_ic_fixed[sample_inds], ".", label="sample")
        plt.xlabel("t_ms")
        plt.ylabel("fb ic_fixed")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.figure()
        plt.errorbar(vbias, fb_sampled_mean, yerr=fb_sampled_std*20)
        plt.xlabel("vbias")
        plt.ylabel("fb")
        plt.grid(True)
        
    return SampledIV(vbias, fb_sampled_mean, 
                     fb_sampled_std, iv.polarity(), 
                     fb_ic_fixed, sample_inds, kink_ind)

for row in range(0,12,3):
    n_fc = len(data.v_fcs)
    n_fc_plot = 1
    plt.figure()
    for polarity in [1,-1]:
        cm = plt.get_cmap("coolwarm", n_fc)
        labels = []
        for ind_fc in np.arange(0,n_fc,n_fc//n_fc_plot):
            v_fc = data.v_fcs[ind_fc]
            siv = get_sampled_iv(ind_fc=ind_fc, row=row,polarity=polarity)
            plt.errorbar(siv.vbias, siv.fb_mean, 
                        yerr=siv.fb_std*20, 
                        label=f"v_fc={v_fc:0.2f}V,p={polarity}",
                        color = cm(ind_fc))
            labels.append(f"v_fc={v_fc:0.2f}V")
    plt.xlabel("-abs(vbias)")
    plt.ylabel("fb")
    plt.grid(True)
    plt.legend(labels,loc="center left")
    plt.tight_layout()
    plt.title(f"20211206, row={row}/(max11), bay=CX, vfc_during_mag=0V")

siv = get_sampled_iv(0,0,-1)
plt.figure()
plt.plot(siv.fb_ic_fixed)
plt.plot(siv.sample_inds, siv.fb_ic_fixed[siv.sample_inds], "o")
plt.xlabel("sample index")
plt.ylabel("sq1 fb (dac units)")
plt.legend(["fb 'ic fixed'", "triangle wave sync points"])
plt.title(f"20211206, row={row}/(max11), bay=CX, vfc_during_mag=0V")


# for row in range(12):
#     plt.figure()
#     for polarity in [1,-1]:
#         cm = plt.get_cmap("coolwarm", n_fc)
#         labels = []
#         try:
#             for ind_fc in np.arange(0,n_fc//2,1):
#                     v_fc = data.v_fcs[ind_fc]
#                     siv = get_sampled_iv(ind_fc=ind_fc, row=row,polarity=polarity)
#                     c  = cm(ind_fc)
#                     if ind_fc%4 == 0:
#                         c = "k"
#                     plt.plot(siv.fb_ic_fixed[siv.sample_inds[-40]:siv.sample_inds[-1]:50], 
#                              color=c, lw=ind_fc%3+1)
#                     labels.append(f"v_fc={v_fc:0.2f}V")
#         except:
#             print(f"{row=} failed plotting")
#     plt.xlabel("sample index (aka time/3.6e-6s)")
#     plt.ylabel("fb")
#     plt.grid(True)
#     plt.legend(labels,loc="center left")
#     plt.tight_layout()
#     plt.title(f"20211206, row={row}/(max11), bay=CX, vfc_during_mag=0V")


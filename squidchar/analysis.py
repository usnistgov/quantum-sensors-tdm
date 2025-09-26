import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from scipy import stats
from dataclasses import dataclass
plt.rcParams['figure.constrained_layout.use']=True
from scipy.signal import decimate
from glob import glob
import re

# Note: most arrays have indices [bias, fb/err, column, row, time]

@dataclass
class DacCircuit:
    r: float = 1 # plain r is always the resistance in series with the voltage source
    dac_bits: int = 14
    dac_volts: float = 1

    def dac_to_v(self, dac):
        return dac * self.dac_volts / (2**self.dac_bits - 1)

    def dac_to_i(self, dac):
        return self.dac_to_v(dac) / self.r

@dataclass
class SeriesArrayCircuit(DacCircuit):
    # A single series array circuit
    m: float = 2.35
    r: float = 5100

    def ssa_fb_to_sq1_i(self, ssa_fb):
        # This is CRATE feedback.
        # Also theoretically the value you get back will have
        # an arbitrary offset.
        return self.dac_to_i(ssa_fb) / self.m

@dataclass
class SquidBiasCircuit(DacCircuit):
    r: float = 5100 
    shunt_r: float = 1
    series_r: float = 0.5
    dac_bits: int = 16
    dac_volts: float = 2.5

    def sq1b_to_i_closed(self, sq1_bias):
        # sq1 is biased via tower
        # when all switches are closed, we can assume all the
        # current goes through the switches.
        return self.dac_to_i(sq1_bias)


@dataclass
class SquidFeedbackCircuit(DacCircuit):
    r: float = 2000

@dataclass
class RowSelectCircuit(DacCircuit):
    r: float = 1000

@dataclass
class InputCircuit(DacCircuit):
    r: float = 10000 


def align_ramp(data, ramp_row=-1):
    """
    Measurements of time series for different bias values will have different
    phases with respect to thie triangle signal. This routine reads the triangle
    and syncs all the data.
    We assume data is an array indexed by [bias, fb/err, column, row, time]
    :param ramp_row: optionally specify which row the triangle signal was echoed to during data collection.
        If this parameter is not specified we assume it's on the last row. 
    """
    for i in range(data.shape[0]):
        data_run = data[i,0] # simplify no. of indices by extracting the only one 
        # we are looking at right now: feedback of whichever bias run are we on
        ramp = data_run[0,ramp_row] # all column ramps should be same, look only at 0
        ramp_min = np.min(ramp) # probably 0 but better to be general
        start_idx = 0
        if ramp[0] == ramp_min and ramp[-1] == ramp_min:
            # this is the edge case.
            # If that conditional is true, np.argmin will return 0 though the beginning
            # of the ramp is actually near the end of the array.
            # So we flag the rest of the data points near the beginning of the array
            # that are equal to the min value
            while ramp[start_idx] == ramp_min:
                start_idx+= 1 # there's certainly a better way to do this but it's not worth it
                # this while loop will run one in 8192 times under normal conditions
                # and will only take one step.
        min_idx = ramp[start_idx:].argmin() + start_idx
        data[i] = np.roll(data[i], -min_idx, axis=3)
    return data


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order, i.e., 1,2,3,10,11,20 instead of 1,10,11,2,20,3
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def load_big_ramp(file_pattern):
    """
    Probably don't use this function because super fine grid sampling is unnecessary.

    Warning: if you're using this loader function, make sure you have around 
    32 GB of ram available. I'll do my best to clean up arrays as they become unused
    but the single data array takes up ~4 GB and we need to do calculations and make 
    other similar sized arrays etc

    With 300 points in the ramp, all 8 columns and 24 rows and 8192 triangle time, on a mac M3,
    I measured peak usage of 13 GB. So could maybe run on 16 GB ram system if you're careful...
    Still seems like a lot though...
    But 300x8x24x8192x4 = 1.8 GiB so even only making like 5 arrays this size gets you in trouble.
    """
    files = sorted(glob(file_pattern),key=natural_keys)
    data_list = []
    for file in files:
        z = np.load(file)
        data_list.append(z["data_array"].astype("float32")) # float32 should have enough 
        # precision for our numbers and use half the memory of the float64 that it's stored as
        # The full bias list should be available in the final file.
    return np.vstack(data_list),z["bias_array"].astype("float32"),z
    # If we don't cast the bias array to float32 also, we'll end up with a lot of float64 arrays later

def mem_audit():
    try:
        snap = tracemalloc.take_snapshot()
        stats = snap.statistics('lineno')
        for i in range(10):
            print(stats[i])
    except RuntimeError:
        print("No audit available. Start tracemalloc first with tracemalloc.start()")
    except NameError:
        pass

def googly_eyes(eyes_xy, eye_size, pupil_size, x_factor, y_factor):
    plt.plot(eyes_xy[:,0],eyes_xy[:,1],'o',markersize=eye_size,color='w',markeredgecolor='k')
    rng = np.random.default_rng()
    r = rng.random(2)
    theta = rng.random(2)*2*np.pi
    translation = (r*np.sin(theta)*x_factor, r*np.cos(theta)*y_factor)
    pupils_xy = eyes_xy + np.array(translation).T
    plt.plot(pupils_xy[:,0],pupils_xy[:,1],'o',markersize=pupil_size,color='k')

def normalize_current_for_ovals(data):
    min_i = np.min(data,axis=-1)
    max_i = np.max(data,axis=-1)
    ms = min_i.shape
    n = data.ndim
    bcast_arg = [data.shape[-1]] + [i for i in ms]
    # bcast_arg is the shape we can broadcast to that's closest to the shape of `data`
    # If data has shape [8,32,8096] then ms = [8,32] and bcast_arg = [8096, 8, 32]
    # Then we need to roll the axes around so that the tiled arrays have the same shape
    # as data. So transpose_arg will be [1, ..., n-1, 0]
    tpose_arg = [i+1 for i in range(n-1)] + [0]
    min_i_tiled = np.broadcast_to(min_i,bcast_arg).transpose(tpose_arg)
    max_i_tiled = np.broadcast_to(max_i,bcast_arg).transpose(tpose_arg)
    
    data_normalized = (data - min_i_tiled)/(max_i_tiled-min_i_tiled)
    return data_normalized

def get_colors(base_cmap="turbo", squish_factor=30, num_colors=10):
    """
    Make two sets of colors, each with num_colors values (usually 10 or 11)
    Each set of colors has num_colors similar colors to represent different rows
    One set contains blue shades to represent minimum values and the other contains
    shades of red to represent maximum values. 
    :param base_cmap: By default, to produce red and blue shades, we use the colormap "turbo"
        But if you want you can do something else. This function selects colors from opposite ends
        of the colormap though, so keep that in mind.
    :param squish_factor: This is the size of the grid to use. Larger squish_factors will result
        in the returned colors being more similar within a set. squish_factor > num_colors*2
    :param num_colors: how many colors in a set. 10 for 2ls muxes, 11 for normal fas muxes.
    """
    cmap = plt.get_cmap(base_cmap)
    colors = cmap(np.linspace(0,1,squish_factor))
    low_colors = colors[0:num_colors]
    hi_colors = colors[-num_colors:][::-1]
    return low_colors, hi_colors


 ###############################################
# Plot functions!                               #
# For examples of what each plot function does, #
# see the accompanying ipython notebook.        #
 ###############################################

def fas_activation_plot(rs_data_ua, tri_ua, cs_ramp_ua, row, col):
    plt.figure()
    plt.imshow(
        rs_data_ua[:,0,col,row,0:4096]-np.min(rs_data_ua[:,0,col,row]),
        aspect="auto",
        interpolation="none",
        origin="lower",
        extent=(min(tri_ua),max(tri_ua),min(cs_ramp_ua),max(cs_ramp_ua))
    )
    plt.xlabel("row select flux [$\\mu$A]")
    plt.ylabel("chip select flux [$\\mu$A]")
    plt.colorbar(label="sq1 response [$\\mu$A]")



def get_fas_biases(rs_data, 
                   cs_ramp, 
                   num_cols=8, 
                   num_rows=24,
                   ignore_rows=[]):
    cs_flux_arr = np.zeros((num_cols,num_rows))
    rs_flux_arr = np.zeros((num_cols,num_rows))
    for col in range(num_cols):
        for row in range(num_rows):
            arr = rs_data[:,0,col,row,0:4096]
            max_idx = np.unravel_index(np.argmax(arr, axis=None), arr.shape)
            cs_flux = cs_ramp[max_idx[0]]
            rs_flux = rs_data[max_idx[0],0,col,-1,max_idx[1]]
            if row in [10,11,22,23]+ignore_rows: # these 2 rows are never connected: 11, 23
                # TODO: rows 10 and 22 ARE connected in 1x11 FAS muxes, but not in 2 level switch muxes
                cs_flux = np.nan 
                rs_flux = np.nan
            cs_flux_arr[col,row] = cs_flux
            rs_flux_arr[col,row] = rs_flux
    return cs_flux_arr, rs_flux_arr

def get_fas_biases_1x11(
    rs_data, 
    num_cols=8, 
    num_rows=24,
    ignore_rows=[]
):
    rs_flux_arr = np.zeros((num_cols,num_rows))
    for col in range(num_cols):
        for row in range(num_rows):
            arr = rs_data[:,0,col,row,0:4096]
            max_idx = np.unravel_index(np.argmax(arr, axis=None), arr.shape)
            rs_flux = rs_data[max_idx[0],0,col,-1,max_idx[1]]
            if row in [11,23]+ignore_rows: # these 2 rows are never connected: 11, 23
                # TODO: rows 10 and 22 ARE connected in 1x11 FAS muxes, but not in 2 level switch muxes
                rs_flux = np.nan
            rs_flux_arr[col,row] = rs_flux
    return  rs_flux_arr


def load_data_and_convert(filename,ramp_row=11,ssa=None,s1b=None):
    ssa = ssa or SeriesArrayCircuit()
    s1b = s1b or SquidBiasCircuit()
    if "part" in filename:
        data,bias,z = load_big_ramp(filename)
    else:
        z = np.load(filename)
        # z = np.load("all_rows_2025_04_16/big_sq1_bias_ramp.npz")
        bias = z["bias_array"]
        data = z["data_array"]
    data = align_ramp(data,ramp_row=ramp_row)
    data_fb = data[:,0]
    data_i = ssa.ssa_fb_to_sq1_i(data_fb)
    bias_i = s1b.sq1b_to_i_closed(bias)*1e6
    data_zero_points = data_i[0,:,:,0] # need one zero point for each row and each column
    tile_arg = (data_fb.shape[0], data_fb.shape[3],data_zero_points.shape[0],data_zero_points.shape[1])
    data_zero_tiled = np.broadcast_to(data_zero_points, tile_arg).transpose([0,2,3,1])  
    # Note that these arrays are called "tiled" because I was using np.tile before. np.broadcast_to is similar,
    # but it makes a view of the array instead of populating an entire array in memory.
    # zero_i = np.median(data_i[0,col,row]) Could do this, but I'd rather take the value when both bias and fb are zero
    #print(min_i)
    data_ua = -(data_i - data_zero_tiled) * 1e6 
    return data_ua, bias_i, data[0,0,0,ramp_row], z

def calculate_icmax(bias, data):
    """
    given a bias array and data array with indices [bias, column, row, time],
    calculate the current modulation depth (peak-to-peak of time series for each bias, column, row)
    and then find the bias with maximum peak to peak for each column row.

    returns: amplitude array, index of icmax, icmax
    """
    amplitude = np.ptp(data, axis=3) # indices now [bias, column, row]
    max_idx = np.argmax(amplitude, axis=0) # indices now [column, row]
    return amplitude, max_idx, bias[max_idx] 

class PlotsWithSameColors:
    """
    To avoid repeatedly calling get_colors or having to pass two color arrays to every function,
    this class stores the two color arrays as member variables and then all the plot functions can
    access them
    """
    def __init__(self, low_colors=None, high_colors=None):
        lc,hc = get_colors()
        if low_colors is not None:
            self.low_colors = low_colors 
        else:
            self.low_colors = lc

        if high_colors is not None:
            self.high_colors = high_colors 
        else:
            self.high_colors = hc # can't do the high_colors or hc trick when numpy arrays are involved

    def fas_activation_plot_1x11(rs_data_ua, tri_ua, chip, col):
        plt.figure()
        for i in range(11):
            if slope=="pos":
                plt.plot(tri_ua[0:4096], rs_data_ua[:,inspect_col,i+inspect_chip*12,0:4096], label=f"RS {i}", c=self.low_colors[i])
            else:
                plt.plot(tri_ua[4096:], rs_data_ua[:,inspect_col,i+inspect_chip*12,4096:], label=f"RS {i}", c=self.low_colors[i])
        plt.xlabel("row select flux [$\\mu$A]")
        plt.ylabel("sq1 response [$\\mu$A]")

    def current_modulation_plot(self, amplitude, icmax, bias_i, inspect_col, inspect_chip, nrows=10):
        plt.figure()
        # icmin = np.zeros(8)
        for i in range(nrows):
            # for j in range(4,len(amplitude)):
            #     subarray = amplitude[0:j]
            #     mad = stats.median_abs_deviation(subarray)
            #     med = np.median(subarray)
            #     deviation = (amplitude[j]-med)/mad
            #     # print(i, bias[j], mad, deviation)
            #     if deviation > 20:
            #         # print(i, bias[j])
            #         icmin[i] = bias[j]
            #         break
            plt.plot(bias_i, amplitude[:,inspect_col,i+inspect_chip*12], label=f"RS {i}", c=self.low_colors[i])
            plt.axvline(icmax[inspect_col,i+inspect_chip*12], color=self.low_colors[i])
        plt.legend()
        plt.title("Current modulation depth")
        plt.xlabel("SQ1 bias [$\\mu$A]")
        plt.ylabel("Response [$\\mu$A]")
        # np.median(bias[ic_idx])

    def i_i_plot(self, data_ua, bias_i, inspect_col, inspect_chip, bad_rs=[],nrows=10):
        plt.figure()
        min_i = np.min(data_ua,axis=3)
        max_i = np.max(data_ua,axis=3)
        for i in range(nrows):
            if i+inspect_chip*12 in bad_rs:
                continue 
            min_i_row = min_i[:,inspect_col,i+inspect_chip*12] #- data_i[0,inspect_col,i+inspect_chip*12,0]
        
            plt.plot(bias_i,min_i_row,label=f"column {i}",c=self.low_colors[i])
        
        for i in range(nrows):
            if i+inspect_chip*12 in bad_rs:
                continue
            max_i_row = max_i[:,inspect_col,i+inspect_chip*12]# - data_i[0,inspect_col,i+inspect_chip*12,0]
        
            plt.plot(bias_i,max_i_row,label=f"column {i}",color=self.high_colors[i])
        plt.plot(np.arange(15),np.arange(15),color='k',linewidth=1,label=f"x=y")
        plt.xlabel("SQ1 Bias [$\\mu$A]")
        plt.ylabel("SQ1 Current [$\\mu$A]")

    def i_v_plot(self, data_ua, shunt_uv, inspect_col, inspect_chip,nrows=10):
        min_i = np.min(data_ua,axis=3)
        max_i = np.max(data_ua,axis=3)
        plt.figure()
        
        # Extract 1 column and 12 rows from the whole data set
        col_idx = inspect_col
        row_start= inspect_chip*12
        
        for i in range(nrows):
            row_idx = row_start + i
            data_i_row = data_ua[:,col_idx,row_idx]
            shunt_v_row = shunt_uv[:,col_idx,row_idx]
            min_i_row = min_i[:,col_idx,row_idx]
            min_idx = np.argmin(data_i_row,axis=1)
            shunt_v_min = shunt_v_row[np.arange(len(min_idx)),min_idx]
            plt.plot(shunt_v_min,min_i_row,label=f"column {i}",c=self.low_colors[i])
        
            max_i_row = max_i[:,col_idx,row_idx] 
            max_idx = np.argmax(data_i_row,axis=1)
            shunt_v_max = shunt_v_row[np.arange(len(max_idx)),max_idx]
            plt.plot(shunt_v_max,max_i_row,label=f"column {i}",c=self.high_colors[i])
        
            # plt.plot(shunt_v,-max_i,label=f"column {i}",color=low_colors[i])
        # plt.plot(np.arange(15),np.arange(15),color='k',linewidth=1,label=f"x=y")
        plt.xlabel("SQ1 Bias [$\\mu$V]")
        plt.ylabel("SQ1 Current [$\\mu$A]")

    def device_resistance_plot(
        self, 
        min_i, 
        max_i, 
        rd_at_iin_min, 
        rd_at_iin_max, 
        inspect_col, 
        inspect_chip,
        nrows=10):
    # Extract 1 column and 12 rows from the whole data set
        col_idx = inspect_col
        row_start= inspect_chip*12
        
        plt.figure()
        for i in range(nrows):
            row_idx = row_start + i
            plt.plot(max_i[1:,col_idx,row_idx],rd_at_iin_max[1:,col_idx,row_idx],c=self.high_colors[i])
            plt.plot(min_i[1:,col_idx,row_idx],rd_at_iin_min[1:,col_idx,row_idx],c=self.low_colors[i])
            # note that the 1: above is because you can't calculate resistance if no current is flowing.
        plt.title("Device normal resistance")
        plt.xlabel("SQ1 current [$\\mu$A]")
        plt.ylabel("SQ1 resistance [$\\Omega$]")

    def device_dynamic_resistance_plot(
        self,min_i, 
        max_i, 
        rdyn_at_iin_min, 
        rdyn_at_iin_max, 
        inspect_col, 
        inspect_chip,
        nrows=10):
        plt.figure()
        col_idx = inspect_col
        row_start= inspect_chip*12
        for i in range(nrows):
            row_idx = row_start + i
            plt.plot(max_i[1:,col_idx,row_idx],rdyn_at_iin_max[1:,col_idx,row_idx],'-',c=self.high_colors[i])
            plt.plot(min_i[1:,col_idx,row_idx],rdyn_at_iin_min[1:,col_idx,row_idx],'-',c=self.low_colors[i])
            # note that the 1: above is because you can't calculate resistance if no current is flowing.
        plt.title("Device dynamic resistance")
        plt.xlabel("SQ1 current [$\\mu$A]")
        plt.ylabel("SQ1 R$_{dyn}$ [$\\Omega$]")

    def rdyn_oval_plot(self,data_normalized, rdyn, ic_idx, inspect_col, inspect_chip, silly=False,nrows=10):
        col_idx = inspect_col
        row_start= inspect_chip*12
        
        plt.figure()
        for i in range(nrows):
            row_idx = row_start + i 
            
            ic_i = ic_idx[col_idx, row_idx] 
            plt.plot(data_normalized[ic_i,col_idx,row_idx],rdyn[ic_i,col_idx,row_idx],'.',c=self.low_colors[i],markersize=0.5)
        plt.xlabel("Normalized current, 0=$I_{min}$ 1=$I_{max}$")
        plt.ylabel("R$_{dyn}$ [$\\Omega$]")
        if silly:
            eyes_xy = np.array([(0.3,16),(0.7,16)])
            googly_eyes(eyes_xy, 50, 25, 0.04, 0.5)

    def squid_curve_input_plot(self, tri_i, d_i, inspect_col, inspect_chip,nrows=10):
        col_idx = inspect_col
        row_start= inspect_chip*12
        plt.figure()
        for i in range(nrows):
            stop_idx = tri_i.shape[0]//2
            device_current = d_i[col_idx,row_start+i,:stop_idx]*1e6
            current_subtracted = device_current-min(device_current)
            plt.plot(tri_i[:stop_idx], current_subtracted,c=self.low_colors[i],label=i)
            plt.xlabel("input current [uA]")
            plt.ylabel("Device current (uA) + arb offset")

    def squid_gain_plot(self, tri_i, gain, inspect_col, inspect_chip,nrows=10):
        plt.figure()
        col_idx = inspect_col
        row_start= inspect_chip*12

        stop_idx = tri_i.shape[0]//2
        plt.axhline(0,color='k',linewidth=1)
        for i in range(nrows):
            
            plt.plot(tri_i[1:stop_idx],
                     gain[col_idx,row_start+i,1:stop_idx]*1e6,
                     label="increasing I",
                     c=self.low_colors[i]
                    )
            #plt.plot(tri_i_filtered[stop_idx+1:],np.gradient(d_i_filtered[1,3,stop_idx+1:]*1e6)/np.gradient(tri_i_filtered[stop_idx+1:]),label="decreasing I")
        plt.xlabel("Input current [$\\mu$A]")
        plt.ylabel("Gain [unitless] = $dI_{SQ1}/dI_{in}$")

    def gain_oval_plot(self, norm_i, gain, inspect_col, inspect_chip, silly=False,nrows=10):
        plt.figure()
        col_idx = inspect_col
        row_start= inspect_chip*12
        for i in range(nrows):
            plt.plot(norm_i[col_idx, row_start+i], gain[col_idx, row_start+i]*1e6, '.', color=self.low_colors[i], markersize=0.5)
        if silly:
            eyes_xy = np.array([(0.3,16),(0.7,16)])
            googly_eyes(eyes_xy, 50, 25, 0.04, 0.5)
        plt.xlabel("Normalized current, 0=$I_{min}$ 1=$I_{max}$")
        plt.ylabel("Squid gain [unitless]")

def calculate_device_voltage(data_ua, bias_i, rshunt):
    bias_broadcast = np.broadcast_to(bias_i, (8,24,8192,bias_i.shape[0]))
    bias_tiled = bias_broadcast.transpose([3,0,1,2]) 
    shunt_uv = (bias_tiled - data_ua) * rshunt# also = voltage across squids
    return shunt_uv

def load_input_ramp(input_ramp_file, ssa=None):
    ssa = ssa or SeriesArrayCircuit()
    z = np.load(input_ramp_file,allow_pickle=True)
    d = np.array([z["data_array"]]) # add an extra index so that I don't have to make align_ramp more robust to number of indices
    d_fb = align_ramp(d)[0,0] # remove the extra index from above and choose to view only feedback (not error signal)
    tri = d_fb[0,-1]
    inp = InputCircuit()
    tri_i = inp.dac_to_i(tri)*1e6
    d_i = ssa.ssa_fb_to_sq1_i(d_fb)
    return tri_i, d_i, z

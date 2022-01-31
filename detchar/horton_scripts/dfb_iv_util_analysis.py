import pylab as plt
import numpy as np
from cringe.tune.analysis import conditionvphi
import h5py
from numpy.polynomial.polynomial import Polynomial



def make_vector_start_at_ind(v, ind):
    out = np.zeros_like(v)
    out = np.zeros_like(v)
    out[:len(v)-ind] = v[ind:]
    out[len(v)-ind:] = v[:ind]
    return out

def add_flux_jumps(fb, phi0_fb, fb_step_threshold):
    """return an array like fb, but with flux jumps resolved and one value removed from the end
    method: find indicies with np.diff(fb) greater ro less than fb_step_threshold
    and add phi0 units with the correct sign"""
    out = fb[:-1]+np.cumsum(np.diff(fb)<-fb_step_threshold)*phi0_fb
    out -= np.cumsum(np.diff(fb)>fb_step_threshold)*phi0_fb
    return out

def avg_n_points_every_m_points_starting_at_o(v, n, m, o):
    assert o+n < m
    out = np.zeros(len(v)//m)
    for i in range(o, o+n):
        out += v[i::m][:len(out)]
    return out/n

def sign_change_inds(y, threshold=400, min_spacing=4):
    """find indicies into y where the slope change sign, the difference in slope is greater than threshold,
    and the previous sign change ind is at least min_spacing away"""
    has_slope_sign_change = np.abs(np.diff(np.array(np.sign(np.diff(y)), dtype="int32")))
    exceeds_threshold = np.abs(np.diff(np.diff(y)))>threshold
    both = np.logical_and(has_slope_sign_change, exceeds_threshold)
    possible_inds = np.where(both)[0]+1
    inds = [possible_inds[0]]
    for i in range(1, len(possible_inds)):
        if possible_inds[i]-possible_inds[i-1] >= min_spacing:
            inds.append(possible_inds[i])
    return np.array(inds)

def _rotate_list(l, n):
    l = list(l)
    return l[n:] + l[:n]

def find_persisent_slope_changes(y, abs_ddy_below=100, abs_dy_above=100, 
    avg_ddy_threshold = 2, merge_region_range=4, debug_plot=False):
    dy = np.diff(y)
    ddy = np.diff(dy, prepend=dy[0])
    # superconducting regions should have the largest magntidue of regions of consistent slope
    possible_sc = np.logical_and(np.abs(ddy)<abs_ddy_below, np.abs(dy)>abs_dy_above)
    # find incicides of starts of possible_sc regions
    possible_sc_inds = np.where(possible_sc)[0]
    _ind = np.where(np.diff(possible_sc_inds, prepend=-1)>1)[0]
    possible_sc_start_inds = possible_sc_inds[_ind]
    possible_sc_end_inds = _rotate_list(possible_sc_inds[_ind-1], 1)
    # check if each region is flat
    sc_region_bounds0 = []
    for a, b in zip(possible_sc_start_inds, possible_sc_end_inds):
        avg_ddy = (dy[b]-dy[a])/(b-a)
        avg_dy = np.mean(dy[a:b])
        avg_ddy_over_dy = avg_ddy/avg_dy
        # print(f"avg_ddy={avg_ddy}")
        # print(f"avg_dy={avg_dy}")
        # print(f"avg_ddy_over_dy={avg_ddy_over_dy}")
        if np.abs(avg_ddy) < avg_ddy_threshold:
            sc_region_bounds0.append(np.array([a,b], dtype="int32"))

    # merge regions with small gaps
    sc_region_bounds = []
    skip = False
    # print("start merge")
    # print(sc_region_bounds0)
    for i in range(len(sc_region_bounds0)):
        a0,b0 = sc_region_bounds0[i]
        if skip:
            skip = False
            continue
        if i < len(sc_region_bounds0)-1: 
            a1,b1 = sc_region_bounds0[i+1]
            # print(f"should merge? ({a0},{b0}) and ({a1},{b1}), diff={a1-b0}")
            if (a1-b0) <= merge_region_range:
                sc_region_bounds.append([a0,b1])
                skip = True
                # print(" yes merge [a0,b1]")
                continue
        # print("no merge, or last index")
        sc_region_bounds.append([a0,b0])
    # print(f"finished with {sc_region_bounds}")


    if debug_plot:
        plt.figure()
        plt.plot(dy, label="dy")
        plt.plot(ddy, label="ddy")
        plt.plot(possible_sc_inds, dy[possible_sc_inds], "o", label="possible_sc_inds")
        plt.plot(possible_sc_start_inds, dy[possible_sc_start_inds], "o", label="possible_sc_start_inds")
        plt.plot(possible_sc_end_inds, dy[possible_sc_end_inds], "o", label="possible_sc_end_inds")
        for i, ab in enumerate(sc_region_bounds):
            plt.plot(ab, dy[ab], label=f"sc region {i}", lw=4)
        for i, ab in enumerate(sc_region_bounds0):
            plt.plot(ab, dy[ab], label=f"sc region pre_merge {i}", lw=2)
        plt.legend()
        
    return sc_region_bounds


def correct_ic_steps_and_split_down_up(fb_sampled_in, tri_sampled, phi0_fb, plot=True):
    fb_sampled = np.copy(fb_sampled_in)
    tri_n_points = len(tri_sampled)

    down_fb = fb_sampled[2:1+tri_n_points//2]
    down_db = tri_sampled[2:1+tri_n_points//2]
    down_fb_orig = np.copy(down_fb)

    up_fb = fb_sampled[tri_n_points//2:] 
    up_db = tri_sampled[tri_n_points//2:]
    up_fb_orig = np.copy(up_fb)

    sc_region_bounds = find_persisent_slope_changes(fb_sampled_in, debug_plot=plot)
    if len(sc_region_bounds) != 2:
        find_persisent_slope_changes(fb_sampled_in, debug_plot=True)
    (down_sc_start, down_sc_end), (up_sc_start, up_sc_end) = sc_region_bounds

    # where the superconducting branches overlap, they should have the same value for the same db
    # lets try just after we go from transition to superconducting on down branch
    down_sc_ind = down_sc_start
    # now find the same db value in the up sc branch
    up_sc_ind = np.where(tri_sampled==tri_sampled[down_sc_ind])[0][1]
    if not plot:
        # if we're going to plot, do this assert later
        assert up_sc_ind > up_sc_start and up_sc_ind < up_sc_end, f"down_sc_ind={down_sc_ind} up_sc_ind={up_sc_ind} sc_region_bounds={sc_region_bounds}"
    nphi0_down = (fb_sampled[up_sc_ind]-fb_sampled[down_sc_ind])/phi0_fb
    ic_step_fb_down = np.round(nphi0_down)*phi0_fb
    fb_sampled[down_sc_end+2:]-=ic_step_fb_down # modify fb_sampled in place, down_fb and friends are views so 
    # are also modified

    # our db is periodic, so the last sample and first sample should be the same
    # we did drop one sample earlier, but close enough
    nphi0_up = (fb_sampled[-1]-fb_sampled[0])/phi0_fb
    ic_step_fb_up = np.round(nphi0_up)*phi0_fb
    if len(sc_region_bounds)==2:
        fb_sampled[up_sc_end+2:]-=ic_step_fb_up

    if plot:
        print(f"nphi0_down={nphi0_down:.2f} nphi0_up={nphi0_up:.2f}")
        print(f"ic_step_fb_down={ic_step_fb_down} ic_step_fb_up={ic_step_fb_up}")
        plt.figure()
        plt.plot(fb_sampled_in/phi0_fb, label="fb_sampled_in")
        plt.plot(fb_sampled/phi0_fb, label="fb_sampled")
        for i, ab in enumerate(sc_region_bounds):
            plt.plot(ab, fb_sampled_in[ab]/phi0_fb, label=f"sc region {i}")
        plt.title("correct_ic_steps_and_split_down_up fig 1")
        plt.ylabel("fb/phi0_fb")
        plt.legend()
        plt.grid(True)

        up_ind = sign_change_inds(up_fb)
        down_ind = sign_change_inds(down_fb)

        up_sc_ind_plot = up_sc_ind - tri_n_points//2
        down_sc_ind_plot = down_sc_ind - 2
        plt.figure()
        plt.plot(down_db, down_fb, label="down_fb")
        plt.plot(down_db, down_fb_orig, label="down_fb_orig")
        plt.plot(up_db, up_fb, label="up_fb")
        plt.plot(up_db, up_fb_orig, label="up_fb_orig")
        plt.plot(down_db[down_sc_ind_plot], down_fb[down_sc_ind_plot], "s", label="down_sc_ind_plot", fillstyle="none")
        plt.plot(up_db[up_sc_ind_plot], up_fb[up_sc_ind_plot], "s", label="up_sc_ind_plot", fillstyle="none")
        plt.title("correct_ic_steps_and_split_down_up fig 2")
        plt.xlabel("db")
        plt.ylabel("fb")
        plt.legend()
        plt.grid(True)

        plt.figure()
        plt.title("correct_ic_steps_and_split_down_up fig 3")
        plt.plot(tri_sampled, label="tri_sampled")
        plt.plot(fb_sampled, label="fb_sampled")
        plt.plot(fb_sampled_in, label="fb_sampled_in")
        plt.plot(up_sc_ind, tri_sampled[up_sc_ind], "o", label="up_sc_ind tri")
        plt.plot(up_sc_ind, fb_sampled[up_sc_ind], "o", label="up_sc_ind fb")
        plt.plot(down_sc_ind, tri_sampled[down_sc_ind], "o", label="down_sc_ind tri")
        plt.plot(down_sc_ind, fb_sampled[down_sc_ind], "o", label="down_sc_ind fb")
        plt.legend()

    # if plotting this is a repeated assert
    assert up_sc_ind > up_sc_start and up_sc_ind < up_sc_end, f"down_sc_ind={down_sc_ind} up_sc_ind={up_sc_ind} sc_region_bounds={sc_region_bounds}"

    return down_fb, down_db, down_fb_orig, up_fb, up_db, up_fb_orig, fb_sampled, sc_region_bounds

def plot_ivs_with_offset(down_fb, down_db, up_fb, up_db, db_offset, fb_offset):
    up_ind = sign_change_inds(up_fb)[1]
    down_ind = sign_change_inds(down_fb)[1]
    plt.figure()
    plt.plot(down_db-db_offset, down_fb-fb_offset, ".", label="first half (down)")
    plt.plot(up_db-db_offset, up_fb-fb_offset, ".", label="2nd half (up)")
    plt.plot(up_db[up_ind]-db_offset, up_fb[up_ind]-fb_offset, "s", label=f"ic at {up_fb[up_ind]-fb_offset} dac units", fillstyle="none")
    plt.plot(down_db[down_ind]-db_offset, down_fb[down_ind]-fb_offset, "s", label=f"ic at {down_fb[down_ind]-fb_offset} dac units", fillstyle="none")
    plt.xlabel("db (from dfb card) dac units")
    plt.ylabel("fb dac units (zero subtracted)")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.title("dfb based iv, measured full IV with no flux jumps\nand Ic with ~1e6 samples, so takes 1e6/samplerate seconds")
    plt.tight_layout()


def fit_for_origin(down_fb, down_db, up_fb, up_db, normal_db_lo_hi, sc_db_lo_hi, plot=True):
    normal_db_lo, normal_db_hi = normal_db_lo_hi
    sc_db_lo, sc_db_hi = sc_db_lo_hi
    # find and fit lines to the normal and superconducting regions for up and down (4 fits)
    normal_inds_down = np.where(np.logical_or(down_db<normal_db_lo, down_db>normal_db_hi))[0]
    pfit_normal_down=Polynomial.fit(down_db[normal_inds_down], down_fb[normal_inds_down], 1, domain=[0,2**14])
    normal_inds_up = np.where(np.logical_or(up_db<normal_db_lo, up_db>normal_db_hi))[0]
    pfit_normal_up=Polynomial.fit(up_db[normal_inds_up], up_fb[normal_inds_up], 1, domain=[0,2**14])
    sc_inds_down = np.where(np.logical_and(down_db>sc_db_lo, down_db<sc_db_hi))[0]
    pfit_sc_down=Polynomial.fit(down_db[sc_inds_down], down_fb[sc_inds_down], 1, domain=[0,2**14])
    sc_inds_up = np.where(np.logical_and(up_db>sc_db_lo, up_db<sc_db_hi))[0]
    pfit_sc_up=Polynomial.fit(up_db[sc_inds_up], up_fb[sc_inds_up], 1, domain=[0,2**14])

    # average the up and down fits for normal and sc, find the intersection
    p_normal = (pfit_normal_up+pfit_normal_down)/2
    p_sc = (pfit_sc_up+pfit_sc_down)/2
    origin_db = (p_normal-p_sc).roots()[0]
    origin_fb = p_sc(origin_db)

    r_para_ddb_per_dfb = 1/p_sc.deriv(1)(0)
    down_db_no_r_para = down_db-origin_db-(down_fb-origin_fb)*r_para_ddb_per_dfb
    up_db_no_r_para = up_db-origin_db-(up_fb-origin_fb)*r_para_ddb_per_dfb

    if plot:
        plt.figure()
        plt.title("fit_for_origin fig 1")
        plt.plot(down_db, down_fb,"o", label="down")
        plt.plot(down_db, pfit_normal_down(down_db), label="down normal fit")
        plt.plot(down_db, pfit_sc_down(down_db), label="down sc fit")
        plt.plot(up_db, up_fb,"o", label="up")
        plt.plot(up_db, pfit_normal_up(up_db), label="up normal fit")
        plt.plot(up_db, pfit_sc_up(up_db), label="up sc fit")
        plt.xlabel("db dac units")
        plt.ylabel("fb dac units")
        plt.legend()

        plt.figure()
        plt.title("fit_for_origin fig 2")
        plt.plot(down_db-origin_db, down_fb-origin_fb,"o", label="down")
        plt.plot(down_db-origin_db, pfit_normal_down(down_db)-origin_fb, label="down normal fit")
        plt.plot(down_db-origin_db, pfit_sc_down(down_db)-origin_fb, label="down sc fit")
        plt.plot(up_db-origin_db, up_fb-origin_fb,"o", label="up")
        plt.plot(up_db-origin_db, pfit_normal_up(up_db)-origin_fb, label="up normal fit")
        plt.plot(up_db-origin_db, pfit_sc_up(up_db)-origin_fb, label="up sc fit")
        plt.xlabel("db dac units")
        plt.ylabel("fb dac units")
        plt.legend()

    return origin_fb, origin_db, down_db_no_r_para, up_db_no_r_para

import pylab as plt
import numpy as np
from cringe.tune.analysis import conditionvphi
import h5py
from numpy.polynomial.polynomial import Polynomial
from dataclasses import dataclass
from dfb_iv_util_analysis import *

plt.ion()
plt.close("all")
h5 = h5py.File("20210527_dfb_iv_vs_field_coil.hdf5", "r")

dwell_power = 11
steps_power = 9
dwell = 2**11
tri_n_points = 2**(steps_power+dwell_power)
phi0_fb = 1675

# incicies are field coil setting, column, row, frame, [err, fb]
# column 0 has detectors
# column 1 is just used to track the triangle used for db
_tri = h5["alldata"][7, 1, 1,:,1]
_fb = h5["alldata"][7, 0, 1,:,1]

# we have one full period, but it doesn't start where we want,
# be the maximum value of the triangle
ind = np.where(_tri==np.amax(_tri))[0][0]
tri = make_vector_start_at_ind(_tri, ind)
fb = make_vector_start_at_ind(_fb, ind)

plt.figure()
plt.plot(fb, label="iv data")
plt.plot(tri, label="db triangle")
plt.xlabel("sample number")
plt.ylabel("dac units")
plt.title("raw data for one iv")

# zoomed in to show keeping lock
plt.figure()
plt.plot(fb, label="iv data")
plt.plot(tri, label="db triangle")
plt.xlabel("sample number")
plt.ylabel("dac units")
plt.xlim(767055, 829880)
plt.ylim(7221, 9218)
plt.title("raw data zoomed in to show how we see relocks")

# add back in phi0s for flux jumps we can resolve
fb2 = fb2 = add_flux_jumps(fb, phi0_fb, fb_step_threshold=1000)
tri2 = tri[:-1]

# sample the last few points before the next step of the triangle
tri_sampled = avg_n_points_every_m_points_starting_at_o(tri2, 4, dwell, dwell-2-4)
fb2_sampled = avg_n_points_every_m_points_starting_at_o(fb2, 4, dwell, dwell-2-4)

# add back phi0s at the two ics
(down_fb, down_db, down_fb_orig, 
    up_fb, up_db, up_fb_orig, fb_sampled,
    sc_region_bounds) = correct_ic_steps_and_split_down_up(fb2_sampled, 
                                                                    tri_sampled, phi0_fb, plot=True)

# find the origin and r_para
origin_fb, origin_db, down_db_no_r_para, up_db_no_r_para = fit_for_origin(down_fb, down_db, up_fb, up_db, normal_db_lo_hi=(2000,14000), 
sc_db_lo_hi=(6000,7000), plot=False)

plot_ivs_with_offset(down_fb, down_db_no_r_para, up_fb, up_db_no_r_para, 
    db_offset=0, fb_offset=origin_fb)

# find ics
def find_ics_fb_units(up_fb, down_fb):
    up_ind = sign_change_inds(up_fb)[2]
    down_ind = sign_change_inds(down_fb)[2]
    up_ic_fb_units = up_fb[up_ind]
    down_ic_fb_units = down_fb[down_ind]
    return up_ic_fb_units, down_ic_fb_units


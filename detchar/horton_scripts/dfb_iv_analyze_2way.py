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
h5null = h5py.File("20210527_dfb_iv_vs_field_coil_null.hdf5", "r")
dwell_power = 11
steps_power = 9
dwell = 2**11
tri_n_points = 2**(steps_power+dwell_power)
phi0_fb = 1675

# incicies are field coil setting, column, row, frame, [err, fb]
# column 0 has detectors
# column 1 is just used to track the triangle used for db
_tri = h5["alldata"][0, 1, 1,:,1]
_fb = h5["alldata"][0, 0, 1,:,1]

def get_ics(_tri, _fb, plot=False, i=1):
    # we have one full period, but it doesn't start where we want,
    # be the maximum value of the triangle
    ind = np.where(_tri==np.amax(_tri))[0][0]
    tri = make_vector_start_at_ind(_tri, ind)
    fb = make_vector_start_at_ind(_fb, ind)

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
                        tri_sampled, phi0_fb, plot=plot)

    # find the origin and r_para
    origin_fb, origin_db, down_db_no_r_para, up_db_no_r_para = fit_for_origin(down_fb, down_db, up_fb, up_db, normal_db_lo_hi=(2000,14000), 
    sc_db_lo_hi=(6000,7000), plot=plot)



    # find ics
    up_ind = sign_change_inds(up_fb)[1]
    down_ind = sign_change_inds(down_fb)[1]
    up_ic_fb_units = up_fb[up_ind]-origin_fb
    down_ic_fb_units = down_fb[down_ind]-origin_fb

    lastplot = i<10
    if lastplot:
        plot_ivs_with_offset(down_fb, down_db_no_r_para, up_fb, up_db_no_r_para, 
            db_offset=0, fb_offset=origin_fb)


    return up_ic_fb_units, down_ic_fb_units, origin_fb, origin_db

row = 0
ics_up = []
ics_down = []
origin_fbs = []
origin_dbs = []
null_fbs = []
for i in range(h5["alldata"].shape[0]):
    _tri = h5["alldata"][i, 1, 1,:,1]
    _fb = h5["alldata"][i, 0, 1,:,1]
    try:
        up_ic_fb_units, down_ic_fb_units, origin_fb, origin_db = get_ics(_tri, _fb, i=i)
    except: 
        print(f"failed on i={i}")
        get_ics(_tri, _fb, plot=True, i=i)
    ics_up.append(up_ic_fb_units)
    ics_down.append(down_ic_fb_units)
    origin_fbs.append(origin_fb)
    origin_dbs.append(origin_db)
    null_fbs.append(np.mean(h5["alldata"][i, 0, 1,:,1]))

plt.figure()
plt.plot(h5["field_coil_dacs"][()], ics_up, "o", label=f"ics up row{row}")
plt.plot(h5["field_coil_dacs"][()], ics_down, "o", label=f"ics down row{row}")
plt.xlabel("field coil dac with 5 k ohm resistance")
plt.ylabel("ic (fb units)")

plt.figure()
plt.plot(null_fbs)
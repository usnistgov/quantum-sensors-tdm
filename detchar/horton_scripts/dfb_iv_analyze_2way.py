import pylab as plt
import numpy as np
from cringe.tune.analysis import conditionvphi
import h5py
from numpy.polynomial.polynomial import Polynomial


plt.ion()
plt.close("all")
h5 = h5py.File("20210527_dfb_iv_vs_field_coil.hdf5", "r")
h5null = h5py.File("20210527_dfb_iv_vs_field_coil_null.hdf5", "r")

_tri = h5["alldata"][0, 1, 2,:,1]
_fb = h5["alldata"][0, 0, 1,:,1]

# we have one full period, but it doesn't start where we want,
# be the maximum value of the triangle
ind = np.where(_tri==np.amax(_tri))[0][0]
def make_vector_start_at_ind(v, ind):
    out = np.zeros_like(v)
    out = np.zeros_like(v)
    out[:len(v)-ind] = v[ind:]
    out[len(v)-ind:] = v[:ind]
    return out

tri = make_vector_start_at_ind(_tri, ind)
fb = make_vector_start_at_ind(_fb, ind)

dwell = 2**11
o = dwell-2
tri_n_points = len(tri)//dwell
phi0_fb = 1675


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
fb2 = fb[:-1]+np.cumsum(np.diff(fb)<-1000)*phi0_fb
fb2 -= np.cumsum(np.diff(fb)>1000)*phi0_fb
tri2 = tri[:-1]

plt.figure()
plt.plot(fb2, label="iv data kinda fixed")
plt.plot(tri, label="db triangle")
plt.xlabel("sample number")
plt.ylabel("dac units")
plt.title("data after removing resolved relocks")

def avg_n_points_every_m_points_starting_at_o(v, n, m, o):
    assert o+n < m
    out = np.zeros(len(v)//m)
    for i in range(o, o+n):
        out += v[i::m][:len(out)]
    return out/n


tri_sampled = avg_n_points_every_m_points_starting_at_o(tri2, 4, dwell, dwell-2-4)
fb2_sampled = avg_n_points_every_m_points_starting_at_o(fb2, 4, dwell, dwell-2-4)



# the downward going arc will give us the IV curve
down_fb = fb2_sampled[2:1+tri_n_points//2]
down_db = tri_sampled[2:1+tri_n_points//2]
down_fb_orig = np.copy(down_fb)

up_fb = fb2_sampled[tri_n_points//2:] # could add +1 here to even out length
up_db = tri_sampled[tri_n_points//2:]
up_fb_orig = np.copy(up_fb)
fb2_sampled_orig = np.copy(fb2_sampled)
# locate sign changes
fb2_ind = np.where(np.abs(np.diff(np.array(np.sign(np.diff(fb2_sampled)), dtype="int32"))))[0]+1


# predict last point of down_db based on slope of first first
slope = (down_db[10]-down_db[0])/10
last_predict_down_db = len(down_db)*slope+down_db[0]

slope_down = (down_fb[10]-down_fb[0])/10
last_predict_down_fb = len(down_fb)*slope_down+down_fb[0]
last_err_down_fb = down_fb[-1]-last_predict_down_fb
nphi0_down = last_err_down_fb/phi0_fb
ic_step_fb_down = np.round(nphi0_down)*phi0_fb
fb2_sampled[fb2_ind[2]+1:]-=ic_step_fb_down # this works because most things are views

slope_up = (up_fb[10]-up_fb[0])/10
last_predict_up_fb = len(up_fb)*slope_up+up_fb[0]
last_err_up_fb = up_fb[-1]-last_predict_up_fb
nphi0_up = last_err_up_fb/phi0_fb
ic_step_fb_up = np.round(nphi0_up)*phi0_fb
fb2_sampled[fb2_ind[7]+1:]-=ic_step_fb_up

up_ind = np.where(np.abs(np.diff(np.array(np.sign(np.diff(up_fb)), dtype="int32"))))[0]+1
down_ind = np.where(np.abs(np.diff(np.array(np.sign(np.diff(down_fb)), dtype="int32"))))[0]+1


plt.figure()
plt.plot(fb2_sampled/phi0_fb, label="fb2_sampled")
plt.plot(fb2_sampled_orig/phi0_fb, label="fb2_sampled_org")
plt.plot(fb2_ind, fb2_sampled[fb2_ind]/phi0_fb, "s", label="fb2_ind", fillstyle="none")
plt.plot(fb2_ind, fb2_sampled_orig[fb2_ind]/phi0_fb, "s", label="fb2_orig_ind", fillstyle="none")
plt.legend()

db_offset = 0*2**13
fb_offset = 0

# up_fb_orig -= (up_fb_orig[0]-down_fb_orig[-1])

plt.figure()
plt.plot(down_fb, label="down_fb")
plt.plot(down_fb_orig, label="down_fb_orig")
plt.plot(down_db, label="down_db")
plt.plot(up_fb, label="up_fb")
plt.plot(up_db, label="up_db")
# plt.plot(up_fb_orig, label="up_fb_orig")
plt.plot(up_ind, up_fb[up_ind], "s", label="up_fb ic ind")
plt.plot(down_ind, down_fb[down_ind], "s", label="down_fb ic ind")
plt.plot(len(down_db)-1, last_predict_down_db, "s", label="down fb last predict")
plt.plot(len(down_fb)-1, last_predict_down_fb, "s", label=f"down fb last predict nphi0={nphi0_down:.2f}")
plt.legend()

# pick out the ic one
up_ind = up_ind[2]
down_ind = down_ind[2]
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
# out = conditionvphi(tri, fb, tridwell=10, tristeps=9, tristepsize=31)

# plt.figure()
# plt.plot(up_fb-down_fb[::-1], label=f"up_fb-down_fb[::-1]")
# plt.plot(up_fb, label="up_fb")
# plt.plot(down_fb[::-1], label="down_fb[::-1]")
# plt.legend()
# plt.ylim(-50,50)
# plt.grid(True)

normal_db_lo, normal_db_hi = 2000,14000
sc_db_lo, sc_db_hi = 6000, 7000
normal_inds_down = np.where(np.logical_or(down_db<normal_db_lo, down_db>normal_db_hi))[0]
pfit_normal_down=Polynomial.fit(down_db[normal_inds_down], down_fb[normal_inds_down], 1, domain=[0,2**14])
normal_inds_up = np.where(np.logical_or(up_db<normal_db_lo, up_db>normal_db_hi))[0]
pfit_normal_up=Polynomial.fit(up_db[normal_inds_up], up_fb[normal_inds_up], 1, domain=[0,2**14])
sc_inds_down = np.where(np.logical_and(down_db>sc_db_lo, down_db<sc_db_hi))[0]
pfit_sc_down=Polynomial.fit(down_db[sc_inds_down], down_fb[sc_inds_down], 1, domain=[0,2**14])
sc_inds_up = np.where(np.logical_and(up_db>sc_db_lo, up_db<sc_db_hi))[0]
pfit_sc_up=Polynomial.fit(up_db[sc_inds_up], up_fb[sc_inds_up], 1, domain=[0,2**14])


plt.figure()
plt.title("fit test")
plt.plot(down_db, down_fb,"o", label="down")
plt.plot(down_db, pfit_normal_down(down_db), label="down normal fit")
plt.plot(down_db, pfit_sc_down(down_db), label="down sc fit")
plt.plot(up_db, up_fb,"o", label="up")
plt.plot(up_db, pfit_normal_up(up_db), label="up normal fit")
plt.plot(up_db, pfit_sc_up(up_db), label="up sc fit")
plt.xlabel("db dac units")
plt.ylabel("fb dac units")
plt.legend()

p_normal = (pfit_normal_up+pfit_normal_down)/2
p_sc = (pfit_sc_up+pfit_sc_down)/2
root_x = (p_normal-p_sc).roots()[0]
root_y = p_sc(root_x)

plt.figure()
plt.title("fit test")
plt.plot(down_db-root_x, down_fb-root_y,"o", label="down")
plt.plot(down_db-root_x, pfit_normal_down(down_db)-root_y, label="down normal fit")
plt.plot(down_db-root_x, pfit_sc_down(down_db)-root_y, label="down sc fit")
plt.plot(up_db-root_x, up_fb-root_y,"o", label="up")
plt.plot(up_db-root_x, pfit_normal_up(up_db)-root_y, label="up normal fit")
plt.plot(up_db-root_x, pfit_sc_up(up_db)-root_y, label="up sc fit")
plt.xlabel("db dac units")
plt.ylabel("fb dac units")
plt.legend()
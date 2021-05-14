import pylab as plt
import numpy as np
from cringe.tune.analysis import conditionvphi
plt.ion()
plt.close("all")
data = np.load("20210507_SSRL_AX_56p6_DFB_IV.npy")

upstart = 164830
upend = 690141
_tri = data[0,2,:,1]
_fb = data[1,1,:,1]

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

dwell = 2**10
o = dwell-2
last = dwell*161
tri_n_points = len(tri)//dwell


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

fb2 = fb[:-1]+np.cumsum(np.diff(fb)<-1000)*1670
fb2 -= np.cumsum(np.diff(fb)>1000)*1670
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
iv_fb = fb2_sampled[:tri_n_points//2]
iv_db = tri_sampled[:tri_n_points//2]

ic_fb = fb2_sampled[tri_n_points//2:]
ic_db = tri_sampled[tri_n_points//2:]
# locate ic by looking for a drop in ic_fb
inds = np.where(np.diff(ic_fb)<-100)[0]
assert len(inds) == 1 # we should only see one ic
ind = inds[0]

offset = ic_fb[0]

plt.figure()
plt.plot(iv_db, iv_fb-offset, ".", label="iv curve")
plt.plot(ic_db, ic_fb-offset, ".", label="ic trace")
plt.plot(ic_db[ind], ic_fb[ind]-offset, "o", label=f"ic at {ic_fb[ind]-offset} dac units")
plt.xlabel("db (from dfb card) dac units")
plt.ylabel("fb dac units (zero subtracted)")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.title("dfb based iv, measured full IV with no flux jumps\nand Ic with ~1e6 samples, so takes 1e6/samplerate seconds")
plt.tight_layout()
# out = conditionvphi(tri, fb, tridwell=10, tristeps=9, tristepsize=31)
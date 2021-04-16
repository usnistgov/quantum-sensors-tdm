'''
test_phasecorr.py

test phase correction algorithm with simulated passband
'''

from fts_utils import *

# define simulated passband
N=1000
f=np.linspace(0,20,N)
fo = 5 # center frequency of passband (icm)
a = 1 # width of passband (icm)
B = np.exp(-1*((f-fo)/a)**2)

# make single-sided interferogram from the passband
I = scipy.fftpack.fft(B)
samp_int = f[2]-f[1]
z = scipy.fftpack.fftfreq(len(B),samp_int)
z = scipy.fftpack.ifftshift(z)
I = scipy.fftpack.ifftshift(I)
I = I + np.random.normal(size=len(I),scale=1) # add noise
phi_err = 0
dzdphi_err = np.pi/10.
I = I*np.exp(1j*(phi_err+dzdphi_err*z)) # add a phase error
#I = I*np.sin(np.pi/4*z)
dexs=np.where(z>-2)[0]
z=z[dexs]
z=z-z[0]
I=np.real(I[dexs])
#I=I+np.random.normal(size=len(I),scale=1) # add noise
# now analyze simulated ifg -----------------------------------------

i2s = IfgToSpectrum()
def test1():
    ''' double sided ifg '''
    # first plot the IFG
    plt.figure(1)
    plt.plot(z,I,'bo-')
    plt.xlabel('$\delta (cm)$')
    plt.ylabel('Response (arb)')

    # work on double sided ifg
    z_ds,I_ds = i2s.get_double_sided_ifg(z,I,zpd_index=None,zpd_value=None,x_to_opd=True,
                             fftpacking=True,plotfig=False)
    f_ds, B_ds = i2s.get_fft(z_ds,I_ds,plotfig=False)
    phi_ds = np.arctan(np.imag(B_ds)/np.real(B_ds))
    #phi_ds = np.angle(B_ds)
    Np=int(len(f_ds)/2)
    B_corr = B_ds*np.exp(-1j*phi_ds)

    # plotting ----------------------------------------------------------

    #f_ds2 = ifftshift(f_ds) ; B_ds = ifftshift(B_ds) ; phi_ds = ifftshift(phi_ds)
    plt.figure(2)
    plt.subplot(211)
    plt.plot(f_ds,np.real(B_ds),'b-')
    plt.plot(f_ds,np.imag(B_ds),'g-')
    plt.plot(f_ds,np.abs(B_ds),'k-')
    plt.plot(f_ds,np.real(B_corr),'b--')
    plt.plot(f_ds,np.imag(B_corr),'g--')
    plt.plot(f_ds,np.abs(B_corr),'k--')
    plt.xlabel('Frequency (icm)')
    plt.ylabel('Response (arb)')
    plt.subplot(212)
    plt.plot(f_ds,phi_ds)
    plt.xlabel('Frequency (icm)')
    plt.ylabel('Phase (rad)')

    plt.figure(3)
    plt.plot(f,B,'k--')
    plt.plot(f_ds[0:Np],np.real(B_corr[0:Np])/(N/2),'o-')
    plt.plot(f_ds[0:Np],np.abs(B_corr[0:Np])/(N/2),'o-')
    plt.xlabel('Frequency (icm)')
    plt.ylabel('Response (arb)')
    plt.legend(('input spec','real recovered spec','abs rec spec'))
    plt.show()


def test2():
    ''' higher res mertz phase correction '''

    fp,Bp = i2s.phase_correction_mertz(z,I,plotfig=False)
    Np=int(len(fp)/2)

    plt.figure(1)
    plt.plot(f,B,'k--')
    plt.plot(fp[0:Np],2*np.real(Bp[0:Np])/(N/2),'o-')
    plt.plot(fp[0:Np],2*np.imag(Bp[0:Np])/(N/2),'o-')
    plt.plot(fp[0:Np],2*np.abs(Bp[0:Np])/(N/2),'o-')
    plt.xlabel('Frequency (icm)')
    plt.ylabel('Response (arb)')
    plt.legend(('input','real rec','imag rec','abs rec'))
    plt.show()

test1()
test2()

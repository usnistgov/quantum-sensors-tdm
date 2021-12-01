import pymeasure
from pymeasure import instruments
from pymeasure.instruments.agilent import Agilent33500
import numpy as np
import nasa_client
import pylab as plt
plt.ion()

a = Agilent33500("TCPIP::A-33522B-01982.local::inst0::INSTR")

def ch2_output_enable(enable_bool):
    a.write(f"OUTP2 {enable_bool:d}")

def ch2_setup():
    a.write(f"SOUR2:FUNC DC")
    a.write(f"SOUR2:VOLT:OFFS 0")
    a.write(f"OUTP2:LOAD INF")
    ch2_output_enable(True)

def ch2_setvolt(v):
    a.write(f"SOUR2:VOLT:OFFS {v:f}")

def ch1_set_ramp_extreme(extreme):
    a.amplitude = np.abs(extreme)
    if extreme>=0:
        a.arb_file = "iv_ramp"
    else:
        a.arb_file = "iv_ramp_neg"

def ch1_ramp_setup(profile):
    a.data_volatile_clear()         # Clear volatile internal memory
    a.data_arb(                     # Send data points of arbitrary waveform
        'iv_ramp',
        profile,          # the wave form
        data_format='float'              # float format takes values from -1 to 1
    )
    a.data_arb(                     # Send data points of arbitrary waveform
        'iv_ramp_neg',
        -profile,          # the wave form
        data_format='float'              # float format takes values from -1 to 1
    )
    a.arb_file = 'iv_ramp'             # Select the transmitted waveform 'test'


def ch1_setup(profile):
    a.write("OUTP2:POL NORM")
    a.shape = "ARB"
    a.arb_srate = 1e4
    a.output = True
    a.output_load = "INF"
    a.burst_state = True
    a.busrt_ncycles = 1
    a.trigger_source ="BUS"
    ch1_ramp_setup(profile)
    ch1_set_ramp_extreme(.5)


def ch1_trigger():
    a.trigger()

def make_ramp_dwell_ramp_profile(n_ramp, n_dwell):
    _ramp = np.linspace(0,1,n_ramp)
    dwell = np.ones(n_dwell)
    profile = np.hstack([_ramp, dwell, _ramp[::-1]])
    return profile


ch2_setup()
ch2_setvolt(.1)
profile = make_ramp_dwell_ramp_profile(n_ramp=19250, n_dwell=500)
ch1_setup(profile)
ch1_set_ramp_extreme(2)
print(f"any errors? {a.check_errors()}")
c = nasa_client.EasyClient()


ch1_trigger()
data = c.getNewData(-0.001, minimumNumPoints=2200000)
col=0
row=0
fb = data[col, row, :, 1]
plt.figure()
plt.plot(fb)


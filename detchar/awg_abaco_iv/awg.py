# import pymeasure
# from pymeasure import instruments
# from pymeasure.instruments.agilent import Agilent33500
import numpy as np
import pylab as plt
import time
import sys, os
import socket

class Agilent33500SCPI():
    def __init__(self, addr="192.168.101.59", port=5025):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.settimeout(1)
        self.s.connect((addr, port))
    
    def write(self, msg):
        self.s.send(f"{msg}\r\n".encode())

    def read(self):
        return self.s.read(1024).decode()
        
    def ask(self, msg):
        self.write(msg)
        return self.read()

    def data_arb(self, arb_name, data_points, data_format='DAC'):
        """
        Uploads an arbitrary trace into the volatile memory of the device. The data_points can be
        given as comma separated 16 bit DAC values (ranging from -32767 to +32767), as comma
        separated floating point values (ranging from -1.0 to +1.0) or as a binary data stream.
        Check the manual for more information. The storage depends on the device type and ranges
        from 8 Sa to 16 MSa (maximum).
        TODO: *Binary is not yet implemented*
        :param arb_name: The name of the trace in the volatile memory. This is used to access the
                         trace.
        :param data_points: Individual points of the trace. The format depends on the format
                            parameter.
                            format = 'DAC' (default): Accepts list of integer values ranging from
                            -32767 to +32767. Minimum of 8 a maximum of 65536 points.
                            format = 'float': Accepts list of floating point values ranging from
                            -1.0 to +1.0. Minimum of 8 a maximum of 65536 points.
                            format = 'binary': Accepts a binary stream of 8 bit data.
        :param data_format: Defines the format of data_points. Can be 'DAC' (default), 'float' or
                            'binary'. See documentation on parameter data_points above.
        """
        if data_format == 'DAC':
            separator = ', '
            data_points_str = [str(item) for item in data_points]  # Turn list entries into strings
            data_string = separator.join(data_points_str)  # Join strings with separator
            print(f"DATA:ARB:DAC {arb_name}, {data_string}")
            self.write(f"DATA:ARB:DAC {arb_name}, {data_string}")
            return
        elif data_format == 'float':
            separator = ', '
            data_points_str = [str(item) for item in data_points]  # Turn list entries into strings
            data_string = separator.join(data_points_str)  # Join strings with separator
            print(f"DATA:ARB {arb_name}, {data_string}")
            self.write(f"DATA:ARB {arb_name}, {data_string}")
            return
        elif data_format == 'binary':
            raise NotImplementedError(
                'The binary format has not yet been implemented. Use "DAC" or "float" instead.')
        else:
            raise ValueError(
                'Undefined format keyword was used. Valid entries are "DAC", "float" and "binary"')

    def check_errors(self):
        """ Read all errors from the instrument.
        :return: list of error entries
        """
        errors = []
        while True:
            err = a.ask("SYST:ERR?")
            if int(err[0]) != 0:
                errors.append(err)
            else:
                break
        return errors
    
a = Agilent33500SCPI()




# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def ch2_output_enable(enable_bool):
    a.write(f"OUTP2 {enable_bool:d}")

def ch2_setvolt(v):
    a.write(f"SOUR2:VOLT:OFFS {v:f}")

def ch2_setup():
    a.write(f"SOUR2:FUNC DC")
    a.write(f"SOUR2:VOLT:OFFS 0")
    a.write(f"OUTP2:LOAD INF")
    ch2_output_enable(True)
    ch2_setvolt(0)

def ch1_set_ramp_extreme(extreme):
    if extreme>=0:
        # a.arb_file = "iv_ramp"
        a.write("FUNC:ARB iv_ramp")
    else:
        # a.arb_file = "iv_ramp_neg"
        a.write("FUNC:ARB iv_ramp_neg")
    # a.amplitude = np.abs(extreme)
    a.write(f"VOLT {np.abs(extreme)}")


def ch1_ramp_setup(profile):
    #a.data_volatile_clear()
    a.write("DATA:VOL:CLE")         # Clear volatile internal memory
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
    # a.arb_file = 'iv_ramp'             # Select the transmitted waveform 'test'
    a.write("FUNC:ARB iv_ramp")


def ch1_setup(profile, srate):
    a.write("OUTP2:POL NORM")
    a.write("FUNC:ARB:FILT OFF")
    # a.shape = "ARB"
    a.write("FUNC ARB")
    # a.arb_srate = srate
    a.write(f"FUNC:ARB:SRAT {srate}")
    # a.output = True
    a.write("OUTP 1")
    # a.output_load = "INF"
    a.write("OUTP:LOAD INF")
    # a.burst_state = True
    # a.write("BURS:STAT 1")
    # a.busrt_ncycles = 1
    a.write("BURS:NCYC 2")
    # a.trigger_source ="BUS"
    a.write("TRIG:SOUR IMM")
    a.write("BURS:INT:PER MIN")
    a.write("OUTP:SYNC:MODE MARK")
    a.write("SOUR:MARK:POINT 0")
    ch1_ramp_setup(profile)
    ch1_set_ramp_extreme(0)


def ch1_trigger(ramp_extreme):
    ch1_set_ramp_extreme(ramp_extreme)
    time_nano = time.time()*1e9
    # a.trigger()
    # a.write("*TRG;*WAI")
    return time_nano

def make_ramp_dwell_ramp_profile(n_ramp, n_dwell, blip_delta):
    _ramp = np.linspace(0,1,n_ramp)
    dwell = np.ones(n_dwell)
    dwell[len(dwell)//2]=1-blip_delta # blipfor alignment
    start_dwell = np.zeros(n_dwell)
    profile = np.hstack([start_dwell, _ramp, dwell, _ramp[::-1], start_dwell])
    return profile

def make_pulse_profile(baseline, peaks, n_dwell):
    profile = []
    for peak in peaks:
        profile += [peak]*n_dwell
        profile += [baseline]*n_dwell
    return np.array(profile)

def add_flux_jumps(fb, phi0_fb, fb_step_threshold):
    """return an array like fb, but with flux jumps resolved and one value removed from the end
    method: find indicies with np.diff(fb) greater ro less than fb_step_threshold
    and add phi0 units with the correct sign"""
    out = fb[1:]+np.cumsum(np.diff(fb)<-fb_step_threshold)*phi0_fb
    out -= np.cumsum(np.diff(fb)>fb_step_threshold)*phi0_fb
    return out

last_flux_jump_threshold_dac_units = None
def set_arl_params(flux_jump_threshold_dac_units):
    # this function is slow due to the cringe timer, so
    # we skip it if its already been done
    global last_flux_jump_threshold_dac_units
    if flux_jump_threshold_dac_units == last_flux_jump_threshold_dac_units:
        print("skpping arl set since flux_jump_threshold_dac_units == last_flux_jump_threshold_dac_units")
        return
    cc.set_arl_params(flux_jump_threshold_dac_units=flux_jump_threshold_dac_units,
        plus_event_reset_delay_frm_units=0, minus_event_reset_delay_frm_units=0)
    last_flux_jump_threshold_dac_units = flux_jump_threshold_dac_units
    print("wait 1 s for cringe timer")
    time.sleep(1) # wait out cringe timer
    print("cringe timer done")

def check_errors():
    errs = check_errors()
    if len(errs)>1:
        print("AWG Errors!")
        print(errs)

def get_fba_offsets():
    reply, fba_offsets = cc.get_fba_offsets()
    assert reply.startswith("ok")
    return fba_offsets




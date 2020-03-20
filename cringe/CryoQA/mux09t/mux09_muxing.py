'''
Written by: Johnathon Gard, Carl Reintsema, Robby Stevens
Purpose:
    To sweep out the bias line for the row select SQUID 1 on MUX09t (old style multiplexers with a second stage
    SQUID then a connection to a lower gain SQUID Series Array.
How to use this class:
    The QA personel will need to launch cringe and tune up the series array, squid 2, and lock on to the second stage
    SQUID. using a DFB card, feed back onto the second stage squid to directly see the SQUID 1 VPhi curve when a
    triangle wave is present on the SQUID1 Feed Back connection.

    List of Steps:
    0.Cool down probe, connect all of the Row select connections to the probe, power up the probe, connect the probe up
      to tune the series array. Connect a signal generator to the SSA FB line to fully sweep out the SQUID VPhi.
    1.Launch cringe and load MrChips configuration
    2.

'''
import sys
import named_serial
import numpy as np
import pylab as pl
import time
'''Local imports for the data processing and bad16 communication'''
import badchan_simple
import daq


class MUX09:

    def __init__(self):
        # Script parameters
        self.number_averages = 20
        self.number_rows = 33
        self.bad16_card_addr = [32, 34, 35]
        self.triangle_period = 2048
        self.icmin_sigma = 3
        self.number_steps = 200
        self.change_pause_s = 0.001
        self.num_points_after_max = 7
        '''Base path of where to save the file'''
        self.base_file_path = '/home/pcuser/Documents/Squid_Screening_Data/MUX09t/'

        self.serialport = named_serial.Serial(port='rack', shared=True)
        self.badchan = badchan_simple.BadChanSimple(chan=0,
                                                    cardaddr=self.bad16_card_addr[0],
                                                    serialport=self.serialport)
        self.daq = daq.Daq(tri_period=self.triangle_period,
                           averages=self.number_averages)
        '''Make variables to hold data'''
        self.baselines = np.zeros((self.number_rows, self.triangle_period))  # There is no reason to save these
        self.baselines_std = np.zeros(self.number_rows)
        self.icmin = np.zeros(self.number_rows)
        self.icmax = np.zeros(self.number_rows)
        '''Generate sweep of valid dac values'''
        self.row_sweep_dac_values = np.linspace(0, 2**14, self.number_steps)
        self.row_sweep_dac_values = np.around(self.row_sweep_dac_values)
        self.row_sweep_dac_values = self.row_sweep_dac_values.astype('int')
        '''Place to store a rows sweep'''
        self.row_average_adcpp_sweep = np.zeros((self.number_rows, self.number_steps))
        '''Set variables to store other data about chip'''
        self.chip_id = None
        self.sq2_icmin = None
        self.sq2_icmax = None
        self.qa_name = None
        self.note = ''

        today = time.localtime()
        self.date = str(today.tm_year) + '_' + str(today.tm_mon) + '_' + str(today.tm_mday)

    def row_decode(self, row):
        if row in range(0, 1):  # DS
            self.badchan.cardaddr = self.bad16_card_addr[2]
            self.badchan.change_channel(row)
        elif row in range(1, 17):               # Rows 0 through 15
            self.badchan.cardaddr = self.bad16_card_addr[0]
            self.badchan.change_channel(row-1)
        elif row in range(17, self.number_rows):              # Rows 16 through 31
            self.badchan.cardaddr = self.bad16_card_addr[1]
            self.badchan.change_channel(row-17)

        else:
            print 'That is not a valid row number passed to row_decode'

    def zero_everything(self):
        print 'Zero all BAD16 DAC Hi values and setup for muxing QA'
        for row in range(self.number_rows):
            self.row_decode(row)
            self.badchan.set_d2a_hi_value(0x0000)
            self.badchan.set_dc(False)
            self.badchan.set_lohi(True)

    def get_baselines(self):
        '''Makes an assumption that no rows are active and that the feed back loop is just noise'''
        print 'Getting basline data to determine ic_min std deviation.'
        fb, err = self.daq.take_data()
        self.baselines = fb[:, :]  # There is no reason to save these
        self.baselines_std = np.std(fb, 1)

    def row_bias_sweeper(self):
        start_time = time.time()
        print 'Starting the row sweep of :' + str(self.number_steps) + ' steps'
        for sweep_point in range(self.number_steps):
            sys.stdout.write('\rSweep point sweep_point %5.0f' % sweep_point)
            sys.stdout.flush()
            for i in range(self.number_rows):
                self.row_decode(i)
                self.badchan.set_d2a_hi_value(self.row_sweep_dac_values[sweep_point])

            time.sleep(self.change_pause_s)
            fb, err = self.daq.take_average_data()

            self.row_average_adcpp_sweep[:, sweep_point] = np.abs(np.max(fb[0:self.number_rows], 1) -
                                                                  np.min(fb[0:self.number_rows], 1))
            # Find IC_min

        end_time = time.time()
        print ''
        print 'Delta Seconds'
        print (end_time-start_time)

    def mux_sweep(self):
        start_time = time.time()

        self.zero_everything()
        self.get_baselines()
        self.row_bias_sweeper()

        end_time = time.time()
        print ''
        print 'Delta Seconds for mux sweep'
        print (end_time - start_time)

    def calculate_ics(self):
        for row in range(self.number_rows):
            have_icmin = False
            for sweep_point in range(self.number_steps):
                if have_icmin is False:
                    if self.row_average_adcpp_sweep[row, sweep_point] >= self.icmin_sigma * self.baselines_std[row]:
                        have_icmin = True
                        self.icmin[row] = self.row_sweep_dac_values[sweep_point]
            if have_icmin is True:
                self.icmax[row] = self.row_sweep_dac_values[np.argmax(self.row_average_adcpp_sweep[row, :])]
            else:
                ''' In the event that no ICmin is found, aka a dead row, set the ic's to be max so that they stick out.'''
                self.icmin[row] = int(2 ** 14)
                self.icmax[row] = int(2 ** 14)


    def mux_spotcheck(self):
        self.zero_everything()
        for row in range(self.number_rows):
            self.row_decode(row)
            self.badchan.set_dc(True)

        for row in range(self.number_rows):
            sys.stdout.write('\rBiasing Row %5.0f' % row)
            sys.stdout.flush()
            self.row_decode(row)
            self.badchan.set_lohi(True)
            self.badchan.set_d2a_hi_value(int(self.icmax[row]))
            raw_input()
            self.badchan.set_lohi(False)

        self.zero_everything()

    def plot_icmin_icmax(self):
        pl.ion()
        pl.figure()
        pl.plot(self.icmax, label='Ic_max')
        pl.plot(self.icmin, label='Ic_min')
        pl.legend()
        pl.title('Ic_min and Ic_max of chip: ' + str(self.chip_id))
        pl.xlabel('row number')
        pl.ylabel('BAD16 DAC setting')

    def set_info(self, chip_id, qa_name):
        self.chip_id = chip_id
        self.qa_name = qa_name

    def set_sq2_data(self, sq2_icmin, sq2_icmax):
        self.sq2_icmin = sq2_icmin
        self.sq2_icmax= sq2_icmax

    def set_notes(self, note_str):
        self.note = note_str

    def save_npz(self):
        if self.chip_id is None:
            print 'You have not set the Chip ID'
            return
        if self.sq2_icmin is None:
            print 'You have not set the SQUID 2 IC_min'
            return
        if self.sq2_icmax is None:
            print 'You have not set the SQUID 2 IC_max'
            return
        if self.qa_name is None:
            print 'Please set your name as the QA person'
            return
        if self.note is '':
            print 'Please input any notes about this chip.'
            print 'Such as what the spot check on the inputs told you'
            return

        today = time.localtime()
        filename = self.chip_id  + '_' + str(today.tm_year) + '_' + str(today.tm_mon) + '_' + str(today.tm_mday)
        np.savez_compressed(self.base_file_path + filename,
                            number_averages=self.number_averages,
                            number_rows=self.number_rows,
                            triangle_period=self.triangle_period,
                            icmin_sigma=self.icmin_sigma,
                            number_steps=self.number_steps,
                            change_pause_s=self.change_pause_s,

                            baselines=self.baselines,  # There is no reason to save these
                            baselines_std=self.baselines_std,
                            icmin=self.icmin,
                            icmax=self.icmax,
                            row_sweep_dac_values=self.row_sweep_dac_values,
                            row_average_adcpp_sweep=self.row_average_adcpp_sweep,

                            chip_id=self.chip_id,
                            date=self.date,
                            sq2_icmin=self.sq2_icmin,
                            sq2_icmax=self.sq2_icmax,
                            qa_name=self.qa_name,
                            note=self.note)

    def load_npz(self, file_name):
        loaded_file = np.load(file_name)
        self.number_averages    = loaded_file['number_averages']
        self.number_rows        = loaded_file['number_rows']
        self.triangle_period    = loaded_file['triangle_period']
        self.icmin_sigma        = loaded_file['icmin_sigma']
        self.number_steps       = loaded_file['number_steps']
        self.change_pause_s     = loaded_file['change_pause_s']

        self.baselines          = loaded_file['baselines']  # There is no reason to load these
        self.baselines_std      = loaded_file['baselines_std']
        self.icmin              = loaded_file['icmin']
        self.icmax              = loaded_file['icmax']
        self.row_sweep_dac_values = loaded_file['row_sweep_dac_values']
        self.row_average_adcpp_sweep = loaded_file['row_average_adcpp_sweep']

        self.chip_id            = loaded_file['chip_id']
        self.date               = loaded_file['date']
        self.sq2_icmin          = loaded_file['sq2_icmin']
        self.sq2_icmax          = loaded_file['sq2_icmax']
        self.qa_name            = loaded_file['qa_name']
        self.note               = loaded_file['note']




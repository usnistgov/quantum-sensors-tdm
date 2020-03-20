'''
Written by: Johnathon Gard, Carl Reintsema, Robby Stevens
Purpose:

'''
import sys
import named_serial
import numpy as np
# import pylab as pl
import time
from statsmodels.nonparametric.smoothers_lowess import lowess
'''Local imports for the data processing, bad16, and tower communication'''
import badchan_simple
import daq
import towerchannel_simple


class Mux:

    def __init__(self, which_columns=[0]):
        '''Script and system parameters'''
        self.save_all_data_flag = False
        self.number_averages_phase1 = 30
        self.number_averages_phase2 = 20
        self.number_averages_phase3 = 30
        self.which_columns = which_columns
        self.number_columns = len(which_columns)
        self.number_rows = 12
        self.bad16_card_rs_addr = [32]
        self.bad16_card_in_addr = [34]
        self.tower_sq1_bias_addr = 11
        self.triangle_period = 8192
        self.icmin_sigma = 3
        self.number_steps = 200  # 16384/200
        self.change_pause_s = 0.001
        self.phase1_sq1_bias = 1200
        '''Base path of where to save the file'''
        self.base_file_path = '/home/pcuser/Documents/Squid_Screening_Data/fas/'
        '''Construct objects to comunicate with the testing setup'''
        self.serialport = named_serial.Serial(port='rack', shared=True)
        self.badchan_rs = badchan_simple.BadChanSimple(chan=0,
                                                       cardaddr=self.bad16_card_rs_addr[0],
                                                       serialport=self.serialport)
        self.badchan_in = badchan_simple.BadChanSimple(chan=0,
                                                       cardaddr=self.bad16_card_in_addr[0],
                                                       serialport=self.serialport)
        self.daq = daq.Daq(tri_period=self.triangle_period,
                           averages=self.number_averages_phase1)
        self.tower = towerchannel_simple.TowerChannel(cardaddr=self.tower_sq1_bias_addr,
                                                      serialport="tower")
        '''Make variables to hold data'''
        self.baselines_std = np.zeros((self.number_columns, self.number_rows))
        self.icmin = np.zeros((self.number_columns, self.number_rows))
        self.icmax = np.zeros((self.number_columns, self.number_rows))
        self.icmod_max = np.zeros((self.number_columns, self.number_rows))
        self.iccol_min = np.zeros(self.number_columns)
        self.iccol_max = np.zeros(self.number_columns)
        self.sq1_row_bias_phase1 = np.zeros((self.number_columns, self.number_rows))
        self.sq1_row_bias_phase2 = np.zeros(self.number_rows)
        self.sq1_col_bias_phase3 = np.zeros(self.number_rows)
        self.sq1_row_select_m = np.zeros((self.number_columns, self.number_rows, 2))
        self.sq1_row_input_m = np.zeros((self.number_columns, self.number_rows, 2))
        self.sq1_row_fb_m = np.zeros((self.number_columns, self.number_rows, 2))
        '''Generate sweep of valid dac values'''
        step_size = 20000/self.number_steps
        self.row_sweep_tower_values = np.linspace(0, 20000-step_size, self.number_steps)
        self.row_sweep_tower_values = np.around(self.row_sweep_tower_values)
        self.row_sweep_tower_values = self.row_sweep_tower_values.astype('int')
        '''Place to store a row sweep data'''
        self.row_sweep_average_mod = np.zeros((self.number_columns, self.number_rows, self.number_steps))
        self.row_sweep_average_min = np.zeros((self.number_columns, self.number_rows, self.number_steps))
        self.row_sweep_average_max = np.zeros((self.number_columns, self.number_rows, self.number_steps))
        self.row_average_mod_sweep = np.zeros((self.number_columns, self.number_rows, self.number_steps))
        self.row_sweep_average_trace = np.zeros((self.number_steps, self.number_columns, self.number_rows,
                                                 self.triangle_period))
        self.feedback_M_data = np.zeros((self.number_columns, self.number_rows, self.triangle_period))
        self.feedback_M_data_folded = np.zeros((self.number_columns, self.number_rows, self.triangle_period/2))
        self.feedback_M_data_smoothed = np.zeros((self.number_columns, self.number_rows, self.triangle_period/2))
        self.input_M_data = np.zeros((self.number_columns, self.number_rows, self.triangle_period))
        self.input_M_data_folded = np.zeros((self.number_columns, self.number_rows, self.triangle_period/2))
        self.input_M_data_smoothed = np.zeros((self.number_columns, self.number_rows, self.triangle_period/2))
        self.rowsel_M_data = np.zeros((self.number_columns, self.number_rows, self.triangle_period))
        self.rowsel_M_data_folded = np.zeros((self.number_columns, self.number_rows, self.triangle_period/2))
        self.rowsel_M_data_smoothed = np.zeros((self.number_columns, self.number_rows, self.triangle_period/2))
        '''Set variables to store other data about chip'''
        self.chip_id = np.zeros(self.number_columns, dtype=object)
        self.chip_id.fill('')
        self.icmax.fill(None)
        self.qa_name = None
        self.note = np.zeros(self.number_columns, dtype=object)

        today = time.localtime()
        self.date = str(today.tm_year) + '_' + str(today.tm_mon) + '_' + str(today.tm_mday)

    def row_decode_rs(self, row):
        if row in range(0, 16):  # DS
            self.badchan_rs.cardaddr = self.bad16_card_rs_addr[0]
            self.badchan_rs.change_channel(row)

        else:
            print('That is not a valid row number passed to row_decode')

    def row_decode_in(self, row):
        if row in range(0, 16):  # DS
            self.badchan_in.cardaddr = self.bad16_card_in_addr[0]
            self.badchan_in.change_channel(row)

        else:
            print('That is not a valid row number passed to row_decode')

    def tower_set_voltage(self, channel, dac_value):
        self.tower.bluebox.channel = channel
        self.tower.set_value(dac_value)

    def zero_everything(self):
        print('Zero all BAD16 DAC Hi values, Tower SQ1 bias and setup for muxing QA')
        for row in range(16):
            self.row_decode_rs(row)
            self.badchan_rs.set_d2a_hi_value(0x0000)
            self.badchan_rs.set_d2a_lo_value(0x0000)
            self.badchan_rs.set_dc(False)
            self.badchan_rs.set_lohi(True)
            self.row_decode_in(row)
            self.badchan_in.set_d2a_hi_value(0x0000)
            self.badchan_in.set_d2a_lo_value(0x0000)
            self.badchan_in.set_dc(False)
            self.badchan_in.set_lohi(True)

        for col in range(8):
            self.tower_set_voltage(col, 0)

    def get_baselines(self):
        '''Makes an assumption that no rows are active and that the feed back loop is just noise'''
        print('Getting basline data to determine ic_min std deviation.')
        fb, err = self.daq.take_data()
        self.baselines_std = np.std(fb, 2)
        for row in range(len(self.baselines_std)):
            if self.baselines_std[0][row] > 20:
                print('The standard deviation for row:' + str(row) + ' is high: ' + str(self.baselines_std[row]))

    def sq1_bias_sweeper(self):
        start_time = time.time()
        print('Starting the row sweep of :' + str(self.number_steps) + ' steps')
        for sweep_point in range(self.number_steps):
            sys.stdout.write('\rSweep point sweep_point %5.0f' % sweep_point)
            sys.stdout.flush()
            for col in range(8):
                self.tower_set_voltage(col, int(self.row_sweep_tower_values[sweep_point]))

            time.sleep(self.change_pause_s)
            fb, err = self.daq.take_average_data()
            self.row_sweep_average_max[:, :, sweep_point] = np.max(fb[:, 0:self.number_rows], 2)
            self.row_sweep_average_min[:, :, sweep_point] = np.min(fb[:, 0:self.number_rows], 2)
            self.row_sweep_average_mod[:, :, sweep_point] = np.abs(np.max(fb[:, 0:self.number_rows], 2) -
                                                                   np.min(fb[:, 0:self.number_rows], 2))
            self.row_sweep_average_trace[sweep_point] = fb
            # Find IC_min

        end_time = time.time()
        print('')
        print('Delta Seconds')
        print((end_time-start_time))

    def calculate_ics(self):
        for col in range(self.number_columns):
            for row in range(self.number_rows-1):
                have_icmin = False
                for sweep_point in range(self.number_steps):
                    if have_icmin is False:
                        if self.row_sweep_average_mod[col, row, sweep_point] >= self.icmin_sigma * \
                                self.baselines_std[col, row]:
                            have_icmin = True
                            self.icmin[col, row] = self.row_sweep_tower_values[sweep_point]
                if have_icmin is True:
                    self.icmod_max[col, row] = self.row_sweep_tower_values[np.argmax(self.row_sweep_average_mod[col,
                                                                                     row, :])]
                else:
                    ''' In the event that no ICmin is found, aka a dead row, set the ic's to be max so that they stick out.'''
                    self.icmin[col, row] = int(2 ** 14)
                    #self.icmod_max[col, row] = int(2 ** 14)
                    self.icmod_max[col, row] = 0

    def set_chip_info(self, column, chip_id):
        self.chip_id[column] = chip_id

    def set_qa_name(self, qa_name):
        self.qa_name = qa_name

    def set_notes(self, column, note_str):
        self.note[column] = note_str

    def windowed_derivative(self, row_data, window_size_even=8):
        window_size = window_size_even
        data_derivative = np.zeros_like(row_data)

        for i in range(len(row_data)):
            end_i0 = i+window_size/2
            end_i1 = i+window_size
            if (end_i1 + 1) < (len(row_data)-1):
                sample0 = np.average(row_data[i:end_i0])
                sample1 = np.average(row_data[i+window_size/2:end_i1])
            else:
                sample0 = 0
                sample1 = 0
            data_derivative[i] = (sample1 - sample0)/window_size

        return data_derivative

    def save_npz(self):
        for col in range(self.number_columns):
            if self.chip_id[col] is None:
                print('You have not set the Chip ID for Column: ' + str(col))
                return

            if self.qa_name is None:
                print('Please set your name as the QA person')
                return

            if self.note[col] is '':
                print('Please input any notes about this Column: ' + str(col))
                return

            if self.save_all_data_flag:
                filename = self.chip_id[col] + '_' + self.date
                np.savez_compressed(self.base_file_path + filename,
                                    save_all_data_flag=self.save_all_data_flag,
                                    number_averages_phase1=self.number_averages_phase1,
                                    number_averages_phase2=self.number_averages_phase2,
                                    number_averages_phase3=self.number_averages_phase3,
                                    number_columns=self.number_columns,
                                    number_rows=self.number_rows,
                                    bad16_card_rs_addr=self.bad16_card_rs_addr,
                                    bad16_card_in_addr=self.bad16_card_in_addr,
                                    tower_sq1_bias_addr=self.tower_sq1_bias_addr,
                                    triangle_period=self.triangle_period,
                                    icmin_sigma=self.icmin_sigma,
                                    number_steps=self.number_steps,
                                    change_pause_s=self.change_pause_s,
                                    phase1_sq1_bias=self.phase1_sq1_bias,

                                    base_file_path=self.base_file_path,
                                    filename=filename,
                                    date=self.date,

                                    baselines_std=self.baselines_std[col],
                                    icmin=self.icmin[col],
                                    icmax=self.icmax[col],
                                    icmod_max=self.icmod_max[col],
                                    iccol_min=self.iccol_min[col],
                                    iccol_max=self.iccol_max[col],

                                    sq1_row_bias_phase1=self.sq1_row_bias_phase1[col],
                                    sq1_row_bias_phase2=self.sq1_row_bias_phase2,
                                    sq1_col_bias_phase3=self.sq1_col_bias_phase3[col],
                                    sq1_row_select_m=self.sq1_row_select_m[col],
                                    sq1_row_input_m=self.sq1_row_input_m[col],
                                    sq1_row_fb_m=self.sq1_row_fb_m[col],

                                    row_sweep_tower_values=self.row_sweep_tower_values,
                                    row_sweep_average_mod=self.row_sweep_average_mod[col],
                                    row_sweep_average_min=self.row_sweep_average_min[col],
                                    row_sweep_average_max=self.row_sweep_average_max[col],
                                    row_average_mod_sweep=self.row_average_mod_sweep[col],
                                    row_sweep_average_trace=self.row_sweep_average_trace[col],
                                    feedback_M_data=self.feedback_M_data[col],
                                    feedback_M_data_folded=self.feedback_M_data_folded[col],
                                    feedback_M_data_smoothed=self.feedback_M_data_smoothed[col],
                                    input_M_data=self.input_M_data[col],
                                    input_M_data_folded=self.input_M_data_folded[col],
                                    input_M_data_smoothed=self.input_M_data_smoothed[col],
                                    rowsel_M_data=self.rowsel_M_data[col],
                                    rowsel_M_data_folded=self.rowsel_M_data_folded[col],
                                    rowsel_M_data_smoothed=self.rowsel_M_data_smoothed[col],
                                    chip_id=self.chip_id[col],
                                    qa_name=self.qa_name,
                                    note=self.note[col])

            else:
                filename = self.chip_id[col] + '_' + self.date
                np.savez_compressed(self.base_file_path + filename,
                                    save_all_data_flag=self.save_all_data_flag,
                                    number_averages_phase1=self.number_averages_phase1,
                                    number_averages_phase2=self.number_averages_phase2,
                                    number_averages_phase3=self.number_averages_phase3,
                                    number_columns=self.number_columns,
                                    number_rows=self.number_rows,
                                    # bad16_card_rs_addr=self.bad16_card_rs_addr,
                                    # bad16_card_in_addr=self.bad16_card_in_addr,
                                    # tower_sq1_bias_addr=self.tower_sq1_bias_addr,
                                    triangle_period=self.triangle_period,
                                    icmin_sigma=self.icmin_sigma,
                                    number_steps=self.number_steps,
                                    change_pause_s=self.change_pause_s,
                                    phase1_sq1_bias=self.phase1_sq1_bias,

                                    base_file_path=self.base_file_path,
                                    filename=filename,
                                    date=self.date,

                                    baselines_std=self.baselines_std[col],
                                    icmin=self.icmin[col],
                                    icmax=self.icmax[col],
                                    icmod_max=self.icmod_max[col],
                                    iccol_min=self.iccol_min[col],
                                    iccol_max=self.iccol_max[col],

                                    sq1_row_bias_phase1=self.sq1_row_bias_phase1[col],
                                    sq1_row_bias_phase2=self.sq1_row_bias_phase2,
                                    sq1_col_bias_phase3=self.sq1_col_bias_phase3[col],
                                    sq1_row_select_m=self.sq1_row_select_m[col],
                                    sq1_row_input_m=self.sq1_row_input_m[col],
                                    sq1_row_fb_m=self.sq1_row_fb_m[col],

                                    row_sweep_tower_values=self.row_sweep_tower_values,
                                    row_sweep_average_mod=self.row_sweep_average_mod[col],
                                    row_sweep_average_min=self.row_sweep_average_min[col],
                                    row_sweep_average_max=self.row_sweep_average_max[col],
                                    row_average_mod_sweep=self.row_average_mod_sweep[col],
                                    # feedback_M_data=self.feedback_M_data,
                                    # feedback_M_data_folded=self.feedback_M_data_folded,
                                    # feedback_M_data_smoothed=self.feedback_M_data_smoothed,
                                    # input_M_data=self.input_M_data,
                                    # input_M_data_folded=self.input_M_data_folded,
                                    # input_M_data_smoothed=self.input_M_data_smoothed,
                                    # rowsel_M_data=self.rowsel_M_data,
                                    # rowsel_M_data_folded=self.rowsel_M_data_folded,
                                    # rowsel_M_data_smoothed=self.rowsel_M_data_smoothed,
                                    chip_id=self.chip_id[col],
                                    qa_name=self.qa_name,
                                    note=self.note[col])

    '''Testing connections:
    DACA: SQ1 Feed Back
    DACb: SSA Feed Back
    '''
    '''Phase 0: Setup and lock on the SQUID series array for further measurements of the connected multiplexers.'''
    # Done by hand for now
    def phase1_0(self, column, iccol_min, iccol_max):
        '''The user need to disconnect all of the inputs and row selects from the probe tower and then sweep the SQ1 bias
        to locate the ic_col_min and max'''
        self.iccol_min[column] = iccol_min
        self.iccol_max[column] = iccol_max

    def phase1_1(self):
        '''Phase 1: Set tower bias to a default value and then sweep the row selects with triangles. This is used to
        determine the optimal flux for the row selection.  Configure Triangle for a full scale triangle. Ensure that
        row11 is set to stream the triangle to the server instead of feedback data.'''
        self.zero_everything()
        self.daq.averages = self.number_averages_phase1
        # Set  tower voltage
        for col in range(self.number_columns):
            self.tower_set_voltage(int(self.which_columns[col]), int(self.phase1_sq1_bias))
        # Set Dc Triangles on Row selects except for RS11 which has no row select.
        for row in range(self.number_rows-1):
            self.row_decode_rs(row)
            self.badchan_rs.set_d2a_hi_value(0)
            self.badchan_rs.set_d2a_lo_value(0)
            self.badchan_rs.set_tri(True)
            self.badchan_rs.set_dc(False)

        '''Take data and roll the data to align to the triangle. Then fold the data over so you are only looking at a
        flux ramp through the SQUID Row select.'''
        fb, err = self.daq.take_average_data()
        roll_number = fb[0, 11, :].argmin()
        for col in range(self.number_columns):
            for row in range(self.number_rows):
                fb[col, row] = np.roll(fb[col, row, :], -roll_number)
            # For now store the raw data

        self.rowsel_M_data = fb
        # Fold data over
        for col in range(self.number_columns):
            for row in range(self.number_rows):
                for i in range(self.triangle_period/2):
                    self.rowsel_M_data_folded[col, row, i] = (fb[col, row, i] +
                                                              fb[col, row, self.triangle_period - i - 1])/2
        '''Extract the optimal row select bias point for each row. And if enough zero crossings in the data extract the
        row select uA/Phi_not dac units'''
        for col in range(self.number_columns):
            for row in range(self.number_rows-1):
                # Old method that would set to the second phi_not of the rowselect.
                # temp = np.argmax(fb[col, row, :self.triangle_period/4])
                # self.sq1_row_bias_phase1[col, row] = fb[col, 11, temp]

                # Smooth the data using built in python methods
                self.rowsel_M_data_smoothed[col, row] = lowess(self.rowsel_M_data_folded[col, row,
                                                               :self.triangle_period/2],
                                                               np.linspace(0, self.triangle_period/2-1,
                                                                           self.triangle_period/2),
                                                               is_sorted=True, frac=0.025, it=0)[:, 1]
                # subtract the mean and determine zero crossings
                zero_crossings = np.where(np.diff(np.signbit(self.rowsel_M_data_smoothed[col, row] -
                                                             (np.min(self.rowsel_M_data_smoothed[col, row]) +
                                                              0.8*(np.max(self.rowsel_M_data_smoothed[col, row]) -
                                                                   np.min(self.rowsel_M_data_smoothed[col, row]))))))[0]
                # zero_crossings = np.where(np.diff(np.signbit(board.rowsel_M_data_smoothed[col, row] -
                #                                              (np.min(board.rowsel_M_data_smoothed[col, row]) +
                #                                               0.8 * (np.max(board.rowsel_M_data_smoothed[col, row]) -
                #                                                      np.min(board.rowsel_M_data_smoothed[col, row]))))))[0]
                # now based off of the first zero crossing pair, determine the max in between the two points.
                if len(zero_crossings) > 2:
                    temp = np.argmax(self.rowsel_M_data_smoothed[col, row, zero_crossings[0]:zero_crossings[1]]) + \
                           zero_crossings[0]
                    self.sq1_row_bias_phase1[col, row] = fb[col, 11, temp]
                else:
                    self.sq1_row_bias_phase1[col, row] = 0  # Broken row select

                if len(zero_crossings) >= 4:
                    self.sq1_row_select_m[col, row, 0] = self.sq1_row_bias_phase1[col, row]
                    temp = np.argmax(self.rowsel_M_data_smoothed[col, row, zero_crossings[2]:zero_crossings[3]]) + \
                           zero_crossings[2]
                    self.sq1_row_select_m[col, row, 1] = fb[col, 11, temp]
                else:
                    pass

    def phase2(self):
        '''Phase 2: Using the values determined from Phase 1, we will now have a triangle on SQ1 FB and sweep tower
        bias. Set row12 to stream the feedback data to determine the Ic_min_column.'''
        # Determine and set the row select bias points for each row.
        # Row bias is picked by taking the highest optimal bias across the columns if doing multiple columns.
        self.zero_everything()
        self.daq.averages = self.number_averages_phase2
        for row in range(self.number_rows-1):
            self.sq1_row_bias_phase2[row] = np.max(self.sq1_row_bias_phase1[:, row])
            self.row_decode_rs(row)
            self.badchan_rs.set_d2a_hi_value(int(self.sq1_row_bias_phase2[row]))
            self.badchan_rs.set_tri(False)
            self.badchan_rs.set_dc(False)
            self.badchan_rs.set_lohi(True)

        self.get_baselines()
        self.sq1_bias_sweeper()
        self.calculate_ics()

    def phase3_0(self):
        '''Phase3 part 0: Now take data to measure the feedback1 coupling to the squids. Set a triangle on the Feedback
        that sweeps out at least two phi_not.'''
        self.zero_everything()
        self.daq.averages = self.number_averages_phase3

        for row in range(self.number_rows-1):
            self.row_decode_rs(row)
            self.badchan_rs.set_d2a_hi_value(int(self.sq1_row_bias_phase2[row]))
            self.badchan_rs.set_tri(False)
            self.badchan_rs.set_dc(False)
            self.badchan_rs.set_lohi(True)
            self.badchan_rs.send_wreg5()
            self.badchan_rs.send_wreg4()
            self.badchan_rs.send_wreg2()

        #  Set the SQ1 Tower bias to the maximum need for the column.
        for col in range(self.number_columns):
            self.sq1_col_bias_phase3[col] = int(np.max(self.icmod_max[col, :]))
            self.tower_set_voltage(int(self.which_columns[col]), int(self.sq1_col_bias_phase3[col]))

        fb, err = self.daq.take_average_data()
        roll_number = fb[0, 11, :].argmin()
        for col in range(self.number_columns):
            for row in range(self.number_rows):
                fb[col, row] = np.roll(fb[col, row, :], -roll_number)

        self.feedback_M_data = fb

        for col in range(self.number_columns):
            for row in range(self.number_columns):
                for i in range(self.triangle_period/2):
                    self.feedback_M_data_folded[col, row, i] = (fb[col, row, i] + fb[col, row, self.triangle_period - i - 1])/2

        for col in range(self.number_columns):
            for row in range(self.number_rows - 1):
                # Smooth the data using built in python methods
                self.feedback_M_data_smoothed[col, row] = lowess(self.feedback_M_data_folded[col, row,
                                                                 :self.triangle_period / 2],
                                                                 np.linspace(0, self.triangle_period / 2 - 1,
                                                                             self.triangle_period / 2),
                                                                 is_sorted=True, frac=0.025, it=0)[:, 1]
                # subtract the mean and determine zero crossings
                zero_crossings = np.where(np.diff(np.signbit(self.feedback_M_data_smoothed[col, row] -
                                                             np.mean(self.feedback_M_data_smoothed[col, row]))))[0]
                # now based off of the first zero crossing pair, determine the max in between the two points.

                if len(zero_crossings) >= 4:
                    temp = np.argmax(fb[col, row, zero_crossings[0]:zero_crossings[1]])
                    self.sq1_row_fb_m[col, row, 0] = fb[col, 11, temp]
                    temp = np.argmax(fb[col, row, zero_crossings[2]:zero_crossings[3]])
                    self.sq1_row_fb_m[col, row, 1] = fb[col, 11, temp]
                else:
                    print("Did not find enough phi_nots swept out on row: " + str(row))

    def phase3_1(self):
        '''Phase3 part 1: Now take data to measure the input coupling to the squids.
        Set very low amplitude triangles on the inputs.'''
        self.zero_everything()
        self.daq.averages = self.number_averages_phase3

        for row in range(self.number_rows-1):
            self.row_decode_rs(row)
            self.badchan_rs.set_d2a_hi_value(int(self.sq1_row_bias_phase2[row]))
            self.badchan_rs.set_tri(False)
            self.badchan_rs.set_dc(False)
            self.badchan_rs.set_lohi(True)
            self.badchan_rs.send_wreg5()
            self.badchan_rs.send_wreg4()
            self.badchan_rs.send_wreg2()
            self.row_decode_in(row)
            self.badchan_in.set_d2a_hi_value(0)
            self.badchan_in.set_d2a_lo_value(0)
            self.badchan_in.set_tri(True)
            self.badchan_in.set_dc(True)
            self.badchan_in.set_lohi(False)
            self.badchan_in.send_wreg5()
            self.badchan_in.send_wreg4()
            self.badchan_in.send_wreg2()

        for col in range(self.number_columns):
            self.sq1_col_bias_phase3[col] = int(np.max(self.icmod_max[col, :]))
            self.tower_set_voltage(int(self.which_columns[col]), int(self.sq1_col_bias_phase3[col]))

        fb, err = self.daq.take_average_data()
        roll_number = fb[0, 11, :].argmin()
        for col in range(self.number_columns):
            for row in range(self.number_rows):
                fb[col, row] = np.roll(fb[col, row, :], -roll_number)

        self.input_M_data = fb
        for col in range(self.number_columns):
            for row in range(self.number_columns):
                for i in range(self.triangle_period/2):
                    self.input_M_data_folded[col, row, i] = (fb[col, row, i] + fb[col, row, self.triangle_period - i - 1])/2

        for col in range(self.number_columns):
            for row in range(self.number_rows - 1):
                # Smooth the data using built in python methods
                self.input_M_data_smoothed[col, row] = lowess(self.input_M_data_folded[col, row,
                                                              :self.triangle_period / 2],
                                                              np.linspace(0, self.triangle_period / 2 - 1,
                                                                          self.triangle_period / 2),
                                                              is_sorted=True, frac=0.025, it=0)[:, 1]
                # subtract the mean and determine zero crossings
                zero_crossings = np.where(np.diff(np.signbit(self.input_M_data_smoothed[col, row] -
                                                             np.mean(self.input_M_data_smoothed[col, row]))))[0]
                zero_crossings = np.where(np.diff(np.signbit(fb[col, row] -
                                                             np.mean(fb[col, row]))))[0]
                # Now
                if len(zero_crossings) >= 4:
                    temp = np.argmax(self.input_M_data_smoothed[col, row, zero_crossings[0]:zero_crossings[1]]) + \
                           zero_crossings[0]
                    self.sq1_row_input_m[col, row, 0] = fb[col, 11, temp]
                    temp = np.argmax(self.input_M_data_smoothed[col, row, zero_crossings[2]:zero_crossings[3]]) + \
                           zero_crossings[2]
                    self.sq1_row_input_m[col, row, 1] = fb[col, 11, temp]
                else:
                    print("Did not find enough phi_nots swept out on row: " + str(row))

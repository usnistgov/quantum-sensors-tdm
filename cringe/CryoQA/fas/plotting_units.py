import sys
import argparse
import IPython
import glob

import numpy as np
import pylab as pl

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Take data from ROACH2, demodulate it, and write the demodulated data '
                                                 'to a file. Can also display various plots.')
    parser.add_argument('list_of_files',
                        type=str,
                        help='A list of paths to some files that will be parsed and plotted')
    parser.add_argument('-i',
                        dest='interactive',
                        action='store_true',
                        help='Enter interactive mode at end of script')
    parser.add_argument('-w',
                        dest='wafer',
                        action='store_true',
                        help='You just want to look at IC_max of all files on the same plot. Overides individule chip '
                             'plots')
    parser.add_argument('-f',
                        dest='fas_summary',
                        action='store_true',
                        help='Makes a plot summarizing the fas switch fb max and min vs sq1 tower bias value for each '
                             'file')
    args = parser.parse_args()

    if args.interactive:
        pl.ion()

    Towerfs = 2**16  # DAC Units
    Towerref = 2.5  # Volts
    DACfs = 2**14  # Dac units
    DACref = 1  # Volts
    R_tower = 5.1e3  # Tower Bias Card resistor
    R_rs = 1.00e3  # HDMI tower resistor Ohms, row select
    R_in = 20.00e3  # HDMI tower resistor Ohms, Input
    R_fb = 5.1e3  # Tower bias card feed back resistor
    scale = 1e6  # Scale to microamps
    factor_rs = DACref*scale/(DACfs*R_rs)
    factor_in = DACref*scale/(DACfs*R_in)
    factor_fb = DACref*scale/(DACfs*R_fb)
    factor_sq1b = Towerref*scale/(Towerfs*R_fb)

    #print 'args.list_of_files'
    #print args.list_of_files
    fnames = glob.glob(str(args.list_of_files))
    fnames.sort()
    print('fnames sorted')
    print(fnames)

    ic_maxs = np.zeros((len(fnames), 33), 'int')
    data_sets = [np.load(f) for f in fnames]

    cmap = pl.get_cmap('jet')
    colors_col = cmap(np.linspace(0, 1.0, len(fnames)))
    colors_row = cmap(np.linspace(0, 1.0, data_sets[0]['row_sweep_average_mod'].shape[0]*2))

    if not args.wafer:
        for i, file_path in enumerate(fnames):

            pl.figure()
            last_part_fname = fnames[i].rsplit('/', 1)[-1]
            '''Make a plot to show the Ic Sweeps for each row overlayed'''
            for j in range(data_sets[i]['row_sweep_average_mod'].shape[0]):
                pl.title('Column Bias Sweep :' + last_part_fname)
                pl.plot(np.linspace(0, 20000-100, 200), data_sets[i]['row_sweep_average_mod'][j, :],
                        label=('Row:' + str(j)),
                        color=colors_row[j])
                pl.xlabel('SQ1 Tower DAC Value')
                pl.ylabel('SQ1 Modulation Depth [FB DAC Units]')
                pl.legend()
            '''Make a plot showing IC_min and IC_max overlayed for each file'''
            pl.figure()
            pl.title('Icmin/mod_max Tower Bias :' + last_part_fname)
            pl.plot(data_sets[i]['icmin'], 'r', label='Ic_min')
            pl.plot(data_sets[i]['icmod_max'], 'b', label='Ic_max')
            pl.xlabel('Row Number')
            pl.ylabel('Tower SQ1 Bias value')
            pl.legend()

            pl.figure()
            pl.title('Icmin/mod_max Tower Bias :' + last_part_fname)
            pl.plot(data_sets[i]['icmin']*factor_sq1b, 'r', label='Ic_min')
            pl.plot(data_sets[i]['icmod_max']*factor_sq1b, 'b', label='Ic_max')
            pl.xlabel('Row Number')
            pl.ylabel('Tower SQ1 Bias value [uA]')
            pl.legend()

    else:
        pl.figure()
        for i in range(len(fnames)):
            ic_maxs[i] = data_sets[i]['icmax']
            pl.plot(ic_maxs[i]*factor_fb, '-*', label=str(data_sets[i]['chip_id']), color=colors_col[i])

        pl.legend()
        pl.xlabel('Row Number')
        pl.ylabel('DAC Bias Current')

    if args.fas_summary:
        for i, file_path in enumerate(fnames):

            pl.figure()
            last_part_fname = fnames[i].rsplit('/', 1)[-1]
            '''Make a plot to show the Ic Sweeps for each row overlayed'''
            x = data_sets[i]['row_sweep_tower_values'] * factor_fb
            #x = np.linspace(0, 20000 - 100, 200) * factor_sq1b
            for j in range(data_sets[i]['row_sweep_average_mod'].shape[0]):
                pl.title('Row Bias Sweep :' + last_part_fname)
                # y = data_sets[i]['row_sweep_average_max'][j, 0] - data_sets[i]['row_sweep_average_max'][j, :] - \
                #     data_sets[i]['row_sweep_average_mod'][j, :]
                y = data_sets[i]['row_sweep_average_min'][j, 0] - data_sets[i]['row_sweep_average_min'][j, :]
                y = y * factor_fb
                pl.plot(x, y, label=('Min_Row:' + str(j)), color=colors_row[j])

            for j in range(data_sets[i]['row_sweep_average_mod'].shape[0]):
                y = data_sets[i]['row_sweep_average_max'][j, 0] - data_sets[i]['row_sweep_average_max'][j, :]
                y = y * factor_fb
                pl.plot(x, y, label=('Max_Row:' + str(j)), color=colors_row[j+data_sets[i]['row_sweep_average_mod'].shape[0]])
                pl.xlabel('SQ1 Tower DAC Value [uA]')
                pl.ylabel('SSA Fb value [uA]')
                pl.legend()

            pl.plot([data_sets[i]['iccol_min']*factor_fb, data_sets[i]['iccol_min']*factor_fb],
                    [0, data_sets[i]['row_sweep_average_max'][11, 199]*factor_fb])
            # pl.plot([8000*factor_fb, 8000*factor_fb],
            #         [0, data_sets[i]['row_sweep_average_max'][11, 199]*factor_fb])

            for j in range(data_sets[i]['row_sweep_average_mod'].shape[0]):
                for row in range(12):
                    Iin_min = (data_sets[i]['row_sweep_average_max'][row, 0] - data_sets[i]['row_sweep_average_max'][row]) * factor_fb/2.4
                    Iin_max = (data_sets[i]['row_sweep_average_min'][row, 0] - data_sets[i]['row_sweep_average_min'][row]) * factor_fb/2.4
                    It_bias = data_sets[i]['row_sweep_tower_values'] * factor_sq1b
                    Rdmax = (It_bias/Iin_max-1) * 1
                    Rdmin = (It_bias/Iin_min-1) * 1
                    pl.plot(Iin_max, Rdmax, label=('Max_Row:' + str(row)), color=colors_row[row])
                    pl.plot(Iin_min, Rdmin, label=('Max_Row:' + str(row)), color=colors_row[row+12])




    if args.interactive:
        IPython.embed()
    else:
        pl.show()
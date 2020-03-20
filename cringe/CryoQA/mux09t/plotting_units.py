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
                        help='You just want to look at IC_max of all files on the same plot')
    args = parser.parse_args()
    if args.interactive:
        pl.ion()

    DACfs = 2**14  # Dac units
    DACref = 1  # Volts
    Rbias = 2.1e3  # HDMI tower resistor Ohms
    scale = 1e6  # Scale to microamps
    factor = DACref*scale/(DACfs*Rbias)


    print('args.list_of_files')
    print(args.list_of_files)
    fnames = glob.glob(str(args.list_of_files))
    fnames.sort()
    print('fnames sorted')
    print(fnames)

    ic_maxs = np.zeros((len(fnames), 33), 'int')
    data_sets = [np.load(f) for f in fnames]

    cmap = pl.get_cmap('jet')
    colors_col = cmap(np.linspace(0, 1.0, len(fnames)))
    colors_row = cmap(np.linspace(0, 1.0, data_sets[0]['row_average_adcpp_sweep'].shape[0]))

    if not args.wafer:
        for i, file_path in enumerate(fnames):

            pl.figure()
            last_part_fname = fnames[i].rsplit('/', 1)[-1]
            '''Make a plot to show the Ic Sweeps for each row overlayed'''
            for j in range(data_sets[i]['row_average_adcpp_sweep'].shape[0]):
                pl.title('Row Bias Sweep :' + last_part_fname)
                pl.plot(data_sets[i]['row_sweep_dac_values'], data_sets[i]['row_average_adcpp_sweep'][j, :],
                        label=('Row:' + str(j)),
                        color=colors_row[j])
                pl.xlabel('BAD16 DAC Value')
                pl.ylabel('SQ1 Modulation Depth [arbs]')
                pl.legend()
            '''Make a plot showing IC_min and IC_max overlayed for each file'''
            pl.figure()
            pl.title('Icmin/max :' + last_part_fname)
            pl.plot(data_sets[i]['icmin'], 'r', label='Ic_min')
            pl.plot(data_sets[i]['icmax'], 'b', label='Ic_max')
            pl.xlabel('Row Number')
            pl.ylabel('BAD16 Bias value')
            pl.legend()

    else:
        pl.figure()
        for i in range(len(fnames)):
            ic_maxs[i] = data_sets[i]['icmax']
            pl.plot(ic_maxs[i]*factor, '-*', label=str(data_sets[i]['chip_id']), color=colors_col[i])

        pl.legend()
        pl.xlabel('Row Number')
        pl.ylabel('DAC Bias Current')

    if args.interactive:
        IPython.embed()
    else:
        pl.show()


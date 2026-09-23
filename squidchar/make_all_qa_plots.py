from squidchar import analysis
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import decimate
import argparse
from progress.bar import IncrementalBar
import yaml

def make_all_plots_pdf(
    sq1_bias_ramp_file, 
    input_ramp_file, 
    outfile, 
    rshunt, 
    silly=False,
    rcw_arr_override=None,
    chip_comments_override=None
):
    np.seterr(divide='ignore', invalid='ignore')
    # col=5
    # chip=1
    # First do all one time calculations
    # Load in file as microamps
    data_ua, bias_i, tri, z = analysis.load_data_and_convert(sq1_bias_ramp_file)
    # overview_plot(data_ua, bias_i, tri, 1, 7) # Overview plot is good for interactive use but not so much for the final product
    amplitude, ic_idx, icmax = analysis.calculate_icmax(bias_i, data_ua)
    shunt_uv = analysis.calculate_device_voltage(data_ua, bias_i, rshunt)

    min_i = np.min(data_ua,axis=3)
    max_i = np.max(data_ua,axis=3)

    bias_tiled_2 = np.broadcast_to(bias_i,(8,24,bias_i.shape[0])).transpose([2,0,1]) # again, bias is the same for all columns and rows.

    # Calculate device resistance at the maximum and minimum points of the modulation
    rd_at_iin_max = (bias_tiled_2/max_i - 1) * rshunt
    rd_at_iin_min = (bias_tiled_2/min_i - 1) * rshunt
    vd_at_iin_max = (bias_tiled_2 - max_i) * rshunt
    vd_at_iin_min = (bias_tiled_2 - min_i) * rshunt

    # Calculate device dynamic resistance at same
    rdyn_at_iin_max = np.gradient(vd_at_iin_max, axis=0) / np.gradient(max_i, axis=0)
    rdyn_at_iin_min = np.gradient(vd_at_iin_min, axis=0) / np.gradient(min_i, axis=0)
    
    ms = min_i.shape
    
    # Calculate dynamic resistance everywhere for the oval plot
    min_i_tiled = np.broadcast_to(min_i,(8192,ms[0],ms[1],ms[2])).transpose([1,2,3,0])
    max_i_tiled = np.broadcast_to(max_i,(8192,ms[0],ms[1],ms[2])).transpose([1,2,3,0])
    rdyn = np.gradient(shunt_uv,axis=0) / np.gradient(data_ua,axis=0)

    data_normalized = (data_ua - min_i_tiled)/(max_i_tiled-min_i_tiled)
    actual_chip_rcw = rcw_arr_override or z["chip_rcw"]
    actual_chip_notes = chip_comments_override or z["chip_notes"]
    pdfpages_per_chip = {}
    nplots = len(actual_chip_rcw)*len(actual_chip_rcw[0])*9 # cols * chips * number of for loops below
    bar = IncrementalBar(
        "Making lots of plots:", 
        max=nplots,
        suffix=' [%(index)d/%(max)d]'
    )
    for chip, col_arr in enumerate(actual_chip_rcw):
        for col, rcw in enumerate(col_arr):
            pdfpages_per_chip[(chip,col)] = PdfPages(outfile.format(f"chip_rcw_{rcw[0]}_{rcw[1]}_{rcw[2]}"))
    cp = analysis.PlotsWithSameColors()

    with PdfPages(outfile.format("current_modulation")) as pdf:
        for chip, array_columns in enumerate(actual_chip_rcw):
            for col, rcw_arr in enumerate(array_columns):
                cp.current_modulation_plot(amplitude, icmax, bias_i, col, chip)
                notes = actual_chip_notes[chip][col]
                plt.title(f"Modulation amplitude for r,c,w={rcw_arr}\n{notes}")
                pdf.savefig()
                pdfpages_per_chip[(chip,col)].savefig()
                plt.close()
                bar.next()
                
    with PdfPages(outfile.format("i_i")) as pdf:
        for chip, array_columns in enumerate(actual_chip_rcw):
            for col, rcw_arr in enumerate(array_columns):
                cp.i_i_plot(data_ua, bias_i, col, chip)
                notes = actual_chip_notes[chip][col]
                plt.title(f"current-current for r,c,w={rcw_arr}\n{notes}")
                pdf.savefig()
                pdfpages_per_chip[(chip,col)].savefig()
                plt.close()
                bar.next()
                
    with PdfPages(outfile.format("i_v")) as pdf:
        for chip, array_columns in enumerate(actual_chip_rcw):
            for col, rcw_arr in enumerate(array_columns):
                cp.i_v_plot(data_ua, shunt_uv, col, chip)
                notes = actual_chip_notes[chip][col]
                plt.title(f"I-V Curve for r,c,w={rcw_arr}\n{notes}")
                pdf.savefig()
                pdfpages_per_chip[(chip,col)].savefig()
                plt.close()
                bar.next()
                
    with PdfPages(outfile.format("R")) as pdf:
        for chip, array_columns in enumerate(actual_chip_rcw):
            for col, rcw_arr in enumerate(array_columns):
                cp.device_resistance_plot(min_i, max_i, rd_at_iin_min, rd_at_iin_max, col, chip)
                notes = actual_chip_notes[chip][col]
                plt.title(f"Device resistance for r,c,w={rcw_arr}\n{notes}")
                pdf.savefig()
                pdfpages_per_chip[(chip,col)].savefig()
                plt.close()
                bar.next()
                
    with PdfPages(outfile.format("R_dyn")) as pdf:
        for chip, array_columns in enumerate(actual_chip_rcw):
            for col, rcw_arr in enumerate(array_columns):
                cp.device_dynamic_resistance_plot(min_i, max_i, rdyn_at_iin_min, rdyn_at_iin_max, col, chip)
                notes = actual_chip_notes[chip][col]
                plt.title(f"Dynamic resistance for r,c,w={rcw_arr}\n{notes}")
                pdf.savefig()
                pdfpages_per_chip[(chip,col)].savefig()
                plt.close()
                bar.next()
                
    rdyn[:,:,0,:] = np.nan # will need to edit this if row 0 gets fixed!

    with PdfPages(outfile.format("R_dyn_oval")) as pdf:
        for chip, array_columns in enumerate(actual_chip_rcw):
            for col, rcw_arr in enumerate(array_columns):
                cp.rdyn_oval_plot(data_normalized, rdyn, ic_idx, col,chip,silly=silly)
                notes = actual_chip_notes[chip][col]
                plt.title(f"Dynamic resistance scatter for r,c,w={rcw_arr}\n{notes}")
                plt.xlim(0,1)
                plt.ylim(4,19)
                if silly:
                    # The oval scatter plots look a lot like mouths, so optionally draw googly eyes on them
                    plt.savefig(outfile.format(f"silly_oval_{chip}_{col}").replace("pdf","png"))
                pdf.savefig()
                pdfpages_per_chip[(chip,col)].savefig()
                plt.close()
                bar.next()

    tri_i, d_i, z_i = analysis.load_input_ramp(input_ramp_file)

    decimations=8
    tri_i_filtered = decimate(tri_i, decimations)
    d_i_filtered = decimate(d_i,decimations,axis=-1)
    gain = np.gradient(d_i_filtered, axis=-1) / np.gradient(tri_i_filtered)
    norm_i = analysis.normalize_current_for_ovals(d_i_filtered)
    
    with PdfPages(outfile.format("input_curve")) as pdf:
        for chip, array_columns in enumerate(actual_chip_rcw):
            for col, rcw_arr in enumerate(array_columns):
                cp.squid_curve_input_plot(tri_i, d_i, col, chip)
                notes = actual_chip_notes[chip][col]
                plt.title(f"Device current vs input for r,c,w={rcw_arr}\n{notes}")
                pdf.savefig()
                pdfpages_per_chip[(chip,col)].savefig()
                plt.close()
                bar.next()
                
    with PdfPages(outfile.format("gain_curve")) as pdf:
        for chip, array_columns in enumerate(actual_chip_rcw):
            for col, rcw_arr in enumerate(array_columns):
                cp.squid_gain_plot(tri_i_filtered, gain, col, chip)
                notes = actual_chip_notes[chip][col]
                plt.title(f"Device gain for r,c,w={rcw_arr}\n{notes}")
                pdf.savefig()
                pdfpages_per_chip[(chip,col)].savefig()
                plt.close()
                bar.next()

    with PdfPages(outfile.format("gain_oval")) as pdf:
        for chip, array_columns in enumerate(actual_chip_rcw):
            for col, rcw_arr in enumerate(array_columns):
                cp.gain_oval_plot(norm_i, gain, col, chip, silly=silly)
                notes = actual_chip_notes[chip][col]
                plt.title(f"Gain vs. Normalized Current for r,c,w={rcw_arr}\n{notes}")
                plt.xlim(0,1)
                plt.ylim(-8,19)
                pdf.savefig()
                pdfpages_per_chip[(chip,col)].savefig()
                plt.close()
                bar.next()
                
    for key in pdfpages_per_chip.keys():
        pdfpages_per_chip[key].close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze squid QA data and make lots of plots!")
    parser.add_argument("sq1_bias_ramp", type=str, help="Squid 1 bias ramp npz file. Should have been taken with row selects properly biased.")
    parser.add_argument("input_ramp", type=str, help="Input ramp npz file.")
    parser.add_argument("outfile_prefix", type=str, help="String appended in front of all output files")
    parser.add_argument("-o", "--chip-info-override", type=str, help="path to optional yaml file containing actual chip serial numbers if you did something wrong during experiment setup")
    parser.add_argument("--silly", action="store_true", help=":D")
    args=parser.parse_args()
    if args.chip_info_override:
        with open(args.chip_info_override, 'r') as yamlfile:
            overrides = yaml.load(yamlfile, Loader=yaml.FullLoader)
        rcw_override = overrides["rcw_arr"]
        comments_override = overrides["comments"]
    else:
        rcw_override = None
        comments_override=None

    make_all_plots_pdf(
        args.sq1_bias_ramp,
        args.input_ramp, 
        f"{args.outfile_prefix}_{{}}.pdf",
        1,
        silly=args.silly,
        rcw_arr_override=rcw_override,
        chip_comments_override=comments_override,

        )
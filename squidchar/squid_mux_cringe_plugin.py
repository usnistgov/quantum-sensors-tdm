"""
Run this script from within the ipython window produced by ``cringe -i``
use ``%run -i this_script.py``

That's the easiest way to do it, since the whole reason
we have ipython is to get access to the ``win`` and ``app`` variables
for the cringe UI.

If you import this script, the only way to access them is to 
pass them as arguments to each and every function call. Something
similar happens with ``%run`` without the ``-i`` flag.

The screening procedure is described in greater detail in an accompanying google doc/PDF
https://docs.google.com/document/d/1ElPe3wK1m9XYn6voxOwFVISBVACSl6Jx8O_w5DDzlyM/edit?usp=sharing

"""

import yaml
from squid_ssa_char.modules import daq as daqlib
import numpy as np
import time
import cringe
from os import path
from pathlib import Path
from datetime import datetime
from progress.bar import IncrementalBar
from squidchar import analysis
from glob import glob

#import line_profiler
# import tracemalloc

 #########################################################
# Helper functions for interacting with CRINGE            #
#                                                         #
# Usually involve several very long lines to do something #
# that seems simple                                       #
 #########################################################
def channel_process(channel):
    if channel == "A":
        a=1 
        b=0
    elif channel == "B":
        a=0
        b=1
    elif channel == "off":
        a=0
        b=0
    else:
        raise RuntimeError("Invalid channel")
    return a,b

def feedback_lock_all(channel):
    """
    Enable feedback lock on all rows of all DFB cards, either A or B as defined by param "channel"
    the dfb/clock card tab must be configured to broadcast and all the other
    cards must be configured to echo.

    :param channel: 'A' or 'B' corresponding to A or B output on the DFBx2
    :type channel: str
    """
    global win, app
    a,b = channel_process(channel)
    win.crate_widgets[0].dfbclk_widget1.master_vector.FBA_button.setChecked(a)
    win.crate_widgets[0].dfbclk_widget1.master_vector.FBB_button.setChecked(b)
    app.processEvents()
    time.sleep(1.0)

def set_triangle(log2_dwell=2, log2_num_steps=10, step_size=6):
    """
    The parameters here are the same as the numbers you'd find in the spin boxes in CRINGE under
    triangle setup.

    :param log2_dwell: log2(number of clock cycles to wait per step)
    :type log2_dwell: int
    :param log2_num_steps: log2(number of steps per ramp)
    :type log2_num_steps: int
    :param step_size: Number of DAC units to step by
    :param step_size: int

    :return: triangle wave period, in clock cycles.
    :rtype: int
    """
    global win, app
    win.dwell.setValue(log2_dwell)
    win.range.setValue(log2_num_steps)
    win.step.setValue(step_size)
    win.tri_idx_button.setChecked(1)
    app.processEvents()
    time.sleep(1.0)
    return(2**log2_dwell * 2**log2_num_steps * 2)

def set_send_mode(channel):
    """
    Based on the "channel" parameter, configure all rows of all DFB cards to send
        A: FBA/err
        B: FBB/err
        AB: FBA/FBB

    As with other functions, this function depends on broadcast and echo having been configured properly.

    :param channel: "A" "B" or "AB", determines what the DFB cards will send
    :type channel: str
    """
    global win, app
    if channel == "A":
        win.crate_widgets[0].dfbclk_widget1.master_vector.data_packet.setCurrentIndex(0)
    elif channel == "B":
        win.crate_widgets[0].dfbclk_widget1.master_vector.data_packet.setCurrentIndex(1)
    elif channel == "AB":
        win.crate_widgets[0].dfbclk_widget1.master_vector.data_packet.setCurrentIndex(2)
    app.processEvents()
    time.sleep(1.0)

def enable_triangle_all(channel):
    """
    Configure all rows of all DFB cards to output the triangle on either A or B
    as set by the "channel" param.
    As with other functions, broadcast and echo must be configured properly.

    :param channel: "A" or "B": where to enable the triangle on all DFBx2 cards.
    :type channel: str
    """
    global win, app
    a,b=channel_process(channel)
    win.crate_widgets[0].dfbclk_widget1.master_vector.TriA_button.setChecked(a)
    win.crate_widgets[0].dfbclk_widget1.master_vector.TriB_button.setChecked(b)
    app.processEvents()
    time.sleep(1.0)

def get_row_vector(widget_index, row_index):
    """
    Get a 'row vector' from a given BAD16 card.
        Widget_index is the tab number (0 indexed) in cringe
        row index is the row number.

    The vector returned represents one row from that cringe tab,
    with options such as DC, LO/HI, Triangle, low state dac output, high state dac ouptut, etc
    see also: cringe/BADASS/badchn_builder.py
    """
    global win
    if row_index=="all":
        vec = win.crate_widgets[widget_index].badrap_widget1.master_vector
    else:
        vec = win.crate_widgets[widget_index].badrap_widget1.chn_vectors[row_index]
    return vec

def set_rs_params(widget_index, row_index='all', dc=0, lohi=0, tri=0):
    """
    For a given row from a given cringe widget, (or all the rows), 
    set the state of the DC, LOW/HI, and TRI buttons.
    By default all will be set to 0 so that only the buttons passed in will
    be set to 1.

    Usually only one button will be set at a time, but it can be useful
    to set DC and TRI in order to put a continuous triangle wave on the row
    selects instead of a 'chopped' triangle wave that is zero when the RS is inactive
    in the 'states' menu.
    """
    global win, app
    vec = get_row_vector(widget_index, row_index)
    vec.dc_button.setChecked(dc)
    vec.LoHi_button.setChecked(lohi)
    vec.Tri_button.setChecked(tri)
    app.processEvents()

def set_rs_dacs(widget_index, dac_low, dac_high, row_index="all"):
    """
    For a given row from a given cringe widget (zero indexed tab number)
    set the low and high state dac output levels. 
    """
    global win, app
    vec = get_row_vector(widget_index, row_index)
    vec.d2a_lo_spin.setValue(dac_low)
    vec.d2a_hi_spin.setValue(dac_high)
    app.processEvents()
    time.sleep(1)

def enable_triangle_all_1_row(channel, row):
    """ Designed to configure a row to echo the triangle onto
    a feedback line for recording into the eventual data product.
    As such, this not only enables the triangle, but also ensures that
    the PID on that line is disabled and the send mode is configured
    to return the triangle signal and the dac offset is 0.
    """
    global win, app
    a,b = channel_process(channel)
    for w in win.crate_widgets:
        if type(w) is cringe.DFBx2.dfbcard.dfbcard:
            row_w1 = w.dfbx2_widget1.state_vectors[row]
            row_w2 = w.dfbx2_widget2.state_vectors[row]
            row_w1.TriA_button.setChecked(a)
            row_w1.TriB_button.setChecked(b)
            row_w2.TriA_button.setChecked(a)
            row_w2.TriB_button.setChecked(b)
            # also disable feedback for that row
            if a:
                row_w1.FBA_button.setChecked(0)
                row_w2.FBA_button.setChecked(0)
                row_w1.d2a_A_spin.setValue(0)
                row_w2.d2a_A_spin.setValue(0)
                row_w1.data_packet.setCurrentIndex(0)
                row_w2.data_packet.setCurrentIndex(0)
            if b:
                row_w1.FBB_button.setChecked(0)
                row_w2.FBB_button.setChecked(0)
                row_w1.d2a_B_spin.setValue(0)
                row_w2.d2a_B_spin.setValue(0)
                row_w1.data_packet.setCurrentIndex(1)
                row_w2.data_packet.setCurrentIndex(1)

    app.processEvents()

 ################################################
# Actual scripts for screening                   #
# Run these during the screening process         #
# these all take an optional argument specifying #
# the configuration file you wish to use         #
 ################################################

def sq1_bias_ramp(
        configfile="mux_config.yaml"
    ):
    """ Run this phase with no row select current/flux in order
    to determine "I_c column". Run it with row selects configured
    properly to get most other params

    Determine the minimum critical current of the flux activated switches
    by ramping sq1 bias with all row select lines off.
    """
    # tracemalloc.start()
    # profile = line_profiler.LineProfiler()
    global win, app
    with open(configfile, 'r') as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    now = datetime.now()
    datestr = now.strftime("%Y-%m-%d-T%H-%M-%S")
    print("Prepare to measure collect squid 1 bias sweep")
    print("Unlock all COLs")
    feedback_lock_all("off")

    sq1bias = cfg["wiring"]["sq1_bias"]

    print("set all SQ1BIAS to 0")
    win.tower_widget.set_card_dac(sq1bias, 1)
    win.tower_widget.set_card_dac(sq1bias, 0)
    app.processEvents()
    time.sleep(1.0)

    print("Setup triangle parameters")
    tri_params = cfg["sq1_bias_ramp"]["triangle_params"]
    tri_pd = set_triangle(**tri_params)
    daq = daqlib.Daq(tri_period=tri_pd)

    print("Activate triangle on sq1_fb")
    sq1fb_loc = cfg["wiring"]["sq1_fb"]
    ssafb_loc = cfg["wiring"]["ssa_fb"]
    # If we're coming from phase1_1, row 11 will have
    # sq1fb triangle on already, so that's fine.
    enable_triangle_all(sq1fb_loc)

    print("Set SEND MODE to ssafb/ERR all COLS/ROWs")
    # Since most rows are configured to send ssafb already,
    # but row 11 is not, after phase1_1, 
    # this will leave row 11 sending sq1fb. Which is fine.
    # But it's probably better to set this explicitly.
    # Also if we're coming from phase1_1 there might be triangles on the row selects.
    # For now I'm going to ignore that because when we set the values for DAC levels
    # for row selects we should also set triangle off.
    set_send_mode(ssafb_loc) # will set all channels
    # to send the ssa feedback and error signals

    print("RESYNC and lock on ssa_fb")
    win.resync_system.click()
    time.sleep(5)
    feedback_lock_all(ssafb_loc)
    try:
        echo_row = cfg["sq1_bias_ramp"]["row_for_triangle_echo"]
        if echo_row: 
            enable_triangle_all_1_row(ssafb_loc, echo_row)
            # turn on triangle on row 11 so that it gets sent
            # as the ssafb signal there
    except KeyError: # row_for_triangle_echo not in config file, 
        pass # don't configure an echo.

    print("Ready to execute phase1_0 measurements / sq1 bias ramp")

    ramp_params = cfg["sq1_bias_ramp"]["bias_ramp"]

    bias_array = np.linspace(
        ramp_params["start_dac"], 
        ramp_params["stop_dac"], 
        ramp_params["num_steps"],
        dtype=int
    )
    part = 0
    start = 0
    #profile.add_function(daq.take_average_data)
    # collect and save data
    data_array = []
    bar = IncrementalBar(
        "Collecting Data:", 
        max=len(bias_array),
        suffix=' [%(index)d/%(max)d, ETA:%(eta_td)s]')
    for i,bias in enumerate(bias_array):
        # print(f"Set bias {bias}")
        win.tower_widget.set_card_dac(sq1bias, bias)
        app.processEvents()
        time.sleep(1)
        #data = profile.wrap_function(daq.take_average_data)()
        data = daq.take_average_data()
        data_array.append(data)
        bar.next()
        if i%100==0 and i!=0 and len(bias_array)>i+40:
            # snapshot=tracemalloc.take_snapshot()
            # stats = snapshot.statistics('lineno')
            # print("\nMemory audit! Top users:")
            # for stat in stats[:10]:
            #     print(stat)
            # Turns out that holding 
            # 100 arrays of 2x8x24x8192 float64s
            # in memory adds up to several GB.
            # so we'll dump them to disk periodically
            bias_save = bias_array[start:i+1]
            start = i+1
            sq1_ramp_save_data(cfg, bias_save, data_array, datestr, part)
            part += 1
            data_array = [] # once there are no
            # references to the old data array 
            # it will be garbage collected.
    #profile.print_stats()
    bar.finish()
    if part == 0:
        part = None
    sq1_ramp_save_data(cfg, bias_array, data_array, datestr, part)
    win.tower_widget.set_card_dac(sq1bias, 0)

    to_interactive()
    # For phase 1_0, since no row selects are active, 
    #all row data on a given column should be 
    #identical, good candidate for further averaging.

def sq1_ramp_save_data(cfg, bias_array, data_array, datestr, part):

    if part is None:
        fname = f"{datestr}_sq1_bias_ramp.npz"
    else:
        fname = f"{datestr}_sq1_bias_ramp_part_{part}.npz"
    Path(cfg['io']['data_folder']).mkdir(parents=True, exist_ok=True)
    fpath = path.join(cfg["io"]["data_folder"], fname)
    np.savez_compressed(
        fpath,
        bias_array=bias_array,
        data_array=data_array,
        **cfg["chip_info"] 
    )

def phase_1_0():
    sq1_bias_ramp()

def phase_1_1(
        configfile="mux_config.yaml"
    ):
    """
    Determine optimal row select DAC values for opening the flux activated
    switches. For 2 level switches this requires a loop in which we ramp the chip-select
    flux 
    """
    global win, app
    with open(configfile, 'r') as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    Path(cfg['io']['data_folder']).mkdir(parents=True, exist_ok=True)
    print("Prepare to measure switch flux quanta")
    print("Unlock all, disable triangle")
    feedback_lock_all("off")
    enable_triangle_all("off")
    tri_params = cfg["phase_1_1"]["triangle_params"]

    print("Configure triangle")
    tri_pd = set_triangle(**tri_params)
    daq = daqlib.Daq(tri_period=tri_pd)

    # We need a row to save the triangle itself to.
    seq_len = win.seqln_spin.value()
    num_rows = cfg['wiring']['num_rows']
    two_lvl = cfg['wiring']['is_two_level']
    num_chips = cfg['wiring']['chips_per_col']

    print(f"Sanity check:")
    print(f"sequence length={seq_len}")
    print(f"num_rows on chip:{num_rows}")
    print(f"is_two_level={two_lvl}")
    print(f"sequence length >= (num_rows {"+ 1 " if two_lvl else ""}+ 1) * chips_per_col")
    extra_rows = 1
    if two_lvl:
        extra_rows += 1
    print(seq_len >= (num_rows + extra_rows)*num_chips)
    print("The other sanity check is: are your HDMIs plugged in?")
    last_row = seq_len - 1
    sq1fb_loc = cfg["wiring"]["sq1_fb"]
    ssafb_loc = cfg["wiring"]["ssa_fb"]

    print("Lock feedback, enable triangle on row selects")
    feedback_lock_all(ssafb_loc)
    set_send_mode(ssafb_loc)
    enable_triangle_all_1_row(sq1fb_loc, last_row) # do this after enabling feedback and setting send mode

    sq1bias = cfg["wiring"]["sq1_bias"]
    win.tower_widget.set_card_dac(sq1bias, cfg["phase_1_1"]["sq1_bias"])
    rs_card = cfg["phase_1_1"]["rs_card"]
    if type(rs_card) is not list:
        rs_card = [rs_card]

    for card in rs_card:
        set_rs_params(card, tri=1)
        set_rs_dacs(card, 0, 0)
    if cfg["wiring"]["is_two_level"]:
        ramp_params = cfg["phase_1_1"]["cs_ramp"]
        cs_index = cfg["wiring"]["chip_select_row"]
        if type(cs_index) is not list:
            cs_index = [cs_index]
        for card, cs in zip(rs_card,cs_index):
            cs_loc_on_card = cs % 16
            set_rs_params(card, row_index=cs_loc_on_card, tri=0)
        cs_ramp = np.linspace(
            ramp_params["start_dac"],
            ramp_params["stop_dac"],
            ramp_params["num_steps"],
            dtype=int
        )
        data_array = []
        bar = IncrementalBar(
            "Collecting Data:", 
            max=len(cs_ramp),
            suffix=' [%(index)d/%(max)d, ETA:%(eta_td)s]')
        bar.start()
        for bias in cs_ramp:
            for card, cs in zip(rs_card,cs_index):
                cs_loc_on_card = cs % 16
                set_rs_dacs(card, 0, bias, row_index=cs_loc_on_card)
            #print(f"set cs flux={bias}")

            data_array.append(daq.take_average_data())
            bar.next()
        bar.finish()
        now = datetime.now()
        datestr = now.strftime("%Y-%m-%d-T%H-%M-%S")
        np.savez_compressed(
            path.join(cfg["io"]["data_folder"], f"{datestr}_rs_ramp.npz"),
            cs_ramp=cs_ramp,
            data_array=data_array,
            **cfg["chip_info"] 
        )
    else:
        now = datetime.now()
        datestr = now.strftime("%Y-%m-%d-T%H-%M-%S")
        np.savez_compressed(
            path.join(cfg["io"]["data_folder"], f"{datestr}_rs_ramp.npz"),
            data_array=daq.take_average_data(),
            **cfg["chip_info"] 
        )
    to_interactive()

def set_fas_flux(
        configfile="mux_config.yaml"
    ):
    """ Analyze the row select ramp taken with stage 1_1 and 
    set the appropriate dac values. Note that this works best if 
    phase 1_1 was configured with a narrow range of dac values so that only one
    period of the switch is evaluated.""" 
    #TODO check this out and make sure it works
    global win, app
    with open(configfile, 'r') as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    is_two_level = cfg["wiring"]["is_two_level"]
    ramp_files = glob(path.join(cfg["io"]["data_folder"], "*rs_ramp.npz"))
    most_recent_file = sorted(ramp_files)[-1]
    stage11_results = np.load(most_recent_file)

    rs_data_unaligned = stage11_results["data_array"]
    if is_two_level:
        rs_data = analysis.align_ramp(rs_data_unaligned)
    else:
        rs_data = analysis.align_ramp(np.array(rs_data_unaligned))
    tri = rs_data[0,0,0,-1]

    num_cols = len(cfg["wiring"]["columns"])
    num_rows = win.seqln_spin.value()
    if is_two_level:
        cs_ramp = stage11_results["cs_ramp"]
        cs_flux_arr, rs_flux_arr = analysis.get_fas_biases(
            rs_data, 
            cs_ramp,
            ignore_rows=cfg['phase_1_1_analysis']['rs_force_zero']
        )

    #Set the chip select fluxes. 
    # From the fluxes that produced the maximum response
    # that were computed above, take the median 
    # for all the Columns and all the RSs where that CS is
    # active.
    cs_seq_map = cfg["wiring"]["cs_seqn_map"]
    for cs in cs_seq_map.keys():
        rs_arr = cs_seq_map[cs]
        optimal_cs_flux = np.nanmedian(cs_flux_arr[:,rs_arr])
        widget_index = cfg["phase_1_1"]["rs_card"][cs // 16]
        row_idx = cs % 16
        set_rs_dacs(widget_index, 0, optimal_cs_flux, row_index=row_idx)
        set_rs_params(widget_index, row_index = row_idx)
    else:
        rs_flux_arr = analysis.get_fas_biases_1x11(rs_data,ignore_rows=cfg['phase_1_1_analysis']['rs_force_zero'])



    #Set the row select fluxes.
    # From the fluxes that produced max response (above),
    # take the median of all columns for each sequence slot
    # and assign that value to the corresponding row select
    # output.
    seq_rs = cfg["wiring"]["seqn_rs_style"]
    if seq_rs == "same":
        seq_rs = np.arange(num_rows)
    elif seq_rs == "true_2ls":
        pass #nyi
        # This will be complicated, because at least with the crate
        # you only get 1 RS bias point per row select line,
        # when in theory there could be multiple different
        # chips addressed bt a given RS line. 
    # else: pass, the user gave us an array
    for seq_slot, rs_active in enumerate(seq_rs):
        if np.count_nonzero(seq_rs==rs_active)>1 and rs_active!=-1:
            Warning("True two level switching is not yet implemented")
        optimal_flux = np.nanmedian(rs_flux_arr[:,seq_slot])
        widget_index=cfg["phase_1_1"]["rs_card"][rs_active//16]
        row_index = rs_active % 16
        if np.isnan(optimal_flux):
            optimal_flux=0
        set_rs_dacs(widget_index,0,int(optimal_flux), row_index=row_index)
        set_rs_params(widget_index, row_index=row_index)

def phase_2(
        configfile="mux_config.yaml"
    ):
    """
    After determining fluxes for FAS, apply those fluxes in cringe and 
    then perform a squid 1 bias ramp again, this time with the switches
    configured realistically.

    For now we'll leave configuring the RS DAC values to the user
    and proceed straight to the bias ramp.

    Actually that basically makes this the same as phase 1_0
    """
    global win, app
    # could set rs dac here. Pseudocode:
    # for i,bias in enumerate(optimal_rs_values):
    #     set_rs_dacs(rs_card, 0, bias, row_index=i)
    sq1_bias_ramp()

def input_ramp(
        configfile="mux_config.yaml"
    ):
    """ with a triangle on the INPUT lines, collect 
    squid 1 response. Still servo on the squid series array.
    Assumes that your FASs are already set up.
    """
    global win, app
    with open(configfile, 'r') as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    print("Disable feedback and triangle")
    feedback_lock_all("off")
    enable_triangle_all("off")
    tri_params = cfg["input_ramp"]["triangle_params"]

    print("Configure triangle")
    tri_pd = set_triangle(**tri_params)
    daq = daqlib.Daq(tri_period=tri_pd)

    print("Lock feedback, enable triangle on inputs")
    sq1fb_loc = cfg["wiring"]["sq1_fb"]
    ssafb_loc = cfg["wiring"]["ssa_fb"]
    seq_len = win.seqln_spin.value()
    last_row = seq_len - 1
    feedback_lock_all(ssafb_loc)
    set_send_mode(ssafb_loc)
    enable_triangle_all_1_row(sq1fb_loc, last_row) # do this after enabling feedback and setting send mode

    sq1bias = cfg["wiring"]["sq1_bias"]
    win.tower_widget.set_card_dac(sq1bias, cfg["input_ramp"]["sq1_bias"])
    rs_card = cfg["input_ramp"]["rs_card"]
    if type(rs_card) is not list:
        rs_card = [rs_card]

    for card in rs_card:
        set_rs_params(card, tri=1, dc=cfg["input_ramp"]["use_dc"])
        set_rs_dacs(card, 0, 0)
    now = datetime.now()
    datestr = now.strftime("%Y-%m-%d-T%H-%M-%S")
    np.savez_compressed(
        path.join(cfg["io"]["data_folder"], f"{datestr}_input_ramp.npz"),
        data_array=daq.take_average_data(),
        **cfg
    )

    to_interactive()

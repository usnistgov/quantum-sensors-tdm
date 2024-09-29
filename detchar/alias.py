from detchar.noise import NoiseAcquire
import argparse
import time
from nasa_client import RpcError
from cringe.cringe_control import CringeControl
import yaml
import collections
from matplotlib import pyplot as plt
from detchar.iv_data import NoiseSweepData
import os
"""
The idea here will be:
1. have Dastard stop data acquisition, 
2. tell Cringe to increase sequence length, 
3. have Dastard start acquisition again 
4. collect noise PSD
5. repeat

The end result will be lots of noise PSDs of different
lengths, which could be annoying.

The challenges could be:
    [x] Can Cringe Control even set the sequence length in the first place?
    [ ] How much resetting do we need to do when the seq length is changed
        so far:
        - mix fraction
        - easyClient needs to reset number of rows etc
        - relock feedback?
        - Do we need new I values??
    [x] Can we be sure that the first X rows stay the same?
    [x] Does it matter what rows we read in the meantime?
    [x] Best way to interface with Dastard? Can Dastard handle multiple connections at once?
    [x] Every time data stops and starts in Dastard Commander the trigger parameters are reset
    [x] when data is stopped and started the mix config is lost

Issues:
    [ ] Dastard Commander will crash when this script is running.

"""

class AliasSweep:
    def __init__(self, noise_acquire, mix_fraction):
        self.na = noise_acquire
        self.ec = self.na.ec
        self.cc = self.na.cc
        self.mix_fraction = mix_fraction

    def set_length(self, new_length):
        try:
            print("Stopping Dastard data to change sequence length...")
            self.ec.stop_data()
        except RpcError as e:
            print(e)

        print("Commanding new seq length in Cringe...")
        self.cc.set_sequence_length(new_length)

        time.sleep(2) # Empirically determined wait time. 1 s is too short.

        print("Attempting to restart Lancero data in Dastard...")
        self.ec.start_data_lancero()
        reported_seq_len = None
        while reported_seq_len != new_length:
            self.ec.messagesSeen = collections.Counter() # Makes sure that the
            # client actually waits for a new status message 
            self.ec._getStatus() # updates things like sample time
            reported_seq_len = self.ec.sequenceLength
            time.sleep(1)
        self.ec.setMix(self.mix_fraction)

    def run(self, seq_len_list):
        noise_data = []
        for i in seq_len_list:
            self.set_length(i)
            self.cc.relock_all_locked_fba(self.na.db_bay)
            noise_data.append(self.na.take(extra_info={"seq_len":i}))
        kludged_noise_sweep_data = NoiseSweepData(
            data = [noise_data],
            column = self.na.column,
            row_sequence = self.na.row_sequence,
            signal_column_index = self.na.column,
            db_cardname = self.na.db_cardname,
            db_tower_channel_str = self.na.db_bay,
            temp_settle_delay_s = 1e-30,
            temp_list_k = [1e-30],
            db_list = [[1]],
            extra_info={
                'seq_len_list': seq_len_list,
                'type': 'Alias Sweep'
            }
        )
        return kludged_noise_sweep_data

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Take noise spectra at varying row-revisit rates')
    parser.add_argument('file', type=str, help="configuration yaml file")
    args = parser.parse_args()
    na = NoiseAcquire.from_yaml(args.file)
    with open(args.file,'r') as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)
    if os.path.isfile(config['io']['SaveTo']):
        raise OSError("File exists")
    alias_sweeper = AliasSweep(na, config['runconfig']['mix_fraction'])
    nsd = alias_sweeper.run(config['runconfig']['seq_len_list'])
    if config['runconfig']['save_data']:
        nsd.to_file(filename=config['io']['SaveTo'])
    if config['runconfig']['show_plot']:
        fig = None
        ax = None
        data_list = nsd.data
        for data in data_list:
            fig, ax = data.plot_avg_psd(fig = fig, ax = ax,row_index = list(range(len(na.row_sequence))))
        plt.show()

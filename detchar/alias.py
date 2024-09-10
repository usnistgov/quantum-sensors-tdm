from detchar.noise import NoiseAcquire

import time
from nasa_client import RpcError
from cringe.cringe_control import CringeControl
import yaml
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
    [ ] Can we be sure that the first X rows stay the same?
    [ ] Does it matter what rows we read in the meantime?
    [x] Best way to interface with Dastard? Can Dastard handle multiple connections at once?
    [x] Every time data stops and starts in Dastard Commander the trigger parameters are reset
    [x] when data is stopped and started the mix config is lost

"""

class AliasSweep:
    def __init__(self, noise_acquire):
        self.na = noise_acquire
        self.ec = self.na.ec
        self.cc = self.na.cc

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
        ec.start_data_lancero()
        ec._getStatus() # updates things like sample time
        ec.setMix(self.mix_fraction)

    def run(self, seq_len_list):
        noise_data = []
        for i in seq_len_list:
            self.set_length(i)
            noise_data.append(self.na.take(extra_info={"seq_len":i}))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Take noise spectra at varying row-revisit rates')
    parser.add_argument('file', type=str, help="configuration yaml file")
    args = parser.parse_args()
    na = NoiseAcquire.from_yaml(args.file)
    with open(args.file,'r') as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)
    alias_sweeper = AliasSweep(na)
    data_list = alias_sweeper.run(config['runconfig']['seq_len_list'])
    if config['runconfig']['show_plot']:
        fig = None
        ax = None
        for data in data_list:
            fig, ax = data.plot_avg_psd(row_index = config['detectors']['Rows'], fig = fig, ax = ax)


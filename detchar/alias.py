from detchar.noise import NoiseAcquire

import time
from nasa_client import EasyClient
from cringe.cringe_control import CringeControl
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
    Can Cringe Control even set the sequence length in the first place?
    How much resetting do we need to do when the seq length is changed
    Can we be sure that the first X rows stay the same?
    Does it matter what rows we read in the meantime?
    Best way to interface with Dastard? Can Dastard handle multiple connections at once?
    Every time data stops and starts in Dastard Commander the trigger parameters are reset
    when data is stopped and started the mix config is lost

"""

print("WIP: Testing Easy Client")

ec = EasyClient()

try:
    ec.stop_data()
except:
    print("Exception probably source not active")
else:
    time.sleep(5)

cc = CringeControl()
cc.set_sequence_length(40)

time.sleep(2)
print("Attempting to start Lancero:")


ec.start_data_lancero()
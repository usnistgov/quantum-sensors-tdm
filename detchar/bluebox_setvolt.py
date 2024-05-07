#! /usr/bin/env python
'''bluebox_setvolt.py '''

from instruments import BlueBox
import sys

vs = BlueBox(port='vbox', version='mrk2',baud=9600, address=0, channel=0, shared=True)
dac=int(sys.argv[1])
vs.setVoltDACUnits(dac)
#vs.setvolt(v) # command repeated because after power cycle 1st command doesn't take
print('Setting bluebox voltage to %.4f V'%(dac/(2**16-1)*6.5535))

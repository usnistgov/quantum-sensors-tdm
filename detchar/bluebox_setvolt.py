#! /usr/bin/env python
'''bluebox_setvolt.py '''

from instruments import bluebox
import sys

vs = bluebox.BlueBox(port='vbox', version='mrk2')
v=float(sys.argv[1])
vs.setvolt(v)
vs.setvolt(v) # command repeated because after power cycle 1st command doesn't take
print('Setting bluebox voltage to %.3f V'%v)

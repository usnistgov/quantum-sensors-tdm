#! /usr/bin/env python
'''test_bluebox.py '''

from instruments import BlueBox 
import sys 

vs = BlueBox(port='vbox', version='mrk2')
v=float(sys.argv[1])
vs.setvolt(v)

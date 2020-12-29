'''test_bluebox.py '''

from instruments import BlueBox 

vs = BlueBox(port='vbox', version='mrk2')
vs.setvolt(0.5)

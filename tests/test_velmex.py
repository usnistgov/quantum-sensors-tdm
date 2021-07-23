from instruments import Velmex 
import numpy as np 
from time import sleep

gm = Velmex(doInit=True)

def test_move_relative():
    rel_move = np.ones(10)*10
    print('current position: ',gm.get_current_position(convert_to_deg=True))
    wait_for_complete = False
    for mv in rel_move:
        print('moving %d degrees relative'%mv)
        gm.move_relative(mv,wait=wait_for_complete)
        sleep(1)
        print('current position: ',gm.get_current_position(convert_to_deg=True))
    gm.move_to_zero_index(wait=wait_for_complete)

def test_home(initial_angle=180):
    wait_for_complete = False
    gm.move_relative(initial_angle,wait=wait_for_complete)
    sleep(gm._motion_time(initial_angle)+3)
    gm.home()

def test_move_abs():
    angles = np.linspace(0,360,5)
    for angle in angles:
        print('Moving to angle = ',angle)
        gm.move_absolute(angle,wait=True)
        print('Current position = ',gm.get_current_position(convert_to_deg=True))
    gm.move_to_zero_index(wait=False)


#test_move_relative()
#test_home(-90)
test_move_abs()


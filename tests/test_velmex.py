from instruments import Velmex 
import numpy as np 
from time import sleep

gm = Velmex(doInit=True)

def test_move_relative():
    rel_move = np.ones(10)*-10
    print('current position: ',gm.get_current_position(convert_to_deg=True))
    wait_for_complete = False
    for mv in rel_move:
        print('moving %d degrees relative'%mv)
        gm.move_relative(mv,wait=wait_for_complete)
        sleep(1)
        print('current position: ',gm.get_current_position(convert_to_deg=True))
    gm.move_to_zero_index(wait=wait_for_complete)

def test_home(initial_angle=180):
    gm.move_relative(initial_angle,wait=True)
    gm.home()

def test_move_abs():
    angles = np.linspace(0,720,10)
    print("motor will move to angles: ",angles)
    for angle in angles:
        print('Moving to angle = ',angle)
        gm.move_absolute(angle,wait=True)
        print('Current position = ',gm.get_current_position(convert_to_deg=True))
    gm.move_to_zero_index(wait=False,verbose=True)

def test_out_of_range():
    gm.move_absolute(-721)

def test_gets():
    print(gm.get_current_position(convert_to_deg=True))
    print(gm.get_current_position_for_motor(motor_id=1,convert_to_deg=False))
    print(gm.get_operating_mode())
    print(gm.get_motor_type(motor_id=1))
    print(gm.get_limitswitch_mode(motor_id=1))
    print(gm.get_current_motor_number())
    print(gm.get_configuration(print_back=True))

#test_gets()
#test_move_relative()
#test_home(90)
#test_move_abs()
test_out_of_range()

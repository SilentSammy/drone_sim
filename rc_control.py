import cv2
import input_man as im
import sim_tools as st
from sim_tools import sim
import math
from sim_drone import SimDrone
from input_man import is_pressed, get_axis, rising_edge, is_toggled
from tello_drone import TelloDrone
from video import show_frame, screenshot, record

# Manual control (keyboard and controller)
def manual_control():
    x, y, z, w = 0, 0, 0, 0
    
    # Get keyboard input using right-handed coordinate system
    k_x = 1 if is_pressed('w') else -1 if is_pressed('s') else 0
    k_y = -1 if is_pressed('d') else 1 if is_pressed('a') else 0
    k_z = 1 if is_pressed('z') else -1 if is_pressed('x') else 0
    k_w = -1 if is_pressed('e') else 1 if is_pressed('q') else 0

    # Get controller input using right-handed coordinate system
    c_x = get_axis('LY')
    c_y = -get_axis('LX')
    c_z = get_axis('RT') - get_axis('LT') # Right trigger to go up, left trigger to go down
    c_w = -get_axis('RX')

    # Get the maximum absolute value of the inputs while keeping the sign
    x = k_x if abs(k_x) > abs(c_x) else c_x
    y = k_y if abs(k_y) > abs(c_y) else c_y
    w = k_w if abs(k_w) > abs(c_w) else c_w
    z = k_z if abs(k_z) > abs(c_z) else c_z

    return x, y, z, w

def flip_control():
    return 'f' if rising_edge('i', 'DPAD_UP') else 'b' if rising_edge('k', 'DPAD_DOWN') else 'l' if rising_edge('j', 'DPAD_LEFT') else 'r' if rising_edge('l', 'DPAD_RIGHT') else None

def process_frame(frame, drawing_frame=None):
    import vision as v
    v.find_arucos(frame, drawing_frame=drawing_frame)

# Client setup
drone = SimDrone()                      # Using simulation drone
# drone = TelloDrone()                  # Using drone's hotspot
# drone = TelloDrone("192.168.137.37")  # Using laptop's hotspot

try:
    sim.startSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        # Get camera image
        if rising_edge('c'):
            drone.cam_idx += 1
        frame = drone.get_frame()
        drawing_frame = frame.copy()

        # Optional screenshot or recording
        if rising_edge('p'):
            screenshot(frame)
        record(frame if is_toggled('o') else None)

        # Takeoff and landing control
        if rising_edge('t'):
            if not drone.flight:
                drone.takeoff()
            else:
                drone.land()

        # Process the frame (e.g., for object detection)
        process_frame(frame, drawing_frame=drawing_frame)

        # Get user input for drone control
        x, y, z, w = 0, 0, 0, 0
        f = flip_control()
        if not f:
            x, y, z, w = manual_control()
        else:
            drone.flip(f)
        drone.send_rc(x, y, z, w)

        show_frame(drawing_frame, 'Drone Camera', scale=0.5)
finally:
    del drone

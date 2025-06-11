import time
import math
import numpy as np
import cv2
import input_man as im
import sim_tools as st
from sim_tools import sim
import math
from input_man import is_pressed, get_axis, rising_edge, is_toggled
from video import show_frame, screenshot, record
from drone_est import DroneEstimator, PnpResult
from drone_control import DroneController, TrajectoryFollower, Choreographer
from ball_detector import BallDetector

# Manual control (keyboard and controller)
def manual_control(keybindings=None):
    # keybindings: [pos_x, neg_x, pos_y, neg_y, pos_z, neg_z, pos_w, neg_w]
    # Defaults:    ['w',   's',   'd',   'a',   'z',   'x',   'e',   'q']
    if keybindings is None:
        keybindings = ['w', 's', 'd', 'a', 'z', 'x', 'e', 'q']
    x = 1 if is_pressed(keybindings[0]) else -1 if is_pressed(keybindings[1]) else 0
    y = 1 if is_pressed(keybindings[3]) else -1 if is_pressed(keybindings[2]) else 0
    z = 1 if is_pressed(keybindings[4]) else -1 if is_pressed(keybindings[5]) else 0
    w = 1 if is_pressed(keybindings[7]) else -1 if is_pressed(keybindings[6]) else 0
    return x, y, z, w

def follow_trajectory(frame, de:DroneEstimator, tf:TrajectoryFollower, drawing_frame=None):
    import matrix_help

    # Get the drone's estimated position and orientation
    result = de.get_drone_transform_nb(frame, drawing_frame=drawing_frame)
    if result is None:
        print("Drone pose estimation failed.")
        return None
    drone_T, res = result
    
    # Get the pose of the current waypoint (x, y, z, yaw)
    waypoint = tf.current_waypoint

    # Visualize the waypoint using the dummy drone
    waypoint_T = matrix_help.vecs_to_matrix(rvec = (0, 0, waypoint[3]), tvec = waypoint[:3])

    # Feed the drone's pose to the trajectory follower
    tf.feed_pose(drone_T)
    
    # Get the command velocities from the trajectory follower
    return tf.move()

def construct_drones(idx):
    if idx == 0:
        from sim_drone import SimDrone
        drone1 = SimDrone(start_sim=use_sim)
        drone2 = SimDrone(
            forward_cam_path='/Quadcopter[1]/visionSensor[0]',
            down_cam_path='/Quadcopter[1]/visionSensor[1]',
            target_path='/Quadcopter[1]/base/target',
        )
    elif idx == 1:
        from tello_drone import TelloDrone
        drone1 = TelloDrone("192.168.137.51")  # Using laptop's hotspot
        drone2 = TelloDrone("192.168.137.52")  # Another Tello drone connected to the same hotspot
    
    drone1.cam_idx = 1
    drone2.cam_idx = 1
    return drone1, drone2

# Camera estimation setup
board = cv2.aruco.CharucoBoard(
        size=(9, 24),
        squareLength=0.1,
        markerLength=0.08,
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    )
K = np.array([[444,   0, 256], [  0, 444, 256], [  0,   0,   1]], dtype=np.float32)
D = np.zeros(5)  # [0, 0, 0, 0, 0]

# Drones setup
use_sim = True
drone1, drone2 = construct_drones(0)  # Change the index to switch between drones
drones = [
    {
        'drone': drone1, 'name': 'Drone1', 'keybindings': ['w', 's', 'd', 'a', 'z', 'x', 'e', 'q'],
        'de': DroneEstimator(K=np.array([[444, 0, 256], [0, 444, 256], [0, 0, 1]], dtype=np.float32), D=np.zeros(5), board=board),
        'tf': TrajectoryFollower(auto_advance=True, waypoints=[
            # Start
            (0, -0.5, 0.72, 0),
            # Rise to Z=1.5
            (0, -0.5, 1.5, 0),
            # Visit corners in counter-clockwise order (start with farthest Y)
            (-0.4,  1.1, 1.5, 0),   # top-left (farthest from start)
            ( 0.4,  1.1, 1.5, 0),   # top-right
            ( 0.4, -1.1, 1.5, 0),   # bottom-right
            (-0.4, -1.1, 1.5, 0),   # bottom-left (nearest to start)
            # Return to center at Z=1.5
            (0, -0.5, 1.5, 0),
            # Lower to start Z
            (0, -0.5, 0.72, 0),
        ]),
    },
    {
        'drone': drone2, 'name': 'Drone2', 'keybindings': ['i', 'k', 'l', 'j', 'm', ',', 'o', 'u'],
        'de': DroneEstimator(K=K, D=D, board=board),
        'tf': TrajectoryFollower(auto_advance=True, waypoints=[
            # Start
            (0, 0.5, 0.72, 0),
            # Rise to Z=1.5
            (0, 0.5, 1.5, 0),
            # Visit corners in clockwise order (start with farthest Y)
            ( 0.4, -1.1, 1.5, 0),   # bottom-right (farthest from start)
            (-0.4, -1.1, 1.5, 0),   # bottom-left
            (-0.4,  1.1, 1.5, 0),   # top-left
            ( 0.4,  1.1, 1.5, 0),   # top-right (nearest to start)
            # Return to center at Z=1.5
            (0, 0.5, 1.5, 0),
            # Lower to start Z
            (0, 0.5, 0.72, 0),
        ]),
    },
]

choreo = Choreographer([drone['tf'] for drone in drones if 'tf' in drone])

try:
    while not use_sim or sim.getSimulationState() != sim.simulation_stopped:
        # start_time = time.time()
        # Get camera image
        frames = [drone['drone'].get_frame() for drone in drones]

        # Takeoff control
        if rising_edge('t'):
            for drone in drones:
                if not drone['drone'].flight:
                    drone['drone'].takeoff()
        
        # Landing control
        if rising_edge('r'):
            for drone in drones:
                if drone['drone'].flight:
                    drone['drone'].land()

        if not is_toggled('f'):
            # Get values for manual drone control
            for i, drone in enumerate(drones):
                man_vels = manual_control(drone['keybindings'])
                drone['drone'].send_rc(*man_vels)
        else:
            choreo.check()
            for i, drone in enumerate(drones):
                # Get values for trajectory following
                frame = frames[i]
                de = drone['de']
                tf = drone.get('tf', None)
                if tf is not None:
                    auto_vels = follow_trajectory(frame, de, tf)
                    if auto_vels is not None:
                        rounded_vels = tuple(round(v, 2) for v in auto_vels)
                        print(f"Vels {drone['name']}: {rounded_vels}")
                        drone['drone'].send_rc(*auto_vels)

        for i, frame in enumerate(frames):
            show_frame(frame, f"{drones[i]['name']} Camera", scale=0.5)
finally:
    # Cleanup
    del drone1
    del drone2

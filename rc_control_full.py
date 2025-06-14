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
from drone_control import DroneController, TrajectoryFollower
from ball_detector import BallDetector

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

def visualize_drone_pose(frame, drawing_frame=None):
    import drone_viz
    res = de.get_drone_transform(frame, drawing_frame=drawing_frame)
    if res is None:
        return
    drone_T, _ = res
    drone_viz.visualize_drone_pose(drone_T)

def match_dummy_pose(frame, drawing_frame=None):
    import drone_viz
    if drone_viz.dummy_drone is None:
        return None

    # Get the drone's estimated position and orientation
    result = de.get_drone_transform(frame, drawing_frame=drawing_frame)
    if result is None:
        print("Drone pose estimation failed.")
        return None
    drone_T, res = result
    dc.feed_pose(drone_T)

    # Get the position and orientation of the dummy object
    dummy_pos = sim.getObjectPosition(drone_viz.dummy_drone, -1)
    dummy_x = dummy_pos[0]  # X position is the first element in the position tuple
    dummy_y = dummy_pos[1]  # Y position is the second element in the position tuple
    dummy_z = dummy_pos[2]  # Z position is the third element in the position tuple
    dummy_yaw = -sim.getObjectOrientation(drone_viz.dummy_drone, -1)[2]  # Yaw is the third element in the orientation tuple

    return dc.move_to( x=dummy_x, y=dummy_y, z=dummy_z, yaw=dummy_yaw )

def follow_trajectory(frame, drawing_frame=None):
    import drone_viz
    import matrix_help
    # Static variable to keep the trajectory follower instance
    # Define a "figure eight" trajectory within the bounds
    waypoints = [
        (0.0, 0.2, 0.5, 0.0),    # Start position (center top)
        (0.25, 0.8, 0.5, math.pi/4),   # Top right loop
        (0.0, 0.0, 0.5, math.pi),      # Cross to center
        (-0.25, -0.8, 0.5, -math.pi/4),# Bottom left loop
        (0.0, 0.2, 0.5, 0.0),          # Return to start
        (0.3, -1.0, 0.5, math.pi/2)    # Far bottom right for a flourish
    ]

    tf = follow_trajectory.tf = TrajectoryFollower(auto_advance=True, waypoints=waypoints) if not hasattr(follow_trajectory, 'tf') else follow_trajectory.tf

    if tf.done:
        drone.land()

    # Get the drone's estimated position and orientation
    result = de.get_drone_transform(frame, drawing_frame=drawing_frame)
    if result is None:
        print("Drone pose estimation failed.")
        return None
    drone_T, res = result
    
    # Get the pose of the current waypoint (x, y, z, yaw)
    waypoint = tf.current_waypoint

    # Visualize the waypoint using the dummy drone
    waypoint_T = matrix_help.vecs_to_matrix(rvec = (0, 0, waypoint[3]), tvec = waypoint[:3]) 
    drone_viz.visualize_drone_pose(waypoint_T)

    # Feed the drone's pose to the trajectory follower
    tf.feed_pose(drone_T)
    
    # Get the command velocities from the trajectory follower
    return tf.move()

def construct_drone(idx):
    if idx == 0:
        from drone import Drone
        cap = cv2.VideoCapture("http://192.168.137.86:4747/video") # Droidcam IP camera
        drone = Drone(
            get_frame=lambda: cv2.rotate(cap.read()[1], cv2.ROTATE_90_CLOCKWISE),
            K = np.array([
                [487.14566155,   0.,         321.7888109 ],
                [  0.,         487.60075097, 239.38896134],
                [  0.,           0.,           1.        ]
            ], dtype=np.float32),
            D = np.array([0.33819757, 1.36709606, -6.17042008, 8.65929659], dtype=np.float32)
        )
    elif idx == 1:
        from sim_drone import SimDrone
        drone = SimDrone(start_sim=use_sim)
    elif idx == 2:
        from tello_drone import TelloDrone
        drone = TelloDrone()                  # Using drone's hotspot
    elif idx == 3:
        from tello_drone import TelloDrone
        drone = TelloDrone("192.168.137.14")  # Using laptop's hotspot
    return drone

# Drone control
dc = DroneController()
de = DroneEstimator(
    K = np.array([
        [226.88179248,   0.,         160.67144106],
        [  0.,         227.22020997, 117.68560542],
        [  0.,           0.,           1.        ]
    ], dtype=np.float32),
    D = np.zeros(5),  # [0, 0, 0, 0, 0]
    board = cv2.aruco.CharucoBoard(
        size=(9, 24),
        squareLength=0.1,
        markerLength=0.08,
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    )
)

de1 = DroneEstimator(
    K = np.array([[444,   0, 256], [  0, 444, 256], [  0,   0,   1]], dtype=np.float32),
    D = np.zeros(5),  # [0, 0, 0, 0, 0]
    board = cv2.aruco.CharucoBoard(
        size=(9, 24),
        squareLength=0.1,
        markerLength=0.08,
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    )
)

# Client setup
use_sim = True
drone = construct_drone(3)  # Change the index to switch between drones
drone.cam_idx = 0           # Start with the dorsal camera

# Control modes
mode = 0
modes = [
    {'desc': 'Manual Control', 'func': None},
    {'desc': 'Pose Estimation', 'func': lambda: visualize_drone_pose(frame, drawing_frame)},
    {'desc': 'Match Dummy Pose', 'func': lambda: match_dummy_pose(frame, drawing_frame)},
    {'desc': 'Follow Trajectory', 'func': lambda: follow_trajectory(frame, drawing_frame)},
]

try:
    cam_idx = 0
    while True:
        # Get camera image
        if rising_edge('c'):
            cam_idx = cam_idx + 1
            drone.change_camera(cam_idx)
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

        # Choose control mode
        for i in range(1, len(modes)+1):
            if rising_edge(str(i)):
                mode = i-1
                print(f"Control mode: {modes[i-1].get('desc', str(mode))}")
                break

        # Get values for drone control
        man_vels = manual_control()
        if mode < len(modes):
            func = modes[mode].get('func', None)
            func_vels = None if func is None else func()

            # Check if func_vels is a tuple of exactly 4 numbers (float or int)
            if func_vels is not None and (
                not isinstance(func_vels, tuple) or
                len(func_vels) != 4 or
                not all(isinstance(v, (float, int, np.floating, np.integer)) for v in func_vels)
            ):
                func_vels = None

            func_vels = np.zeros(4) if func_vels is None else func_vels

        total_vels = np.add(man_vels, func_vels)
        total_vels = np.clip(total_vels, -1.0, 1.0)
        
        drone.send_rc(*total_vels)

        show_frame(drawing_frame, 'Drone Camera', scale=0.75)
        # print("execution time:", time.time() - start_time)
finally:
    # Cleanup
    del drone

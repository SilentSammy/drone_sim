import numpy as np
import cv2
import input_man as im
import sim_tools as st
from sim_tools import sim
import math
from input_man import is_pressed, get_axis, rising_edge, is_toggled
from video import show_frame, screenshot, record
from drone_est import DroneEstimator

def process_frame(frame, drawing_frame=None):
    drone_T = de.get_camera_transform(frame, drawing_frame=drawing_frame)
    if drone_T is None:
        return

    # 4) Send the result directly to CoppeliaSim
    T_coppelia = st.np_to_coppelia_T(drone_T)
    sim.setObjectMatrix(drone_repr, T_coppelia, sim.handle_world)

de = DroneEstimator(
    K = np.array([[444,   0, 256], [  0, 444, 256], [  0,   0,   1]], dtype=np.float32),
    D = np.zeros(5),  # [0, 0, 0, 0, 0]
    board = cv2.aruco.CharucoBoard(
        size=(9, 12),
        squareLength=0.1,
        markerLength=0.08,
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    )
)
cam = sim.getObject('/CamTest/visionSensor')
drone_repr = sim.getObject('/PoseViz/Dummy')

try:
    sim.startSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        frame = st.get_image(cam)
        drawing_frame = frame.copy()

        # Optional screenshot or recording
        if rising_edge('p'):
            screenshot(frame)
        record(frame if is_toggled('o') else None)

        # Process the frame (e.g., for object detection)
        process_frame(frame, drawing_frame=drawing_frame)

        show_frame(drawing_frame, 'Test Camera', scale=0.75)
finally:
    # Cleanup
    sim.stopSimulation()

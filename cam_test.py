import numpy as np
import cv2
import input_man as im
import sim_tools as st
from sim_tools import sim
import math
from input_man import is_pressed, get_axis, rising_edge, is_toggled
from video import show_frame, screenshot, record
from vision import find_arucos, estimate_marker_pose, get_camera_pose, vecs_to_matrix, matrix_to_vecs, rvec_to_euler, estimate_grid_origin_pose, Rz90, Rz180, Rz270, Rz270_local

K = np.array([[444,   0, 256], [  0, 444, 256], [  0,   0,   1]], dtype=np.float32)
D = np.zeros(5)  # [0, 0, 0, 0, 0]
cam = sim.getObject('/CamTest/visionSensor')
drone_repr = sim.getObject('/PoseViz/Dummy')
marker_Ts = {
    0: vecs_to_matrix([0, 0, 0], [0, 0, 0]),
    1: vecs_to_matrix([0, 0, 0], [-0.5, +0.5, 0]),
    2: vecs_to_matrix([0, 0, 0], [+0.5, +0.5, 0]),
    4: vecs_to_matrix([0, 0, 0], [-0.5, -0.5, 0]),
    3: vecs_to_matrix([0, 0, 0], [+0.5, -0.5, 0]),
}

def process_frame(frame, drawing_frame=None):
    # Find ArUco markers in the frame
    corners, ids = find_arucos(frame, drawing_frame=drawing_frame)
    if ids is None:
        return
    ids = ids.flatten()
    
    # Choose the center-most marker as the reference
    centers = [np.mean(corner[0], axis=0) for corner in corners]
    dists = [np.linalg.norm(center - np.array([frame.shape[1] / 2, frame.shape[0] / 2])) for center in centers]
    idx = np.argmin(dists)
    marker_corners = corners[idx]
    marker_id = ids[idx]
    rvec, tvec = estimate_grid_origin_pose(marker_corners, marker_id, 0.1, marker_Ts, K, D)

    # get the camera pose relative to the grid
    cam_rvec, (x, y, z) = get_camera_pose(rvec, tvec)

    # Convert Rodrigues vector to Euler angles
    rx, ry, rz = rvec_to_euler(cam_rvec)

    # Move the camera representation to the estimated position
    st.move_object(drone_repr, x=x, y=y, z=z)
    st.orient_object(drone_repr, alpha=rx, beta=-ry, gamma=-rz)

    # Display on the drawing frame
    if drawing_frame is not None:
        rxd = math.degrees(rx)
        ryd = math.degrees(ry)
        rzd = math.degrees(rz)
        cv2.putText(drawing_frame, f"ID: {marker_id}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(drawing_frame, f"Pos: ({x:.2f}, {y:.2f}, {z:.2f})", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(drawing_frame, f"Rot: ({rxd:.2f}, {ryd:.2f}, {rzd:.2f})", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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

        show_frame(drawing_frame, 'Test Camera', scale=0.5)
finally:
    # Cleanup
    sim.stopSimulation()

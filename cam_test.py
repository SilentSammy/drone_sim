import numpy as np
import cv2
import input_man as im
import sim_tools as st
from sim_tools import sim
import math
from input_man import is_pressed, get_axis, rising_edge, is_toggled
from video import show_frame, screenshot, record

K = np.array([[444,   0, 256], [  0, 444, 256], [  0,   0,   1]], dtype=np.float32)
D = np.zeros(5)  # [0, 0, 0, 0, 0]
cam = sim.getObject('/visionSensor')
cam_repr = sim.getObject('/Cone')

T_cv2sim = np.array([
    [0,  1, 0, 0],   # X_cv -> Y_sim
    [-1, 0, 0, 0],   # Y_cv -> -X_sim
    [0,  0, 1, 0],   # Z_cv -> Z_sim
    [0,  0, 0, 1]
], dtype=np.float32)

def process_frame(frame, drawing_frame=None):
    from vision import find_arucos, estimate_marker_pose, get_camera_pose, vecs_to_matrix, matrix_to_vecs, rvec_to_euler, matrix_to_euler_vecs
    corners, ids = find_arucos(frame, drawing_frame=drawing_frame)

    if ids is not None:
        # Find the index of the lowest id
        min_idx = np.argmin(ids.flatten())
        marker_corners = corners[min_idx]

        # get its pose relative to the camera
        rvec, tvec = estimate_marker_pose(marker_corners, 0.1, K, D)

        # get the camera pose relative to the marker
        cam_rvec, cam_tvec = get_camera_pose(rvec, tvec)

        # Transform to the right-handed coordinate system properly, using matrix operations
        T_cam_in_marker = vecs_to_matrix(cam_rvec, cam_tvec)
        T_cam_in_sim = T_cv2sim @ T_cam_in_marker
        # rvec, tvec = matrix_to_vecs(T_cam_in_sim)

        # Unpack matrix to Euler angles and position
        eu_rvec, tvec = matrix_to_euler_vecs(T_cam_in_sim)
        rx, ry, rz = eu_rvec
        x, y, z = tvec

        # Move the camera representation to the estimated position, crudely transforming the coordinates
        st.move_object(cam_repr, x=x, y=y, z=z)
        st.orient_object(cam_repr, alpha=ry, beta=-rx+math.pi, gamma=rz)

        # Display on the drawing frame
        if drawing_frame is not None:
            rxd = math.degrees(rx)
            ryd = math.degrees(ry)
            rzd = math.degrees(rz)
            cv2.putText(drawing_frame, f"ID: {ids[0][0]}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
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

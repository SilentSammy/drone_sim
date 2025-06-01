import numpy as np
import cv2
import input_man as im
import sim_tools as st
from sim_tools import sim
import math
from input_man import is_pressed, get_axis, rising_edge, is_toggled
from video import show_frame, screenshot, record
import vision
from matrix_help import ( reverse_xyz_to_zyx_4x4, extract_euler_zyx, Rx, Ry, Rz, Rx180, vecs_to_matrix )

def fix_camera_transform(cam_T):
    """
    Given a 4×4 camera‐to‐board transform cam_T (after rot_x_180 adjustments),
    this will:
      1) apply the “mirror‐Y” step you already had to fix Z
      2) call reverse_xyz_to_zyx_4x4() to fix X and Z
      3) extract the Y Euler angle from the new rotation block, negate it,
         and recompose so that clockwise Y → counterclockwise Y only.
    Returns the edited 4×4 cam_T.
    """
    
    def rot_x_180():
        """Returns a 4x4 matrix for a 180-degree rotation about the X axis."""
        R = np.eye(4)
        R[1, 1] = -1
        R[2, 2] = -1
        return R
    
    # 1) Pivot the camera transform around the X axis by 180 degrees, then rotate it along the X axis by 180 degrees.
    cam_T = rot_x_180() @ cam_T @ rot_x_180()

    # 2) Split out R and t
    R = cam_T[:3, :3].copy()
    t = cam_T[:3,  3].copy()

    # 3) Mirror‐Y reflection to fix Z (keep t unchanged)
    mirror_y_3 = np.diag([1, -1,  1])
    R = mirror_y_3 @ R @ mirror_y_3

    # 4) Reassemble cam_T with R_fixed‐Z, same translation
    cam_T[:3, :3] = R
    cam_T[:3,  3] = t

    # 5) Fix X (180° swap) is assumed done already; now reorder X/Y/Z
    #    so that X and Z come out correct. This is your existing line:
    cam_T = reverse_xyz_to_zyx_4x4(cam_T)

    # 6) At this point, cam_T[:3,:3] is R_rev = Rx(α)·Ry(β)·Rz(γ),
    #    and you found that the Y‐rotation was still backward.
    #    So now we extract (α,β,γ) under the “intrinsic ZYX” convention,
    #    flip β → –β, and recompose exactly Rx·Ry·Rz.

    # 6a) Extract the “ZYX” Euler angles from R_rev
    R_rev = cam_T[:3, :3].copy()
    t_rev = cam_T[:3,  3].copy()

    alpha, beta, gamma = extract_euler_zyx(R_rev)
    # Now flip only the Y‐angle (pitch)
    beta = -beta

    # 6b) Rebuild R_fixed = Rx(alpha)·Ry(beta)·Rz(gamma)
    R_fixed = Rx(alpha) @ Ry(beta) @ Rz(gamma)

    # 6c) Reinsert into cam_T with the same translation
    cam_T[:3, :3] = R_fixed
    cam_T[:3,  3] = t_rev

    return cam_T

def get_camera_transform(frame, drawing_frame=None):
    # detect the charuco board
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_det.detectBoard(frame)
    if charuco_ids is None:
        return

    # get the board pose relative to the camera
    res = vision.get_board_pose(board, K, D, charuco_corners, charuco_ids, center=True)
    if res is None:
        return
    board_rvec, board_tvec = res

    # Convert rvec, tvec to a 4x4 transformation matrix
    board_T = vecs_to_matrix(board_rvec, board_tvec)

    # get the camera pose relative to the board by inverting the transformation matrix
    cam_T = np.linalg.inv(board_T)

    # Adjust the camera transformation to match our coordinate system
    # …after you’ve done rot_x_180() adjustments…
    cam_T = fix_camera_transform(cam_T)

    # Display on the drawing frame
    if drawing_frame is not None:
        cv2.aruco.drawDetectedCornersCharuco( drawing_frame, charuco_corners, charuco_ids )
    
    return cam_T

def process_frame(frame, drawing_frame=None):
    cam_T = get_camera_transform(frame, drawing_frame=drawing_frame)
    if cam_T is None:
        return
    T_coppelia = st.np_to_coppelia_T(cam_T)
    sim.setObjectMatrix(drone_repr, T_coppelia, sim.handle_world)

K = np.array([[444,   0, 256], [  0, 444, 256], [  0,   0,   1]], dtype=np.float32)
D = np.zeros(5)  # [0, 0, 0, 0, 0]
cam = sim.getObject('/CamTest/visionSensor')
drone_repr = sim.getObject('/PoseViz/Dummy')
board = cv2.aruco.CharucoBoard(
    size=(9, 12),
    squareLength=0.1,
    markerLength=0.08,
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
)
charuco_det = cv2.aruco.CharucoDetector(board)

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

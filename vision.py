import numpy as np
import math
import cv2

# ARUCO MARKER STUFF
def find_arucos(frame, drawing_frame=None):
    # Detect markers using the new ArucoDetector API
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None and drawing_frame is not None:
        cv2.aruco.drawDetectedMarkers(drawing_frame, corners, ids)
    
    return corners, ids

def estimate_marker_pose(marker_corners, marker_length, camera_matrix, dist_coeffs):
    """
    Estimate a single ArUco marker pose using cv2.solvePnP.

    Returns
    -------
    rvec : np.ndarray, shape (3,)
        Rotation vector (marker in camera coordinates).
    tvec : np.ndarray, shape (3,)
        Translation vector (marker in camera coordinates).
    """
    # --- 1. Prepare image points: reshape to (4,2) float32 ---
    img_pts = np.asarray(marker_corners, dtype=np.float32)
    if img_pts.ndim == 3 and img_pts.shape[0] == 1:
        img_pts = img_pts[0]
    img_pts = img_pts.reshape((4, 2))

    # --- 2. Define object points in marker coordinate frame (Z=0 plane) ---
    half_len = marker_length / 2.0
    obj_pts = np.array([
        [-half_len,  half_len, 0.0],
        [ half_len,  half_len, 0.0],
        [ half_len, -half_len, 0.0],
        [-half_len, -half_len, 0.0],
    ], dtype=np.float32)

    # --- 3. Solve PnP ---
    success, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not success:
        raise RuntimeError("solvePnP failed to find a pose for this marker")

    # Flatten vectors for convenience
    rvec = rvec.flatten()
    tvec = tvec.flatten()
    return rvec, tvec

def estimate_grid_pose(markers_corners, marker_ids, marker_length, marker_world_vecs, camera_matrix, dist_coeffs):
    """
    Estimate the camera pose in the grid/world frame using the marker with the lowest id.
    Returns (rvec, tvec) of the camera in the grid/world frame.
    """
    # Convert marker_world_vecs to transformation matrices
    marker_world_Ts = {marker_id: vecs_to_matrix(np.array(rvec, dtype=np.float32), np.array(tvec, dtype=np.float32))
                       for marker_id, (rvec, tvec) in marker_world_vecs.items()}

    # Find the marker with the lowest id among those detected
    min_idx = np.argmin(marker_ids.flatten())
    marker_id = int(marker_ids[min_idx][0])
    marker_corners = markers_corners[min_idx]

    # Estimate marker pose in camera frame
    rvec, tvec = estimate_marker_pose(marker_corners, marker_length, camera_matrix, dist_coeffs)
    T_marker_in_cam = vecs_to_matrix(rvec, tvec)
    T_cam_in_marker = np.linalg.inv(T_marker_in_cam)

    # Get the world pose of this marker
    T_marker_in_world = marker_world_Ts[marker_id]

    # Camera pose in world frame
    T_cam_in_world = T_marker_in_world @ T_cam_in_marker

    # Convert to rvec, tvec
    cam_rvec, cam_tvec = matrix_to_vecs(T_cam_in_world)
    return cam_rvec, cam_tvec

def estimate_grid_origin_pose(marker_corners, marker_id, marker_length, marker_world_Ts, camera_matrix, dist_coeffs):
    """
    Estimate the pose (rvec, tvec) of the grid origin (marker 0) in the camera frame,
    using the marker with the lowest visible id.
    """

    # Estimate marker pose in camera frame
    rvec, tvec = estimate_marker_pose(marker_corners, marker_length, camera_matrix, dist_coeffs)
    T_marker_in_cam = vecs_to_matrix(rvec, tvec)

    # Get the world pose of this marker
    T_marker_in_world = marker_world_Ts[marker_id]

    # Compute the transform from grid (marker 0) to camera: T_grid_in_cam = T_marker_in_cam @ inv(T_marker_in_world) @ T_grid_in_world
    T_grid_in_cam = T_marker_in_cam @ np.linalg.inv(T_marker_in_world)

    # Convert to rvec, tvec
    grid_rvec, grid_tvec = matrix_to_vecs(T_grid_in_cam)
    return grid_rvec, grid_tvec

def get_camera_pose(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    t_inv = -R_inv @ tvec.reshape(3, 1)
    cam_rvec, _ = cv2.Rodrigues(R_inv)
    cam_tvec = t_inv.flatten()
    return cam_rvec.flatten(), cam_tvec

# MATRIX AND VECTOR STUFF
def vecs_to_matrix(rvec, tvec):
    """Convert rvec, tvec to a 4x4 transformation matrix."""
    rvec = np.asarray(rvec, dtype=np.float32)
    tvec = np.asarray(tvec, dtype=np.float32)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

def matrix_to_vecs(T):
    """Convert a 4x4 transformation matrix to rvec, tvec."""
    R = T[:3, :3]
    tvec = T[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten(), tvec.flatten()

def rvec_to_euler(rvec):
    """Convert a rotation vector to Euler angles (XYZ convention, radians)."""
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0.0
    return rx, ry, rz

def rotz(angle_rad):
    """
    Create a 4x4 homogeneous rotation matrix for a rotation of angle_rad (radians) about the Z axis.
    """
    Rz = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0, 0, 1]
    ])
    T = np.eye(4)
    T[:3, :3] = Rz
    return T

# GLOBALS
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
Rz90 = rotz(np.pi/2)  # Precomputed rotation matrix for 90 degrees around Z-axis
Rz180 = rotz(np.pi)    # Precomputed rotation matrix for 180 degrees around Z-axis
Rz270 = rotz(3*np.pi/2)  # Precomputed rotation matrix for 270 degrees around Z-axis
Rz90_local = vecs_to_matrix([0, 0, np.pi/2], [0, 0, 0])  # 90° local rotation
Rz180_local = vecs_to_matrix([0, 0, np.pi], [0, 0, 0])  # 180° local rotation
Rz270_local = vecs_to_matrix([0, 0, 3*np.pi/2], [0, 0, 0])  # 270° local rotation
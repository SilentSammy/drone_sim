import numpy as np
import math
import cv2

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

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

def get_camera_pose(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    t_inv = -R_inv @ tvec.reshape(3, 1)
    cam_rvec, _ = cv2.Rodrigues(R_inv)
    cam_tvec = t_inv.flatten()
    return cam_rvec.flatten(), cam_tvec

def vecs_to_matrix(rvec, tvec):
    """Convert rvec, tvec to a 4x4 transformation matrix."""
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

def matrix_to_euler_vecs(T):
    """
    Convert a 4x4 transformation matrix to Euler angles (rx, ry, rz) and translation vector (x, y, z).
    """
    x, y, z = T[:3, 3]
    R = T[:3, :3]
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

    euler_rvec = (rx, ry, rz)
    tvec = (x, y, z)
    return euler_rvec, tvec

# def rt_to_matrix(rvec, tvec):
#     R, _ = cv2.Rodrigues(rvec)
#     T = np.eye(4)
#     T[:3, :3] = R
#     T[:3, 3] = tvec.flatten()
#     return T

# def matrix_to_pose(T):
#     x, y, z = T[:3, 3]
#     R = T[:3, :3]
#     # Convert R to Euler angles (XYZ convention)
#     sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
#     singular = sy < 1e-6
#     if not singular:
#         rx = math.atan2(R[2, 1], R[2, 2])
#         ry = math.atan2(-R[2, 0], sy)
#         rz = math.atan2(R[1, 0], R[0, 0])
#     else:
#         rx = math.atan2(-R[1, 2], R[1, 1])
#         ry = math.atan2(-R[2, 0], sy)
#         rz = 0.0
#     return x, y, z, rx, ry, rz

# def pose_vecs_to_euler(rvec, tvec):
#     """
#     Convert rvec, tvec to (x, y, z, rx, ry, rz) with Euler angles (XYZ convention).
#     """
#     x, y, z = tvec
#     R, _ = cv2.Rodrigues(rvec)
#     sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
#     singular = sy < 1e-6

#     if not singular:
#         rx = math.atan2(R[2, 1], R[2, 2])
#         ry = math.atan2(-R[2, 0], sy)
#         rz = math.atan2(R[1, 0], R[0, 0])
#     else:
#         rx = math.atan2(-R[1, 2], R[1, 1])
#         ry = math.atan2(-R[2, 0], sy)
#         rz = 0.0

#     return x, y, z, rx, ry, rz


def test(frame, drawing_frame, K, D):
    corners, ids = find_arucos(frame, drawing_frame=drawing_frame)

    if ids is not None:
        # Find the index of the lowest id
        min_idx = np.argmin(ids.flatten())
        marker_corners = corners[min_idx]

        # get its pose relative to the camera
        rvec, tvec = estimate_marker_pose(marker_corners, 0.1, K, D)

        # get the camera pose relative to the marker
        cam_rvec, cam_tvec = get_camera_pose(rvec, tvec)

        # convert to (x, y, z, rx, ry, rz)
        cam_pose = pose_vecs_to_euler(cam_rvec, cam_tvec)
        x, y, z, rx, ry, rz = cam_pose

        # display
        if drawing_frame is not None:
            rxd = math.degrees(rx)
            ryd = math.degrees(ry)
            rzd = math.degrees(rz)
            cv2.putText(drawing_frame, f"ID: {ids[0][0]}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(drawing_frame, f"Pos: ({x:.2f}, {y:.2f}, {z:.2f})", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(drawing_frame, f"Rot: ({rxd:.2f}, {ryd:.2f}, {rzd:.2f})", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
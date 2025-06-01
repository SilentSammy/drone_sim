import threading
import numpy as np
import cv2
import input_man as im
import sim_tools as st
from sim_tools import sim
import math
from input_man import is_pressed, get_axis, rising_edge, is_toggled
from video import show_frame, screenshot, record
import vision
from matrix_help import ( reverse_xyz_to_zyx_4x4, extract_euler_zyx, Rx, Ry, Rz, vecs_to_matrix )

# K = np.array([[444,   0, 256], [  0, 444, 256], [  0,   0,   1]], dtype=np.float32)
# D = np.zeros(5)  # [0, 0, 0, 0, 0]
# board = cv2.aruco.CharucoBoard(
#     size=(9, 12),
#     squareLength=0.1,
#     markerLength=0.08,
#     dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
# )

class DroneEstimator:
    def __init__(self, board, K, D=None):
        self.board = board
        self.K = K
        self.D = D if D is not None else np.zeros(5)
        self.detector = cv2.aruco.CharucoDetector(board)

        # For background threading
        self._lock = threading.Lock()
        self._thread_running = False
        self._last_cam_T = None
    
    def fix_camera_transform(self, cam_T):
        """
        Given a 4×4 camera‐to‐board transform cam_T (after rot_x_180 adjustments),
        this will:
          1) apply the “mirror‐Y” step you already had to fix Z
          2) call reverse_xyz_to_zyx_4x4() to fix X and Z
          3) extract the Y Euler angle from the new rotation block, negate it,
             and recompose so that clockwise Y → counterclockwise Y only.
        Returns the edited 4×4 cam_T.
        """
        # 1) Pivot the camera transform around the X axis by 180 degrees, then rotate it along the X axis by 180 degrees.
        cam_T = Rx180 @ cam_T @ Rx180

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

    def get_camera_transform(self, frame, drawing_frame=None):
        # detect the charuco board
        charuco_corners, charuco_ids, marker_corners, marker_ids = self.detector.detectBoard(frame)
        if charuco_ids is None:
            return None

        # get the board pose relative to the camera
        res = vision.get_board_pose(self.board, self.K, self.D, charuco_corners, charuco_ids, center=True)
        if res is None:
            return None
        board_rvec, board_tvec = res

        # Convert rvec, tvec to a 4x4 transformation matrix
        board_T = vecs_to_matrix(board_rvec, board_tvec)

        # get the camera pose relative to the board by inverting the transformation matrix
        cam_T = np.linalg.inv(board_T)

        # Adjust the camera transformation to match our coordinate system
        cam_T = self.fix_camera_transform(cam_T)

        # Display on the drawing frame
        if drawing_frame is not None:
            cv2.aruco.drawDetectedCornersCharuco(drawing_frame, charuco_corners, charuco_ids, cornerColor=(0, 255, 255))
        
        return cam_T

    def get_drone_transform(self, frame, drawing_frame=None):
        return self.get_camera_transform(frame, drawing_frame=drawing_frame)

    def get_drone_transform_nb(self, frame, drawing_frame=None):
            """
            Return the most-recent camera transform. If a background
            thread isn’t running, launch one to compute a new transform.
            While that thread is running, return the last known transform.
            """
            with self._lock:
                # If no thread is active, start one with the current frame
                if not self._thread_running:
                    self._thread_running = True

                    def worker(f, d):
                        try:
                            new_T = self.get_drone_transform(f, drawing_frame=d)
                            with self._lock:
                                self._last_cam_T = new_T
                        finally:
                            with self._lock:
                                self._thread_running = False

                    thread = threading.Thread(
                        target=worker, args=(frame.copy(), None if drawing_frame is None else drawing_frame.copy())
                    )
                    thread.daemon = True
                    thread.start()

                # Return last-known transform (might be None on first call)
                return self._last_cam_T
        
def rot_x_180():
    """Returns a 4x4 matrix for a 180-degree rotation about the X axis."""
    R = np.eye(4)
    R[1, 1] = -1
    R[2, 2] = -1
    return R

Rx180 = rot_x_180()
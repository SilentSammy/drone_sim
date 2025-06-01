import numpy as np
import math
import cv2

# POSE ESTIMATION
def estimate_marker_pose(
    marker_corners,
    marker_length: float,
    camera_matrix,
    dist_coeffs,
    pnp_method=cv2.SOLVEPNP_IPPE_SQUARE
):
    # --- 0. Validate & coerce intrinsics ---
    K = np.asarray(camera_matrix, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"camera_matrix must be 3×3, got {K.shape}")
    D = np.asarray(dist_coeffs, dtype=np.float64).flatten()
    if D.size < 4:
        raise ValueError(f"dist_coeffs must have ≥4 entries, got {D.size}")

    # --- 1. Normalize corner input to a (4,2) float32 array ---
    pts = marker_corners
    # Unwrap lists
    if isinstance(pts, (list, tuple)):
        pts = pts[0]
    pts = np.asarray(pts, dtype=np.float32)
    # Handle shapes like (1,4,2) or (4,1,2)
    if pts.ndim == 3:
        if pts.shape[0] == 1:
            pts = pts[0]
        elif pts.shape[1] == 1:
            pts = pts[:, 0, :]
    pts = pts.reshape(4, 2)

    # --- 2. Build the 3D object points for a square marker of side=marker_length ---
    h = marker_length / 2.0
    objp = np.array([
        [-h,  h, 0],
        [ h,  h, 0],
        [ h, -h, 0],
        [-h, -h, 0]
    ], dtype=np.float32)

    # --- 3. Solve PnP ---
    success, rvec, tvec = cv2.solvePnP(
        objp,       # 3D points in marker frame
        pts,        # 2D image corners
        K,          # camera intrinsics
        D,          # distortion
        flags=pnp_method
    )
    if not success:
        raise RuntimeError("solvePnP failed to find a pose for this marker")

    return rvec.flatten(), tvec.flatten()

def get_camera_pose(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    t_inv = -R_inv @ tvec.reshape(3, 1)
    cam_rvec, _ = cv2.Rodrigues(R_inv)
    cam_tvec = t_inv.flatten()
    return cam_rvec.flatten(), cam_tvec

def get_board_pose(
    board: cv2.aruco.CharucoBoard,
    K: np.ndarray,
    D: np.ndarray,
    charuco_corners: np.ndarray,
    charuco_ids: np.ndarray,
    center: bool = False
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Estimate the Charuco‐board pose, optionally recentering the translation
    so that the board’s center is treated as the origin instead of its top-left corner.

    Parameters
    ----------
    board : cv2.aruco.CharucoBoard
        The ChArUco board object.
    K : array-like of shape (3,3)
        Camera intrinsic matrix.
    D : array-like of length >= 4
        Distortion coefficients.
    charuco_corners : array-like, shape (N,1,2) or (N,2)
        Detected 2D corner positions (pixels) from CharucoDetector.detectBoard().
    charuco_ids : array-like, shape (N,1) or (N,)
        IDs of those detected Charuco corners.
    center : bool
        If True, shift the translation so that the pose’s origin lies at the board’s center
        rather than at its top-left corner.

    Returns
    -------
    (rvec, tvec) : tuple of ndarray
        - rvec : ndarray of shape (3,)
            Rodrigues rotation vector (board→camera).
        - tvec : ndarray of shape (3,)
            Translation vector (board→camera), optionally recentered to the board’s center.
        Returns None if insufficient corners or solvePnP fails.
    """
    # 1. Match 3D–2D points: each obj_pt is a (X,Y,0) in board coords
    obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
    # obj_pts: (N×3), img_pts: (N×2) {{}}  # See Board.matchImagePoints docs

    # 2. Need at least 6 points for the default CV_ITERATIVE solver
    if obj_pts.shape[0] < 6:
        return None

    # 3. SolvePnP: get rotation & translation from board frame → camera frame
    success, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        K,
        D,
        flags=cv2.SOLVEPNP_ITERATIVE
    )  # For n_pts ≥ 6, ITERATIVE is recommended {{:contentReference[oaicite:7]{index=7}}}
    if not success:
        return None

    rvec = rvec.flatten()
    tvec = tvec.flatten()

    # 4. If centering requested, compute the board’s geometric center in board coords:
    if center:
        # 4a. Query the number of squares in each direction
        squaresX, squaresY = board.getChessboardSize()
        sq_len = board.getSquareLength()
        # The farthest interior chess‐corner sits at ((squaresX-1)*sq_len, (squaresY-1)*sq_len, 0)
        # So center (halfway) is:
        center_board = np.array([
            (squaresX - 1) * sq_len / 2.0,
            (squaresY - 1) * sq_len / 2.0,
            0.0
        ], dtype=np.float64)  # {{:contentReference[oaicite:8]{index=8}}}

        # 4b. Convert rvec → rotation matrix R (3×3)
        R_mat, _ = cv2.Rodrigues(rvec)  # board→camera rotation {{:contentReference[oaicite:9]{index=9}}}

        # 4c. Find where that board‐center lives in camera coords:
        #     X_center_cam = R_mat @ center_board + tvec
        offset_cam = R_mat.dot(center_board) + tvec

        # 4d. By overwriting tvec with (offset_cam), we shift origin to board‐center
        tvec = offset_cam

    return rvec, tvec
import math
import matrix_help as mh
from simple_pid import PID

class DroneController:
    def __init__(self):
        # Current drone pose (4×4 transform)
        self.drone_T = None

        # Initialize PID controllers for x, y, z, and yaw (all setpoints = 0)
        # Tuning gains can be adjusted as needed
        self.x_pid   = PID(2.5, 0, 0.25, setpoint=0, output_limits=(-1, 1))
        self.y_pid   = PID(2.5, 0, 0.25, setpoint=0, output_limits=(-1, 1))
        self.z_pid   = PID(2.5, 0, 0.25, setpoint=0, output_limits=(-1, 1))
        self.yaw_pid = PID(2.5, 0, 0.25, setpoint=0, output_limits=(-1, 1))

    def feed_pose(self, drone_T):
        """
        Feed the drone's estimated 4×4 transform to the controller.
        This should be called every time you get a new pose estimate.
        """
        self.drone_T = drone_T

    def move_to(self, x: float, y: float, z: float, yaw: float):
        """
        Compute control outputs (in the drone's local frame) to move toward
        the desired global pose (x, y, z, yaw). Returns (x_control, y_control, z_control, w_control).

        Parameters
        ----------
        x, y, z : float
            Desired global position.
        yaw : float
            Desired global yaw (radians).

        Returns
        -------
        x_control : float
            Thrust/control in the drone's local X direction (forward).
        y_control : float
            Thrust/control in the drone's local Y direction (right).
        z_control : float
            Thrust/control in the drone's local Z direction (up).
        w_control : float
            Yaw‐control (positive = rotate CCW in local frame).
        """
        if self.drone_T is None:
            # No pose yet; output zero controls
            return 0.0, 0.0, 0.0, 0.0

        # 1) Extract current drone pose: rotation vector (Euler) and translation
        d_rvec, d_tvec = mh.matrix_to_vecs(self.drone_T)
        current_x   = d_tvec[0]
        current_y   = d_tvec[1]
        current_z   = d_tvec[2]
        current_yaw = d_rvec[2]  # assume matrix_to_vecs gives Euler [roll, pitch, yaw]

        # 2) Compute global errors (desired minus current)
        x_err   = x - current_x
        y_err   = y - current_y
        z_err   = z - current_z
        yaw_err = _normalize_angle(yaw - current_yaw)

        # 3) Convert the global (x, y) error into the drone's local frame
        x_err, y_err = _global_to_local(x_err, y_err, -current_yaw)

        # 4) Feed errors into PID controllers (setpoint is zero, so pass error directly)
        #    Each PID returns a control in the local frame
        x_control = self.x_pid(x_err)    # forward/back
        y_control = self.y_pid(-y_err)    # right/left
        z_control = self.z_pid(-z_err)        # up/down
        w_control = self.yaw_pid(yaw_err)    # yaw rate

        return y_control, x_control, z_control, w_control

def _normalize_angle(angle: float) -> float:
    """
    Normalize an angle to the range (–π, +π].
    """
    a = (angle + math.pi) % (2 * math.pi)
    if a <= 0:
        a += 2 * math.pi
    return a - math.pi

def _global_to_local(x_global: float, y_global: float, yaw: float) -> tuple[float, float]:
    """
    Convert a vector from the global frame to the drone’s local frame, given the drone’s yaw.

    Parameters
    ----------
    x_global : float
        X component in the global coordinate frame.
    y_global : float
        Y component in the global coordinate frame.
    yaw : float
        Drone’s yaw angle (radians) measured from the global X axis (positive CCW).

    Returns
    -------
    x_local : float
        X component in the drone’s local frame.
    y_local : float
        Y component in the drone’s local frame.
    """
    c = math.cos(yaw)
    s = math.sin(yaw)

    # Rotate the global (x, y) by –yaw to get local coordinates:
    x_local =  c * x_global + s * y_global
    y_local = -s * x_global + c * y_global
    return x_local, y_local

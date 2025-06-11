import math
import matrix_help as mh
from simple_pid import PID

class DroneController:
    def __init__(self):
        # Current drone pose (4×4 transform)
        self.drone_T = None

        # Initialize PID controllers for x, y, z, and yaw (all setpoints = 0)
        # Tuning gains can be adjusted as needed
        self.x_pid   = PID(2.5, 0, 0, setpoint=0, output_limits=(-1, 1))
        self.y_pid   = PID(2.5, 0, 0, setpoint=0, output_limits=(-1, 1))
        self.z_pid   = PID(2.5, 0, 0, setpoint=0, output_limits=(-1, 1))
        self.yaw_pid = PID(2.5, 0, 0, setpoint=0, output_limits=(-1, 1))

    def feed_pose(self, drone_T):
        """
        Feed the drone's estimated 4×4 transform to the controller.
        This should be called every time you get a new pose estimate.
        """
        self.drone_T = drone_T

    def move_to(self, x: float, y: float, z: float, yaw: float, debug: bool = False):
        """
        Compute control outputs (in the drone's local frame) to move toward
        the desired global pose (x, y, z, yaw). Returns (x_control, y_control, z_control, w_control).
        """
        if self.drone_T is None:
            if debug:
                print("[DroneController] No pose available. Returning zero controls.")
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
        x_err_local, y_err_local = _global_to_local(x_err, y_err, -current_yaw)

        # 4) Feed errors into PID controllers (setpoint is zero, so pass error directly)
        #    Each PID returns a control in the local frame
        x_control = self.x_pid(x_err_local)    # forward/back
        y_control = self.y_pid(-y_err_local)   # right/left
        z_control = self.z_pid(-z_err)         # up/down
        w_control = self.yaw_pid(yaw_err)      # yaw rate

        if debug:
            print(f"\n[DroneController] move_to Debug")
            print(f"Target: x={x}, y={y}, z={z}, yaw={yaw:.2f}")
            print(f"Current: x={current_x:.2f}, y={current_y:.2f}, z={current_z:.2f}, yaw={current_yaw:.2f}")
            print(f"Errors (global): x_err={x_err:.3f}, y_err={y_err:.3f}, z_err={z_err:.3f}, yaw_err={yaw_err:.3f}")
            print(f"Errors (local): x_err_local={x_err_local:.3f}, y_err_local={y_err_local:.3f}")
            print(f"Control outputs: y_ctrl={y_control:.3f}, x_ctrl={x_control:.3f}, z_ctrl={z_control:.3f}, w_ctrl={w_control:.3f}")
            print(f"--------------------------------\n")

        return y_control, x_control, z_control, w_control


class TrajectoryFollower:
    def __init__(self, waypoints=None, loop=False, auto_advance=True, position_tolerance=0.2, yaw_tolerance=math.radians(30)):
        self.controller = DroneController()
        self.waypoints = waypoints or default_waypoints
        self.loop = loop
        self.auto_advance = auto_advance
        self.position_tolerance = position_tolerance
        self.yaw_tolerance = yaw_tolerance
        self.current_idx = 0

    @property
    def done(self):
        """True if all waypoints have been reached (only if loop is False)."""
        return (not self.loop) and (self.current_idx >= len(self.waypoints) - 1) and self.reached
    
    @property
    def reached(self):
        """True if the current waypoint is reached (live check)."""
        if not self.waypoints or self.controller.drone_T is None:
            return False

        x, y, z, yaw = self.waypoints[self.current_idx]
        d_rvec, d_tvec = self.controller.drone_T[:3, :3], self.controller.drone_T[:3, 3]
        pos_dist = math.sqrt((x - d_tvec[0])**2 + (y - d_tvec[1])**2 + (z - d_tvec[2])**2)
        try:
            current_yaw = math.atan2(d_rvec[1,0], d_rvec[0,0])
        except Exception:
            current_yaw = 0.0
        yaw_err = abs(_normalize_angle(yaw - current_yaw))

        # Optionally, you may want to cache the last control outputs for stability check
        # ctrl_thresh = 0.05
        # Use last control outputs if available, else assume not stable
        # last_ctrls = getattr(self, "_last_ctrls", (float('inf'),)*4)
        # stable = all(abs(val) < ctrl_thresh for val in last_ctrls)

        # return pos_dist < self.position_tolerance and yaw_err < self.yaw_tolerance and stable
        return pos_dist < self.position_tolerance and yaw_err < self.yaw_tolerance

    @property
    def current_waypoint(self):
        if self.current_idx < len(self.waypoints):
            return self.waypoints[self.current_idx]
        return None

    def feed_pose(self, drone_T):
        self.controller.feed_pose(drone_T)

    def start_next(self):
        if self.current_idx < len(self.waypoints) - 1:
            self.current_idx += 1
        elif self.loop:
            self.current_idx = 0

    def move(self, debug: bool = True):
        if not self.waypoints:
            print("No waypoints set.")
            return 0.0, 0.0, 0.0, 0.0

        x, y, z, yaw = self.waypoints[self.current_idx]
        y_ctrl, x_ctrl, z_ctrl, w_ctrl = self.controller.move_to(x, y, z, yaw, debug=debug)

        # Cache last control outputs for stability check in reached property
        self._last_ctrls = (y_ctrl, x_ctrl, z_ctrl, w_ctrl)

        # Auto-advance if reached
        if self.reached and self.auto_advance:
            if debug:
                print("Waypoint reached and stable. Advancing to next waypoint...")
            self.start_next()

        return y_ctrl, x_ctrl, z_ctrl, w_ctrl


class Choreographer:
    """
    Synchronizes a list of TrajectoryFollowers so that all must reach their current waypoint
    before any are allowed to advance to the next.
    """
    def __init__(self, followers):
        """
        followers: list of TrajectoryFollower instances
        """
        self.followers = followers

        # Disable auto-advance for all followers
        for f in self.followers:
            f.auto_advance = False

    def check(self):
        """
        Call this in your main loop. If all followers have reached their current waypoint,
        advances all to the next waypoint.
        """
        if all(f.reached for f in self.followers):
            print("All followers reached their current waypoint. Advancing to next waypoint...")
            for f in self.followers:
                f.start_next()


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

# Yaw-only trajectory: drone stays at (0, 0, 1.0) and rotates in place
default_waypoints = [
    (0.0, 0.0, 1.0, 0.0),           # Start at yaw 0
    (0.0, 0.0, 1.0, math.pi/2),     # Rotate to 90 degrees
    (0.0, 0.0, 1.0, math.pi),       # Rotate to 180 degrees
    (0.0, 0.0, 1.0, -math.pi/2),    # Rotate to -90 degrees
    (0.0, 0.0, 1.0, 0.0),           # Return to yaw 0
]

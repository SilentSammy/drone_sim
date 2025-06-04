import numpy as np
import math

class Drone:
    def __init__(self, get_frame=None, K=None, D=None):
        self.flight = True
        self.prev_x = 0
        self.prev_y = 0
        self.prev_w = 0
        self.prev_z = 0
        self.transl_mag = 0.1
        self.thr_mag = 0.15
        self.yaw_mag = math.radians(15)
        self._cam_idx = 0
        self.K = K
        self.D = D
        self._get_frame = get_frame
        self._dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    @property
    def cam_idx(self):
        return self._cam_idx
    @cam_idx.setter
    def cam_idx(self, value):
        self._cam_idx = value
        self._change_camera(value)

    def send_rc(self, x, y, z, w):
        if (x == self.prev_x and y == self.prev_y and z == self.prev_z and w == self.prev_w):
            return
        
        if not self.flight:
            print("Drone is not in flight mode. Please take off first.")
            return

        self.prev_x = x
        self.prev_y = y
        self.prev_w = w
        self.prev_z = z

        x = x * self.transl_mag
        y = y * self.transl_mag
        z = z * self.thr_mag
        w = w * self.yaw_mag
        self._apply_rc(x, y, z, w)

    # --- ABSTRACT METHODS ---
    def _apply_rc(self, x, y, z, w):
        print(f"Applying RC: x={x}, y={y}, z={z}, w={w}")

    def get_frame(self):
        if self._get_frame is not None:
            return self._get_frame()
        return self._dummy_frame

    # --- OPTIONAL METHODS ---
    def _change_camera(self, cam_idx):
        print(f"Changing camera to {cam_idx}...")

    def takeoff(self):
        print("Taking off...")
    
    def land(self):
        print("Landing...")
    
    def flip(self, direction):
        print(f"Flipping {direction}...")

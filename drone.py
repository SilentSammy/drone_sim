import math

class Drone:
    def __init__(self):
        self.flight = False
        self.prev_x = 0
        self.prev_y = 0
        self.prev_w = 0
        self.prev_z = 0
        self.transl_mag = 0.1
        self.thr_mag = 0.15
        self.yaw_mag = math.radians(15)

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
        raise NotImplementedError("_apply_movement must be implemented by subclasses")

    def get_frame(self):
        raise NotImplementedError("get_frame must be implemented by subclasses")

    # OPTIONAL METHODS
    def change_camera(self, cam_idx):
        print(f"Changing camera to {cam_idx}...")

    def takeoff(self):
        print("Taking off...")
    
    def land(self):
        print("Landing...")
    
    def flip(self, direction):
        print(f"Flipping {direction}...")

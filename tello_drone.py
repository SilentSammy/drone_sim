import cv2
from djitellopy import Tello
from drone import Drone

class TelloDrone(Drone):
    def __init__(self, ip=None):
        super().__init__()
        
        # Magnitudes for tello drone control
        self.transl_mag = 100
        self.thr_mag = 100
        self.yaw_mag = 100

        # Initialize Tello drone
        self.tello = Tello(host=ip) if ip else Tello()
        self.tello.connect()
        self.tello.streamon()

        # The Tello drone starts landed
        self.flight = False

        # Query battery
        bat = self.tello.query_battery()
        print("Battery:", bat)

    # MANDATORY METHODS
    def get_frame(self):
        frame = self.tello.get_frame_read().frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def _apply_rc(self, x, y, z, w):
        # Convert to right-handed coordinate system
        x= int(x)
        y = -int(y)
        z = int(z)
        w = -int(w)

        self.tello.send_rc_control(
            forward_backward_velocity=x,
            left_right_velocity=y,
            up_down_velocity=z,
            yaw_velocity=w
        )
    
    # OPTIONAL METHODS
    def change_camera(self, cam_idx):
        if cam_idx % 2 == 0:
            self.tello.set_video_direction(Tello.CAMERA_FORWARD)
        else:
            self.tello.set_video_direction(Tello.CAMERA_DOWNWARD)

    def takeoff(self):
        if not self.flight:
            print("Tello drone taking off...")
            self.tello.takeoff()
            self.flight = True
            print("Tello drone took off!")
    
    def land(self):
        if self.flight:
            self.tello.land()
            self.flight = False
            print("Tello drone landing...")

    def flip(self, direction):
        self.tello.flip(direction)

    def __del__(self):
        if self.flight:
            self.tello.land()
        self.tello.streamoff()
        self.tello.end()
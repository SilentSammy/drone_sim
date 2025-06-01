import cv2
import numpy as np
import input_man as im
import sim_tools as st
from sim_tools import sim
import math
from drone import Drone

class SimDrone(Drone):
    def __init__(
        self,
        start_sim=True,
        forward_cam_path='/Quadcopter/visionSensor[0]',
        down_cam_path='/Quadcopter/visionSensor[1]',
        target_path='/Quadcopter/base/target'
    ):
        super().__init__()

        # Magnitudes for simulated drone control
        self.transl_mag = 0.1
        self.thr_mag = 0.15
        self.yaw_mag = math.radians(15)

        # Initialize camera parameters
        self.cams = [
            ( sim.getObject(forward_cam_path), np.array([[624,   0, 480], [  0, 624, 360], [  0,   0,   1]], dtype=np.float32) ),
            ( sim.getObject(down_cam_path), np.array([[444,   0, 256], [  0, 444, 256], [  0,   0,   1]], dtype=np.float32) ),
        ]
        self.cam = self.cams[0][0]  # Start with the forward camera
        self.K = self.cams[0][1]  # Forward camera intrinsic matrix
        self.D = np.zeros(5)  # [0, 0, 0, 0, 0]

        # Start the simulation
        self.target = sim.getObject(target_path)
        if start_sim:
            sim.startSimulation()

        # The simulated drone is always in flight mode unlike the real drone
        self.flight = True

    def _change_camera(self, cam_idx):
        self.cam = self.cams[cam_idx % len(self.cams)][0]
        self.K = self.cams[cam_idx % len(self.cams)][1]
        print("Changing camera to", "forward" if cam_idx % 2 == 0 else "down")

    def _apply_rc(self, x, y, z, w):
        st.move_object_local(self.target, x=x, y=y, z=z)
        st.orient_object_local(self.target, gamma=w)
    
    def get_frame(self):
        frame = st.get_image(self.cam)
        return frame

    def __del__(self):
        sim.stopSimulation()
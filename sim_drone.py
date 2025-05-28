import cv2
import input_man as im
import sim_tools as st
from sim_tools import sim
import math
from drone import Drone

class SimDrone(Drone):
    def __init__(self):
        super().__init__()
        
        # Magnitudes for simulated drone control
        self.transl_mag = 0.1
        self.thr_mag = 0.15
        self.yaw_mag = math.radians(15)

        # Start the simulation
        self.forw_cam = sim.getObject('/Quadcopter/visionSensor[0]')
        self.down_cam = sim.getObject('/Quadcopter/visionSensor[1]')
        self.cam = self.forw_cam
        self.target = sim.getObject('/target')
        sim.startSimulation()
        
        # The simulated drone is always in flight mode unlike the real drone
        self.flight = True

    def _change_camera(self, cam_idx):
        if cam_idx % 2 == 0:
            self.cam = self.forw_cam
        else:
            self.cam = self.down_cam
        print("Changing camera to", "forward" if cam_idx % 2 == 0 else "down")

    def _apply_rc(self, x, y, z, w):
        st.move_object_local(self.target, x=x, y=y, z=z)
        st.orient_object_local(self.target, gamma=w)
    
    def get_frame(self):
        frame = st.get_image(self.cam)
        return frame

    def __del__(self):
        sim.stopSimulation()
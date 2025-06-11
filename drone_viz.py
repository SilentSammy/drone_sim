import sim_tools as st
from sim_tools import sim
from drone_est import DroneEstimator

def visualize_drone_pose(drone_T):
    if dummy_drone is None or drone_T is None:
        return
    st.set_matrix(dummy_drone, drone_T)

def visualize_ball_pose(ball_pos_xy):
    if dummy_ball is None or ball_pos_xy is None:
        return
    sim.setObjectPosition(dummy_ball, sim.handle_world, [ball_pos_xy[0], -ball_pos_xy[1], 0.05])  # Set Z to a small value above the ground

# Attempt to get the drone representation object
try:
    dummy_drone = sim.getObject('/PoseViz/Dummy')
except Exception as e:
    print("Error getting Dummy drone object. Try to re-open the CoppeliaSim scene and restart.", e)
    dummy_drone = None

# Attempt to get the ball representation object
try:
    dummy_ball = sim.getObject('/PoseViz/Sphere')
except Exception as e:
    print("Error getting Dummy ball object. Try to re-open the CoppeliaSim scene and restart.", e)
    dummy_ball = None

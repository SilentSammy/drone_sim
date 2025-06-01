import sim_tools as st
from sim_tools import sim
from drone_est import DroneEstimator

def visualize_drone_pose(drone_T):
    if dummy_drone is None or drone_T is None:
        return
    T_coppelia = st.np_to_coppelia_T(drone_T)
    sim.setObjectMatrix(dummy_drone, T_coppelia, sim.handle_world)

# Attempt to get the drone representation object
try:
    dummy_drone = sim.getObject('/PoseViz/Dummy')
except Exception as e:
    print("Error getting Dummy drone object. Try to re-open the CoppeliaSim scene and restart.", e)
    dummy_drone = None

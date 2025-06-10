from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from pynput import keyboard
from simple_pid import PID
import numpy as np
import time
import math
import cv2

client = RemoteAPIClient('localhost', 23000)
sim = client.getObject('sim')

def orient_object(object_handle, alpha=None, beta=None, gamma=None):
    """Sets an object's orientation to specific angles (in radians)."""
    orientation = sim.getObjectOrientation(object_handle, -1)
    orientation[2] = gamma if gamma is not None else orientation[2]
    orientation[1] = beta if beta is not None else orientation[1]
    orientation[0] = alpha if alpha is not None else orientation[0]
    sim.setObjectOrientation(object_handle, -1, orientation)

def orient_object_local(object_handle, alpha=None, beta=None, gamma=None):
    """Sets an object's orientation to specific angles (in radians) relative to its parent frame."""
    parent_handle = sim.getObjectParent(object_handle)
    orientation = sim.getObjectOrientation(object_handle, parent_handle)
    orientation[0] = alpha if alpha is not None else orientation[0]
    orientation[1] = beta if beta is not None else orientation[1]
    orientation[2] = gamma if gamma is not None else orientation[2]
    sim.setObjectOrientation(object_handle, parent_handle, orientation)

def move_object(object_handle, x=None, y=None, z=None):
    """Teleports an object to a specific position."""
    position = sim.getObjectPosition(object_handle, -1)
    position[0] = x if x is not None else position[0]
    position[1] = y if y is not None else position[1]
    position[2] = z if z is not None else position[2]
    sim.setObjectPosition(object_handle, -1, position)

def move_object_local(object_handle, x=None, y=None, z=None):
    """Teleports an object to a specific position relative to its parent frame."""
    parent_handle = sim.getObjectParent(object_handle)
    position = sim.getObjectPosition(object_handle, parent_handle)
    position[0] = x if x is not None else position[0]
    position[1] = y if y is not None else position[1]
    position[2] = z if z is not None else position[2]
    sim.setObjectPosition(object_handle, parent_handle, position)

def translate_object(object_handle, x=0, y=0, z=0):
    """Adds to an object's position."""
    position = sim.getObjectPosition(object_handle, -1)
    sim.setObjectPosition(object_handle, -1, [position[0] + x, position[1] + y, position[2] + z])

def rotate_object(object_handle, alpha=0, beta=0, gamma=0):
    """Adds to an object's orientation (in radians)."""
    orientation = sim.getObjectOrientation(object_handle, -1)
    sim.setObjectOrientation(object_handle, -1, [orientation[0] + alpha, orientation[1] + beta, orientation[2] + gamma])

def np_to_coppelia_T(T):
    # T is a 4x4 numpy array
    # CoppeliaSim expects columns of the rotation matrix and the translation vector
    return [
        T[0,0], T[1,0], T[2,0], T[0,3],  # first column + x
        T[0,1], T[1,1], T[2,1], T[1,3],  # second column + y
        T[0,2], T[1,2], T[2,2], T[2,3],  # third column + z
    ]

def get_image(vision_sensor_handle):
    sim.handleVisionSensor(vision_sensor_handle)
    img, resolution = sim.getVisionSensorImg(vision_sensor_handle)
    img = np.frombuffer(img, dtype=np.uint8).reshape((resolution[1], resolution[0], 3))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.flip(img, 0)
    return img

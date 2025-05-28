import numpy as np
import math
import cv2

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

def find_arucos(frame, drawing_frame=None):
    # Detect markers using the new ArucoDetector API
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None and drawing_frame is not None:
        cv2.aruco.drawDetectedMarkers(drawing_frame, corners, ids)
    
    return corners, ids
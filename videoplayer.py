import os
import numpy as np
import math
import cv2
import time
import keybrd
from collections import deque
from ball_detector import BallDetector
from video import show_frame, screenshot, record, VideoPlayer

bd = BallDetector()

pipeline = [
    ("To Gray", lambda: bd.to_gray(frame, drawing_frame)),
    ("Canny", lambda: bd.get_edges(frame, drawing_frame)),
    ("Dilate Edges", lambda: bd.dilate_edges(frame, drawing_frame)),
    ("Find Black Contours", lambda: bd.find_black_contours(frame, drawing_frame)),
    ("Fit Ellipses", lambda: bd.fit_ellipses(frame, drawing_frame)),
    ("Find Best Ellipse", lambda: bd.find_best_ellipse(frame, drawing_frame))
]

if __name__ == "__main__":
    import keybrd
    vp = VideoPlayer(r"videos\output_2025-06-03_18-51-57.mp4")  # Path to the video file
    # vp = VideoPlayer(cv2.VideoCapture(r"http://192.168.137.141:4747/video"))  # Use the camera stream
    re = keybrd.rising_edge # Function to check if a key is pressed once
    pr = keybrd.is_pressed  # Function to check if a key is held down
    tg = keybrd.is_toggled  # Function to check if a key is toggled
    layers = pipeline
    layer = 1
    
    while True:
        # Get current frame
        vp.time_step()
        vp.move(1 if pr('d') else -1 if pr('a') else 0)  # Move forward/backward
        vp.move((1 if pr('e') else -1 if pr('q') else 0) * 10)  # Fast forward/backward
        vp.step(1 if re('w') else -1 if re('s') else 0)  # Step forward/backward
        mask = None
        frame = vp.get_frame()
        drawing_frame = frame.copy()

        # Optional screenshot or recording
        if re('p'):
            screenshot(frame)
        record(frame if tg('o') else None)

        # Print the current frame
        print(f"Frame {vp.frame_idx}/{vp.frame_count} ", end='')

        # Choose layer to show
        for i in range(1, 10):
            if re(str(i)):
                layer = i
                break

        # Choose the layer to show. Layer 1 is do nothing. Layer 2 is index 0 in the pipeline, etc.
        if layer >= 2 and layer <= len(layers) + 1:
            name, func = layers[layer - 2]
            print(name, end=', ')
            func()

        print()
    
        if re('p'): # Save the current frame as an image.
            output_file = f"frame_{vp.frame_idx}_layer_{layer}.png"
            cv2.imwrite(output_file, drawing_frame)
            print(f"Saved frame {vp.frame_idx} as {output_file}")

        # Show
        vp.show_frame(drawing_frame, "Frame")

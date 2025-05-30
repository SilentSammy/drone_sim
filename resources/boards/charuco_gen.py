#!/usr/bin/env python3
"""
Render a 2×2 ArUco GridBoard to a PNG image.
"""

import cv2
import os

def main():
    # 1) Load your dictionary and build the board
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard(
        size=(9, 12),
        squareLength=0.1,
        markerLength=0.08,
        dictionary=aruco_dict
    )

    # 2) Draw the board into a square image (600×600 px here)
    img_size = 1200  # you can change this to scale your output
    board_img = board.generateImage(
        (img_size, img_size),  # size of the output image
        marginSize=0          # margin around the board in pixels
    )

    # 3) Save alongside this script
    out_file = os.path.join(os.path.dirname(__file__), "gridboard.png")
    cv2.imwrite(out_file, board_img)
    print(f"Saved board image to: {out_file}")

if __name__ == "__main__":
    main()

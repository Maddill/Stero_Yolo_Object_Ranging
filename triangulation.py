import sys
import cv2
import numpy as np
import time


def find_depth(right_point, left_point, frame_right, frame_left, baseline, f, alpha):

    # CONVERT FOCAL LENGTH f FROM [mm] TO [pixel]:
    height_right, width_right, depth_right = frame_right.shape[0:3]
    height_left, width_left, depth_left = frame_left.shape[0:3]

    if width_right == width_left:
        f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi/180)
        # f_pixel = 640

    else:
        print('Left and right camera frames do not have the same pixel width')

    x_right = right_point[0]
    x_left = left_point[0]

    # CALCULATE THE DISPARITY:
    # Displacement between left and right frames [pixels]
    disparity = x_left-x_right

    # CALCULATE DEPTH z:
    zDepth = ((baseline*f_pixel)/ (2*np.tan(alpha/2)*disparity))/100 if disparity != 0 else 0 # Depth in [cm]
    # zDepth = (baseline*f_pixel / disparity) if disparity != 0 else 0
    return np.abs(zDepth)

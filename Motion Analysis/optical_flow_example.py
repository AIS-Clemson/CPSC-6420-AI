#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:59:31 2024

@author: maxgray
"""

import cv2
import numpy as np

# Path to the video file
video_path = "dataset/JP_10.MOV"  # Replace with your video path

# Open the video
cap = cv2.VideoCapture(video_path)

# Read the first frame and convert it to grayscale
ret, frame1 = cap.read()
if not ret:
    print("Error: Cannot read the video.")
    cap.release()
    exit()

gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    # Read the next frame
    ret, frame2 = cap.read()
    if not ret:
        break

    # Convert the second frame to grayscale
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    # Compute the magnitude and angle of the flow
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Create a mask for visualizing the flow
    hsv_mask = np.zeros_like(frame1)
    hsv_mask[..., 1] = 255

    # Use angle to set hue and magnitude to set value
    hsv_mask[..., 0] = angle * 180 / np.pi / 2
    hsv_mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV to BGR for visualization
    # rgb_flow = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
    rgb_flow = cv2.applyColorMap(hsv_mask[..., 2], cv2.COLORMAP_JET)


    # Display the optical flow visualization
    cv2.imshow("Dense Optical Flow", rgb_flow)

    # Update the previous frame
    gray1 = gray2

    # Break on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

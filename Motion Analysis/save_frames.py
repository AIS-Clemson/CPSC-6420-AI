#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:54:21 2024

@author: maxgray
"""

import cv2

# Path to the video file
video_path = "dataset/JP_10.MOV"  # Replace with your video path

# Open the video
cap = cv2.VideoCapture(video_path)

# Read the first two frames
ret, frame1 = cap.read()
if ret:
    cv2.imwrite("frame_1.png", frame1)  # Save the first frame

ret, frame2 = cap.read()
if ret:
    cv2.imwrite("frame_2.png", frame2)  # Save the second frame

cap.release()
print("Frames saved as frame1.jpg and frame2.jpg.")

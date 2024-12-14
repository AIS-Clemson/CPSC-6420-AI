# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:44:10 2024

@author: MaxGr
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

import cv2
import numpy as np
from collections import deque

# video_path = "dataset/JP_10.MOV"  # Replace with your video path
# video_path = 'Videos/Bruno Mars - Treasure (Official Music Video).mp4'
# video_path = 'Videos/Tokyo Japan - Shinjuku Summer Night Walk 2024 • 4K HDR.mp4'
video_path = 'Videos/The Wildest POV Videos From 2021.mp4'

# Take first frame and find corners in it
cap = cv2.VideoCapture(video_path)
ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
img_height, img_width = gray1.shape


# Parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=1000, qualityLevel=0.1, minDistance=7, blockSize=7)
# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


# Create a mask image for drawing purposes
mask = np.zeros_like(frame1)
keypoints_list = deque()

scale = 1.0

frame_id = 0
while cap.isOpened():
    ret, frame2 = cap.read()
    print(frame_id)
    frame_id += 1
    if not ret: break
    # if frame_id % 5 in [1,2,3,4]: continue

    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    mask.fill(0)

    # Calculate optical flow
    p1 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
    p2, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p1, None, **lk_params)
    
    # if p1 is None: continue
    
    # Select good points
    num_keypoints = 1000
    good_new = p2[st == 1][:num_keypoints]
    good_old = p1[st == 1][:num_keypoints]
    magnitude = err[st == 1][:num_keypoints]
    
    # Draw the tracks
    for i, (new, old, mag) in enumerate(zip(good_new, good_old, magnitude)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        cv2.circle(mask, (a, b), 5, (0, 0, 255), -1)
        # cv2.circle(mask, (a, b), 3, (0, 255, 0), -1)
        cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)

        keypoints_list.append([a,b,c,d])
        if len(keypoints_list) % 1000 == 0:
            keypoints_list.popleft()
            
    for points in keypoints_list:
        a,b,c,d = points
        # cv2.circle(mask, (c, d), 1, (0, 255, 0), -1)
        # cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)

    img = cv2.add(frame2, mask)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # cv2.putText(img,f'{frame_id}',(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color[mark_danger], 2, cv2.LINE_AA)
    cv2.imshow('Sparse Optical Flow', img)

    if cv2.waitKey(15) & 0xFF == ord('q'):
        break
    
    # Now update the previous frame and previous points
    gray1 = gray2
    
cap.release()
cv2.destroyAllWindows()

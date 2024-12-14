# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 17:49:03 2024

@author: MaxGr
"""

import cv2
import numpy as np

# Path to the video file
# video_path = "path/to/your/video.mp4"  # Replace with your video path
# video_path = 'Videos/Tokyo Japan - Shinjuku Summer Night Walk 2024 • 4K HDR.mp4'
# video_path = 'Videos/15 Most Unbelievable Traffic Camera Moments.mp4'
# video_path = 'Videos/Shopping, People, Commerce, Mall, Many, Crowd, Walking   Free Stock video footage   YouTube.webm'
# video_path = 'Videos/VIDEVO - Static Times Square Shot.mp4'
# video_path = 'Videos/4K Hongya Cave, Chongqing (Static Shot).mp4'
video_path = 'Videos/Static shot of a busy road in Copenhagen, Denmark.webm'

# Open the video
cap = cv2.VideoCapture(video_path)

# Read the first frame
ret, frame1 = cap.read()
if not ret:
    print("Error: Cannot read the video.")
    cap.release()
    exit()

# Convert the first frame to grayscale
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    ret, frame2 = cap.read()
    if not ret:
        break

    # Convert the second frame to grayscale
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate frame difference
    diff = cv2.absdiff(gray1, gray2)

    # Threshold the difference to create a binary mask
    _, saliency_map = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Apply colormap for visualization
    saliency_colored = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    # saliency_colored = cv2.applyColorMap(saliency_map, cv2.COLORMAP_BONE)


    # Combine the saliency map with the original frame
    combined_frame = cv2.addWeighted(frame2, 0.2, saliency_colored, 0.9, 0)

    # Display the saliency map and combined frame
    # cv2.imshow("Saliency Map", saliency_colored)
    cv2.imshow("Combined Frame with Saliency", combined_frame)

    # Update the previous frame
    gray1 = gray2

    # Break on pressing 'q'
    if cv2.waitKey(15) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

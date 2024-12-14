# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 15:09:24 2024

@author: MaxGr
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO
import torch
print(torch.cuda.is_available())

# Determine the appropriate device (supports MPS for Apple devices)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
print('Use device: ', device)
    
# Choose your model here, I suggest use yolo11s
model = YOLO("yolo11s.pt")
# video_path = "path/to/video.mp4"
# video_path = "dataset/JP_10.MOV"  # Replace with your video path
video_path = 'Videos/Tokyo Japan - Shinjuku Summer Night Walk 2024 • 4K HDR.mp4'

cap = cv2.VideoCapture(video_path)
track_history = defaultdict(lambda: [])

scale = 1

frame_idx = 0
while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1
    if frame_idx%5 in [1,2,3,4]: continue
    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if success:
        results = model.track(frame, persist=True)
        boxes = results[0].boxes.xywh.cpu()
        if len(boxes)>1:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            annotated_frame = results[0].plot(line_width=1)
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                # cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
        cv2.imshow("YOLO11 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()









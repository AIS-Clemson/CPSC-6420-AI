#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:38:25 2024

@author: maxgray
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

import torch
from transformers import DPTForDepthEstimation, DPTFeatureExtractor
from PIL import Image
import cv2
import numpy as np

# Determine the appropriate device (supports MPS for Apple devices)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load the DPT model and feature extractor from Hugging Face
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device)
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

# Helper function to visualize depth map
def normalize_depth(depth):
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = (depth * 255).astype(np.uint8)
    return depth

# ** Single Image Input **
image_path = "image_5.webp"  # Replace with your image path
image = Image.open(image_path).convert("RGB")

# Preprocess image
inputs = feature_extractor(images=image, return_tensors="pt").to(device)

# Predict depth
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth.squeeze().cpu().numpy()

# Normalize depth map
normalized_depth = normalize_depth(predicted_depth)

# Display depth map using OpenCV
cv2.imshow("Depth Map - Single Image", cv2.applyColorMap(normalized_depth, cv2.COLORMAP_PLASMA))
cv2.waitKey(0)
cv2.destroyAllWindows()

# ** Video Input **
video_path = "dataset/JP_10.MOV"  # Replace with your video path
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL Image
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Preprocess frame
    inputs = feature_extractor(images=pil_frame, return_tensors="pt").to(device)

    # Predict depth
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth.squeeze().cpu().numpy()

    # Normalize depth map for visualization
    normalized_depth = normalize_depth(predicted_depth)
    # depth_map_colored = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_PLASMA)
    depth_map_colored = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)

    # Combine original frame and depth map for display
    depth_map_colored = cv2.resize(depth_map_colored, frame.shape[:2][::-1])
    combined_frame = cv2.addWeighted(frame, 0.2, depth_map_colored, 0.8, 0)

    # Display the depth map and combined frame
    # cv2.imshow("Depth Map - Video", depth_map_colored)
    cv2.imshow("Combined Frame - Video", combined_frame)

    # Break on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




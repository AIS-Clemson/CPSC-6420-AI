import cv2
import numpy as np
from collections import deque
import os
from datetime import datetime
from helper_functions import img_uint8, img_resize, generate_gaussian_image

import torch

# Load YOLOv5 model for pet detection
def load_yolo_model(weights_path):
    # Load the YOLOv5 model using PyTorch
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    return model

# Perform pet detection on a frame
def detect_pets(frame, model, confidence_threshold=0.1):
    # Run inference using the PyTorch YOLO model
    results = model(frame)  # YOLO model processes the frame directly
    detections = results.xyxy[0].cpu().numpy()  # Extract detections in (x1, y1, x2, y2, confidence, class) format

    boxes = []
    for detection in detections:
        confidence = detection[4]
        if confidence > confidence_threshold:
            # Extract bounding box coordinates
            box = detection[:4].astype("int")
            boxes.append(box)
    return boxes

# Update heat map with detected pet locations
def update_heat_map(heat_map, boxes, image_size, sigma=10):
    for box in boxes:
        x_center = (box[0] + box[2]) // 2
        y_center = (box[1] + box[3]) // 2
        gaussian = generate_gaussian_image(image_size, image_size, (x_center, y_center), sigma)
        heat_map += gaussian
    return heat_map

def process_video(input_video_path, output_video_path, model, output_size=(1024, 512), heat_map_size=512):
    # Initialize video capture
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(output_video_path, fourcc, fps, output_size)

    # Initialize heat map and other variables
    heat_map = np.zeros((heat_map_size, heat_map_size), dtype=float)
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop when video ends

        frame_id += 1
        # Resize frame for processing
        resized_frame = img_resize(frame, heat_map_size)

        # Detect pets
        boxes = detect_pets(resized_frame, model)

        # Update heat map
        heat_map = update_heat_map(heat_map, boxes, heat_map_size)

        # Create visual overlays
        overlay = img_uint8(heat_map)
        overlay = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)
        combined = cv2.addWeighted(resized_frame, 0.6, overlay, 0.4, 0)

        # Add bounding boxes to the overlay
        for box in boxes:
            cv2.rectangle(combined, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # Resize combined frame to match output size
        combined_resized = cv2.resize(combined, output_size)

        # Write frame to output video
        output.write(combined_resized)

        # Display the frame (optional for debugging)
        cv2.imshow("Pet Tracking with Heat Map", combined_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    output.release()
    cv2.destroyAllWindows()

    # Save heat map as an image
    cv2.imwrite("final_heat_map.png", img_uint8(heat_map))

if __name__ == "__main__":
    # Paths
    input_video = "input_video.mp4" #change depending on mp4 name you are inputting

    output_video = "output_video.mp4"

    yolo_weights = "yolov5s.pt"  # Update with your model path
    #yolo_weights = "yolov5m.pt"  # ideal for prioritizing accuracy (especially with cattle video in repository)

    # Load YOLO model
    yolo_net = load_yolo_model(yolo_weights)

    # Process video and generate output
    process_video(input_video, output_video, yolo_net)

    print("Processing complete. Output saved.")
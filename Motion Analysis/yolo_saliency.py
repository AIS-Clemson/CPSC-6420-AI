
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Load the YOLO model with segmentation capabilities
model = YOLO("yolo11n-seg.pt")

# Open the video file
video_path = "./Videos/Tokyo Japan - Shinjuku Summer Night Walk 2024 • 4K HDR.mp4"
# video_path = "./Videos/Shopping, People, Commerce, Mall, Many, Crowd, Walking   Free Stock video footage   YouTube.webm"

cap = cv2.VideoCapture(video_path)

# Retrieve video properties: width, height, and frames per second
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Corrected code to specify MJPG codec
out = cv2.VideoWriter("output1.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (w, h))

# Dictionary to store tracking history with default empty lists
track_history = defaultdict(lambda: [])

# Read the first frame for saliency calculation
ret, frame1 = cap.read()
if not ret:
    print("Error: Cannot read the video.")
    cap.release()
    exit()

# Convert the first frame to grayscale
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

frame_idx = 0
while cap.isOpened():
    ret, frame2 = cap.read()

    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    frame_idx += 1
    
    if frame_idx%5 in [0,1,2,3]: continue

    # Convert the second frame to grayscale
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate frame difference for saliency
    diff = cv2.absdiff(gray1, gray2)

    # Threshold the difference to create a binary mask
    _, saliency_map = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)

    # Apply colormap for visualization
    saliency_colored = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)

    # Combine the saliency map with the original frame
    combined_frame = cv2.addWeighted(frame2, 0.2, saliency_colored, 0.9, 0)

    # Create an annotator object to draw on the frame
    annotator = Annotator(frame2, line_width=2)

    # Perform object tracking on the current frame using YOLO
    results = model.track(frame2, persist=True)

    # Check if tracking IDs and masks are present in the results
    if results[0].boxes.id is not None and results[0].masks is not None:
        # Extract masks and tracking IDs
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Annotate each mask with its corresponding tracking ID and color
        for mask, track_id in zip(masks, track_ids):
            annotator.seg_bbox(mask=mask, mask_color=colors(int(track_id), True), label=str(track_id))

    # Get the annotated frame
    annotated_frame = annotator.result()

    # Combine the annotated frame with the saliency map
    final_output = cv2.addWeighted(annotated_frame, 0.5, combined_frame, 0.5, 0)

    # Write the final output frame to the video
    out.write(final_output)

    # Display the final output frame
    cv2.imshow("Output", final_output)

    # Update the previous frame for saliency calculation
    gray1 = gray2

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video writer and capture objects, and close all OpenCV windows
out.release()
cap.release()
cv2.destroyAllWindows()
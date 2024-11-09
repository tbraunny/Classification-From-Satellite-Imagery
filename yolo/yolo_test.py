from ultralytics import YOLO
import cv2
import glob
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# Load YOLO model
model = YOLO("yolo11x-obb.pt")

# Directory containing the extracted images
image_dir = Path('datasets/planesnet/scenes/scenes/')

output_dir = "datasets/yolo infrence dataset"

# Iterate over images in the directory
for image_path in image_dir.glob('*.png'):
    # Read the image
    img = cv2.imread(str(image_path))

    # Perform inference
    results = model(img)

    # Extract bounding boxes and labels
    for i, (box, label) in enumerate(zip(results.xyxy[0], results.names)):
        if label == 'airplane':
            x_min, y_min, x_max, y_max = map(int, box[:4])

            # Crop the detected airplane
            cropped_img = img[y_min:y_max, x_min:x_max]

            # Resize to 20x20 pixels
            resized_img = cv2.resize(cropped_img, (20, 20))

            # Save the cropped image
            output_path = output_dir / f"{image_path.stem}_plane_{i}.png"
            cv2.imwrite(str(output_path), resized_img)

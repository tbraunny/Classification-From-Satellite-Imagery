from ultralytics import YOLO
import cv2
import glob
import matplotlib.pyplot as plt
import torch
import json

# Load YOLO model
model = YOLO("yolo11x-obb.pt")

#List of images to process
train_data = glob.glob("satellite dataset/*.png") + glob.glob("satellite dataset/*.jpg") + glob.glob("planesnet/scenes/scenes/*.png")
json_path = "datasets/planesnet/planesnet.json"
image_files = glob.glob("datasets/planesnet/scenes/scenes/*.png")
print(image_files)

output_dir = "datasets/yolo infrence dataset"

#ID for 'airplane'
airplane_class_id = 4  

#Train model on small images
model.train(data=train_data, imgsz=1280, epochs=100)

for image_name in image_files:
    print(image_name)
    print("Processesing...")
    #Run inference
    results = model(image_name)

    #Load the original image (not segmented image for now)
    image_with_airplanes = results[0].orig_img.copy()

    #Draw bounding boxes for airplanes
    if results[0].boxes is not None:
        for box in results[0].boxes:
            if int(box.cls) == airplane_class_id: #Check if ID is 'airplane'
                x1, y1, x2, y2 = map(int, box.xyxy[0]) #Convert to integer coordinates
                color = (0, 255, 0) #Green color for the bounding box
                thickness = 2 #Thickness of the bounding box lines
                image_with_airplanes = cv2.rectangle(image_with_airplanes, (x1, y1), (x2, y2), color, thickness)

    # #Save image
    # output_name = f"detected_{image_name}"
    # cv2.imwrite("yolo_output_images/", image_with_airplanes)

    #Display image
    plt.imshow(cv2.cvtColor(image_with_airplanes, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

print("Processing complete")
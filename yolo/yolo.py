from ultralytics import YOLO
import cv2
import glob
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import shutil
import yaml

# Check for GPU availability
if torch.cuda.is_available():
    print("CUDA available, running on GPU")
    device = torch.device("cuda")
else:
    print("***GPU unavailable, running on CPU***")
    device = torch.device("cpu")

class YOLOModel:
    def __init__(self, model_path, output_dir):
        self.model = YOLO(model_path).to(device)
        self.output_dir = output_dir
        self.airplane_class_id = 4  # Assuming 'airplane' ID

    def train(self, train_data, img_size=1280, epochs=100, batch_size=16, num_workers=8):
        self.model.train(data=train_data, imgsz=img_size, epochs=epochs, batch=batch_size, workers=num_workers)


    def detect_and_draw_boxes(self, image_path):
        results = self.model(image_path)
        image_with_boxes = results[0].orig_img.copy()

        if results[0].boxes is not None:
            for box in results[0].boxes:
                if int(box.cls) == self.airplane_class_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    image_with_boxes = cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return image_with_boxes

    def display_image(self, image):
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

class ImageDataset(Dataset):
    def __init__(self, image_list, label, transform=None):
        self.image_list = image_list
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label

class ImageAugmentor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop((128, 128), scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor()
        ])

    def augment_and_load_images(self, image_list, label, batch_size=32):
        dataset = ImageDataset(image_list, label, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

class DatasetPreprocessor:
    def __init__(self, plane_images, non_plane_images, output_dir):
        self.plane_images = plane_images
        self.non_plane_images = non_plane_images
        self.output_dir = output_dir
        self.train_dir = os.path.join(output_dir, "train")
        self.val_dir = os.path.join(output_dir, "val")

        # # Clear existing directories to start fresh
        # self.clear_directory(self.train_dir)
        # self.clear_directory(self.val_dir)
        
        # Create directories for train and val
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)
        os.makedirs(os.path.join(self.train_dir, "planes"), exist_ok=True)
        os.makedirs(os.path.join(self.train_dir, "no_planes"), exist_ok=True)
        os.makedirs(os.path.join(self.val_dir, "planes"), exist_ok=True)
        os.makedirs(os.path.join(self.val_dir, "no_planes"), exist_ok=True)

    def clear_directory(self, directory):
        """Remove all contents of the directory if it exists."""
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

    def create_training_validation_split(self, split_ratio=0.8):
        plane_train_split = int(len(self.plane_images) * split_ratio)
        non_plane_train_split = int(len(self.non_plane_images) * split_ratio)

        plane_train_images = self.plane_images[:plane_train_split]
        plane_val_images = self.plane_images[plane_train_split:]

        non_plane_train_images = self.non_plane_images[:non_plane_train_split]
        non_plane_val_images = self.non_plane_images[non_plane_train_split:]

        # Save images in respective directories
        print("Saving train and validation images to directories...")
        self.save_images(plane_train_images, os.path.join(self.train_dir, "planes"))
        self.save_images(non_plane_train_images, os.path.join(self.train_dir, "no_planes"))
        self.save_images(plane_val_images, os.path.join(self.val_dir, "planes"))
        self.save_images(non_plane_val_images, os.path.join(self.val_dir, "no_planes"))

        # Combine train and validation images for return if needed
        train_images = plane_train_images + non_plane_train_images
        val_images = plane_val_images + non_plane_val_images

        return train_images, val_images

    
    def save_images(self, images, destination_folder):
        for idx, image_path in enumerate(images):
            image = cv2.imread(image_path)
            output_filename = os.path.join(destination_folder, f"{os.path.basename(image_path)}")
            cv2.imwrite(output_filename, image)


class InferenceProcessor:
    def __init__(self, yolo_model, image_files):
        self.yolo_model = yolo_model
        self.image_files = image_files

    def process_images(self):

        for image_path in self.image_files:
            print(f"Processing {image_path}...")
            image_with_boxes = self.yolo_model.detect_and_draw_boxes(image_path)
            self.yolo_model.display_image(image_with_boxes)
        print("Processing complete")

# Main Execution
if __name__ == "__main__":
    # Define paths
    model_path = "models/yolo11x-obb.pt"
    output_dir = "datasets/yolo_inference_dataset"
    config_path = "yolo/train_config.yaml"  # Path to the YAML file created above

    yolo_model = YOLOModel(model_path, output_dir)
    
    plane_images = glob.glob("datasets/planesnet/scenes/planesnet/planesnet/1_*.png")
    non_plane_images = glob.glob("datasets/planesnet/scenes/planesnet/planesnet/0_*.png")

    import rasterio
    from rasterio.transform import Affine

    # Open the scene image
    scene = rasterio.open('scene.tif')
    transform = scene.transform

    # Function to convert lon/lat to pixel coordinates
    def lonlat_to_pixel(lon, lat):
        x, y = scene.index(lon, lat)
        return x, y

    # Preprocess and augment images using GPU
    print("Preprocessing and augmenting images...")
    augmentor = ImageAugmentor()
    plane_loader = augmentor.augment_and_load_images(plane_images, label=1, batch_size=32)
    non_plane_loader = augmentor.augment_and_load_images(non_plane_images, label=0, batch_size=32)
    
    # # Set up training data and train the model
    # print("Setting up model...")
    # preprocessor = DatasetPreprocessor(plane_images, non_plane_images, output_dir)
    # train_images, val_images = preprocessor.create_training_validation_split()

    # Paths relative to the location of train_config.yaml
    data_config = {
        'train': os.path.abspath(os.path.join(output_dir, 'train')),
        'val': os.path.abspath(os.path.join(output_dir, 'val')),
        'nc': 2,  # number of classes
        'names': ['no_plane', 'plane']  # class names
    }

    # Save the YAML file
    with open(config_path, 'w') as f:
        yaml.dump(data_config, f)

    print("Training model...")
    yolo_model.train(config_path, img_size=1280, epochs=100, batch_size=16, num_workers=8)

    # Run inference on new images
    print("Running inference...")
    image_files = glob.glob("datasets/LHR/*.png")  # Set the actual path to images
    scenes_files = glob.glob("datasets/planesnet/scenes/scenes")  # Set the actual path to images

    
    inference_processor = InferenceProcessor(yolo_model, scenes_files)
    # inference_processor.process_images()
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision
import torch.optim as optim
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import cv2
import time
import os
from matplotlib import pyplot as plt

program_start_time = time.time()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Running on device: {device}")

# Dataset class with image augmentation
class PlanesDataset(Dataset):
    def __init__(self, data_path, transform=None):
        with open(data_path, 'r') as f:
            planesnet = json.load(f)
        
        self.images = np.array(planesnet['data']) / 255.0
        self.images = self.images.reshape([-1, 3, 20, 20]).transpose([0, 2, 3, 1])
        self.labels = np.array(planesnet['labels'])
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.fromarray((self.images[idx] * 255).astype(np.uint8))
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        if label == 1:
            boxes = torch.tensor([[0, 0, 20, 20]], dtype=torch.float32)
            labels = torch.tensor([1], dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}

        return image, target

class FasterRCNNTrainer:
    def __init__(self, num_classes=2, model_path='models/fasterrcnn_planes.pth'):
        self.model_path = model_path
        self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = self._initialize_model(num_classes)
        self.model.to(device)
    
    def _initialize_model(self, num_classes):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=self.weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def train(self, dataloader, num_epochs=10, learning_rate=0.005):
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
        self.model.train()
        total_images_processed = 0

        train_start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            for i, (images, targets) in enumerate(dataloader):
                batch_start_time = time.time()
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                epoch_loss += losses.item()

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                batch_size = len(images)
                total_images_processed += batch_size

                batch_elapsed_time = time.time() - batch_start_time
                images_per_minute = (batch_size / batch_elapsed_time) * 60

                elapsed_train_time = time.time() - train_start_time
                minutes = int(elapsed_train_time // 60)
                seconds = int(elapsed_train_time % 60)

                print(f"\rEpoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(dataloader)}, "
                      f"Images/min: {images_per_minute:.2f}, "
                      f"Train time: {minutes} minutes {seconds} seconds.", end="")
            print("\n")
            print(f"\rLoss at Epoch {epoch+1}: {epoch_loss:.4f}", end="")

        total_train_time = time.time() - train_start_time
        avg_images_per_minute = (total_images_processed / total_train_time) * 60

        print(f"\nTraining complete. Average Images Processed per Minute: {avg_images_per_minute:.2f}")
        self.save_model()

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.model.eval()
        print("Model loaded for inference.")

    def get_model(self):
        return self.model

class InferenceEngine:
    def __init__(self, model, weights):
        self.model = model
        self.model.eval()
        self.transforms = weights.transforms()
    
    def detect_planes(self, image, threshold=0.5):
        image_tensor = self.transforms(image).to(device)
        with torch.no_grad():
            predictions = self.model([image_tensor])
        boxes = predictions[0]['boxes']
        scores = predictions[0]['scores']
        # Filter out detections below the threshold
        keep = scores >= threshold
        boxes = boxes[keep].cpu().numpy().astype(int)
        scores = scores[keep].cpu().numpy()
        return boxes, scores
    
    def draw_boxes(self, image, boxes):
        image = np.array(image)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return image

def collate_fn(batch):
    return tuple(zip(*batch))

# Function for standard sliding window with overlap
def sliding_window_with_overlap(image, window_size=(20, 20), overlap=0.2):
    width, height = image.size
    step_size = int(window_size[0] * (1 - overlap))
    patches = []
    coordinates = []

    for y in range(0, height - window_size[1] + 1, step_size):
        for x in range(0, width - window_size[0] + 1, step_size):
            patch = image.crop((x, y, x + window_size[0], y + window_size[1]))
            patches.append(patch)
            coordinates.append((x, y))

    return patches, coordinates

# Inference on patches with NMS
def infer_on_patches(model, patches, coordinates, batch_size=64, threshold=0.5, iou_threshold=0.3):
    detected_boxes = []
    scores = []
    model.eval()
    
    transforms = T.Compose([T.ToTensor()])
    for i in range(0, len(patches), batch_size):
        batch_patches = patches[i:i+batch_size]
        batch_coords = coordinates[i:i+batch_size]
        batch_tensors = [transforms(patch).to(device) for patch in batch_patches]
        
        with torch.no_grad():
            predictions = model(batch_tensors)
        
        for prediction, (x_offset, y_offset) in zip(predictions, batch_coords):
            if prediction['scores'].numel() > 0 and prediction['scores'][0] >= threshold:
                x1, y1, x2, y2 = prediction['boxes'][0].cpu().numpy()
                x1_full = x1 + x_offset
                y1_full = y1 + y_offset
                x2_full = x2 + x_offset
                y2_full = y2 + y_offset
                detected_boxes.append([x1_full, y1_full, x2_full, y2_full])
                scores.append(prediction['scores'][0].cpu().item())

    # Apply NMS
    if detected_boxes:
        boxes_tensor = torch.tensor(detected_boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        keep = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold)
        detected_boxes = boxes_tensor[keep].cpu().numpy()
        scores = scores_tensor[keep].cpu().numpy()

    return detected_boxes, scores

def main(train=True):
    print("Initializing dataset...")
    train_transform = T.Compose([
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.RandomRotation(degrees=45),
        T.ToTensor(),
    ])
    dataset = PlanesDataset(data_path='datasets/planesnet/planesnet.json', transform=train_transform)
    dataloader = DataLoader(
        dataset,
        batch_size=12,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    model_path = 'models/fasterrcnn_planes.pth'
    trainer = FasterRCNNTrainer(model_path=model_path)

    if train or not os.path.exists(model_path):
        print("Training model...")
        trainer.train(dataloader, num_epochs=10)
    else:
        print("Loading model...")
        trainer.load_model()

    model = trainer.get_model()
    weights = trainer.weights

    print("Running inference with sliding window and overlap...")
    full_image = Image.open('datasets/planesnet/scenes/scenes/scene_1.png').convert("RGB")
    inference_engine = InferenceEngine(model, weights)
    
    # Generate patches using sliding window with overlap
    patches, coordinates = sliding_window_with_overlap(full_image, overlap=0.2)
    
    # Run inference on patches in batches with NMS
    boxes, scores = infer_on_patches(model, patches, coordinates, batch_size=64, threshold=0.5, iou_threshold=0.3)
    
    # Draw bounding boxes on the original image
    print("Drawing bounding boxes...")
    result_image = inference_engine.draw_boxes(full_image, boxes)
    plt.figure(figsize=(12, 8))
    plt.imshow(result_image)
    plt.axis('off')
    plt.show()
    
    # Print elapsed time
    elapsed_time = time.time() - program_start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Program ran for {minutes} minutes and {seconds} seconds.")

if __name__ == "__main__":
    main(train=False)  # Set to True if you want to retrain the model

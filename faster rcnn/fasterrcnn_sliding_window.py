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
from matplotlib import pyplot as plt
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PlanesDataset(Dataset):
    def __init__(self, data_path, transform=None):
        with open(data_path, 'r') as f:
            data = json.load(f)
        self.images = np.array(data['data']).reshape(-1, 3, 20, 20) / 255.0
        self.labels = data['labels']
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray((self.images[idx].transpose(1, 2, 0) * 255).astype(np.uint8))
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        target = {'boxes': torch.tensor([[0, 0, 20, 20]], dtype=torch.float32) if label == 1 else torch.zeros((0, 4)),
                  'labels': torch.tensor([label] if label == 1 else [], dtype=torch.int64)}
        return image, target

class PlaneDetector:
    def __init__(self, model_path='fasterrcnn_sliding_window.pth'):
        self.model_path = model_path
        self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = self._init_model()
        self.model.to(device)

    def _init_model(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=self.weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        return model

    def train(self, dataloader, epochs=10, lr=0.005):
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        
        for epoch in range(epochs):
            start_time = time.time()
            total_loss = 0
            
            for batch_idx, (images, targets) in enumerate(dataloader):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                loss = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Print progress for current batch
                elapsed = time.time() - start_time
                print(f"\rEpoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], "
                      f"Loss: {loss.item():.4f}, Elapsed: {elapsed:.2f}s", end="")

            print(f"\nEpoch [{epoch+1}/{epochs}] completed, Total Loss: {total_loss:.4f}, "
                  f"Time: {time.time() - start_time:.2f}s\n")

        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def load(self):
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        print("Model loaded for inference.")

class SlidingWindowInference:
    def __init__(self, model, window_size=(20, 20), overlap=0.2, threshold=0.5):
        self.model = model
        self.window_size = window_size
        self.step_size = int(window_size[0] * (1 - overlap))
        self.threshold = threshold
        self.transform = T.ToTensor()

    def sliding_windows(self, image):
        width, height = image.size
        for y in range(0, height - self.window_size[1] + 1, self.step_size):
            for x in range(0, width - self.window_size[0] + 1, self.step_size):
                yield image.crop((x, y, x + self.window_size[0], y + self.window_size[1])), x, y

    def detect_planes(self, image):
        boxes = []
        start_time = time.time()
        for i, (patch, x, y) in enumerate(self.sliding_windows(image)):
            patch_tensor = self.transform(patch).to(device).unsqueeze(0)
            with torch.no_grad():
                outputs = self.model(patch_tensor)
            scores = outputs[0]['scores'].cpu().numpy()
            patch_boxes = outputs[0]['boxes'].cpu().numpy()
            
            for score, box in zip(scores, patch_boxes):
                if score >= self.threshold:
                    boxes.append([box[0] + x, box[1] + y, box[2] + x, box[3] + y])

            # Print progress for current sliding window patch
            elapsed = time.time() - start_time
            print(f"\rSliding Window [{i+1}], Detections: {len(boxes)}, Elapsed: {elapsed:.2f}s", end="")
        
        print("\nSliding window inference completed.")
        return boxes

    def draw_boxes(self, image, boxes):
        image = np.array(image)
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        return Image.fromarray(image)

def main(train=False):
    dataset_path = 'datasets/planesnet/planesnet.json'
    image_path = 'datasets/planesnet/scenes/scenes/scene_1.png'
    model_path = 'models/fasterrcnn_planes.pth'

    transform = T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(10), T.ToTensor()])
    dataset = PlanesDataset(dataset_path, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=14, 
        shuffle=True,
        collate_fn=lambda batch: tuple(zip(*batch)),
        num_workers=18,
        pin_memory=True
        )

    detector = PlaneDetector(model_path)
    if train:
        detector.train(dataloader)
    else:
        detector.load()

    full_image = Image.open(image_path).convert("RGB")
    inference = SlidingWindowInference(detector.model)
    with torch.no_grad():
        boxes = inference.detect_planes(full_image)
    result_image = inference.draw_boxes(full_image, boxes)
    
    plt.imshow(result_image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main(train=True)  # Set to True to train the model

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision
import torch.optim as optim
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import cv2
from matplotlib import pyplot as plt
import time
from torchvision.models.detection.rpn import AnchorGenerator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from glob import glob
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

# # Dataset class with multi-scale training and augmentation
# class PlanesDataset(Dataset):
#     def __init__(self, data_path, transform=None):
#         with open(data_path, 'r') as f:
#             data = json.load(f)
#         self.images = np.array(data['data']).reshape(-1, 3, 20, 20) / 255.0
#         self.labels = data['labels']
#         self.transform = transform

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         image = Image.fromarray((self.images[idx].transpose(1, 2, 0) * 255).astype(np.uint8))
#         label = self.labels[idx]
#         if self.transform:
#             image = self.transform(image)
#         target = {'boxes': torch.tensor([[0, 0, 20, 20]], dtype=torch.float32) if label == 1 else torch.zeros((0, 4)),
#                   'labels': torch.tensor([label] if label == 1 else [], dtype=torch.int64)}
#         return image, target

class PlanesDataset(Dataset):
    def __init__(self, data_path, tensor_dir="processed_tensors", transform=None):
        self.data_path = data_path
        self.tensor_dir = tensor_dir
        self.transform = transform
        self.images, self.labels = self.load_or_process_images()

    def load_or_process_images(self):
        # Check if the tensor directory exists and contains tensors
        os.makedirs(self.tensor_dir, exist_ok=True)
        tensor_files = glob(os.path.join(self.tensor_dir, "*.pt"))
        
        if tensor_files:
            print("Loading preprocessed tensors from directory...")
            images = torch.load(os.path.join(self.tensor_dir, "images.pt"))
            labels = torch.load(os.path.join(self.tensor_dir, "labels.pt"))
        else:
            print("No preprocessed tensors found, processing images from data path...")
            images, labels = [], []

            for label in range(2):
                imgs_path = os.path.join(self.data_path, f"{label}_*")
                matching_files = glob(imgs_path)

                if not matching_files:
                    print(f"Warning: No files found for label {label} in path {imgs_path}")
                i = 1
                for file in matching_files:
                    img = cv2.imread(file)
                    if img is None:
                        print(f"Warning: Could not read image file {file}. Skipping.")
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    images.append(img)
                    labels.append(label)
                    print(f"\rLoaded {i}/{len(matching_files)} in {label} label ", end="")

                print(f"Loaded {len(images)} images for label {label}")

            images = np.array(images)
            labels = np.array(labels)

            if images.size > 0:
                images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
                labels = torch.tensor(labels, dtype=torch.int64)
            else:
                images = torch.empty(0, 3, 20, 20, dtype=torch.float32)
                labels = torch.empty(0, dtype=torch.int64)

            print("Dataset processing complete, saving tensors to disk...")
            torch.save(images, os.path.join(self.tensor_dir, "images.pt"))
            torch.save(labels, os.path.join(self.tensor_dir, "labels.pt"))
            print(f"Tensors saved in {self.tensor_dir}")

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]

        if self.transform:
            image = T.ToPILImage()(image)
            image = self.transform(image)
        
        target = {
            'boxes': torch.tensor([[0, 0, 20, 20]], dtype=torch.float32) if label == 1 else torch.zeros((0, 4)),
            'labels': torch.tensor([label], dtype=torch.int64) if label == 1 else torch.zeros((0,), dtype=torch.int64)
        }
        
        return image, target

class PlaneDetector:
    def __init__(self, model_path='fasterrcnn.pth'):
        self.model_path = model_path
        self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = self._init_model()
        self.model.to(device)

    def _init_model(self):
        # Define smaller anchor sizes for detecting small objects
        anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

        # Initialize Faster R-CNN with custom anchors
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=self.weights, rpn_anchor_generator=anchor_generator)
        
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        return model

    def train(self, train_loader, val_loader, epochs=10, lr=0.001, patience=3):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_loss = 0
            
            self.model.train()
            for batch_idx, (images, targets) in enumerate(train_loader):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model(images, targets)
                loss = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                # Log batch progress
                print(f"\rEpoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}", end="")

            # Validation check
            val_loss = self.evaluate(val_loader)
            print(f"\nEpoch [{epoch+1}/{epochs}], Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Time: {time.time() - epoch_start_time:.2f}s")

            # Early stopping and checkpoint saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                self.save_model()
                print(f"Model saved with improved validation loss: {best_val_loss:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered. No improvement in validation loss for {patience} epochs.")
                    break

    def evaluate(self, loader):
        self.model.train()
        total_val_loss = 0

        with torch.no_grad():
            for images, targets in loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Get a list of loss dictionaries, one for each image
                loss_dict_list = self.model(images, targets)
                
                # Sum the losses for each dictionary in the list
                batch_loss = sum(sum(loss for loss in loss_dict.values()) for loss_dict in loss_dict_list)
                
                total_val_loss += batch_loss

        avg_val_loss = total_val_loss / len(loader)
        return avg_val_loss


    def test(self, test_loader):
        self.model.eval()
        all_labels, all_preds = [], []

        with torch.no_grad():
            for images, targets in test_loader:
                images = [img.to(device) for img in images]
                outputs = self.model(images)
                
                for target, output in zip(targets, outputs):
                    true_label = target['labels'].cpu().numpy()
                    pred_label = (output['scores'] > 0.5).cpu().numpy().astype(int)  # Apply threshold for predictions
                    
                    all_labels.extend(true_label)
                    all_preds.extend(pred_label)

        # Calculate accuracy and print metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    def save_model(self):
        """Saves the model to the specified path."""
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Loads the model from the specified path."""
        self.model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.model.eval()
        print(f"Model loaded from {self.model_path}")

class InferenceEngine:
    def __init__(self, model, tile_size=(512, 512), overlap=0.1, threshold=0.5):
        self.model = model
        self.tile_size = tile_size
        self.step_size = int(tile_size[0] * (1 - overlap))
        self.threshold = threshold
        self.transform = T.ToTensor()

    def load_images_from_dir(self, image_dir):
        """Loads all .png images from the specified directory."""
        image_paths = glob.glob(os.path.join(image_dir, "*.png"))
        images = []
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            images.append((image_path, image))
        print(f"Loaded {len(images)} images from {image_dir}")
        return images

    def tile_image(self, image):
        """Divides a single image into overlapping tiles."""
        width, height = image.size
        tiles = []
        for y in range(0, height - self.tile_size[1] + 1, self.step_size):
            for x in range(0, width - self.tile_size[0] + 1, self.step_size):
                tile = image.crop((x, y, x + self.tile_size[0], y + self.tile_size[1]))
                tiles.append((tile, x, y))
        return tiles

    def detect_on_tiles(self, image):
        """Runs detection on all tiles of an image."""
        tiles = self.tile_image(image)
        detections = []
        start_time = time.time()

        for i, (tile, x_offset, y_offset) in enumerate(tiles):
            tile_tensor = self.transform(tile).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = self.model(tile_tensor)[0]

            for box, score, label in zip(outputs['boxes'], outputs['scores'], outputs['labels']):
                if score >= self.threshold:
                    adjusted_box = box + torch.tensor([x_offset, y_offset, x_offset, y_offset], device=box.device)
                    detections.append((adjusted_box, score, label))

            # Log progress for each tile
            elapsed = time.time() - start_time
            print(f"\rTile [{i+1}/{len(tiles)}], Detections: {len(detections)}, Elapsed: {elapsed:.2f}s", end="")

        print("\nInference on tiles completed.")
        return detections

    def run_inference_on_directory(self, image_dir):
        """Performs inference on all images in the specified directory."""
        images = self.load_images_from_dir(image_dir)
        
        for image_path, image in images:
            print(f"Processing {image_path}")
            detections = self.detect_on_tiles(image)
            result_image = self.draw_boxes(image, detections)

            # Display the result or save it
            result_image.show()
            result_image.save(os.path.join(image_dir, "results", os.path.basename(image_path)))
            print(f"Saved results for {image_path}")
            break

    def draw_boxes(self, image, boxes):
        """Draws bounding boxes on the image."""
        image_np = np.array(image)
        for (box, score, label) in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return Image.fromarray(image_np)
    
# Main function to train, validate, and test the model
def main(train=True):
    train_path = 'datasets/planesnet/scenes/planesnet/planesnet'
    train_tensor_path = 'datasets/planesnet_tensor/scenes/planesnet/planesnet'
    infrence_path = 'datasets/planesnet/scenes/scenes/'
    model_path = 'models/fasterrcnn_planes.pth'

    transform = T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(10), T.ToTensor()])
    
    # Load dataset and split into train, val, and test sets
    dataset = PlanesDataset(train_path, tensor_dir=train_tensor_path , transform=transform)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=12 , shuffle=True, num_workers=18, pin_memory=True, collate_fn=lambda batch: tuple(zip(*batch)))
    val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False, num_workers=18, pin_memory=True, collate_fn=lambda batch: tuple(zip(*batch)))
    test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=18, pin_memory=True, collate_fn=lambda batch: tuple(zip(*batch)))

    detector = PlaneDetector(model_path)

    if train:
        detector.train(train_loader, val_loader, epochs=20, lr=0.001, patience=3)
    else:
        detector.load()
        
    inference = InferenceEngine(detector.model)
    inference.run_inference_on_directory(infrence_path)
    
    # plt.imshow(result_image)
    # plt.axis('off')
    # plt.show()

    # Test the model
    print("\nEvaluating on Test Set...")
    detector.test(test_loader)

if __name__ == "__main__":
    main(train=True)  # Set to False to skip training and load the pre-trained model
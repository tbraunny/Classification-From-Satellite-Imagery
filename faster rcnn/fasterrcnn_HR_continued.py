import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import torchvision.transforms.functional as F
import matplotlib.patches as patches
from pycocotools.coco import COCO


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

# # Define the Dataset class
# class HRPlanesDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.images_dir = os.path.join(root_dir, 'images')
#         self.labels_dir = os.path.join(root_dir, 'labels')
#         self.image_files = [f for f in os.listdir(self.images_dir) 
#                             if f.endswith(('.jpeg', '.jpg', '.png'))]
#         self.image_files.sort()
#         self.valid_indices = []
#         for idx, img_name in enumerate(self.image_files):
#             label_name = os.path.splitext(img_name)[0] + '.txt'
#             label_path = os.path.join(self.labels_dir, label_name)
#             if os.path.exists(label_path):
#                 with open(label_path, 'r') as f:
#                     if any(line.strip() for line in f):
#                         self.valid_indices.append(idx)
        

#     def __len__(self):
#         return len(self.valid_indices)

#     def __getitem__(self, idx):
#         idx = self.valid_indices[idx]
#         img_name = self.image_files[idx]
#         img_path = os.path.join(self.images_dir, img_name)
#         image = Image.open(img_path).convert('RGB')
#         img_width, img_height = image.size
#         label_name = os.path.splitext(img_name)[0] + '.txt'
#         label_path = os.path.join(self.labels_dir, label_name)
#         boxes = []
#         labels = []
        
#         if os.path.exists(label_path):
#             with open(label_path, 'r') as f:
#                 for line in f:
#                     parts = line.strip().split()
#                     if len(parts) != 5:
#                         continue
#                     cls_id, x_center, y_center, width, height = map(float, parts)
#                     labels.append(int(cls_id))
#                     x_center *= img_width
#                     y_center *= img_height
#                     width *= img_width
#                     height *= img_height
#                     x_min = x_center - width / 2
#                     y_min = y_center - height / 2
#                     x_max = x_center + width / 2
#                     y_max = y_center + height / 2
#                     boxes.append([x_min, y_min, x_max, y_max])

#         # Skip samples with no bounding boxes
#         if len(boxes) == 0:
#             return self.__getitem__((idx + 1) % len(self.image_files))  # Fetch the next sample
        
#         boxes = torch.tensor(boxes, dtype=torch.float32)
#         labels = torch.tensor(labels, dtype=torch.int64)
#         target = {'boxes': boxes, 'labels': labels}

#         if self.transform:
#             image = self.transform(image)

#         return image, target

class COCOPlanesDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.image_ids = [
            img_id for img_id in self.coco.getImgIds()
            if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.coco.imgs[image_id]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        
        # Load image
        image = Image.open(img_path).convert("RGB")

        # Load annotations
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)

        boxes = []
        labels = []
        for ann in annotations:
            x, y, width, height = ann['bbox']
            boxes.append([x, y, x + width, y + height])
            labels.append(ann['category_id'])

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            image = self.transform(image)

        return image, target



# Define the PlaneDetector class with training and validation methods
class PlaneDetector:
    def __init__(self, num_classes=2, model_path='models/fasterrcnn.pth'):
        self.model_path = model_path
        self.model = self._init_model(num_classes)
        self.model.to(device)

    def _init_model(self, num_classes):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def train_model(self, train_loader, valid_loader, epochs=10, patience=3, lr=0.005):
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0

            for batch_idx, (images, targets) in enumerate(train_loader):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                optimizer.zero_grad()

                # Calculate loss
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()
                train_loss += losses.item()

                print(f"\rEpoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {losses.item():.4f}", end="")

            train_loss /= len(train_loader)

            # Validation loop
            val_loss = self.evaluate(valid_loader)
            print(f"\nEpoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model()
            else:
                patience_counter += 1
                print(f"Patience Counter: {patience_counter}")
                if patience_counter >= patience:
                    print("Stopping early due to overfitting.")
                    break
        print(f"Training complete. Saved model to {self.model_path}")

    def evaluate(self, data_loader):
        self.model.train()
        val_loss = 0
        with torch.no_grad():
            for images, targets in data_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Calculate loss
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

        val_loss /= len(data_loader)
        return val_loss

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=device)) #, weights_only=True
        self.model.eval()
        print("Model loaded for inference.")

class InferenceEngine:
    def __init__(self, model_path='models/fasterrcnn.pth', num_classes=2, threshold=0.5, output_dir='Inference/faster rcnn'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._init_model(num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        self.threshold = threshold
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print("Model loaded for inference.")

    def _init_model(self, num_classes):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model
                
    def predict(self, data_loader):
        results = []
        with torch.no_grad():
            for batch in data_loader:
                # Handle case where only images are in the batch
                if isinstance(batch, tuple) and len(batch) == 2:
                    images, _ = batch
                else:
                    images = batch

                # Flatten `images` if it's a nested list and filter out non-tensor items
                if isinstance(images, list):
                    images = [img for sublist in images for img in (sublist if isinstance(sublist, list) else [sublist])]
                    images = [img for img in images if isinstance(img, torch.Tensor)]  # Filter out any dicts or non-tensors

                # Move images to device
                images = [img.to(self.device) for img in images]

                # Perform inference
                outputs = self.model(images)

                for img, output in zip(images, outputs):
                    boxes = output['boxes'][output['scores'] >= self.threshold]
                    scores = output['scores'][output['scores'] >= self.threshold]
                    results.append((img, boxes, scores))

        return results
    
    def visualize_predictions(self, results, mean, std):
        for i, (img, boxes, scores) in enumerate(results):


            # Denormalize the image
            img = denormalize(img.cpu(), mean, std)


            img = F.to_pil_image(img)
            fig, ax = plt.subplots(1)
            ax.imshow(img)

            for box, score in zip(boxes.cpu(), scores.cpu()):
                x_min, y_min, x_max, y_max = box
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x_min, y_min, f'{score:.2f}', color='yellow', fontsize=10, weight='bold')

            plt.axis('off')
            output_path = os.path.join(self.output_dir, f"prediction_{i}.png")
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
            print(f"\rSaved prediction to {output_path}", end="")

def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=False, collate_fn=collate_fn_for_mean_std)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    for images in loader:
        batch_samples = len(images)
        images = torch.stack(images)  # Stack images into a tensor
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images
    return mean, std

def denormalize(image, mean, std):
    # Convert mean and std to tensors and reshape them to match the image dimensions
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    # Reverse the normalization: image = (image * std) + mean
    denormalized_image = image * std + mean
    
    # Clip values to be within the range [0, 1] to avoid display issues
    denormalized_image = torch.clamp(denormalized_image, 0, 1)
    return denormalized_image

# Custom collate function
def collate_fn(batch):
    images, targets = zip(*batch)
    images = list(images)
    targets = list(targets)
    return images, targets


def collate_fn_for_mean_std(batch):
    images, _ = zip(*batch)  # Ignore targets
    return images

def collate_fn_for_unlabeled(batch):
    return batch  # Returns only images, no targets


class UnlabeledImagesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]
        self.image_files.sort()
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image  # No target for unlabeled images

def main(train=True):

    # Create your dataset without transformations
    dataset_for_stats = COCOPlanesDataset(
        root_dir='datasets/HRPlanes/train/images',
        annotation_file='datasets/HRPlanes_coco/train/_annotations.coco.json',
        transform=T.ToTensor()
    )
    mean, std = compute_mean_std(dataset_for_stats)
    mean = mean.tolist()
    std = std.tolist()
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

    # train_dataset = HRPlanesDataset(root_dir='datasets/HRPlanes/train', transform=transform)
    # valid_dataset = HRPlanesDataset(root_dir='datasets/HRPlanes/valid', transform=transform)
    # test_dataset = HRPlanesDataset(root_dir='datasets/HRPlanes/test', transform=transform)
    # Instantiate the dataset
    train_dataset = COCOPlanesDataset(root_dir='datasets/HRPlanes/train/images',
                                      annotation_file='datasets/HRPlanes_coco/train/_annotations.coco.json',
                                      transform=transform)
    
    valid_dataset = COCOPlanesDataset(root_dir='datasets/HRPlanes/valid/images',
                                      annotation_file='datasets/HRPlanes_coco/valid/_annotations.coco.json',
                                      transform=transform)
    
    test_dataset = COCOPlanesDataset(root_dir='datasets/HRPlanes/test/images',
                                     annotation_file='datasets/HRPlanes_coco/test/_annotations.coco.json',
                                     transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=18, shuffle=True, pin_memory=True, num_workers=10, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=18, shuffle=False, pin_memory=True, num_workers=10, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=18, shuffle=False, pin_memory=True, num_workers=10, collate_fn=collate_fn)
    
    model_path = 'models/fasterrcnn.pth'
    if not os.path.exists('models/fasterrcnn.pth'):
        print("Model file not found. Make sure the model is trained and saved.")
        return

    detector = PlaneDetector(num_classes=2, model_path=model_path)
    if train:
        detector.train_model(train_loader, valid_loader, epochs=20, patience=3, lr=0.01)

    # # Test model on HRPlanes Test dataset
    output_path = "Inference/faster rcnn/HRPlanes test"
    inference_engine = InferenceEngine(model_path=model_path, num_classes=2, threshold=0.95, output_dir=output_path)
    results = inference_engine.predict(test_loader)
    inference_engine.visualize_predictions(results,mean,std)

    # Instantiate the unlabeled dataset for inference
    unlabeled_dataset = UnlabeledImagesDataset(root_dir='datasets/planesnet/scenes/scenes/', transform=transform)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=18, shuffle=False, pin_memory=True, num_workers=10, collate_fn=collate_fn_for_unlabeled)

    # Inference on unlabeled images
    inference_engine = InferenceEngine(model_path=model_path, num_classes=2, threshold=0.95, output_dir="Inference/faster rcnn/unlabeled_images")
    unlabeled_results = inference_engine.predict(unlabeled_loader)
    inference_engine.visualize_predictions(unlabeled_results, mean, std)






# Main block
if __name__ == '__main__':
    main(train=False)
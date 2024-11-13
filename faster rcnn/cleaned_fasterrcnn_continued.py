from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import matplotlib.patches as patches
from pycocotools.coco import COCO
import torchmetrics

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)




class COCOPlanesDataset(Dataset):
    """
    Custom Dataset class for loading images and annotations in COCO format.
    """
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir

        # Silence annoying coco prints
        from contextlib import redirect_stdout
        with redirect_stdout(open(os.devnull, "w")):
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


class UnlabeledImagesDataset(Dataset):
    """
    Dataset class for loading unlabeled images for inference.
    """
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

        place_holder = None
        return image, place_holder


def compute_mean_std(dataset):
    """
    Compute the mean and standard deviation of a dataset for normalization.
    """
    loader = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=False, collate_fn=collate_fn)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    for images, _ in loader:
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
    """
    Denormalize an image tensor using the provided mean and std.
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    denormalized_image = image * std + mean
    denormalized_image = torch.clamp(denormalized_image, 0, 1)
    return denormalized_image

# Custom collate functions
# HRPlanes images have a varying number of bounding boxes so custom collate needed
def collate_fn(batch):
    images, targets = zip(*batch)
    images = list(images)
    targets = list(targets)
    return images, targets

def collate_fn_for_mean_std(batch):
    images, _ = zip(*batch)
    return images



class Inference:
    def __init__(self, model, device, threshold=0.5, output_dir="./"):

        self.model = model
        self.threshold=threshold
        self.output_dir=output_dir
        self.device = device
        self.num_classes = 2

    def predict(self, dataloader):
        self.model.eval()
        results = []
        with torch.no_grad():
            for images, _ in dataloader:

                # Move images to device
                images = [img.to(self.device) for img in images]


                # Perform inference
                outputs = self.model(images)

                for img, output in zip(images, outputs):
                    boxes = output['boxes'][output['scores'] >= self.threshold]
                    scores = output['scores'][output['scores'] >= self.threshold]
                    results.append((img, boxes, scores))

        return results


    def evaluate(self, dataloader):
        """
        Compute Top-1, Top-5 accuracy, and Mean Average Precision (mAP).
        """
        self.model.eval()

        map_metric = torchmetrics.detection.mean_ap.MeanAveragePrecision(box_format='xyxy', iou_type='bbox').to(self.device)
        
        top1_correct = 0
        top5_correct = 0
        total_samples = 0
        # Set model to evaluation mode
        results = []

        with torch.no_grad():
            for images, targets in dataloader:

                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Run inference
                outputs = self.model(images)

                for img, output in zip(images, outputs):
                    boxes = output['boxes'][output['scores'] >= self.threshold]
                    scores = output['scores'][output['scores'] >= self.threshold]
                    results.append((img, boxes, scores))

                for target, output in zip(targets, outputs):
                    true_labels = target['labels']
                    pred_scores = output['scores']
                    pred_labels = output['labels']

                    # Update mAP metric
                    map_metric.update([output], [target])

                    # Apply confidence threshold
                    mask = pred_scores >= self.threshold
                    filtered_scores = pred_scores[mask]
                    filtered_labels = pred_labels[mask]

                    # Top-1 and Top-5 Accuracy
                    if len(filtered_labels) > 0:
                        # Sort predictions by confidence
                        sorted_indices = filtered_scores.argsort(descending=True)
                        sorted_labels = filtered_labels[sorted_indices]

                        # Top-1: Check if top prediction is correct
                        top1_correct += (sorted_labels[0] == true_labels[0]).item()

                        # Top-5: Check if true label is in top 5 predictions
                        top5_correct += (true_labels[0] in sorted_labels[:5])

                    total_samples += 1
    

        # Calculate metrics
        top1_accuracy = top1_correct / total_samples
        top5_accuracy = top5_correct / total_samples
        mean_ap = map_metric.compute()['map'].item()

        # Print Results
        print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
        print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
        print(f"Mean Average Precision (mAP): {mean_ap:.4f}")

        self.model.train()

        return results

    def save_predictions(self, results, mean, std):
        """
        Visualize predictions and save the images with bounding boxes.
        """
        for i, (img, boxes, scores) in enumerate(results):
            # Denormalize the image
            img = denormalize(img.cpu(), mean, std)

            img = F.to_pil_image(img)
            fig, ax = plt.subplots(1)
            ax.imshow(img)

            for box, score in zip(boxes.cpu(), scores.cpu()):
                x_min, y_min, x_max, y_max = box
                rect = patches.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x_min, y_min, f'{score:.2f}',
                        color='yellow', fontsize=10, weight='bold'
                )

            plt.axis('off')
            output_path = os.path.join(self.output_dir, f"prediction_{i}.png")
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
            print(f"\rSaved prediction to {output_path} ", end="")



class PlaneDetector:
    """
    Class for initializing, training, and evaluating the object detection model.
    """
    def __init__(self, num_classes=2, model_path='models/fasterrcnn.pth'):
        self.model_path = model_path
        self.model = self._init_model(num_classes)
        self.model.to(device)

    def _init_model(self, num_classes):
        """
        Initialize the Faster R-CNN model with the specified number of classes.
        """
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # Replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def train_model(self, train_loader, valid_loader, epochs=10, patience=3, lr=0.005):
        """
        Train the model using the provided training and validation data loaders.
        """
        self.model.train()

        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
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

                print(f"\rEpoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {losses.item():.4f} ", end="")

            train_loss /= len(train_loader)

            # Validation loop
            val_loss = self.validation_check(valid_loader)
            print(f"\nEpoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} ")

            # Early stopping based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model()
            else:
                patience_counter += 1
                print(f"Patience Counter: {patience_counter}")
                if patience_counter >= patience:
                    print("Stopped training due to overfitting.")
                    break
        print(f"Training complete. Saved model to {self.model_path}")

    def validation_check(self, data_loader):
        """
        Evaluate the model on the validation set and compute the loss.
        """
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
        """
        Save the trained model to disk.
        """
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")
    def load_model(self):
        """
        Load the model from disk for inference.
        """
        self.model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        print("Model loaded for inference.")



def main(train=True, test=True, inference=True):
    # Define file paths
    HRPlanes_path_train = 'datasets/HRPlanes_coco/train/'
    annotation_HRPlanes_path_train = 'datasets/HRPlanes_coco/train/_annotations.coco.json'

    HRPlanes_path_valid = 'datasets/HRPlanes_coco/valid/'
    annotation_HRPlanes_path_valid = 'datasets/HRPlanes_coco/valid/_annotations.coco.json'

    HRPlanes_path_test = 'datasets/HRPlanes_coco/test/'
    annotation_HRPlanes_path_test = 'datasets/HRPlanes_coco/test/_annotations.coco.json'

    external_images_path = 'datasets/external images/'

    model_path = 'models/fasterrcnn.pth'
    os.makedirs('models/', exist_ok=True)

    test_output_path = "inference outputs/faster rcnn/HRPlanes test"
    os.makedirs(test_output_path, exist_ok=True)

    external_output_path = "inference outputs/faster rcnn/external images"
    os.makedirs(external_output_path, exist_ok=True)

    # Create dataset without transformations for mean,std calculation
    dataset_for_stats = COCOPlanesDataset(
        root_dir=HRPlanes_path_train,
        annotation_file=annotation_HRPlanes_path_train,
        transform=T.ToTensor()
    )

    # Calculate custom mean,std for normalization
    mean, std = compute_mean_std(dataset_for_stats)
    mean = mean.tolist()
    std = std.tolist()

    # Define Transformation of dataset
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

    # Instantiate the dataset
    train_dataset = COCOPlanesDataset(root_dir=HRPlanes_path_train,
                                      annotation_file=annotation_HRPlanes_path_train,
                                      transform=transform)
    
    valid_dataset = COCOPlanesDataset(root_dir=HRPlanes_path_valid,
                                      annotation_file=annotation_HRPlanes_path_valid,
                                      transform=transform)
    
    test_dataset = COCOPlanesDataset(root_dir=HRPlanes_path_test,
                                     annotation_file=annotation_HRPlanes_path_test,
                                     transform=transform)
    
    unlabeled_dataset = UnlabeledImagesDataset(root_dir=external_images_path,
                                                transform=transform)

    # Load data to dataloaders
    train_loader = DataLoader(train_dataset, batch_size=18, shuffle=True, pin_memory=True, num_workers=10, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=18, shuffle=False, pin_memory=True, num_workers=10, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=18, shuffle=False, pin_memory=True, num_workers=10, collate_fn=collate_fn)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=18, shuffle=False, pin_memory=True, num_workers=10, collate_fn=collate_fn)
    


    # Train model and save
    detector = PlaneDetector(num_classes=2, model_path=model_path)

    if train:
        detector.train_model(train_loader, valid_loader, epochs=50, patience=3, lr=0.01)
    # Check if model file exists
    if not os.path.exists(model_path):
        print("Model file not found. Make sure the model is trained and saved.")
        return

    # Load the trained model
    model = detector.model
    detector.load_model()

    # # Load the trained model
    # model = PlaneDetector(num_classes=2, model_path=model_path).model

    # model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # Create Evaluator instance and compute metrics
    evaluator = Inference(model, device, threshold = 0.97, output_dir=test_output_path)

    if test:
        eval_results = evaluator.evaluate(test_loader)
        evaluator.save_predictions(eval_results, mean, std)

    if inference:
        eval_results = evaluator.predict(unlabeled_loader)
        evaluator.save_predictions(eval_results, mean, std)

if __name__ == '__main__':
    main(train=True, test=True, inference=True)

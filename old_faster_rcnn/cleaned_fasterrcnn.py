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
import torchvision.transforms.functional as F
import matplotlib.patches as patches
from pycocotools.coco import COCO

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

        return image


def compute_mean_std(dataset):
    """
    Compute the mean and standard deviation of a dataset for normalization.
    """
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
    """
    Denormalize an image tensor using the provided mean and std.
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    denormalized_image = image * std + mean
    denormalized_image = torch.clamp(denormalized_image, 0, 1)
    return denormalized_image

# Custom collate functions
# HRPlanes images have a varying number of bounding boxes
def collate_fn(batch):
    images, targets = zip(*batch)
    images = list(images)
    targets = list(targets)
    return images, targets


def collate_fn_for_mean_std(batch):
    images, _ = zip(*batch)
    return images

def collate_fn_unlabeled(batch):
    return [image for image in batch]


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

                print(f"\rEpoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {losses.item():.4f} ", end="")

            train_loss /= len(train_loader)

            # Validation loop
            val_loss = self.evaluate(valid_loader)
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

    def evaluate(self, data_loader):
        """
        Evaluate the model on the validation set and compute the loss.
        """
        self.model.train()  # Set model to evaluation mode
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
        # self.model.train()
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


class InferenceEngine:
    """
    Class for performing inference and visualization using a trained model.
    """
    def __init__(self, model_path='models/fasterrcnn.pth', num_classes=2, threshold=0.5, output_dir='Inference/faster rcnn'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._init_model(num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        self.threshold = threshold
        self.output_dir = output_dir

    def _init_model(self, num_classes):
        """
        Initialize the Faster R-CNN model with the specified number of classes.
        """
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # Replace the pre-trained head with a new one
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
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=18, shuffle=False, pin_memory=True, num_workers=10, collate_fn=collate_fn_unlabeled)
    


    # Train model and save
    if train:
        detector = PlaneDetector(num_classes=2, model_path=model_path)
        detector.train_model(train_loader, valid_loader, epochs=50, patience=3, lr=0.01)

    # Check if model file exists
    if not os.path.exists(model_path):
        print("Model file not found. Make sure the model is trained and saved.")
        return



    # Inference model on HRPlanes test dataset and save bound boxed images to dir
    if test:
        test_inference_engine = InferenceEngine(model_path=model_path, num_classes=2, threshold=0.97, output_dir=test_output_path)
        test_results = test_inference_engine.predict(test_loader)
        test_inference_engine.visualize_predictions(test_results,mean,std)



    # Perform inference and save bound boxed images to dir
    if inference:
        external_inference_engine = InferenceEngine(model_path=model_path, num_classes=2, threshold=0.5, output_dir=external_output_path)
        external_images_results = external_inference_engine.predict(unlabeled_loader)
        external_inference_engine.visualize_predictions(external_images_results, mean, std)


if __name__ == '__main__':
    main(train=True, test=True, inference=True)

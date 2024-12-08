import torch
import torchvision
import torchvision.transforms as T
from torchvision.transforms.functional import crop
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image


import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
from contextlib import redirect_stdout

from faster_rcnn.config import (
    TRAIN_DIR,TRAIN_ANNOTATIONS,LOG_MODEL_PATH,DEVICE
)


class FasterRCNNModel:
    def __init__(self, num_classes, backbone='resnet50', pretrained=True):
        self.num_classes = num_classes
        self.backbone = backbone
        self.pretrained = pretrained
        self.model = self._initialize_model()
        self.model.to(DEVICE)

    def _initialize_model(self):
        # Initialize a Faster R-CNN model with a ResNet-50-FPN backbone
        if self.backbone == 'resnet50':
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=self.pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
        
        # Replace the box predictor to match num_classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        return model

    def extract_region_proposals(self, images, max_proposals=100):
        print(f"Extracting region proposals for {len(images)} image(s)...")
        self.model.eval()
        with torch.no_grad():
            images = [T.ToTensor()(img).to(DEVICE) for img in images]
            image_list, _ = self.model.transform(images)
            features = self.model.backbone(image_list.tensors)
            proposals, scores = self.model.rpn(image_list, features)

        # Process each image's proposals
        batch_proposals = []
        for proposal in proposals:
            batch_proposals.append(proposal[:max_proposals])
        print(f"Extracted {sum(len(p) for p in batch_proposals)} total proposals.")
        return batch_proposals






def prepare_data(train_dataset, faster_rcnn, num_samples=100):
    features = []
    labels = []
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

    for i in range(num_samples):
        print(f"Processing sample {i + 1}/{num_samples}...")  # Progress indicator

        image, target = train_dataset[i]  # image is a PIL Image
        proposals = faster_rcnn.extract_region_proposals([image])[0]

        # Convert the image to a NumPy array (RGB)
        image_np = np.array(image)  # Expected shape: (H, W, 3)

        for box in proposals:
            x_min, y_min, x_max, y_max = box.int().tolist()
            
            # Crop the region
            cropped_region = image_np[y_min:y_max, x_min:x_max, :]
            
            # Convert to PIL Image
            cropped = Image.fromarray(cropped_region.astype(np.uint8))  # Ensure uint8 type
            
            # Apply the transformations
            resized = transform(cropped).flatten()
            features.append(resized.numpy())

            # Label: 1 if there's at least one annotated object in the image, else 0
            labels.append(1 if target["boxes"].size(0) > 0 else 0)

    print("Data preparation complete.")  # Final message
    return np.array(features), np.array(labels)



class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)


import os

def visualize_detections(image, boxes, scores, labels, output_dir, image_id, threshold=0.5):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving detection visualization for image {image_id}...")  # Progress indicator

    # Convert tensor to NumPy array if needed (assumes image is torch.Tensor)
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()

    # Plot the image
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Draw each box
    for box, score, label in zip(boxes, scores, labels):
        if score > threshold:
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                x_min, y_min - 10,
                f'{label}: {score:.2f}', color='yellow',
                fontsize=10, weight='bold'
            )

    plt.axis('off')
    output_path = os.path.join(output_dir, f"detection_{image_id}.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved detection visualization to {output_path}")  # Log output file



class COCOCustomDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Suppress COCO stdout
        with redirect_stdout(open(os.devnull, "w")):
            self.coco = COCO(annotation_file)

        # Filter out images with no annotations
        self.image_ids = [
            img_id for img_id in self.coco.getImgIds()
            if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        ]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Retry logic in case of missing files or errors
        while True:
            try:
                image_id = self.image_ids[idx]
                img_info = self.coco.loadImgs(image_id)[0]
                img_path = os.path.join(self.root_dir, img_info['file_name'])

                # Load the image as a PIL image and ensure RGB
                image = Image.open(img_path).convert("RGB")

                # Load annotations for the image
                ann_ids = self.coco.getAnnIds(imgIds=image_id)
                anns = self.coco.loadAnns(ann_ids)

                boxes = []
                labels = []
                for ann in anns:
                    x, y, w, h = ann['bbox']
                    boxes.append([x, y, x + w, y + h])
                    labels.append(ann['category_id'])

                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)
                target = {"boxes": boxes, "labels": labels}

                # If a transform is provided, apply it
                # Typically transform might be T.ToTensor(), which expects a PIL image
                if self.transform:
                    image = self.transform(image)

                return image, target

            except FileNotFoundError:
                # If the file is missing, move to the next index
                idx = (idx + 1) % len(self.image_ids)
            except Exception:
                # On any other error, also move to the next index
                idx = (idx + 1) % len(self.image_ids)

if __name__ == "__main__":
    print("Loading dataset...")
    train_dataset = COCOCustomDataset(root_dir=TRAIN_DIR, annotation_file=TRAIN_ANNOTATIONS, transform=None)
    print(f"Dataset loaded. Total samples: {len(train_dataset)}")

    print("Initializing Faster R-CNN model...")
    faster_rcnn = FasterRCNNModel(num_classes=2)

    print("Preparing data for logistic regression...")
    num_samples = 20  # Reduce the number of samples
    X, y = prepare_data(train_dataset, faster_rcnn, num_samples=num_samples)

    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training logistic regression model...")
    log_reg = LogisticRegressionModel()
    log_reg.train(X_train, y_train)

    print("Evaluating logistic regression model...")
    accuracy = log_reg.evaluate(X_test, y_test)
    print(f"Logistic Regression Accuracy: {accuracy:.2f}")

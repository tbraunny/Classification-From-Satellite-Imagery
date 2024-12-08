import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from pycocotools.cocoeval import COCOeval
import json
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


class ObjectDetectionModel:
    """
    Generalized Object Detection Model using Faster R-CNN.
    Supports custom configurations for object detection tasks.
    """

    def __init__(self, num_classes, model_path='models/faster_rcnn.pth', backbone='resnet50', pretrained=True):
        """
        Initialize the object detection model.
        :param num_classes: Number of object categories (including background).
        :param model_path: Path to save or load the model.
        :param backbone: Backbone network for the Faster R-CNN.
        :param pretrained: Whether to use pretrained weights for the backbone.
        """
        print(f"Num classes: {num_classes}")
        self.num_classes = num_classes
        self.model_path = model_path
        self.backbone = backbone
        self.pretrained = pretrained
        self.model = self._initialize_model()
        self.model.to(DEVICE)

    def _initialize_model(self):
        """
        Initialize the Faster R-CNN model with the specified backbone and custom head.
        """
        if self.backbone == 'resnet50':
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if self.pretrained else None
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")

        # Replace the classification head with a new one
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        return model


    def train(self, train_loader, val_loader, num_epochs=10, learning_rate=0.005, patience=3):
        """
        Train the model using the provided data loaders.
        :param train_loader: DataLoader for training data.
        :param val_loader: DataLoader for validation data.
        :param num_epochs: Number of training epochs.
        :param learning_rate: Learning rate for the optimizer.
        :param patience: Number of epochs to wait for validation loss improvement before early stopping.
        """
        self.model.train()

        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):            
            # train_loss = self._train_one_epoch(train_loader, optimizer, epoch, num_epochs)
            total_loss = 0

            for batch_idx, (images, targets) in enumerate(train_loader):
                images = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                optimizer.zero_grad()

                # Calculate loss
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()
                total_loss += losses.item()

                print(f"\rEpoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {losses.item():.4f} ", end="")

            total_loss /+ len(train_loader)
            val_loss = self._validate(val_loader)
            scheduler.step(val_loss)

            print(f"\nEpoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f} ")


            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model()
            else:
                patience_counter += 1
                print(f"Patience Counter: {patience_counter}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    def _validate(self, val_loader):
        """
        Validate the model on the validation dataset.
        """
        total_loss = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()

        return total_loss / len(val_loader)

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):

        self.model.load_state_dict(torch.load(self.model_path, map_location=DEVICE,weights_only=True))
        self.model.to(DEVICE)
        self.model.eval()
        print("Model loaded for inference.")


    def evaluate_model(self,model, data_loader, coco_gt, output_json="detections.json", confidence_threshold=0.5):
        model.eval()
        results = []
        image_ids = data_loader.dataset.image_ids

        with torch.no_grad():
            for img_idx, (images, targets) in enumerate(data_loader):
                img_ids = [data_loader.dataset.image_ids[idx] for idx in range(img_idx * data_loader.batch_size, 
                                                                            min((img_idx+1)*data_loader.batch_size,
                                                                                len(data_loader.dataset)))]
                images = [img.to(DEVICE) for img in images]
                outputs = model(images)

                for i, output in enumerate(outputs):
                    boxes = output['boxes'].cpu().numpy()
                    scores = output['scores'].cpu().numpy()
                    labels = output['labels'].cpu().numpy()

                    # Filter out low-confidence detections
                    keep = scores >= confidence_threshold
                    boxes = boxes[keep]
                    scores = scores[keep]
                    labels = labels[keep]

                    # Convert boxes to COCO format: [x, y, w, h]
                    for box, score, label in zip(boxes, scores, labels):
                        x1, y1, x2, y2 = box
                        w = x2 - x1
                        h = y2 - y1
                        result = {
                            "image_id": img_ids[i],
                            "category_id": int(label),
                            "bbox": [float(x1), float(y1), float(w), float(h)],
                            "score": float(score)
                        }
                        results.append(result)

        # Save predictions to file
        with open(output_json, "w") as f:
            json.dump(results, f, indent=4)

        # Load results in COCO and run evaluation
        coco_dt = coco_gt.loadRes(output_json)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats
        return stats

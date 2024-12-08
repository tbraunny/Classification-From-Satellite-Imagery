import torch
from torch.utils.data import DataLoader
from faster_rcnn.config import TRAIN_DIR, TRAIN_ANNOTATIONS, VAL_DIR, VAL_ANNOTATIONS, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, PATIENCE
from faster_rcnn.datasets.coco_custom_dataset import COCOCustomDataset
from faster_rcnn.models.object_detection_model import ObjectDetectionModel
from faster_rcnn.transforms.augmentations import get_train_transform, get_test_transform
from faster_rcnn.utils.collate import collate_fn
import os

def train_model():
    os.makedirs("models", exist_ok=True)

    train_dataset = COCOCustomDataset(
        root_dir=TRAIN_DIR,
        annotation_file=TRAIN_ANNOTATIONS,
        transform=get_train_transform()
    )

    val_dataset = COCOCustomDataset(
        root_dir=VAL_DIR,
        annotation_file=VAL_ANNOTATIONS,
        transform=get_test_transform()
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4, collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4, collate_fn=collate_fn
    )

    model = ObjectDetectionModel(
        num_classes=len(train_dataset.coco.getCatIds()),
        model_path="models/faster_rcnn.pth",
        backbone="resnet50",
        pretrained=True
    )

    model.train_model(
        train_loader, val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        patience=PATIENCE
    )

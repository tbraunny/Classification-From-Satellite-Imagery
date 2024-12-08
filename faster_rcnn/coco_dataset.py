import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
from pycocotools.coco import COCO
from contextlib import redirect_stdout

class COCOCustomDataset(Dataset):
    """
    Custom Dataset class for COCO-format datasets.
    Handles ambiguous categories, bounding boxes, and optional segmentation masks.
    """
    def __init__(self, root_dir, annotation_file, transform=None):
        """
        Initialize the dataset.
        :param root_dir: Root directory where images are stored.
        :param annotation_file: Path to the COCO annotation file (JSON format).
        :param transform: Transformations to be applied to the images.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Suppress COCO API prints
        with redirect_stdout(open(os.devnull, "w")):
            self.coco = COCO(annotation_file)
        
        # Get image IDs with annotations
        self.image_ids = [
            img_id for img_id in self.coco.getImgIds()
            if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        ]

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        while True:
            try:
                image_id = self.image_ids[idx]
                img_info = self.coco.imgs[image_id]
                img_path = os.path.join(self.root_dir, img_info['file_name'])

                # Load image
                image = Image.open(img_path).convert("RGB")
                image = np.array(image)  # Albumentations requires NumPy arrays

                # Load annotations
                annotation_ids = self.coco.getAnnIds(imgIds=image_id)
                annotations = self.coco.loadAnns(annotation_ids)

                # Initialize bounding boxes and labels
                boxes = []
                labels = []

                for ann in annotations:
                    x, y, width, height = ann['bbox']
                    boxes.append([x, y, x + width, y + height])
                    labels.append(ann['category_id'])

                # Apply transformations
                if self.transform:
                    transformed = self.transform(
                        image=image,
                        bboxes=boxes if boxes else [],
                        labels=labels if labels else []
                    )
                    image = transformed["image"]
                    boxes = transformed["bboxes"]
                    labels = transformed["labels"]

    
                # Convert to PyTorch tensors
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)

                target = {"boxes": boxes, "labels": labels}
                return image, target


                """ Working for visual but needs to be formatted different"""
                # # Apply Albumentations transformations
                # if self.transform:
                #     transformed = self.transform(
                #         image=image,
                #         bboxes=boxes if boxes else [],
                #         category_ids=category_ids if category_ids else []
                #     )
                #     image = transformed["image"]
                #     boxes = transformed["bboxes"]
                #     category_ids = transformed["category_ids"]

    

                # # Convert to PyTorch tensors
                # boxes = torch.as_tensor(boxes, dtype=torch.float32)
                # category_ids = torch.as_tensor(category_ids, dtype=torch.int64)

                # target = {"boxes": boxes, "labels": category_ids}
                # return image, target

            except FileNotFoundError:
                print(f"Skipping missing file at index {idx}")
                idx = (idx + 1) % len(self.image_ids)

            except Exception as e:
                print(f"Error processing index {idx}: {e}")
                idx = (idx + 1) % len(self.image_ids)

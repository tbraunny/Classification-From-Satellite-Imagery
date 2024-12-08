import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
from contextlib import redirect_stdout

class COCOCustomDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        with redirect_stdout(open(os.devnull, "w")):
            self.coco = COCO(annotation_file)
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

                image = Image.open(img_path).convert("RGB")
                image = np.array(image)

                ann_ids = self.coco.getAnnIds(imgIds=image_id)
                anns = self.coco.loadAnns(ann_ids)

                boxes = []
                labels = []

                for ann in anns:
                    x, y, w, h = ann['bbox']
                    boxes.append([x, y, x + w, y + h])
                    labels.append(ann['category_id'])

                if self.transform:
                    transformed = self.transform(image=image, bboxes=boxes, labels=labels)
                    image = transformed["image"]
                    boxes = transformed["bboxes"]
                    labels = transformed["labels"]

                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)

                target = {"boxes": boxes, "labels": labels}

                return image, target

            except FileNotFoundError:
                idx = (idx + 1) % len(self.image_ids)
            except Exception as e:
                idx = (idx + 1) % len(self.image_ids)

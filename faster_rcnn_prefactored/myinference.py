import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F




DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", DEVICE)


class Inference:
    def __init__(self, model, confidence_threshold=0.5, output_dir="./"):

        self.model = model.model
        self.threshold=confidence_threshold
        self.output_dir=output_dir
        self.device = DEVICE
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
    
    def save_predictions(self, results, mean, std):
        """
        Visualize predictions and save the images with bounding boxes.
        """
        for i, (img, boxes, scores) in enumerate(results):
            # Denormalize the image
            img = denormalize_image(img.cpu(), mean, std)

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
        image = np.array(image)  # Convert PIL image to numpy array for Albumentations

        if self.transform:
            augmented = self.transform(image=image)  # Apply the transform with the named argument
            image = augmented["image"]

        place_holder = None
        return image, place_holder


def denormalize_image(image, mean, std):
    """
    Denormalize an image.
    :param image: Tensor of shape (C, H, W).
    :param mean: Tuple of means for each channel.
    :param std: Tuple of standard deviations for each channel.
    :return: Denormalized image as a numpy array.
    """
    image = image.cpu().numpy().transpose(1, 2, 0)  # Convert to (H, W, C) format
    image = (image * std) + mean  # Denormalize
    image = (image * 255).clip(0, 255).astype(np.uint8)  # Convert to uint8
    return image


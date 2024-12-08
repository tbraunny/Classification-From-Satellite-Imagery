import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as F
from faster_rcnn.config import DEVICE, MEAN, STD
from faster_rcnn.utils.image_utils import denormalize_image

class InferenceRunner:
    def __init__(self, model, threshold, output_dir):
        self.model = model.model
        self.threshold = threshold
        self.output_dir = output_dir
        self.device = DEVICE

    def predict(self, dataloader):
        self.model.eval()
        results = []
        with torch.no_grad():
            for images, _ in dataloader:
                images = [img.to(self.device) for img in images]
                outputs = self.model(images)

                for img, output in zip(images, outputs):
                    boxes = output['boxes'][output['scores'] >= self.threshold]
                    scores = output['scores'][output['scores'] >= self.threshold]
                    results.append((img.cpu(), boxes.cpu(), scores.cpu()))
        return results

    def save_predictions(self, results):
        os.makedirs(self.output_dir, exist_ok=True)
        for i, (img, boxes, scores) in enumerate(results):
            img = denormalize_image(img, MEAN, STD)
            fig, ax = plt.subplots(1)
            ax.imshow(img)

            for box, score in zip(boxes, scores):
                x_min, y_min, x_max, y_max = box
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                         linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x_min, y_min, f'{score:.2f}', color='yellow', fontsize=10, weight='bold')

            plt.axis('off')
            output_path = os.path.join(self.output_dir, f"prediction_{i}.png")
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
            print(f"Saved prediction to {output_path}")

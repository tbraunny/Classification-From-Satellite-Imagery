import matplotlib.pyplot as plt
import cv2
import os
import matplotlib.patches as patches
import numpy as np


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


def save_predictions(predictions, dataset, output_dir, label_map=None, confidence_threshold=0.5):
    """
    Save predictions to disk as images with bounding boxes.
    :param predictions: List of predictions from the model.
    :param dataset: Dataset used for inference.
    :param output_dir: Directory to save the output images.
    :param label_map: Optional dictionary mapping label IDs to category names.
    :param confidence_threshold: Minimum confidence score to display a bounding box.
    """
    os.makedirs(output_dir, exist_ok=True)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    for idx, prediction in enumerate(predictions):
        # Get the original image and denormalize it
        image, _ = dataset[idx]
        denormalized_image = denormalize_image(image, mean, std)

        # Convert to RGB for visualization
        image_rgb = cv2.cvtColor(denormalized_image, cv2.COLOR_BGR2RGB)

        # Create a figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_rgb)

        # Plot bounding boxes
        boxes = prediction['boxes']
        labels = prediction['labels']
        scores = prediction['scores']
        
        for box, label, score in zip(boxes, labels, scores):
            if score < confidence_threshold:
                continue  # Skip low-confidence detections

            x_min, y_min, x_max, y_max = box.cpu().numpy()
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Map label ID to category name if available
            label_name = label_map[label.item()] if label_map else str(label.item())
            ax.text(
                x_min, y_min - 10, f"{label_name} ({score:.2f})",
                color='red', fontsize=12, backgroundcolor='white'
            )

        ax.axis('off')

        # Save the image
        output_path = os.path.join(output_dir, f"prediction_{idx}.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved predictions to {output_dir}")


""" Uses coco annotations and not converted annotations"""

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    x_min, y_min, w, h = bbox
    x_min, x_max = int(x_min), int(x_min + w)
    y_min, y_max = int(y_min), int(y_min + h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    ((text_width, text_height), _) = cv2.getTextSize(
        class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
    )
    cv2.rectangle(
        img,
        (x_min, y_min - int(1.3 * text_height)),
        (x_min + text_width, y_min),
        BOX_COLOR, -1
    )
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img

def visualize(image, bboxes, category_ids, category_id_to_name, output_dir="visualizations/", file_prefix="visualized"):
    os.makedirs(output_dir, exist_ok=True)
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name.get(category_id, str(category_id))
        img = visualize_bbox(img, bbox, class_name)
    existing_files = os.listdir(output_dir)
    file_indices = [int(f.split('_')[-1].split('.')[0]) for f in existing_files if f.startswith(file_prefix) and f.endswith(".jpg")]
    next_index = max(file_indices, default=0) + 1
    output_path = os.path.join(output_dir, f"{file_prefix}_{next_index:03d}.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Saved visualized image to {output_path}")


def before_after(train_dataset, train_dataset_no_transform):
    category_id_to_name = {0: 'Aircraft'}

    # Let's pick a specific index to visualize both before and after
    # Make sure this index is valid (e.g., 0)
    idx = 0  
    # Get the original image and targets (no transform)
    orig_img, orig_target = train_dataset_no_transform[idx]
    # Get the transformed image and targets
    trans_img, trans_target = train_dataset[idx]

    # Before Transformation Visualization
    # orig_img is a NumPy array (H, W, C)
    # orig_target["boxes"] is a set of coordinates in [x_min, y_min, w, h] format
    orig_bboxes = orig_target["boxes"].numpy()
    # For 'before' image, we have original scale bboxes. No normalization needed.
    # Category IDs
    orig_labels = orig_target["labels"].numpy()

    # Visualize "before"
    visualize(orig_img, orig_bboxes, orig_labels, category_id_to_name, output_dir="visualizations", file_prefix="before")

    # After Transformation Visualization
    # trans_img is likely a tensor in (C, H, W) format due to ToTensorV2
    # Convert it back to numpy for visualization
    if isinstance(trans_img, torch.Tensor):
        trans_img_np = trans_img.permute(1, 2, 0).cpu().numpy()
        # Denormalization step if you want to see the image in original colors:
        # Since we normalized using mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225), undo that:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        trans_img_np = (trans_img_np * std + mean)
        trans_img_np = np.clip(trans_img_np, 0, 1)
        trans_img_np = (trans_img_np * 255).astype(np.uint8)
    else:
        # If already numpy, just copy
        trans_img_np = trans_img.copy()

    trans_bboxes = trans_target["boxes"].cpu().numpy()
    # The code in the dataset converts them already to absolute pixel coordinates,
    # but since they came from Albumentations in "coco" format, let's confirm they are [x_min, y_min, x_max, y_max].
    # The code you currently have changes them to [x_min_pixel, y_min_pixel, x_max_pixel, y_max_pixel].
    # visualize function expects coco format: [x_min, y_min, w, h]
    # So we need to convert [x_min, y_min, x_max, y_max] back to [x_min, y_min, w, h]:
    coco_bboxes = []
    for box in trans_bboxes:
        x_min, y_min, x_max, y_max = box
        w = x_max - x_min
        h = y_max - y_min
        coco_bboxes.append([x_min, y_min, w, h])
    coco_bboxes = np.array(coco_bboxes)

    trans_labels = trans_target["labels"].cpu().numpy()

    # Visualize "after"
    visualize(trans_img_np, trans_bboxes, trans_labels, category_id_to_name, output_dir="output/faster_rcnn/box_check/", file_prefix="after")

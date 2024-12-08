import os
from torch.utils.data import DataLoader
from coco_dataset import COCOCustomDataset
from object_detection_model import ObjectDetectionModel
import albumentations as A
from albumentations.pytorch import ToTensorV2
from myinference import Inference, UnlabeledImagesDataset


# HRPlanes images have a varying number of bounding boxes so custom collate needed
def collate_fn_HRPlanes(batch):
    images, targets = zip(*batch)
    images = list(images)
    targets = list(targets)
    return images, targets

def get_mean():
    return (0.485, 0.456, 0.406)
def get_std():
    return (0.229, 0.224, 0.225)

def get_augmentations():
    return A.Compose(
        [
            A.RandomScale(scale_limit=(-0.8, 0.5), p=0.9),
            A.RandomSizedBBoxSafeCrop(height=512, width=512, p=0.5),
            A.Resize(height=512, width=512, p=1.0),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Normalize(mean=get_mean(), std=get_std()),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


def get_test_augmentations():
    return A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )

def get_inference_augmentations():
    return A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )



def main(train=True, test=True, inference=True):
       
    # Define dataset paths
    train_dir = "data/HRPlanes_coco/train/"
    train_annotations = "data/HRPlanes_coco/train/_annotations.coco.json"

    val_dir = "data/HRPlanes_coco/valid/"
    val_annotations = "data/HRPlanes_coco/valid/_annotations.coco.json"

    test_dir = "data/HRPlanes_coco/test/"
    test_annotations = "data/HRPlanes_coco/test/_annotations.coco.json"

    inference_dir = "data/scenes/"

    model_path = "model_weights/faster_rcnn.pth"
    os.makedirs("model_weights/", exist_ok=True)

    test_output_path = "output/faster_rcnn/test/"
    os.makedirs(test_output_path, exist_ok=True)
    inference_output_path = "output/faster_rcnn/test/"
    os.makedirs(inference_output_path, exist_ok=True)

    # Define Albumentations transformations
    transform = get_augmentations()
    transform_test = get_test_augmentations()


    # Initialize datasets
    train_dataset = COCOCustomDataset(
        root_dir=train_dir,
        annotation_file=train_annotations,
        transform=transform
    )

    val_dataset = COCOCustomDataset(
        root_dir=val_dir,
        annotation_file=val_annotations,
        transform=transform_test
    )

    test_dataset = COCOCustomDataset(
        root_dir=test_dir,
        annotation_file=test_annotations,
        transform=transform_test
    )

    batch_size = 10
   
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4, collate_fn=collate_fn_HRPlanes
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, collate_fn=collate_fn_HRPlanes
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, collate_fn=collate_fn_HRPlanes
    )

    # Initialize the model
    model = ObjectDetectionModel(
        # num_classes=train_dataset.coco.getCatIds().__len__() + 1,  # Add 1 for background class
        num_classes=train_dataset.coco.getCatIds().__len__(),
        model_path=model_path,
        backbone="resnet50", 
        pretrained=True
    )

    
    if train:
        # Train the model
        print("Starting Training...")
        model.train(train_loader, val_loader, num_epochs=50, learning_rate=0.005, patience=3)
        # Save the trained model
        model.save_model()

    # Load the model for inference
    model.load_model()

    if test:
        # Perform inference on the test dataset
        print("\Evaluating...")
        stats = model.evaluate_model(model.model, test_loader, test_dataset.coco, output_json="output/faster_rcnn/eval_metrics.json", confidence_threshold=0.93)
        print("Evaluation Stats:", stats)

    if inference:
        print("\nStarting Inference...")
        evaluator = Inference(model, confidence_threshold = 0.80, output_dir=inference_output_path)

        inference_transform = get_inference_augmentations()

        unlabeled_dataset = UnlabeledImagesDataset(root_dir=inference_dir,
                                                transform=inference_transform)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=10, shuffle=False, pin_memory=True, num_workers=4, collate_fn=collate_fn_HRPlanes)

        eval_results = evaluator.predict(unlabeled_loader)
        evaluator.save_predictions(eval_results, get_mean(), get_std())


if __name__ == '__main__':
    main(train=False, test=True, inference=True)

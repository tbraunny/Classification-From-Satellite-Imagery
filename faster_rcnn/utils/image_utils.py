import numpy as np

def denormalize_image(image, mean, std):
    image = image.numpy().transpose(1, 2, 0)
    image = (image * std) + mean
    image = (image * 255).clip(0, 255).astype(np.uint8)
    return image

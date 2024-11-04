import numpy as np
import pandas as pd
import torch as nn
import os
from glob import glob
import cv2

# load dataset
def load_dataset():
    basepath = 'C:\\Undergraduate\\Year 4\\CS 482\\final_project\\planesnet\\planesnet'
    images , labels = [] , []

    # loop thru each class to find all files with name starting with label_*
    for label in range(2):
        imgs_path = os.path.join(basepath , f"{label}_*")

        # prep image for addition to list
        for file in imgs_path:
            img = cv2.imread(file)
            img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB) # convert BGR to RGB
            images.append(img)
            labels.append(label)

    # convert to np arrays for easier handling
    images = np.array(images , dtpye=np.int64)
    labels = np.array(labels , dtpye=np.int64)

    return images , labels

# define model network



# train model



# predict



# evaluate


if __name__ == '__main__':
    images , labels = load_dataset()
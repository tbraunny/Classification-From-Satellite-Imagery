import torch
import os
import cv2
import glob
import numpy as np
from torch.utils.data import TensorDataset , DataLoader
from sklearn.model_selection import train_test_split


def load_dataset():

    # if planesnet_tensors.pt exists, load the file
    saved_models_path = "data/saved_models"
    img_tensors = os.path.join(saved_models_path , "planesnet_tensors.pt")

    if os.path.exists(img_tensors):
        # Load the tensors
        print(f"Loading tensor data...")
        data = torch.load(img_tensors , weights_only=False)
        images, labels = data[0], data[1]  

        print(f"Tensor data loaded")  
    else:
        print("Loading image data...")
        data_path = "data/planesnet"
        images , labels = [] , []

        # loop thru each class to find all files with name starting with label_*
        for label in range(2):
            imgs_path = os.path.join(data_path , f"{label}_*")

            # prep image for addition to list
            for file in glob.glob(imgs_path):
                img = cv2.imread(file)
                img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB) # convert BGR to RGB
                images.append(img)
                labels.append(label)

            print(f"{label}\t{len(images)}")

        # convert to np arrays for easier handling
        images = np.array(images , dtype=np.int64)
        labels = np.array(labels , dtype=np.int64)

        images_tensor = torch.tensor(images)
        labels_tensor = torch.tensor(labels)

        # Save tensors to a file
        torch.save((images_tensor, labels_tensor) , os.path.join(saved_models_path , 'planesnet_tensors.pt'))

    return images , labels

def data_preprocess(X , y , flag=False):
    X = X / 255 # normalize
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.3 , random_state=42)

    if (flag == True): # only return test
        return X_test , y_test

    train_data = TensorDataset(X_train , y_train)
    test_data = TensorDataset(X_test , y_test)

    train_dataloader = DataLoader(train_data , batch_size=32 , shuffle=True)
    test_dataloader = DataLoader(test_data , batch_size=32 , shuffle=True)

    return train_dataloader , test_dataloader
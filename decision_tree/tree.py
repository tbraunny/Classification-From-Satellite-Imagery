from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset , DataLoader
import matplotlib.pyplot as plt
import os
from glob import glob
import cv2
import numpy as np
from xgboost import XGBClassifier

def load_dataset():
    print("Loading image data...")
    basepath = os.path.abspath(os.path.join('../dataset'))
    images , labels = [] , []

    # loop through all data
    for label in range(2):
        imgs_path = os.path.join(basepath , f"{label}_*")

        # prep each image for the model
        for file in glob(imgs_path):
            img = cv2.imread(file)
            img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(label)

    # convert to np arrays for easier handling
    images = np.array(images , dtype=np.int64)
    labels = np.array(labels , dtype=np.int64)
    
    print("Data loaded")
    return images , labels

def data_preprocess(X , y):
    X = X / 255 # normalize
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.3 , random_state=42)

    # flatten for xgboost
    X = X.reshape(X.shape[0], -1)

    # convert to data loaders
    train_data = TensorDataset(X_train , y_train)
    test_data = TensorDataset(X_test , y_test)

    train_dataloader = DataLoader(train_data , batch_size=32 , shuffle=True)
    test_dataloader = DataLoader(test_data , batch_size=32 , shuffle=True)

    return train_dataloader , test_dataloader

class DecisionTree():
    def __init__(self , max_depth=5 , lr=0.1 , n_est = 100):
        self.model = XGBClassifier(
            max_depth = max_depth , 
            learning_rate = lr , 
            n_estimators = n_est , 
            objective = 'binary:logistic' , 
            use_label_encoder = False , 
            verbosity = 0
        )

def train_tree(train , test):
    max_depth = 10
    lr = 0.01
    n_est = 1000

    print("Training decision tree model...")
    tree = DecisionTree(max_depth , lr , n_est)
    tree.train()

def plots(train_loss , test_loss , train_accuracy , test_accuracy):
    # loss
    plt.plot(train_loss , color='blue' , label="Train")
    plt.plot(test_loss , color='red' , label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Epochs vs. Loss")
    plt.legend()
    plt.savefig("tree_loss.png")
    plt.show()

    # accuracy
    plt.plot(train_accuracy , color='blue' , label="Train")
    plt.plot(test_accuracy , color='red' , label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Epochs vs. Accuracy")
    plt.legend()
    plt.savefig("tree_accuracy.png")
    plt.show()

if __name__ == '__main__':
    print("decision tree w xg boost")
    X , y = load_dataset()
    train , test = data_preprocess(X , y)
    train_loss , test_loss , train_accuracy , test_accuracy = train_tree(train , test)
    train_loss , test_loss , train_accuracy , test_accuracy = np.array(train_loss) , np.array(test_loss) , np.array(train_accuracy) , np.array(test_accuracy)
    print("Accuracy: " , test_accuracy[-1 , 0])

    plots(train_loss , test_loss , train_accuracy , test_accuracy)
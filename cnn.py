import numpy as np
import pandas as pd
import torch
import torch as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import cv2
from torch.utils.data import TensorDataset , DataLoader
from glob import glob
from sklearn import train_test_split


# load dataset, return images & labels as numpy arrays
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

def data_preprocess(X , y):
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.3 , random_state=42)

    X_train , X_test = torch.tensor(X_train) , torch.tensor(X_test)
    y_train , y_test = torch.tensor(y_train) , torch.tensor(y_test)

    train_data = TensorDataset(X_train , y_train)
    test_data = TensorDataset(X_test , y_test)

    train_dataloader = DataLoader(train_data , batch_size=64 , shuffle=True)
    test_dataloader = DataLoader(test_data , batch_size=64 , shuffle=True)

    return train_dataloader , test_dataloader

# define model network
class CNN():
    def __init__(self):
        super(self , CNN).__init__()

        # layers + dropout
        self.conv1 = nn.conv2d(1 , 32 , 3 , 1)
        self.conv2 = nn.conv2d(32 , 64 , 3 , 1)
        self.fc1 = nn.Linear(9216 , 128)
        self.fc2 = nn.Linear(128 , 2)
        self.dropout1 = nn.dropout(0.25)
        self.dropout2 = nn.dropout(0.5)

    def forward(self , x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x , 2)
        x = self.dropout1(x)
        x = nn.flatten(x , 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x)

        return output


# train model
def train_cnn(X_train , y_train):
    cnn = CNN()

    # hyperparameters
    num_epochs = 100
    learning_rate = 0.001
    criterion = F.nll_loss() # placeholder
    optimizer = nn.optim.Adam(cnn.parameters() , learning_rate)
    train_losses = []

    for epoch in range(num_epochs):
        cnn.train()
        optimizer.zero_grad()
        outputs = cnn(X_train.float())
        loss = criterion(outputs , y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item()) # store loss

    y_pred = []
    return y_pred


def evaluate(pred):
    accuracy = 0
    return accuracy

'''
def save_model():
    torch.save({
            'epoch': num_epochs,
            'model_state_dict': mlp.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_losses[-1],
            }, f'./planesnet_weights.pt')
'''

if __name__ == '__main__':
    images , labels = load_dataset()
    train , test = data_preprocess(images , labels)
    predictions = train_cnn(test)
    accuracy = evaluate(predictions)
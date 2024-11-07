import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import cv2
from torch.utils.data import TensorDataset , DataLoader
from glob import glob
from sklearn.model_selection import train_test_split

# load dataset, return images & labels as numpy arrays
def load_dataset():

    # if planesnet_tensors.pt exists, load those instead
    if os.path.exists("planesnet_tensors.pt"):
        # Load the tensors
        print(f"Loading tensor data...")
        data = torch.load("planesnet_tensors.pt" , weights_only=False)
        images, labels = data[0], data[1]    
    else:
        print("Loading image data...")
        basepath = 'C:\\Undergraduate\\Year 4\\CS 482\\final_project\\planesnet\\planesnet'
        #basepath = '/content/drive/MyDrive/planesnet/'
        images , labels = [] , []

        # loop thru each class to find all files with name starting with label_*
        for label in range(2):
            imgs_path = os.path.join(basepath , f"{label}_*")

            # prep image for addition to list
            for file in glob(imgs_path):
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
        torch.save((images_tensor, labels_tensor), f'./planesnet_tensors.pt')

    return images , labels

def data_preprocess(X , y):
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.3 , random_state=42)

    train_data = TensorDataset(X_train , y_train)
    test_data = TensorDataset(X_test , y_test)

    train_dataloader = DataLoader(train_data , batch_size=256 , shuffle=True)
    test_dataloader = DataLoader(test_data , batch_size=256 , shuffle=True)

    return train_dataloader , test_dataloader

# define model network
class CNN(nn.Module):
    def __init__(self):
        super(CNN , self).__init__()

        # layers + dropout
        self.conv1 = nn.Conv2d(20 , 32 , kernel_size=3 , stride=1 , padding=1)
        self.conv2 = nn.Conv2d(32 , 64 , kernel_size=3 , stride=1 , padding=1)
        self.fc1 = nn.Linear(640 , 128)
        self.fc2 = nn.Linear(128 , 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self , x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.bn1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.bn2(x)
        x = F.max_pool2d(x , 2)

        x = x.view(x.size(0) , -1) # Flatten the tensor

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc2(x)
        output = F.log_softmax(x , dim=1)

        return output


# train model
def train_cnn(train_loader , test_loader , device):
    cnn = CNN().to(device)

    # hyperparameters
    num_epochs = 10
    learning_rate = 0.01
    #criterion = F.nll_loss()
    optimizer = optim.Adam(cnn.parameters() , learning_rate)
    train_losses , test_losses = [] , []

    for epoch in range(num_epochs):
        cnn.train()
        running_loss = 0
        for batch_idx , (data , target) in enumerate(train_loader):
          optimizer.zero_grad()
          outputs = cnn(data.float())
          loss = F.nll_loss(outputs , target)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()

        train_losses.append(running_loss) # store loss
        print(f"Epoch {epoch + 1} \t Loss: {running_loss}")
      
    # predict
    num_correct = 0
    cnn.eval()
    with torch.no_grad():
      for (data , target) in test_loader:
        output = cnn(data.float())
        loss = F.nll_loss(output , target)
        test_losses.append(loss.item())
        pred = output.argmax(dim=1, keepdim=True)
        num_correct += pred.eq(target.view_as(pred)).sum().item()

    return num_correct / len(test_loader.dataset)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: " , device)
    images , labels = load_dataset()
    train , test = data_preprocess(images , labels)
    accuracy = train_cnn(train , test , device)
    print("Accuracy: " , accuracy)

    #save_model()
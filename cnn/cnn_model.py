import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset , DataLoader
from glob import glob
from sklearn.model_selection import train_test_split



# load dataset, return images & labels as numpy arrays
def load_dataset():

    # DEPRECATED if planesnet_tensors.pt exists, load the file
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

# normalize & reshape
def data_preprocess(X , y):
    X = X / 255 # normalize
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.3 , random_state=42)

    train_data = TensorDataset(X_train , y_train)
    test_data = TensorDataset(X_test , y_test)

    train_dataloader = DataLoader(train_data , batch_size=32 , shuffle=True)
    test_dataloader = DataLoader(test_data , batch_size=32 , shuffle=True)

    return train_dataloader , test_dataloader

# define model network
class CNN(nn.Module):
    def __init__(self):
        super(CNN , self).__init__()

        # layers + dropout
        self.conv1 = nn.Conv2d(20 , 32 , kernel_size=3 , stride=1 , padding=1)
        self.conv2 = nn.Conv2d(32 , 64 , kernel_size=3 , stride=1 , padding=1)
        self.fc1 = nn.Linear(640 , 128)
        self.fc2 = nn.Linear(128 , 64)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.25)

    def forward(self , x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        #x = F.max_pool2d(x , 2)
        x = self.bn1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x , 2)
        x = self.bn2(x)

        x = x.view(x.size(0) , -1) # Flatten the tensor

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc2(x)
        output = F.log_softmax(x , dim=1)

        return output

# train model
def train_cnn(train_loader , test_loader):
    cnn = CNN()

    # hyperparameters
    num_epochs = 50
    learning_rate = 0.0001
    optimizer = optim.Adam(cnn.parameters() , learning_rate)
    train_loss , test_loss , train_accuracy , test_accuracy = [] , [] , [] , []
    

    for epoch in range(num_epochs):
        cnn.train()
        running_loss , num_correct = 0 , 0

        for (data , target) in train_loader:
          optimizer.zero_grad()
          train_outputs = cnn(data.float())
          loss = F.nll_loss(train_outputs , target)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()
          pred = train_outputs.argmax(dim=1, keepdim=True)
          num_correct += pred.eq(target.view_as(pred)).sum().item()

        # train accuracy
        train_acc = num_correct / len(train_loader.dataset)
        train_accuracy.append((train_acc , epoch))
        train_loss.append((running_loss , epoch))

        print(f"Epoch {epoch + 1}: Train Loss: {running_loss} \t Train Accuracy: {train_acc}")

        # test accuracy
        cnn.eval()
        running_loss , num_correct = 0 , 0
        with torch.no_grad():
            for (data , target) in test_loader:
                test_outputs = cnn(data.float())
                loss = F.nll_loss(test_outputs , target)
                running_loss += loss.item()
                pred = test_outputs.argmax(dim=1, keepdim=True)
                num_correct += pred.eq(target.view_as(pred)).sum().item()
        test_acc = num_correct / len(test_loader.dataset)

        # store accuracies & losses
        test_accuracy.append((test_acc , epoch))
        test_loss.append((running_loss , epoch))
        print(f"\t Test Loss: {running_loss} \t Test Accuracy: {test_acc}\n") # print loss, accuracy

    torch.save({
            'epoch': num_epochs,
            'model_state_dict': cnn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss[-1],
            }, f'./planesnet_weights.pt')

    return train_loss , test_loss , train_accuracy , test_accuracy

# plot accuracy & loss of models
def plots(train_loss , test_loss , train_accuracy , test_accuracy):
    # loss
    plt.plot(train_loss , color='blue' , label="Train")
    plt.plot(test_loss , color='red' , label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Epochs vs. Loss")
    plt.legend()
    plt.savefig("cnn_loss.png")
    plt.show()

    # accuracy
    plt.plot(train_accuracy , color='blue' , label="Train")
    plt.plot(test_accuracy , color='red' , label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Epochs vs. Accuracy")
    plt.legend()
    plt.savefig("cnn_accuracy.png")
    plt.show()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: " , device)
    images , labels = load_dataset()
    train , test = data_preprocess(images , labels)
    train_loss , test_loss , train_accuracy , test_accuracy = train_cnn(train , test)
    train_loss , test_loss , train_accuracy , test_accuracy = np.array(train_loss) , np.array(test_loss) , np.array(train_accuracy) , np.array(test_accuracy)

    print("Accuracy: " , test_accuracy[-1 , 0])

    plots(train_loss , test_loss , train_accuracy , test_accuracy)
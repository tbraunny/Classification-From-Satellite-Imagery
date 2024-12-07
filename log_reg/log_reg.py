from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset , DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np

def load_tensors():
    if os.path.exists("planesnet_tensors.pt"):
        print("Loading data...")
        data = torch.load("planesnet_tensors.pt" , weights_only=False)
        images , labels = data[0] , data[1]
    else:
        print("Error: File does not exist")
    
    print("Data Loaded")
    return images , labels

def data_preprocess(X , y):
    X = X / 255 # normalize
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.3 , random_state=42)

    train_data = TensorDataset(X_train , y_train)
    test_data = TensorDataset(X_test , y_test)

    train_dataloader = DataLoader(train_data , batch_size=32 , shuffle=True)
    test_dataloader = DataLoader(test_data , batch_size=32 , shuffle=True)
    
    return train_dataloader , test_dataloader

class Log_Reg(nn.Module):
    def __init__(self):
        super(Log_Reg , self).__init__()
        self.linear = nn.Linear(1200 , 2)

    def forward(self , x):
        #x = x.view(x.size(0) , -1)
        x = x.reshape(x.size(0) , -1)
        x = self.linear(x)
        return F.log_softmax(x , dim=1)
    
def train_logisitc_regression(train_loader , test_loader):
    logistic_regression = Log_Reg()

    train_loss , test_loss , train_accuracy , test_accuracy = [] , [] , [] , []

    num_epochs = 100
    learning_rate = 1e-3
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(params=logistic_regression.parameters() , lr=learning_rate)
    
    for epoch in range(num_epochs):
        logistic_regression.train()

        running_loss , num_correct = 0 , 0

        for (data , target) in train_loader:
            optimizer.zero_grad()
            train_outputs = logistic_regression(data.float())
            loss = criterion(train_outputs , target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pred = train_outputs.argmax(dim=1 , keepdim=True)
            num_correct += pred.eq(target.view_as(pred)).sum().item()

        train_acc = num_correct / len(train_loader.dataset)
        train_accuracy.append((train_acc , epoch))
        train_loss.append((running_loss , epoch))
        print(f"Epoch {epoch + 1} : Train Loss {running_loss:.6f} : Train Accuracy {train_acc:.6f}")

        logistic_regression.eval()
        running_loss , num_correct = 0 , 0
        with torch.no_grad():
            for (data , target) in test_loader:
                test_outputs = logistic_regression(data.float())
                loss = criterion(test_outputs , target)
                running_loss += loss.item()
                pred = test_outputs.argmax(dim=1, keepdim=True)
                num_correct += pred.eq(target.view_as(pred)).sum().item()
        test_acc = num_correct / len(test_loader.dataset)

        test_loss.append((running_loss , epoch))
        test_accuracy.append((test_acc , epoch))
        print(f"\t  Test Loss {running_loss:.6f} : Test Accuracy {test_acc:.6f}") # print loss, accuracy

        torch.save({
            'epoch': num_epochs , 
            'model_state_dict': logistic_regression.state_dict() ,
            'loss': train_loss[-1],
            }, f'./log_reg_weights.pt'
        )

    return train_accuracy , train_loss , test_accuracy , test_loss

# plot accuracy & loss of models
def plots(train_loss , test_loss , train_accuracy , test_accuracy):
    # loss
    plt.plot(train_loss , color='blue' , label="Train")
    plt.plot(test_loss , color='red' , label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Epochs vs. Loss")
    plt.legend()
    plt.savefig("log_reg_loss.png")
    plt.show()

    # accuracy
    plt.plot(train_accuracy , color='blue' , label="Train")
    plt.plot(test_accuracy , color='red' , label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Epochs vs. Accuracy")
    plt.legend()
    plt.savefig("log_reg_accuracy.png")
    plt.show()

if __name__ == '__main__':
    X , y = load_tensors()
    train , test = data_preprocess(X , y)
    train_acc , train_loss , test_acc , test_loss = train_logisitc_regression(train , test)
    train_loss , test_loss , train_acc , test_acc = np.array(train_loss) , np.array(test_loss) , np.array(train_acc) , np.array(test_acc)
    print("Final Test Accuracy" , test_acc[-1 , 0])

    plots(train_loss , test_loss , train_acc , test_acc)
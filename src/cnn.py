import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay , classification_report
import matplotlib.pyplot as plt

# define model network
class CNN(nn.Module):
    def __init__(self):
        super(CNN , self).__init__()
        self.conv1 = nn.Conv2d(3 , 20, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(20 , 64 , 3 , 1 , padding=1)
        self.fc1 = nn.Linear(1600 , 128)
        self.fc2 = nn.Linear(128 , 2)
        self.bn1 = nn.BatchNorm2d(20)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.25)

    def forward(self , x):
        # convolution 1
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x , 2)
        x = self.bn1(x)
        x = self.dropout1(x)

        # convolution 2
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x , 2)
        x = self.bn2(x)

        x = x.reshape(x.size(0) , -1) # Flatten the tensor

        # fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x , dim=1)

        return output

# train model
def train_model(train_loader , test_loader):
    print("Training model...")
    cnn = CNN()

    # hyperparameters
    num_epochs = 80
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
        y_pred , y_test = [] , []
        running_loss , num_correct = 0 , 0
        with torch.no_grad():
            for (data , target) in test_loader:
                test_outputs = cnn(data.float())
                loss = F.nll_loss(test_outputs , target)
                running_loss += loss.item()
                pred = test_outputs.argmax(dim=1, keepdim=True)
                y_pred.append(pred)
                y_test.append(target.view_as(pred))
                num_correct += pred.eq(target.view_as(pred)).sum().item()
        test_acc = num_correct / len(test_loader.dataset)

        # store accuracies & losses
        test_accuracy.append((test_acc , epoch))
        test_loss.append((running_loss , epoch))
        print(f"\t Test Loss: {running_loss} \t Test Accuracy: {test_acc}\n") # print loss, accuracy

    y_test = torch.cat(y_test).cpu().numpy() # flatten for analysis
    y_pred = torch.cat(y_pred).cpu().numpy()
    cm = confusion_matrix(y_test , y_pred)
    cm = ConfusionMatrixDisplay(confusion_matrix=cm)
    cm.plot()
    print(classification_report(y_test , y_pred))

    plt.savefig("results/confusion_matrix_cnn.png")

    # save_path = "results"
    # torch.save({
    #         'epoch': num_epochs,
    #         'model_state_dict': cnn.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'loss': train_loss[-1],
    #         }, os.path.join(save_path , 'cnn_weights.pt'))
    
    # print("Training complete")

    return train_loss , test_loss , train_accuracy , test_accuracy
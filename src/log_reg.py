import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
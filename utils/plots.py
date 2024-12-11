import matplotlib.pyplot as plt
import os

# plot loss & accuracy, save to results
def plot(x1 , x2 , model , mode):
    save_dir = "results"

    data1 = [item[0] for item in x1]
    epochs1 = [item[1] for item in x1] 
    data2 = [item[0] for item in x2]
    epochs2 = [item[1] for item in x2] 

    # general
    plt.figure()
    plt.plot(epochs1 , data1 , color='blue' , label="Train")
    plt.plot(epochs2 , data2 , color='red' , label="Test")
    plt.xlabel("Epochs")
    plt.ylabel(mode)
    plt.title(f'Epochs vs. {mode} {model.__name__}')
    plt.legend()
    loss_path = os.path.join(save_dir , f'{mode}_plot_{model.__name__}.png')
    plt.savefig(loss_path)

    # # accuracy
    # plt.figure()
    # plt.plot(train_accuracy , color='blue' , label="Train Accuracy")
    # plt.plot(test_accuracy , color='red' , label="Test Accuracy")
    # plt.xlabel("Epochs")
    # plt.ylabel("Accuracy")
    # plt.title(f'Epochs vs. Accuracy {model.__name__}')    
    # plt.legend()
    # accuracy_path = os.path.join(save_dir , f'accuracy_plot_{model.__name__}.png')
    # plt.savefig(accuracy_path)
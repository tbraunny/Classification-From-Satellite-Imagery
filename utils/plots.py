import matplotlib.pyplot as plt
import os

# plot loss & accuracy, save to results
def plot(train_loss , test_loss , train_accuracy , test_accuracy):
    save_dir = "../results"

    # loss
    plt.plot(train_loss , color='blue' , label="Train")
    plt.plot(test_loss , color='red' , label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Epochs vs. Loss")
    plt.legend()

    loss_path = os.path.join(save_dir , "loss_plot.png")
    plt.savefig(loss_path)
    plt.show()

    # accuracy
    plt.plot(train_accuracy , color='blue' , label="Train")
    plt.plot(test_accuracy , color='red' , label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Epochs vs. Accuracy")
    plt.legend()

    accuracy_path = os.path.join(save_dir , "accuracy_plot.png")
    plt.savefig(accuracy_path)
    plt.show()
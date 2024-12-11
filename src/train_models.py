import utils
import cnn
import log_reg
import xgb
import utils.plots
import utils.process_dataset
#import xgb # not fully integrated yet, runnable from src/xgb.py

# uncomment the model you would like to re-train
#model_to_train = log_reg
#model_to_train = cnn
model_to_train = xgb

##############################################

if (model_to_train == xgb):
    print("training xgboost")
    xgb.classifier.main()

else:
    X , y = utils.process_dataset.load_dataset()
    train , test = utils.process_dataset.data_preprocess(X , y)
    train_loss , test_loss , train_accuracy , test_accuracy = model_to_train.train_model(train , test) # model saved to data/saved_models

    utils.plots.plot(train_loss , test_loss , model_to_train , mode='Loss') # plots saved to /results
    utils.plots.plot(train_accuracy , test_accuracy , model_to_train , mode='Accuracy')
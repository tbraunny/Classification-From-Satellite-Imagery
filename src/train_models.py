import cnn
import log_reg
import utils.plots
import utils.process_dataset
import xgb
import utils

model_to_train = xgb # specify


X , y = utils.process_dataset.load_dataset()
train , test = utils.process_dataset.data_preprocess(X , y)
train_loss , test_loss , train_accuracy , test_accuracy = model_to_train.train_model(train , test) # model saved to data/saved_models
utils.plots.plot(train_loss , test_loss , train_accuracy , test_accuracy) # plots saved to /results
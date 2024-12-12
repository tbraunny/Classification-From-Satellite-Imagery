import utils.process_dataset
from xgboost import XGBClassifier
from numpy import hstack
import log_reg
import cnn
import torch
import utils
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay , classification_report

class classifier():
    def __init__(self):
        print("Running XGBoost Classifier...")

    def main():
        save_dir = "data/saved_models"

        print("Loading models...")
        log_reg_model = log_reg.Log_Reg()
        checkpoint = torch.load(os.path.join(save_dir , 'log_reg_weights.pt') , weights_only=True)
        log_reg_model.load_state_dict(checkpoint['model_state_dict'])

        cnn_model = cnn.CNN()
        checkpoint = torch.load(os.path.join(save_dir , 'cnn_weights.pt') , weights_only=True)
        cnn_model.load_state_dict(checkpoint['model_state_dict'])
        print("Models loaded")

        X , y = utils.process_dataset.load_dataset()
        X_train , X_test , y_train , y_test = utils.process_dataset.data_preprocess(X , y , flag=True) # flag only return test
        
        X_train = X_train.reshape(X_train.size(0), -1)  # Reshape to (24200, 1200)
        X_test = X_test.reshape(X_test.size(0), -1)

        print("Training XGBoost...")
        model = XGBClassifier()
        model.fit(X_train , y_train)
        print("XGBoost trained")
        y_pred = model.predict(X_test)
        y_pred = [round(value) for value in y_pred] # grab 0 or 1

        print(classification_report(y_test , y_pred))
        joblib.dump(model , os.path.join(save_dir , "xgboost.pkl")) # save model to data/saved_models

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(os.path.join("results/confusion_matrix_xgb.png"))

        # y_pred , y_train = [] , []
        # print("Training...")
        # with torch.no_grad():
        #     for (data , target) in train:
        #         pred_log = log_reg_model(data.float())
        #         pred_cnn = log_reg_model(data.float())
        #         pred_log= pred_log.argmax(dim=1 , keepdim=True)
        #         pred_cnn = pred_cnn.argmax(dim=1 , keepdim=True) # calc conf to reduce false positives
        #         y_pred.append(np.vstack([pred_log , pred_cnn]))
        #         y_train.append(target.view_as(pred_log))

        # y_train = np.array(y_train).squeeze()
        # y_train = y_train[:,-1]
        # y_pred = np.array(y_pred).squeeze()
        # y_pred = y_pred[:,-1]

        # xgb_model = XGBClassifier()
        # xgb_model.fit(y_pred.reshape(-1,1) , y_train)
        # joblib.dump(xgb_model, os.path.join(save_dir , "revised_xgboost.pkl")) # save model to data/saved_models

        # print("Training complete")

        # # Evaluate model
        # #xgb_model.eval()
        # y_pred , y_test = [] , []
        # count = 0
        
        # print("Size " , len(test.dataset))
        # with torch.no_grad():
        #     for (data , target) in test:
        #         count += 1
        #         pred_log = log_reg_model(data.float())
        #         pred_cnn = log_reg_model(data.float())
        #         pred_log= pred_log.argmax(dim=1 , keepdim=True)
        #         pred_cnn = pred_cnn.argmax(dim=1 , keepdim=True) # calc conf to reduce false positives
        #         y_pred.append(np.vstack([pred_log , pred_cnn]))
        #         y_test.append(target.view_as(pred_log))
        # print(np.array(y_pred).shape)
        # y_test = np.array(y_test).squeeze()
        # y_test = y_test[:,-1]
        # y_pred = np.array(y_pred).squeeze()
        # y_pred = y_pred[:,-1]

        # y_pred = xgb_model.predict(y_pred.reshape(-1,1))
        # print(y_pred.shape)
        # print(y_test.shape)

        # print(classification_report(y_test, y_pred))

        # # Confusion matrix
        # cm = confusion_matrix(y_test, y_pred)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        # disp.plot()
        # plt.savefig(os.path.join("results/revised_confusion_matrix_xgb.png"))
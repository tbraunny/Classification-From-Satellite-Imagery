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
    def __init__():
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
        X_test , y_test = utils.process_dataset.data_preprocess(X , y , flag=True) # flag only return test

        pred1 , pred2 = 0 , 0

        with torch.no_grad():
            X_test = X_test.permute(0, 1, 3, 2)
            pred1 = log_reg_model(X_test.float()).numpy()
            pred2 = cnn_model(X_test.float()).numpy()

        xgb_features = hstack([pred1, pred2])

        xgb_model = XGBClassifier()
        xgb_model.fit(xgb_features, y_test)
        joblib.dump(xgb_model, os.path.join(save_dir , "xgboost.pkl")) # save model to data/saved_models

        # Evaluate model
        y_pred = xgb_model.predict(xgb_features)
        print(classification_report(y_test, y_pred))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(os.path.join("results/confusion_matrix_xgb.png"))
import utils.process_dataset
from xgboost import XGBClassifier
from numpy import hstack
import log_reg
import cnn
import torch
import utils
import joblib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

save_dir = "data/saved_models"

print("Loading models...")
log_reg = log_reg.Log_Reg()
checkpoint = torch.load(os.path.join(save_dir , 'log_reg_weights.pt') , weights_only=True)
log_reg.load_state_dict(checkpoint['model_state_dict'])

cnn = cnn.CNN()
checkpoint = torch.load(os.path.join(save_dir , 'cnn_weights.pt') , weights_only=True)
cnn.load_state_dict(checkpoint['model_state_dict'])
print("Models loaded")

X , y = utils.process_dataset.load_dataset()
X_test , y_test = utils.process_dataset.data_preprocess(X , y , flag=True) # flag only return test

pred1 , pred2 = 0 , 0

with torch.no_grad():
    X_test = X_test.permute(0, 3, 1, 2)
    pred1 = log_reg(X_test.float()).numpy()
    pred2 = cnn(X_test.float()).numpy()

xgb_features = hstack([pred1, pred2])

xgb_model = XGBClassifier()
xgb_model.fit(xgb_features, y_test)

joblib.dump(xgb_model, os.path.join(save_dir , "xgboost.pkl"))
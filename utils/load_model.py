import torch
import os
import pickle

model_path = "data/saved_models"

def load_model(model_obj , saved_model_file , flag=False):
    print("Loading model...")

    if (flag):
        saved_model = os.path.join(model_path , saved_model_file)

        with open(saved_model, 'rb') as f:
            model_obj = pickle.load(f)
    else:
        saved_model_file = os.path.join(model_path , saved_model_file)
        checkpoint = torch.load(saved_model_file , weights_only=True) # load only the weights at checkpoint
        model_obj.load_state_dict(checkpoint['model_state_dict']) # load the model state dictionary
        print("Model loaded")

    return model_obj

def load_xgb(saved_model_file):
    saved_model = os.path.join(model_path , saved_model_file)

    with open(saved_model, 'rb') as f:
        model = pickle.load(f)
    return model
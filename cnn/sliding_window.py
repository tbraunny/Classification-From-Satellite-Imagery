import numpy as np
import cv2
import os
import torch
import matplotlib.pyplot as plt
import cnn_model
import log_reg.log_reg
import decision_tree.tree
import time
from matplotlib import patches

def load_scene(num):
    print("Loading scene data...")
    basepath = "C:\\Undergraduate\\Year 4\\CS 482\\final_project\\scenes"
    #img_path = os.path.join(basepath , f"scene_{num}.png")
    img_path = os.path.join(basepath , f"scene_{num}.png")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB) # convert BGR to RGB

    return img

def load_model(file):
    print("Loading model...")
    model = cnn_model.CNN() # import model from original py file
    checkpoint = torch.load(file , weights_only=True) # load only the weights at checkpoint
    model.load_state_dict(checkpoint['model_state_dict']) # load the model state dictionary
    print("Model loaded")

    return model
    
def evaluate_scene(model , scene , stride):
    print("Analyzing scene...")
    print(f"\tScene shape: {scene.shape[0]} x {scene.shape[1]}")
    height , width , _ = scene.shape

    plt.figure(figsize=(25 , 25))
    sub_img = plt.subplot(1 , 1 , 1)
    sub_img.imshow(scene)

    start = time.time() # track segmentation time
    model.eval() # set model to evaluation
    
    for h in range(0 , height - 20 , stride):
        print(f"\r\tEvaluating row {(h // stride) + 1} of {((height - 20) // stride)}" , end="")

        for w in range(0 , width - 20 , stride):
            img_box = []
            img_box.append(scene[h:h + 20 , w:w + 20]) # append pixels in window
            img_box = np.array(img_box)
            img_box = torch.tensor(img_box , dtype=torch.int64).permute(0 , 3 , 1 , 2) # convert to proper size for evalution
            img_box = img_box / 255 # apply same transformation
            img_box = img_box.view(1 , 3 , 20 , 20) # ensure same window size
            prediction = model(img_box)
            confidence = torch.softmax(prediction , dim=1)[0 , 1].item() # calc conf to reduce false positives
            if confidence > 0.84:
                prediction = prediction.argmax(dim=1 , keepdim=True) # determine positive/negative prediction
                if prediction == 1:
                    sub_img.add_patch(patches.Rectangle((w , h) , 20 , 20 , edgecolor = 'blue' , facecolor='none')) # segment positive prediction             

    end = time.time()
    print(f"\nScene Evaluated in {(end - start):.3f} seconds")
    plt.savefig('scene_evalution_test.png')
    plt.show()

if __name__ == '__main__':
    scene_num = 1 # specify scene number to test on
    stride = 3 # specify the granularity of evaluations
    file = '/models/cnn_weights.pt' # specify model file to load
    cnn_model = load_model(file)
    scene = load_scene(scene_num)
    evaluate_scene(cnn_model , scene , stride)
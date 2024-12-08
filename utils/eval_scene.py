import numpy as np
import cv2
import os
import torch
import matplotlib.pyplot as plt
import time
from matplotlib import patches

def load_scene(scene_num):
    print("Loading scene data...")
    basepath = "data/scenes/"
    #img_path = os.path.join(basepath , f"scene_{num}.png")
    img_path = os.path.join(basepath , f"scene_{scene_num}.png")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB) # convert BGR to RGB

    return img
    
def evaluate_scene(model , scene , stride , check_confidence=False , flag=False):
    print("Analyzing scene...")
    print(f"\tScene shape: {scene.shape[0]} x {scene.shape[1]}")
    height , width , _ = scene.shape
    
    plt.figure(figsize=(25 , 25))
    plt.ion()
    sub_img = plt.subplot(1 , 1 , 1)
    sub_img = plt.gca()
    sub_img.imshow(scene)
    confidence = 0

    start = time.time() # track segmentation time
    
    if (not flag):
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
            if (flag):
                img_box = img_box.flatten()[:4].reshape(1, -1)
                prediction = model.predict(img_box)
            else:
                prediction = model(img_box)

            if (check_confidence): # calc confidence for CNN model
                confidence = torch.softmax(prediction , dim=1)[0 , 1].item() # calc conf to reduce false positives
            else:
                confidence = 1

            if confidence > 0.84:
                if isinstance(prediction , torch.Tensor):
                    prediction = prediction.argmax(dim=1 , keepdim=True) # determine positive/negative prediction
                else:
                    prediction = np.argmax(prediction , keepdims=True)
                if prediction == 1:
                    sub_img.add_patch(patches.Rectangle((w , h) , 20 , 20 , edgecolor = 'blue' , facecolor='none')) # segment positive prediction  

                    plt.draw()
                    # might need to adjust           

    end = time.time()
    print(f"\nScene Evaluated in {(end - start):.3f} seconds")
    plt.ioff()
    plt.show()
    save_path = "results"
    plt.savefig(os.path.join(save_path , f'scene_evalution_{type(model).__name__}.png'))
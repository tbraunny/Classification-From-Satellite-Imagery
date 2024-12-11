import numpy as np
import cv2
import os
import torch
import matplotlib.pyplot as plt
import time
from matplotlib import patches

class SceneEvaluator(): # for good practice, eliminate global variables
    def __init__(self):
        self.scene_num = None

    def set_scene_num(self , scene_num):
        self.scene_num = scene_num

    def get_scene_num(self):
        return self.scene_num
    
    def load_scene(self , scene_num):
        print("Loading scene data...")
        basepath = "data/scenes/"
        #img_path = os.path.join(basepath , f"scene_{num}.png")
        img_path = os.path.join(basepath , f"scene_{scene_num}.png")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB) # convert BGR to RGB
        self.set_scene_num(scene_num)
        print("Scene data loaded")

        return img
    
    def evaluate_scene(self , model , scene , stride , log , cnn , flag=False):
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
            #print(f"\r\tEvaluating row {(h // stride)} of {((height - 20) // stride)}" , end="")
            for w in range(0 , width - 20 , stride):
                img_box = []
                img_box.append(scene[h:h + 20 , w:w + 20]) # append pixels in window
                img_box = np.array(img_box)
                img_box = torch.tensor(img_box , dtype=torch.int64).permute(0 , 3 , 1 , 2) # convert to proper size for evalution
                img_box = img_box / 255 # apply same transformation
                img_box = img_box.view(1 , 3 , 20 , 20) # ensure same window size
                
                if (flag): # predict xgb model if flag raised
                    log_pred = log(img_box.float()).detach().numpy()
                    cnn_pred = cnn(img_box.float()).detach().numpy()
                    xgb_features = np.hstack([log_pred , cnn_pred])
                    prediction = model.predict_proba(xgb_features)
                    confidence = prediction[:,1] # store probability of class 1 as confidence
                    prediction = 1 # set prediction to 1 (its always 1, idk why)
                else: # for logistic & cnn
                    prediction = model(img_box)
                    confidence = torch.softmax(prediction , dim=1)[0 , 1].item() # calc conf to reduce false positives
                    prediction = prediction.argmax(dim=1 , keepdim=True) # determine positive/negative prediction

                if confidence > 0.88:  # set to 0.75 for scene_6                  
                    if prediction == 1:                     
                        sub_img.add_patch(patches.Rectangle((w , h) , 20 , 20 , edgecolor = 'blue' , facecolor='none')) # segment positive prediction  

                        plt.draw()
                        #plt.pause(0.05)          

        end = time.time()
        print(f"\nScene Evaluated in {(end - start):.3f} seconds")
        plt.ioff()
        plt.show()
        save_path = "results"
        scene_num = self.get_scene_num()
        plt.savefig(os.path.join(save_path , f'scene{scene_num}_evalution_{type(model).__name__}.png'))
        print("Image saved to: " , os.path.join(save_path , f'scene{scene_num}_evalution_{type(model).__name__}.png'))
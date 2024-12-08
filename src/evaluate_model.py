import utils
import utils.eval_scene
import utils.load_model
import cnn
import log_reg
import xgb

# specify which model you would like to run
# - 1 : xgboost
# - 2 : log_reg
# - 3 : cnn
model_selection = 3
scene_num = 1

############################################
stride = 3
pkl_flag = False
confidence = False

if (model_selection == 1):
    model = xgb
    pkl_flag = True # if evaluating xgboost
    saved_model_file = 'xgboost.pkl'
elif (model_selection == 2):
    model = log_reg.Log_Reg()
    saved_model_file = 'log_reg_weights.pt'
elif (model_selection == 3):
    model = cnn.CNN()
    check_confidence = True
    saved_model_file = 'cnn_weights.pt'
else:
    print("invalid option")
    exit(0)

model_obj = utils.load_model.load_model(model , saved_model_file , flag=pkl_flag)
scene = utils.eval_scene.load_scene(scene_num)
utils.eval_scene.evaluate_scene(model_obj , scene , stride , check_confidence=confidence , flag=pkl_flag) # final image shown in results/
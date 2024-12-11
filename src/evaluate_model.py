import utils
from utils.eval_scene import SceneEvaluator
import utils.load_model
import cnn
import log_reg
import xgb

# specify which model you would like to run:
# - 1 : xgboost
# - 2 : log_reg
# - 3 : cnn
model_selection = 3

# scene numbers:
# - 1 : San Francisco Intl (SFO)
# - 2 : San Diego Intl (SAN)
# - 3 : Los Angeles Intl (LAX)
# - 4 : Southern California Logistics (SCLA) Airport Boneyard 
# - 5 : New Jersey Coast (Nov 2021) 39°28'20.2"N 74°18'49.7"W (15,000ft)
scene_num = 1

############################################
stride = 3
pkl_flag = False
eval = SceneEvaluator()

if (model_selection == 1):
    print("Evaluating XGBoost...")
    model = xgb
    pkl_flag = True # if evaluating xgboost
    saved_model_file = 'revised_xgboost.pkl'
elif (model_selection == 2):
    print("Evaluating Logistic Regression...")
    model = log_reg.Log_Reg()
    saved_model_file = 'log_reg_weights.pt'
elif (model_selection == 3):
    print("Evaluating CNN...")
    model = cnn.CNN()
    saved_model_file = 'cnn_weights.pt'
else:
    print("invalid option")
    exit(0)

model_obj = utils.load_model.load_model(model , saved_model_file , flag=pkl_flag)
scene = eval.load_scene(scene_num)
eval.evaluate_scene(model_obj , scene , stride , log=log_reg.Log_Reg() , cnn=cnn.CNN() , flag=pkl_flag) # final image shown in results/

import pandas as pd
import numpy as np
import sklearn.tree 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score,roc_curve
import lightgbm as lgb


#Finally we run the model on the full dataset
def full_sample_predict(saved_model,data_full_sample,y_full_sample):
    #Load the saved model
    bst = lgb.Booster(model_file=saved_model)
    #Now we evaluate on the full sample
    y_pred_final = bst.predict(data_full_sample)
    #Full sample predicitons
    full_sample_preds = pd.DataFrame(y_pred_final,index=data_full_sample.index)
    #evaluate on validation data
    auc_full_model = roc_auc_score(y_full_sample,y_pred_final)
    print(f"Full sample training - The ROC AUC of model's prediction is: {auc_full_model}")
    return y_pred_final,auc_full_model
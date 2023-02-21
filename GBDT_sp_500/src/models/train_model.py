
import pandas as pd
import numpy as np
import sklearn.tree 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb


#First rounds
def train_first_rounds(params,lgb_train,num_boost,valid_set,feature_num,x_train,y_train_target):
    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=num_boost,
                valid_sets=valid_set,
                feature_name=feature_num,
                ) 
    #ROC AUC for the training model
    y_pred_train = gbm.predict(x_train)
    auc_train_model = roc_auc_score(y_train_target,y_pred_train)
    print(f"Training - The ROC AUC of model's prediction is: {auc_train_model}")
    return gbm,y_pred_train,auc_train_model

#Train next rounds with learning decay
def train_second_rounds(params,lgb_train,num_boost,prev_model,valid_set,feature_num,x_train,y_train_target,learning_param):
    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=num_boost,
                init_model=prev_model,
                valid_sets=valid_set,
                feature_name=feature_num,
                callbacks = [lgb.reset_parameter(learning_rate=lambda iter:learning_param*(0.99 ** iter))]
                )
    #ROC AUC for the training model
    y_pred_train = gbm.predict(x_train)
    auc_train_model = roc_auc_score(y_train_target,y_pred_train)
    print(f"Training - The ROC AUC of model's prediction is: {auc_train_model}")
    return gbm,y_pred_train,auc_train_model


#Train next rounds with early stopping call back
def train_third_rounds(params,lgb_train,num_boost,prev_model,valid_set,feature_num,stopping_rounds,x_train,y_train_target):
    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=num_boost,
                init_model=prev_model,
                valid_sets=valid_set,
                feature_name=feature_num,
                callbacks = [lgb.early_stopping(stopping_rounds=stopping_rounds)])
    #ROC AUC for the training model
    y_pred_train = gbm.predict(x_train)
    auc_train_model = roc_auc_score(y_train_target,y_pred_train)
    print(f"Training - The ROC AUC of model's prediction is: {auc_train_model}")
    return gbm,y_pred_train,auc_train_model

#Validate
def validate(saved_model,x_valid,y_valid_target):
    #load model to predict on validation data
    bst = lgb.Booster(model_file=saved_model)
    y_pred_valid = bst.predict(x_valid)
    #evaluate on validation data
    auc_loaded_model = roc_auc_score(y_valid_target,y_pred_valid)
    print(f"Validation - The ROC AUC of model's prediction is: {auc_loaded_model}")
    return bst,y_pred_valid,auc_loaded_model

#Validate with best iteration
#def validate_best(saved_model,x_valid,y_valid_target):
    #load model to predict on validation data
#    bst = lgb.Booster(model_file=saved_model)
#    y_pred_valid = bst.predict(x_valid,num_iteration=gbm.best_iteration)
    #evaluate on validation data
#    auc_loaded_model = roc_auc_score(y_valid_target,y_pred_valid)
#    print(f"Validation - The ROC AUC of model's prediction is: {auc_loaded_model}")
#    return bst,y_pred_valid,auc_loaded_model

#Check final model against test data
def test_final_model(saved_model,x_test,y_test_target,model_name):
    #load model to predict on test data
    bst = lgb.Booster(model_file=saved_model)
    # Now we evaluate on the test dataset
    y_pred_test = bst.predict(x_test)
    #evaluate on validation data
    auc_test_model = roc_auc_score(y_test_target,y_pred_test)
    print(f"Test - The ROC AUC of model's prediction is: {auc_test_model}")
    return bst,y_pred_test,auc_test_model

def full_sample_model(params,lgb_train,num_boost,valid_set,feature_num,stopping_rounds,data_full_sample,y_full_sample,learning_param):
    #First rounds
    
    output,y_pred_full1,auc_full_model1 = train_first_rounds(params,lgb_train,num_boost,valid_set,feature_num,data_full_sample,y_full_sample)

    #Second rounds
    prev_model = output
    output,y_pred_full2,auc_full_model2 = train_second_rounds(params,lgb_train,num_boost,prev_model,valid_set,feature_num,data_full_sample,y_full_sample,learning_param)

    #Third rounds
    prev_model = output
    output,y_pred_full3,auc_full_model3 = train_third_rounds(params,lgb_train,num_boost,prev_model,valid_set,feature_num,stopping_rounds,data_full_sample,y_full_sample)

    return output
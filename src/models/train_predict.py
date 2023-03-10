import pandas as pd
import lightgbm as lgb
import shap
from datetime import datetime,timedelta
import pickle
import src
from src.data import download
from src.visualization import eda,results
from src.custom_funcs import helper_funcs
from src.features import create_features
from src.models import train_model as train
from src.models import predict_model as predict


def train_test_pickle(x_train,x_valid,y_train,y_valid,x_test,y_test,col_index,num_boost,f_horizon,params,feature_num,model_name,full_sample_data,full_sample_y,full_data,f_names,percentile):
    #create a dataset for lightgbm
    lgb_train= lgb.Dataset(x_train,y_train.iloc[:,col_index],free_raw_data=False)
    lgb_eval = lgb.Dataset(x_valid,y_valid.iloc[:,col_index],reference=lgb_train,free_raw_data=False)
    y_train_target = y_train.iloc[:,col_index]
    y_valid_target = y_valid.iloc[:,col_index]
    y_test_target = y_test.iloc[:,col_index]
    valid_set = lgb_train
    ######Start training first 10 rounds ######
    gbm,y_pred_train,auc_train_model = train.train_first_rounds(params,lgb_train,num_boost,valid_set,feature_num,x_train,y_train_target)
    gbm.save_model(model_name + '.txt')
    saved_model = model_name + '.txt'

    ######Validate ######

    bst,y_pred_valid,auc_loaded_model = train.validate(saved_model,x_valid,y_valid_target)

    ######Start training next 10 rounds with learning decay ######
    prev_model = gbm
    valid_set = lgb_eval
    learning_param = 0.05
    gbm,y_pred_train,auc_train_model = train.train_second_rounds(params,lgb_train,num_boost,prev_model,valid_set,feature_num,x_train,y_train_target,learning_param)
    gbm.save_model(model_name+'_learning_decay.txt')
    saved_model = model_name+'_learning_decay.txt'

    ######Validate ######

    bst,y_pred_valid,auc_loaded_model = train.validate(saved_model,x_valid,y_valid_target)

    ######Start training next 10 rounds with early stopping ######
    prev_model = gbm
    stopping_rounds = 5
    gbm,y_pred_train,auc_train_model = train.train_third_rounds(params,lgb_train,num_boost,prev_model,valid_set,feature_num,stopping_rounds,x_train,y_train_target)
    gbm.save_model(model_name+'_early_stopping.txt')
    saved_model = model_name+'_early_stopping.txt'

    ######Validate ######
    bst,y_pred_valid,auc_loaded_model = train.validate(saved_model,x_valid,y_valid_target)

    ######Test ########
    bst,y_pred_test,auc_test_model = train.test_final_model(saved_model,x_test,y_test_target,model_name)

    ######Full sample model run ######
    lgb_train = lgb.Dataset(full_sample_data,full_sample_y.iloc[:,col_index],free_raw_data=False)
    valid_set = lgb_train
    data_full_sample = full_sample_data
    y_full_sample = full_sample_y.iloc[:,col_index]

    gbm = train.full_sample_model(params,lgb_train,num_boost,valid_set,feature_num,stopping_rounds,data_full_sample,y_full_sample,learning_param)
    gbm.save_model(model_name +'_full_sample.txt')
    saved_model = model_name +'_full_sample.txt'

    y_pred_final,auc_full_model =  predict.full_sample_predict(saved_model,data_full_sample,y_full_sample)

    #######Pickled#######
    pkl_name = str(f_horizon) + 'd' + str(col_index+1) + '.pkl'
    with open(pkl_name,'wb') as fout:
        pickle.dump(gbm,fout)

    ######Sample including the last f_horizon days#######
    with open(pkl_name,'rb') as fin:
        pkl_model = pickle.load(fin)

    y_pred = pkl_model.predict(full_data)


    ######Feature importance######
    #print(f"Fetaure importances: {list(gbm.feature_importance())}")
    #SHAP values
    results.plot_shap(gbm,full_data,f_names,model_name)

    ######plot model predictions#####
    
    results.plot_predictions(f_horizon,y_pred,percentile,full_data,model_name)
    
  
    results.plot_roc_curve(y_full_sample,y_pred_final,auc_full_model,model_name)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.tree 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from yahoo_fin.stock_info import get_earnings_history
from fredapi import Fred
from dotenv import load_dotenv
import os
import pandas_datareader as web
import lightgbm as lgb
import shap
from datetime import datetime,timedelta
import pickle




########################################Download data####################################################

def download_data(fred,series_from_fred,start_fred,ticker_yf,start_yf,interval,series_names):
    series_list = []
    for s in range(len(series_from_fred)):
        series_list.append(fred.get_series(series_from_fred[s],observation_start=start_fred))
    yf_series = yf.download(tickers=ticker_yf,start=start_yf,interval=interval)
    yf_series = yf_series['Adj Close']
    series_list = pd.DataFrame(series_list).T 
    series_list.columns = series_names
    return series_list,yf_series

########################################Derive####################################################
def interpolate_na(data):
    if data.isnull().sum().sum()==0:
        return data
    else:
        return data.interpolate(limit_direction='both')

def derive_features(fred_series,sp_data):
        
    #Create return variable - target
    sp_data = interpolate_na(sp_data) #Check for NA and interpolate if any found
    sp_return = sp_data.pct_change()
    sp_return5 = sp_data.pct_change(periods=5) #5days ahead
    sp_return10 = sp_data.pct_change(periods=10) #10 days ahead
    sp_return15 = sp_data.pct_change(periods=15) #15 days ahead
    #Remove NaNs
    sp_return5 = sp_return5.dropna()
    sp_return10 = sp_return10.dropna()
    sp_return15 = sp_return15.dropna()
    #Rolling-window variance S&P 500 returns
    sp_var5 = sp_return.rolling(5).var()
    sp_var10 = sp_return.rolling(10).var()
    sp_var21 = sp_return.rolling(21).var()
    #Corporate spreads
    ig_spread = interpolate_na(fred_series['ig_spread'])
    hy_spread = interpolate_na(fred_series['hy_spread'])
    #Change in spreads
    ig_change = ig_spread.diff()*100
    hy_change = hy_spread.diff()*100
    #Change in wti
    wti = interpolate_na(fred_series['wti'])
    wti_change = wti.pct_change()
    #Yields
    treas_3m = interpolate_na(fred_series['treas_3m'])
    treas_5y = interpolate_na(fred_series['treas_5y'])
    treas_10y = interpolate_na(fred_series['treas_10y'])
    treas_30y = interpolate_na(fred_series['treas_30y'])
    #Change in yields
    treas3m_change = treas_3m.diff()*100 #Change in bps
    treas5y_change = treas_5y.diff()*100
    treas10y_change = treas_10y.diff()*100
    treas30y_change = treas_30y.diff()*100
    #Breakeven yields
    binf_5y = interpolate_na(fred_series['binf_5y'])
    binf_10y = interpolate_na(fred_series['binf_10y'])
    #Change in breakeven yields
    binf5_change = binf_5y.diff()*100
    binf10_change = binf_10y.diff()*100
    #2s10s and 3m10s
    y10_y2 = interpolate_na(fred_series['y10_y2'])
    y10_m3 = interpolate_na(fred_series['y10_m3'])
    #Change
    y10_y2change = y10_y2.diff()*100
    y10_m3change = y10_m3.diff()*100
    # VIX and change
    vix = interpolate_na(fred_series['vix'])
    vix_change = vix.pct_change()
    #ccy returns
    eur = interpolate_na(fred_series['eur'])
    jpy = interpolate_na(fred_series['jpy'])
    gbp = interpolate_na(fred_series['gbp'])
    cny = interpolate_na(fred_series['cny'])
    #ccy change
    eur_change = eur.pct_change()
    jpy_change = jpy.pct_change()
    gbp_change = gbp.pct_change()
    cny_change = cny.pct_change()
    data_0 = pd.concat([sp_var5,sp_var10,sp_var21,ig_spread,ig_change,hy_spread,hy_change,wti_change,treas3m_change,treas5y_change,treas10y_change,treas30y_change,binf5_change,binf10_change,y10_y2,y10_m3,y10_m3change,y10_y2change,vix_change,eur_change,jpy_change,gbp_change,cny_change],axis=1)
    data_0 = interpolate_na(data_0)
    num_data,num_feature = data_0.shape
    feature_num = [f'f_{col}' for col in range(num_feature)]
    data_0.columns = feature_num
    data_0 = data_0.drop(index=data_0.index[0],axis=0)

    return data_0,feature_num,sp_return5,sp_return10,sp_return15





#######################################Preparing target and features data################################
def create_target_series(data,percentile_list,f_horizon):
    target_list = []
    for x in range(len(percentile_list)):
        target_list.append (np.where(data<np.percentile(data,percentile_list[x]),1,0))
    targets_df = pd.DataFrame(target_list).T
    targets_df.index = data.index
    targets_df = targets_df.shift(-f_horizon)
    targets_df = targets_df.iloc[:-f_horizon,:]
    return targets_df

def reindex_features_data(data,f_horizon,target_data):
    data_reindex = data.iloc[:-f_horizon,:] 
    common_idx = target_data.index.intersection(data_reindex.index)
    data_reindex = data_reindex.loc[common_idx]
    return data_reindex

########################################################## Split data ######################################
# Split data into training, validation and test set
def data_split_size(data, train_share):
    train_size = int(len(data)*train_share)
    valid_size = int((len(data)-train_size)/2)
    test_size = valid_size
    return train_size,valid_size,test_size
def data_split(x_data,y_data,train_size,valid_size,test_size):
    x_train = x_data.iloc[:train_size,:]
    y_train = y_data.iloc[:train_size,:]
    x_valid = x_data.iloc[train_size:train_size+valid_size,:]
    y_valid = y_data.iloc[train_size:train_size+valid_size,:]
    x_test = x_data.iloc[-test_size:,:]
    y_test = y_data.iloc[-test_size:,:]
    return x_train,y_train,x_valid,y_valid,x_test,y_test
########################################################## Light GBM ###########################################
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
def test_final_model(saved_model,x_test,y_test_target):
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
    return full_sample_preds,auc_full_model

#Plot SHAP Values
def plot_shap(final_model,data,feature_names):
    final_model.params['objective'] = 'binary'
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(data)
    #print(f"shap values: {shap_values}")
    #Average shap values
    shap.summary_plot(shap_values,data,feature_names=feature_names)

#Plot model predictions
def plot_predictions(f_horizon,predictions,percentile,data):
    oos_dates = []
    for x in range(1,f_horizon+1):
        oos_dates.append(data.index[-1]+timedelta(days=x))

    oos_dates = pd.to_datetime(oos_dates)
    oos_dates = oos_dates.strftime('%Y-%m-%d')
    oof_update = pd.DataFrame(predictions[-f_horizon:],index=oos_dates)
    fig, ax = plt.subplots(figsize=(15,15),nrows = 3,ncols=1)
    fig.suptitle(f'Probability of return < {percentile}th percentile in next {f_horizon} days',x=0.5,y=0.9)
    #First subplot
    predictions_df = pd.DataFrame(predictions,index=data_0.index)
    ax[0].plot(predictions_df,color = 'grey',label='full sample')
    ax[0].legend()
    
    #Second subplot
    ax[1].plot(predictions_df.iloc[-21:],color='grey',label='last 21 dates')
    ax[1].legend()
    
    #Out of sample forecasts
    ax[2].plot(oof_update,color='grey',label='out of sample forecasts',marker='o')
    ax[2].legend()
    ax[2].set_title(f'{f_horizon}-day ahead projections')



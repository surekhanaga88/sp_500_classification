import pandas as pd
import numpy as np
from fredapi import Fred
from dotenv import load_dotenv
import os
from os.path import dirname, abspath
import src
from src.data import download
from src.features import create_features
from src.models import params,predict_model,train_model,train_predict
from src.visualization import eda,results
from src.custom_funcs import helper_funcs
import importlib
import yfinance as yf

#Steps to cover:
# Import the csv files from data/raw
# create features
# create y and data and save these in data/processed
# train test split
# run the model

#Download data from FRED API
load_dotenv() #Create environment variable
API_KEY = os.getenv("API_KEY") #Get API_KEY from .env file
fred = Fred(api_key = API_KEY)
###*****Don't change order of the series names*****###
series_fred = ['BAMLC0A0CM','BAMLH0A0HYM2','DCOILWTICO','DTB3','DGS5','DGS10','DGS30','T5YIE','T10YIE','T10Y2Y','T10Y3M','VIXCLS','DEXUSEU','DEXJPUS','DEXUSUK','DEXCHUS']
start_fred = '1/1/1995'
ticker_yf = '^GSPC'
start_yf = '1995-01-01'
interval='1d'
series_names = ['ig_spread','hy_spread','wti','treas_3m','treas_5y','treas_10y','treas_30y','binf_5y','binf_10y','y10_y2','y10_m3','vix','eur','jpy','gbp','cny']

def download_data(fred,series_from_fred,start_fred,ticker_yf,start_yf,interval,series_names):
    series_list = []
    for s in range(len(series_from_fred)):
        series_list.append(fred.get_series(series_from_fred[s],observation_start=start_fred))
    yf_series = yf.download(tickers=ticker_yf,start=start_yf,interval=interval)
    yf_series = yf_series['Adj Close']
    series_list = pd.DataFrame(series_list).T 
    series_list.columns = series_names
    return series_list,yf_series

fred_series,sp_data = download_data(fred,series_fred,start_fred,ticker_yf,start_yf,interval,series_names)

d = dirname(dirname(abspath(__file__)))

save_path_fred = os.path.join(d,'data','raw','fred.csv')
save_path_sp = os.path.join(d,'data','raw','sp_data.csv')

fred_series.to_csv(save_path_fred)
sp_data.to_csv(save_path_sp)

save_path_y = os.path.join(d,'data','processed','y.csv')
save_path_x = os.path.join(d,'data','processed','data.csv')


#Derive features

data_0,feature_num,sp_return,sp_return5,sp_return10,sp_return15 = create_features.derive_features(fred_series,sp_data)

#Create target series

percentile_list = [5,10,15,20,25]

y_10d = helper_funcs.create_target_series(sp_return5,percentile_list,f_horizon=10)

#Reindex features data

data_10d = helper_funcs.reindex_features_data(data_0,10,y_10d)

# Save processed data

y_10d.to_csv(save_path_y)

data_10d.to_csv(save_path_x)

#Create feature list

f_names = ['sp_var5','sp_var10','sp_var21','ig_spread','ig_change','hy_spread','hy_change','wti_change','treas3m_change','treas5y_change','treas10y_change','treas30y_change','binf5_change','binf10_change','y10_y2','y10_m3','y10_m3change','y10_y2change','vix_change','eur_change','jpy_change','gbp_change','cny_change']

features_names = pd.DataFrame(f_names,feature_num)

#Train-test split

train_size,valid_size,test_size = helper_funcs.data_split_size(data_10d,0.75)


x_train,y_train,x_valid,y_valid,x_test,y_test = helper_funcs.data_split(data_10d,y_10d,train_size,valid_size,test_size)

# User chooses percentile

p = input("Choose percentile : 5,10,15,20,25 ")

#col_index = 0 (5 percentile),1 (10 percentile),2 (15 percentile),3 (20 percentile),4 (25 percentile)

if (p==5):
    col_index=0
elif(p==10):
    col_index=1
elif(p==15):
    col_index=2
elif(p==20):
    col_index=3
else:
    col_index=4

importlib.reload(params) #IMP: Reload the params.py script before using the attribute params_common
params = params.params_common
train_predict.train_test_pickle(x_train,x_valid,y_train,y_valid,x_test,y_test,col_index=col_index,num_boost=10,f_horizon=10,params=params,feature_num=feature_num,model_name = 'model_10d',full_sample_data = data_10d,full_sample_y = y_10d,full_data = data_0,f_names = f_names,percentile = p)
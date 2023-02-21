import pandas as pd
import numpy as np




######################################Interpolate missing values ##########################################

def interpolate_na(data):
    if data.isnull().sum().sum()==0:
        return data
    else:
        return data.interpolate(limit_direction='both')


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
    if f_horizon==0:
        data_reindex = data
    else:
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



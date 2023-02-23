import pandas as pd
import numpy as np
import src
from src.custom_funcs import helper_funcs


def derive_features(fred,sp):
        
    #Create return variable - target
    sp = helper_funcs.interpolate_na(sp) #Check for NA and interpolate if any found
    sp_return = sp.pct_change()
    sp_return5 = sp.pct_change(periods=5) #5days ahead
    sp_return10 = sp.pct_change(periods=10) #10 days ahead
    sp_return15 = sp.pct_change(periods=15) #15 days ahead
    #Remove NaNs
    sp_return5 = sp_return5.dropna()
    sp_return10 = sp_return10.dropna()
    sp_return15 = sp_return15.dropna()
    #Rolling-window variance S&P 500 returns
    sp_var5 = sp_return.rolling(5).var()
    sp_var10 = sp_return.rolling(10).var()
    sp_var21 = sp_return.rolling(21).var()
    #Corporate spreads
    ig_spread = helper_funcs.interpolate_na(fred['ig_spread'])
    hy_spread = helper_funcs.interpolate_na(fred['hy_spread'])
    #Change in spreads
    ig_change = ig_spread.diff()*100
    hy_change = hy_spread.diff()*100
    #Change in wti
    wti = helper_funcs.interpolate_na(fred['wti'])
    wti_change = wti.pct_change()
    #Yields
    treas_3m = helper_funcs.interpolate_na(fred['treas_3m'])
    treas_5y = helper_funcs.interpolate_na(fred['treas_5y'])
    treas_10y = helper_funcs.interpolate_na(fred['treas_10y'])
    treas_30y = helper_funcs.interpolate_na(fred['treas_30y'])
    #Change in yields
    treas3m_change = treas_3m.diff()*100 #Change in bps
    treas5y_change = treas_5y.diff()*100
    treas10y_change = treas_10y.diff()*100
    treas30y_change = treas_30y.diff()*100
    #Breakeven yields
    binf_5y = helper_funcs.interpolate_na(fred['binf_5y'])
    binf_10y = helper_funcs.interpolate_na(fred['binf_10y'])
    #Change in breakeven yields
    binf5_change = binf_5y.diff()*100
    binf10_change = binf_10y.diff()*100
    #2s10s and 3m10s
    y10_y2 = helper_funcs.interpolate_na(fred['y10_y2'])
    y10_m3 = helper_funcs.interpolate_na(fred['y10_m3'])
    #Change
    y10_y2change = y10_y2.diff()*100
    y10_m3change = y10_m3.diff()*100
    # VIX and change
    vix = helper_funcs.interpolate_na(fred['vix'])
    vix_change = vix.pct_change()
    #ccy returns
    eur = helper_funcs.interpolate_na(fred['eur'])
    jpy = helper_funcs.interpolate_na(fred['jpy'])
    gbp = helper_funcs.interpolate_na(fred['gbp'])
    cny = helper_funcs.interpolate_na(fred['cny'])
    #ccy change
    eur_change = eur.pct_change()
    jpy_change = jpy.pct_change()
    gbp_change = gbp.pct_change()
    cny_change = cny.pct_change()
    data_features = pd.concat([sp_var5,sp_var10,sp_var21,ig_spread,ig_change,hy_spread,hy_change,wti_change,treas3m_change,treas5y_change,treas10y_change,treas30y_change,binf5_change,binf10_change,y10_y2,y10_m3,y10_m3change,y10_y2change,vix_change,eur_change,jpy_change,gbp_change,cny_change],axis=1)
    data_features = helper_funcs.interpolate_na(data_features)
    num_data,num_feature = data_features.shape
    feature_num = [f'f_{col}' for col in range(num_feature)]
    data_features.columns = feature_num
    data_features = data_features.drop(index=data_features.index[0],axis=0)

    return data_features,feature_num,sp_return,sp_return5,sp_return10,sp_return15



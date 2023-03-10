import pandas as pd
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv
import os
from os.path import dirname, abspath


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

d = dirname(dirname(dirname(abspath(__file__))))

save_path_fred = os.path.join(d,'data','raw','fred.csv')
save_path_sp = os.path.join(d,'data','raw','sp_data.csv')

fred_series.to_csv(save_path_fred)
sp_data.to_csv(save_path_sp)

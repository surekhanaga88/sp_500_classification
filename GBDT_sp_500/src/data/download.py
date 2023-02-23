import pandas as pd
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv
import os


def download_data(fred,series_from_fred,start_fred,ticker_yf,start_yf,interval,series_names):
    series_list = []
    for s in range(len(series_from_fred)):
        series_list.append(fred.get_series(series_from_fred[s],observation_start=start_fred))
    yf_series = yf.download(tickers=ticker_yf,start=start_yf,interval=interval)
    yf_series = yf_series['Adj Close']
    series_list = pd.DataFrame(series_list).T 
    series_list.columns = series_names
    return series_list,yf_series



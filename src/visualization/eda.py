import pandas as pd
import numpy as np
from numpy import mean, std
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import statsmodels.api as sm
import scipy as sci
from scipy.stats import skew, shapiro
from sklearn.preprocessing import StandardScaler, Normalizer
from math import sqrt
import src
from src.custom_funcs import helper_funcs

# Take a prelim look at the distribution of past returns


def distribution_returns(sp_df):
    #Create SP returns
    sp_return = sp_df.pct_change()
    sp_return5 = sp_df.pct_change(periods=5) #5days ahead
    sp_return10 = sp_df.pct_change(periods=10) #10 days ahead
    sp_return15 = sp_df.pct_change(periods=15) #15 days ahead

    #Plot SP returns
    fig, axs = plt.subplots(1,2,figsize=(13,6))
    plot01 = axs[0].plot(sp_return)
    plot02 = axs[1].hist(sp_return,30,alpha=0.5)
    plt.show()
    print("Distribution summary: ")
    sp_return = helper_funcs.interpolate_na(sp_return)
    print(sp_return.describe())


def scatter_plot(data,target):
    df_scatter = pd.concat([data,target])
    sns.pairplot(df_scatter,height=20)
    


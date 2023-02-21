
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.preprocessing import StandardScaler
import shap
from datetime import datetime,timedelta


def plot_roc_curve(y_full_sample,y_pred_final,auc_full_model,model_name):
    false_pos_rate,true_pos_rate,thresholds = roc_curve(y_full_sample,y_pred_final)
    fig = plt.figure(figsize=(10,8),dpi=100)
    plt.axis('scaled')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title("ROC Curve - Full sample")
    plt.plot(false_pos_rate,true_pos_rate,'grey')
    plt.text(0.95, 0.05, 'AUC = %0.4f' % auc_full_model, ha='right')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    roc_plot_name = model_name + '_roc_curve.png'
    plt.savefig(roc_plot_name,dpi=700)

#Plot SHAP Values
def plot_shap(final_model,data,feature_names,model_name):
    final_model.params['objective'] = 'binary'
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(data)
    #print(f"shap values: {shap_values}")
    #Average shap values
    shap.summary_plot(shap_values,data,feature_names=feature_names,show=False)
    shap_plot_name = model_name + '_shap.png'
    plt.savefig(shap_plot_name,dpi=700)
    #plt.show()

#Plot model predictions
def plot_predictions(f_horizon,predictions,percentile,data,model_name):
    oos_dates = []
    for x in range(1,f_horizon+1):
        oos_dates.append(data.index[-1]+timedelta(days=x))

    oos_dates = pd.to_datetime(oos_dates)
    oos_dates = oos_dates.strftime('%Y-%m-%d')
    oof_update = pd.DataFrame(predictions[-f_horizon:],index=oos_dates)
    fig, ax = plt.subplots(figsize=(15,15),nrows = 3,ncols=1)
    fig.suptitle(f'Probability of return < {percentile}th percentile in next {f_horizon} days',x=0.5,y=0.9)
    #First subplot
    predictions_df = pd.DataFrame(predictions,index=data.index)
    ax[0].plot(predictions_df,color = 'grey',label='full sample')
    ax[0].legend()
    
    #Second subplot
    ax[1].plot(predictions_df.iloc[-21:],color='grey',label='last 21 dates')
    ax[1].legend()
    
    #Out of sample forecasts
    ax[2].plot(oof_update,color='grey',label='out of sample forecasts',marker='o')
    ax[2].legend()
    ax[2].set_title(f'{f_horizon}-day ahead projections')

    pred_plot_name = model_name + '_preds.png'
    plt.savefig(pred_plot_name,dpi=700)
    #plt.show()

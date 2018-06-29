import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math
from utils import XyScaler
from linear import rmse
from tabulate import tabulate
from linear import linear_modeling_test
import statsmodels.api as sm
import seaborn as sns

def roc_curve(predictions, labels, thresholds):
    Recall = []
    FPR = []
    Precision = []
    Accuracy = []
    Spec = []

    for score in thresholds:
        pred_int = (predictions > score).astype(int)
        labels_bin = (labels > 0).astype(int)
        tp = ((pred_int == 1) & (labels_bin == 1)).sum()
        fn = ((pred_int == 0) & (labels_bin == 1)).sum()
        fp = ((pred_int == 1) & (labels_bin == 0)).sum()
        tn = ((pred_int == 0) & (labels_bin == 0)).sum()
        tpr = tp/(fn + tp)
        fpr = fp/(tn + fp)
        pre = tp/(fp + tp)
        specificity = tn/(tn+fp)
        acc = (tp + tn)/(tp + tn + fp + fn)
        FPR.append(fpr)
        Recall.append(tpr)
        Precision.append(pre)
        Accuracy.append(acc)
        Spec.append(specificity)
    return np.array(Precision), np.array(Recall), np.array(Accuracy), np.array(Spec), np.array(FPR)

def roc_graph(y_hat,y, thresholds, name):
    prec, rec, acc, spec, fpr = roc_curve(y_hat, y, thresholds)
    fig, ax = plt.subplots()
    ax.plot(fpr, rec)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity, Recall)")
    ax.set_title("ROC plot of {}".format(name))
    plt.savefig("images/ROC of {}".format(name))
    plt.close()

def model_regress(X,y,base_estimator,X_final, y_final):
    #standardizer = XyScaler()
    #standardizer.fit(X, y)
    #X_std, y_std = standardizer.transform(X, y)
    # Fit estimator
    estimator = base_estimator
    estimator.fit(X, y)
    coeff = estimator.coef_
    #X_std_final, y_std_final = standardizer.transform(X_final, y_final)
    y_hat_final = estimator.predict(X_final)
    #X_final, y_hat_final = standardizer.inverse_transform(X_std_final,y_hat_std_final)
    rmse_final = rmse(y_final,y_hat_final)
    # Return coefficients
    return coeff, y_hat_final, rmse_final

if __name__ == "__main__":
    #using ridge regression with dataframe 3 from model set C to get model and coefficients
    df3 = pd.read_pickle('data/reg_model_data_final3.pkl')
    endogc = df3['label_h_point_spread'].values
    exogsc2 = df3.drop(['label_h_point_spread','label_home_winner','year','home_fgp_var','home_ppg_var',
                        'away_fgp_var','away_ppg_var','home_ps_home','away_ps_away',
                        'home_fgpct','away_fgpct','home_3ppct','away_3ppct',
                        'awayteam_awaywp','hometeam_pt_sprd','hometeam_opp_ppg',
                        'awayteam_pt_sprd','awayteam_opp_ppg','home_pyth_wd','away_pyth_wd',
                        'hometeam_ps_var', 'awayteam_ps_var','home_ftpct','away_ftpct',
                        'hpsq','apsq','hometeam_homewp','home_3pa_perposs','away_3pa_perposs'],axis=1)
    #Applying that model to hold-out test data from NCAA Tournaments
    df_final_test = pd.read_pickle('data/tourney_model_data_final3.pkl')
    endog_final = df_final_test['label_h_point_spread'].values
    exogs_final = df_final_test.drop(['label_h_point_spread','label_home_winner','year','home_fgp_var','home_ppg_var',
                        'away_fgp_var','away_ppg_var','home_ps_home','away_ps_away',
                        'home_fgpct','away_fgpct','home_3ppct','away_3ppct',
                        'awayteam_awaywp','hometeam_pt_sprd','hometeam_opp_ppg',
                        'awayteam_pt_sprd','awayteam_opp_ppg','home_pyth_wd','away_pyth_wd',
                        'hometeam_ps_var', 'awayteam_ps_var','home_ftpct','away_ftpct',
                        'hpsq','apsq','hometeam_homewp','home_3pa_perposs','away_3pa_perposs','DATE','Home','Away'],axis=1)

    coeff, y_hat_final, rmse_final = model_regress(exogsc2, endogc,LinearRegression(), exogs_final, endog_final)

    coeff_dict = {"Variables": exogsc2.columns.values, "Coefficients": coeff}
    coeff_df = pd.DataFrame(data = coeff_dict)
    print(tabulate(coeff_df.round(4), headers='keys', tablefmt='pipe'))

    #Tabling and Graphing ROC
    thresh = np.arange(-5,5)
    y_prec, y_rec, y_acc, y_spec, y_fpr = roc_curve(y_hat_final, endog_final, thresh)
    roc_dict = {"Precision":y_prec, "Recall": y_rec, "Accuracy": y_acc, "Specificity": y_spec, "False Positive Rate": y_fpr}
    roc_df = pd.DataFrame(data = roc_dict, index = thresh)
    print(tabulate(roc_df.round(3), headers='keys', tablefmt='pipe'))

    thresh2 = np.sort(y_hat_final)
    y_prec2, y_rec2, y_acc2, y_spec2, y_fpr2 = roc_curve(y_hat_final, endog_final, thresh2)
    roc_graph(y_hat_final, endog_final, thresh2, 'Final NCAA Game Prediction Using Pointspread')

    df_final_test['Predicted_Point_Spread'] = y_hat_final
    df_final_test.rename(index = str, columns={'label_h_point_spread':'Point_Spread_Actual','label_home_winner':'Home_team_winner'}, inplace=True)
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
    sns.lmplot('Predicted_Point_Spread','Point_Spread_Actual', fit_reg=False, data = df_final_test, hue = 'Home_team_winner')
    plt.savefig('images/real_v_predicted1.png')
    plt.close()
    sns.lmplot('Predicted_Point_Spread','Point_Spread_Actual',  data = df_final_test)
    plt.savefig('images/real_v_predicted2.png')
    plt.close()

import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import matplotlib.pyplot as plt
import math
from utils import XyScaler
from linear import rmse
import warnings
import pdb

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
    return Precision, Recall, Accuracy, Spec, FPR

def roc_graph(y_hat,y, thresholds, name):
    prec, rec, acc, spec, fpr = roc_curve(y_hat, y, thresholds)
    fig, ax = plt.subplots()
    ax.plot(fpr, rec)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity, Recall)")
    ax.set_title("ROC plot of {}".format(name))
    plt.savefig("images/ROC of {}".format(name))

def model_regress(X,y,base_estimator):
    standardizer = XyScaler()
    standardizer.fit(X, y)
    X_std, y_std = standardizer.transform(X, y)
    # Fit estimator
    estimator = base_estimator
    estimator.fit(X_std, y_std)
    coeff = estimator.coef_
    # Measure performance
    y_hat_std = estimator.predict(X_std)
    X,y = standardizer.inverse_transform(X_std,y_std)
    X, y_hat = standardizer.inverse_transform(X_std,y_hat_std)
    rmse_final = rmse(y,y_hat)
    return y, y_hat, coeff, rmse_final

if __name__ == "__main__":
    df_orig = pd.read_pickle('data/reg_model_data_final1.pkl')
    endoga = df_orig['label_h_point_spread'].values
    exogsa0 = df_orig.drop(['label_h_point_spread','label_home_winner'],axis=1)
    y, y_hat, coeff, rmse_final = model_regress(exogsa0, endoga,Lasso(alpha = 0.011288))
    coeff_dict = dict(zip(exogsa0.columns,coeff))

    #Tabling and Graphing ROC
    thresh = np.arange(-5,5)
    y_prec, y_rec, y_acc, y_spec, y_fpr = roc_curve(y_hat, endoga, thresh)

    thresh2 = np.sort(y_hat)
    y_prec2, y_rec2, y_acc2, y_spec2, y_fpr2 = roc_curve(y_hat, endoga, thresh2)
    roc_graph(y_hat, y, thresh2, 'NCAA Game Prediction Using Pointspread')

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
import statsmodels.api as sm

def rmse(y, y_hat):
    '''Takes in y and y_hat and provides root mean squared error for them'''
    return math.sqrt(np.mean((y  - y_hat)**2))

def cv(X, y, base_estimator, n_folds, random_seed=154):
    """Estimate the in and out-of-sample error of a model using cross validation.

    Parameters
    ----------

    X: np.array
      Matrix of predictors.

    y: np.array
      Target array.

    base_estimator: sklearn model object.
      The estimator to fit.  Must have fit and predict methods.

    n_folds: int
      The number of folds in the cross validation.

    random_seed: int
      A seed for the random number generator, for repeatability.

    Returns
    -------

    train_cv_errors, test_cv_errors: tuple of arrays
      The training and testing errors for each fold of cross validation.
    """
    kf = KFold(n_splits=n_folds, random_state=random_seed)
    test_cv_errors, train_cv_errors = np.empty(n_folds), np.empty(n_folds)
    for idx, (train, test) in enumerate(kf.split(X)):
        # Split into train and test
        X_cv_train, y_cv_train = X[train], y[train]
        X_cv_test, y_cv_test = X[test], y[test]
        # Standardize data.
        standardizer = XyScaler()
        standardizer.fit(X_cv_train, y_cv_train)
        X_cv_train_std, y_cv_train_std = standardizer.transform(X_cv_train, y_cv_train)
        X_cv_test_std, y_cv_test_std = standardizer.transform(X_cv_test, y_cv_test)
        # Fit estimator
        estimator = base_estimator
        estimator.fit(X_cv_train_std, y_cv_train_std)
        # Measure performance
        y_hat_train = estimator.predict(X_cv_train_std)
        y_hat_test = estimator.predict(X_cv_test_std)
        # Calclate the error metrics
        train_cv_errors[idx] = rmse(y_cv_train_std, y_hat_train)
        test_cv_errors[idx] = rmse(y_cv_test_std, y_hat_test)
    return train_cv_errors, test_cv_errors

def linear_modeling_test(exogs, endog):
    exogs = sm.add_constant(exogs)
    test = sm.OLS(endog, exogs)
    test2 = test.fit()
    print(test2.summary())

def linear_modeling_multiple(exogs_array, endog):
    ''' Parameters:
            exogs_list: list of different of exogenous features for different models
            endog: label to regress on
        Returns:
            list of mean rmse for each model, train and test
            '''
    train_rmse_list = []
    test_rmse_list = []
    for exogs in exogs_array:
        linear_modeling_test(exogs,endog)
        exogs = exogs.values
        train_rmse, test_rmse = cv(exogs,endog,lr,20)
        train_rmse_list.append(train_rmse.mean())
        test_rmse_list.append(test_rmse.mean())
    return train_rmse_list, test_rmse_list

if __name__ == "__main__":
    df_orig = pd.read_pickle('data/reg_model_data_final.pkl')
    lr = LinearRegression()
    #Linear Modeling - test regression to look at coefficients
    endog = df_orig['label_h_point_spread'].values
    exogs0 = df_orig.drop(['label_h_point_spread','label_home_winner'],axis=1)#.values
    exogs1 = df_orig.drop(['label_h_point_spread','label_home_winner','home_ppg','home_fgapg','home_ftapg','home_ftpct','home_fgp_var','home_ppg_var','away_ftapg','away_ppg_var','away_3ppct','away_ftpct','awayteam_awaywp','hometeam_opp_ppg','pyth_wd_away','awayteam_ps_var','home_pace','away_pace'],axis=1)
    exogs2 = df_orig.drop(['label_h_point_spread','label_home_winner','home_ppg','home_fgapg','home_ftapg','home_ftpct','home_fgp_var','home_ppg_var','away_ftapg','away_ppg_var','away_3ppct','away_ftpct','awayteam_awaywp','hometeam_opp_ppg','pyth_wd_away','awayteam_ps_var','home_pace','away_pace','home_drebpg','home_fgpct','away_fgpct','home_3ppct','awayteam_opp_ppg'],axis=1)
    exogs3 = df_orig.drop(['label_h_point_spread','label_home_winner','home_ppg','home_fgapg','home_ftapg','home_ftpct','home_fgp_var','home_ppg_var','away_ftapg','away_ppg_var','away_3ppct','away_ftpct','awayteam_awaywp','hometeam_opp_ppg','pyth_wd_away','awayteam_ps_var','home_pace','away_pace','home_drebpg','home_fgpct','away_fgpct','home_3ppct','awayteam_opp_ppg','home_foulpg','home_3papg','away_3papg','home_stlpg'],axis=1)
    exogs_list = [exogs0,exogs1,exogs2,exogs3]
    train_rmse_list, test_rmse_list =  linear_modeling_multiple(exogs_list,endog)
    # linear_modeling_test(exogs0,endog)
    # linear_modeling_test(exogs1,endog)
    # linear_modeling_test(exogs2,endog)
    # linear_modeling_test(exogs3,endog)

    #LR1
    # exogs0 = df_orig.drop(['label_h_point_spread','label_home_winner'],axis=1).values
    # exogs1 = df_orig.drop()
    # exogs_array = [exogs0, exogs1, exogs2, exogs3, exogs4]
    # train_rmse1, test_rmse1 = cv(exogs0,endog,lr,20)

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
    ''' Parameters:
            exogs: Exogenous values - pandas dataframe
            endog: Endogenous value - pandas series
        Returns:
            Nothing - prints summary of test for inspection of coefficients.
        '''
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
        #linear_modeling_test(exogs,endog)#Only use for visual inspection of coefficients
        exogs = exogs.values
        train_rmse, test_rmse = cv(exogs,endog,lr,20)
        train_rmse_list.append(train_rmse.mean())
        test_rmse_list.append(test_rmse.mean())
    return train_rmse_list, test_rmse_list

def train_at_various_alphas(X, y, model, alphas, n_folds=10, **kwargs):
    """Train a regularized regression model using cross validation at various values of alpha.
    Code adapted from regularization assignment solutions.

    Parameters
    ----------

    X: np.array - Matrix of predictors.

    y: np.array - Target array.

    model: sklearn model class - A class in sklearn that can be used to create a regularized regression object.  Options are `Ridge` and `Lasso`.

    alphas: numpy array - An array of regularization parameters.

    n_folds: int - Number of cross validation folds.

    Returns
    -------

    cv_errors_train, cv_errors_test: tuple of DataFrame
      DataFrames containing the training and testing errors for each value of
      alpha and each cross validation fold.  Each row represents a CV fold,
      and each column a value of alpha.
    """
    cv_errors_train = pd.DataFrame(np.empty(shape=(n_folds, len(alphas))),
                                     columns=alphas)
    cv_errors_test = pd.DataFrame(np.empty(shape=(n_folds, len(alphas))),
                                        columns=alphas)
    for alpha in alphas:
        train_fold_errors, test_fold_errors = cv(X, y, model(alpha=alpha, **kwargs), n_folds=n_folds)
        cv_errors_train.loc[:, alpha] = train_fold_errors
        cv_errors_test.loc[:, alpha] = test_fold_errors
    return cv_errors_train, cv_errors_test

def get_optimal_alpha(mean_cv_errors_test):
    alphas = mean_cv_errors_test.index
    optimal_idx = np.argmin(mean_cv_errors_test.values)
    optimal_alpha = alphas[optimal_idx]
    return optimal_alpha

def plot_errors_alphas(alphas, opt_alpha, mean_train_errors, mean_test_errors, title):
    ''' Plots rmse for test and train sets over range of alphas.

        Parameters:
            alphas: Alphas to plot as x values (Array)
            opt_alpha: Optimal alpha value (Float)
            mean_train_errors: mean errors for training set (Array)
            mean_test_errors: mean errors for testing set (Array)
            title: title for graph (string)

        Returns: Nothing, but saves figure
        '''
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(np.log10(alphas), mean_train_errors)
    ax.plot(np.log10(alphas), mean_test_errors)
    ax.axvline(np.log10(opt_alpha), color='grey')
    ax.set_title(title)
    ax.set_xlabel(r"$\log(\alpha)$")
    ax.set_ylabel("RMSE")
    plt.savefig('images/{}.png'.format(title))

if __name__ == "__main__":
    df_orig = pd.read_pickle('data/reg_model_data_final.pkl')
    lr = LinearRegression()

    #Linear Modeling - OLS regressions
    endog = df_orig['label_h_point_spread'].values
    exogs0 = df_orig.drop(['label_h_point_spread','label_home_winner'],axis=1)#.values
    exogs1 = df_orig.drop(['label_h_point_spread','label_home_winner','home_ppg','home_fgapg','home_ftapg','home_ftpct','home_fgp_var','home_ppg_var','away_ftapg','away_ppg_var','away_3ppct','away_ftpct','awayteam_awaywp','hometeam_opp_ppg','pyth_wd_away','awayteam_ps_var','home_pace','away_pace'],axis=1)
    exogs2 = df_orig.drop(['label_h_point_spread','label_home_winner','home_ppg','home_fgapg','home_ftapg','home_ftpct','home_fgp_var','home_ppg_var','away_ftapg','away_ppg_var','away_3ppct','away_ftpct','awayteam_awaywp','hometeam_opp_ppg','pyth_wd_away','awayteam_ps_var','home_pace','away_pace','home_drebpg','home_fgpct','away_fgpct','home_3ppct','awayteam_opp_ppg'],axis=1)
    exogs3 = df_orig.drop(['label_h_point_spread','label_home_winner','home_ppg','home_fgapg','home_ftapg','home_ftpct','home_fgp_var','home_ppg_var','away_ftapg','away_ppg_var','away_3ppct','away_ftpct','awayteam_awaywp','hometeam_opp_ppg','pyth_wd_away','awayteam_ps_var','home_pace','away_pace','home_drebpg','home_fgpct','away_fgpct','home_3ppct','awayteam_opp_ppg','home_foulpg','home_3papg','away_3papg','home_stlpg'],axis=1)
    exogs_list = [exogs0,exogs1,exogs2,exogs3]
    train_rmse_list, test_rmse_list =  linear_modeling_multiple(exogs_list,endog)
    ols_train_rmse = np.array(train_rmse_list).min()
    ols_test_rmse = np.array(test_rmse_list).min()
    print("OLS")
    print("Minimum Training CV standardized rmse is for Exogs{0}:  {1:2.4f}".format(np.array(train_rmse_list).argmin(),ols_train_rmse))
    print("Minimum Testing CV standardized rmse is Exogs{0}:  {1:2.4f}".format(np.array(test_rmse_list).argmin(),ols_test_rmse))


    #LR1: Lasso Regression
    lasso_alphas = np.logspace(-6, 1, 20)
    lasso_cv_errors_train, lasso_cv_errors_test = train_at_various_alphas(exogs0.values, endog, Lasso, lasso_alphas, n_folds=10)
    lasso_mean_test_cv_errors = lasso_cv_errors_test.mean(axis=0)
    lasso_mean_train_cv_errors = lasso_cv_errors_train.mean(axis=0)
    lasso_opt_alpha = get_optimal_alpha(lasso_mean_test_cv_errors)
    lasso_train_rmse = lasso_mean_train_cv_errors.min()
    lasso_test_rmse = lasso_mean_test_cv_errors.min()
    print("Lasso")
    print("Minimum Training CV standardized rmse is {:2.4f}".format(lasso_train_rmse))
    print("Minimum Testing CV standardized rmse is {0:2.4f} for Alpha = {1:2.6}".format(lasso_test_rmse, lasso_opt_alpha))
    plot_errors_alphas(lasso_alphas,lasso_opt_alpha,lasso_mean_train_cv_errors,lasso_mean_test_cv_errors,'Lasso_RMSE_over_series_of_alphas_v1')

    #LR2: Ridge Regression
    ridge_alphas = np.logspace(-1,5)
    ridge_cv_errors_train, ridge_cv_errors_test = train_at_various_alphas(exogs0.values, endog, Ridge, ridge_alphas, n_folds=10)
    ridge_mean_test_cv_errors = ridge_cv_errors_test.mean(axis=0)
    ridge_mean_train_cv_errors = ridge_cv_errors_train.mean(axis=0)
    ridge_opt_alpha = get_optimal_alpha(ridge_mean_test_cv_errors)
    ridge_train_rmse = ridge_mean_train_cv_errors.min()
    ridge_test_rmse = ridge_mean_test_cv_errors.min()
    print("Ridge")
    print("Minimum Training CV standardized rmse is {:2.4f}".format(ridge_train_rmse))
    print("Minimum Testing CV standardized rmse is {0:2.4f} for Alpha = {1:2.6}".format(ridge_test_rmse, ridge_opt_alpha))
    plot_errors_alphas(ridge_alphas,ridge_opt_alpha,ridge_mean_train_cv_errors,ridge_mean_test_cv_errors,'Ridge_RMSE_over_series_of_alphas_v1')

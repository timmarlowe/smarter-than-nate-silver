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
import warnings

def warn(*args, **kwargs):
    pass

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
        y_hat_train_std = estimator.predict(X_cv_train_std)
        y_hat_test_std = estimator.predict(X_cv_test_std)
        # Calclate the error metrics
        train_cv_errors[idx] = rmse(y_cv_train_std, y_hat_train_std)
        test_cv_errors[idx] = rmse(y_cv_test_std, y_hat_test_std)
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

def linear_modeling_multiple(exogs_array, endog, loud):
    ''' Parameters:
            exogs_list: list of different of exogenous features for different models
            endog: label to regress on
            loud: choice of whether to loudly regress - takes 1 or 0
        Returns:
            list of mean rmse for each model, train and test
            Minimum cv mean for each model, train and test
            '''
    train_rmse_list = []
    test_rmse_list = []
    for exogs in exogs_array:
        if loud == 1:
            linear_modeling_test(exogs,endog)
        exogs = exogs.values
        train_rmse, test_rmse = cv(exogs,endog,lr,20)
        train_rmse_list.append(train_rmse.mean())
        test_rmse_list.append(test_rmse.mean())
    ols_train_rmse = np.array(train_rmse_list).min()
    ols_test_rmse = np.array(test_rmse_list).min()
    print("OLS")
    print("Minimum Training CV standardized rmse is for Exogs{0}:  {1:2.4f}".format(np.array(train_rmse_list).argmin(),ols_train_rmse))
    print("Minimum Testing CV standardized rmse is for Exogs{0}:  {1:2.4f}\n".format(np.array(test_rmse_list).argmin(),ols_test_rmse))
    return train_rmse_list, test_rmse_list, ols_train_rmse, ols_test_rmse

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

def regularization_func(exogs, endog, base_estimator, alphas,  title, n_folds=10):
    '''Regularizing function - returns estimates, graphs, for all regularizing functions.
        Parameters:
            exogs (pd dataframe),
            endog (pd series),
            base estimator (regularization estimator function),
            alphas (list of alphas)
            title (string) - for graphs and outputs
            n_folds - number of k-fold cvs
        Returns:
            Pd dataframe of training errors
            Pd dataframe of testing errors
            Minimum training error (float)
            Minmum testing error (float)
            Optimal alpha for minimum testing error (float)
        '''
    alphas = np.logspace(-6, 1, 20)
    cv_errors_train, cv_errors_test = train_at_various_alphas(exogs.values, endog, base_estimator, alphas, n_folds=10)
    mean_test_cv_errors = cv_errors_test.mean(axis=0)
    mean_train_cv_errors = cv_errors_train.mean(axis=0)
    opt_alpha = get_optimal_alpha(mean_test_cv_errors)
    train_rmse = mean_train_cv_errors.min()
    test_rmse = mean_test_cv_errors.min()
    plot_errors_alphas(alphas, opt_alpha, mean_train_cv_errors, mean_test_cv_errors, title)
    print(title)
    print("Minimum Training CV standardized rmse is {:2.4f}".format(train_rmse))
    print("Minimum Testing CV standardized rmse is {0:2.4f} for Alpha = {1:2.6}\n".format(test_rmse, opt_alpha))
    return cv_errors_train, cv_errors_test, train_rmse, test_rmse, opt_alpha

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
    warnings.warn = warn

    lr = LinearRegression()

    print('Data vA: Standard ds:\n')
    #Original model - straightforward
    #Linear Modeling - OLS regressions
    df_orig = pd.read_pickle('data/reg_model_data_final1.pkl')
    endoga = df_orig['label_h_point_spread'].values
    exogsa0 = df_orig.drop(['label_h_point_spread','label_home_winner'],axis=1)
    exogsa1 = df_orig.drop(['label_h_point_spread','label_home_winner','home_ppg','home_fgapg','home_ftapg','home_ftpct','home_fgp_var','home_ppg_var','away_ftapg','away_ppg_var','away_3ppct','away_ftpct','awayteam_awaywp','hometeam_opp_ppg','away_pyth_wd','awayteam_ps_var','home_pace','away_pace'],axis=1)
    exogsa2 = df_orig.drop(['label_h_point_spread','label_home_winner','home_ppg','home_fgapg','home_ftapg','home_ftpct','home_fgp_var','home_ppg_var','away_ftapg','away_ppg_var','away_3ppct','away_ftpct','awayteam_awaywp','hometeam_opp_ppg','away_pyth_wd','awayteam_ps_var','home_pace','away_pace','home_drebpg','home_fgpct','away_fgpct','home_3ppct','awayteam_opp_ppg'],axis=1)
    exogsa3 = df_orig.drop(['label_h_point_spread','label_home_winner','home_ppg','home_fgapg','home_ftapg','home_ftpct','home_fgp_var','home_ppg_var','away_ftapg','away_ppg_var','away_3ppct','away_ftpct','awayteam_awaywp','hometeam_opp_ppg','away_pyth_wd','awayteam_ps_var','home_pace','away_pace','home_drebpg','home_fgpct','away_fgpct','home_3ppct','awayteam_opp_ppg','home_foulpg','home_3papg','away_3papg','home_stlpg'],axis=1)
    exogs_alist = [exogsa0,exogsa1,exogsa2,exogsa3]
    ols_train_rmse_list_a, ols_test_rmse_list_a, ols_train_rmse_a, ols_test_rmse_a =  linear_modeling_multiple(exogs_alist,endoga,0)


    # #LR1: Lasso Regression
    lasso_alphas = np.logspace(-8, 1, 20)
    lasso_cv_errors_train_a, lasso_cv_errors_test_a, lasso_train_rmse_a, lasso_test_rmse_a, lasso_opt_alpha_a = regularization_func(exogsa0, endoga, Lasso, lasso_alphas,'Lasso_RMSE_vA')

    # #LR2: Ridge Regression
    ridge_alphas = np.logspace(-1,5,20)
    ridge_cv_errors_train_a, ridge_cv_errors_test_a, ridge_train_rmse_a, ridge_test_rmse_a, ridge_opt_alpha_a = regularization_func(exogsa0, endoga, Ridge, ridge_alphas,'Ridge_RMSE_vA')

    print('Data vB: Difference between home and away:\n')
    #Second set of models - on 'home minus away' dataset
    #Linear Modeling - OLS regressions
    df_ha = pd.read_pickle('data/reg_model_data_final_homeaway.pkl')
    endogb = df_ha['label_h_point_spread'].values
    exogsb0 = df_ha.drop(['label_h_point_spread','label_home_winner','diff_3ppct','diff_fgpct'],axis=1)
    exogsb1 = df_ha.drop(['label_h_point_spread','label_home_winner','diff_ftapg','diff_ppg_var','diff_3ppct','diff_pace','diff_fgpct'],axis=1)
    exogsb2 = df_ha.drop(['label_h_point_spread','label_home_winner','diff_ftapg','diff_ppg_var','diff_3ppct','diff_pace','diff_ftpct','diff_fgpct'],axis=1)
    exogs_blist = [exogsb0, exogsb1, exogsb2]
    ols_train_rmse_list_b, ols_test_rmse_list_b, ols_train_rmse_b, ols_test_rmse_b =  linear_modeling_multiple(exogs_blist,endogb,0)

    #LR1: Lasso Regression
    lasso_cv_errors_train_b, lasso_cv_errors_test_b, lasso_train_rmse_b, lasso_test_rmse_b, lasso_opt_alpha_b = regularization_func(exogsb0, endogb, Lasso, lasso_alphas,'Lasso_RMSE_vB')

    #LR2: Ridge Regression
    ridge_cv_errors_train_b, ridge_cv_errors_test_b, ridge_train_rmse_b, ridge_test_rmse_b, ridge_opt_alpha_b = regularization_func(exogsb0, endogb, Ridge, ridge_alphas,'Ridge_RMSE_vB')

    print('Data vC: Per Possession Stats and some Polynomials:\n')
    #Third set of models - on 'per possession' dataset
    #Linear Modeling - OLS regressions
    df_3 = pd.read_pickle('data/reg_model_data_final3.pkl')
    endogc = df_3['label_h_point_spread'].values
    exogsc0 = df_3.drop(['label_h_point_spread','label_home_winner','home_fgpct','away_fgpct','home_3ppct','away_3ppct'],axis=1)
    exogsc1 = df_3.drop(['label_h_point_spread','label_home_winner','year','home_fgp_var','home_ppg_var',
                        'away_fgp_var','away_ppg_var','home_ps_home','away_ps_away',
                        'home_fgpct','away_fgpct','home_3ppct','away_3ppct',
                        'awayteam_awaywp','hometeam_pt_sprd','hometeam_opp_ppg',
                        'awayteam_pt_sprd','awayteam_opp_ppg','home_pyth_wd','away_pyth_wd',
                        'hometeam_ps_var', 'awayteam_ps_var','home_ftpct','away_ftpct',
                        'hpsq','apsq'],axis=1)
    exogsc2 = df_3.drop(['label_h_point_spread','label_home_winner','year','home_fgp_var','home_ppg_var',
                        'away_fgp_var','away_ppg_var','home_ps_home','away_ps_away',
                        'home_fgpct','away_fgpct','home_3ppct','away_3ppct',
                        'awayteam_awaywp','hometeam_pt_sprd','hometeam_opp_ppg',
                        'awayteam_pt_sprd','awayteam_opp_ppg','home_pyth_wd','away_pyth_wd',
                        'hometeam_ps_var', 'awayteam_ps_var','home_ftpct','away_ftpct',
                        'hpsq','apsq','hometeam_homewp','home_3pa_perposs','away_3pa_perposs'],axis=1)

    exogs_clist = [exogsc0, exogsc1, exogsc2]
    ols_train_rmse_list_c, ols_test_rmse_list_c, ols_train_rmse_c, ols_test_rmse_c =  linear_modeling_multiple(exogs_clist,endogc,0)

    #LR1: Lasso Regression
    lasso_cv_errors_train_c, lasso_cv_errors_test_c, lasso_train_rmse_c, lasso_test_rmse_c, lasso_opt_alpha_c = regularization_func(exogsc0, endogc, Lasso, lasso_alphas,'Lasso_RMSE_vC')

    #LR2: Ridge Regression
    ridge_cv_errors_train_c, ridge_cv_errors_test_c, ridge_train_rmse_c, ridge_test_rmse_c, ridge_opt_alpha_c = regularization_func(exogsc0, endogc, Ridge, ridge_alphas,'Ridge_RMSE_vC')

    #CV accuracy rate of top three models (calling the winner)

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns

def hist_matrix(df):
    ''' Input: Dataframe of features
        Output: histogram plot comparing features
        '''
    histplot = df.hist(figsize=(20,20))
    plt.savefig(r"images/features_labels_histplot.png")
    plt.close()

def home_minus_away(df,varlst):
    ''' Input: Dataframe with home and away stats
        Output: Dataframe with home stats minus away stats
        '''
    df2 = df.copy()
    for var in varlst:
        df2['diff{}'.format(var)] = df['home{}'.format(var)] - df['away{}'.format(var)]
        df2.drop(['home{}'.format(var),'away{}'.format(var)], axis=1, inplace=True)
    return df2

def scatter_df(df):
    df_samp = df.sample(n=500)
    df_samp.drop(['home_ps_home','away_ps_away','hometeam_homewp','awayteam_awaywp','home_pyth_wd','away_pyth_wd','diff_drebpg','diff_fgpct','diff_3ppct','diff_ft_factor','diffteam_pt_sprd'], axis=1, inplace=True)
    scatplot = scatter_matrix(df_samp, alpha = 0.2, figsize = (30,30), diagonal='kde')
    plt.savefig(r'images/features_scatter_matrix.png')
    plt.close()

def scatter_outcomes(df):
    df_samp = df.sample(n=500)
    df_samp.drop(['home_pyth_wd','away_pyth_wd','year','label_home_winner'], axis=1, inplace=True)
    collist = list(df_samp.columns.values)
    collist.pop(0)
    for col in collist:
        sns_plot = sns.lmplot(x=col, y='label_h_point_spread', data = df_samp)
        plt.savefig(r"images/{}_and_home_point_spread.png".format(col))
        plt.close()

if __name__ == "__main__":
    df_orig = pd.read_pickle('data/reg_model_data_final1.pkl')
    df1 = df_orig.copy()
    hist_matrix(df1)
    df2 = df_orig.copy()
    varlist = ['_bpg','_drebpg','_orebpg','_foulpg','_ppg','_stlpg','_3papg','_fgapg','_ftapg','_apg','_fgp_var','_ppg_var','_fgpct','_3ppct','_ftpct','_efgpct','team_wp','team_pt_sprd','team_opp_ppg','team_ps_var','_tovpct','_ft_factor','_pace']
    df2= home_minus_away(df2,varlist)
    scatter_df(df2)
    df3 = df_orig.copy()
    df3 = home_minus_away(df3,varlist)
    scatter_outcomes(df3)

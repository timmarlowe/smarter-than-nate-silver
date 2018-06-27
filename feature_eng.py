import pandas as pd
import numpy as np
import pdb


def name_cleanup(df):
    ''' Input: Dataframe with each row being a game and both home and away team season previous stats calculated
        Output: Dataframe in same form with labels names
        '''
    df.rename(index = str, columns={'home_point_spread':'label_h_point_spread','home_winner':'label_home_winner'}, inplace=True)
    return df

def shooting(df):
    ''' Input: Dataframe with each row being a game and both home and away team season previous stats calculated
        Output: Dataframe in same form with shooting percentage calculated for home and away teams
        '''
    #FG %
    df['home_fgpct'] = df['home_fgmpg']/df['home_fgapg']
    df['away_fgpct'] = df['away_fgmpg']/df['away_fgapg']
    #3p%
    df['home_3ppct'] = df['home_3pmpg']/df['home_3papg']
    df['away_3ppct'] = df['away_3pmpg']/df['away_3papg']
    #FT%
    df['home_ftpct'] = df['home_ftmpg']/df['home_ftapg']
    df['away_ftpct'] = df['away_ftmpg']/df['away_ftapg']
    #EFG%
    df['home_efgpct'] = (df['home_fgmpg']+.5*df['home_3pmpg'])/df['home_fgapg']
    df['away_efgpct'] = (df['away_fgmpg']+.5*df['away_3pmpg'])/df['away_fgapg']
    #dropping unneeded columns
    df.drop(['home_fgmpg','away_fgmpg','home_3pmpg','away_3pmpg'], axis = 1,inplace=True)
    return df

def record(df):
    ''' Input: Dataframe with each row being a game and both home and away team season previous stats calculated
        Output: Dataframe in same form with W-L record stats calculated for home and away teams
        '''
    #calculating winning percentage for teams
    #Home team
    df['hometeam_wins'] = df['home_wins_home'] + df['away_games_home'] - df['away_losses_home']
    df['hometeam_games'] = df['home_games_home'] + df['away_games_home']
    df['hometeam_wp'] = df['hometeam_wins']/df['hometeam_games']
    df['hometeam_homewp'] = df['home_wins_home']/df['home_games_home']
    #Away team
    df['awayteam_wins'] = df['home_wins_away'] + df['away_games_away'] - df['away_losses_away']
    df['awayteam_games'] = df['home_games_away'] + df['away_games_away']
    df['awayteam_wp'] = df['awayteam_wins']/df['awayteam_games']
    df['awayteam_awaywp'] = 1-(df['away_losses_away']/df['away_games_away'])
    #dropping unneeded columns
    df.drop(['hometeam_wins','home_wins_home','away_losses_home','awayteam_wins','home_wins_away','away_losses_away'], axis = 1,inplace=True)
    return df

def point_spread(df):
    ''' Input: Dataframe with each row being a game and both home and away team season previous stats calculated
        Output: Dataframe in same form with point spread stats calculated for home and away teams
        '''
    #Avg points spread in games
    #Home
    df['hometeam_pt_sprd'] = ((df['home_games_home'] * df['home_ps_home'])+(df['away_games_home'] * df['away_ps_home']))/df['hometeam_games']
    df['hometeam_opp_ppg'] = df['home_ppg'] - df['hometeam_pt_sprd']
    #Away
    df['awayteam_pt_sprd'] = ((df['home_games_away'] * df['home_ps_away'])+(df['away_games_away'] * df['away_ps_away']))/df['awayteam_games']
    df['awayteam_opp_ppg'] = df['away_ppg'] - df['awayteam_pt_sprd']
    #Pythagorean expected wins
    df['pyth_wp_home'] = (df['home_ppg']**14)/((df['home_ppg']**14)+(df['hometeam_opp_ppg']**14))
    df['pyth_wp_away'] = (df['away_ppg']**14)/((df['away_ppg']**14)+(df['awayteam_opp_ppg']**14))
    #Point Spread Variance
    df['hometeam_ps_var'] = ((df['hometeam_games']-1)*(df['home_ps_var_home']+df['away_ps_var_home'])+((df['hometeam_games']/2)*((df['home_ps_home']-df['away_ps_home'])**2)))/(2*df['hometeam_games']-1)
    df['awayteam_ps_var'] = ((df['awayteam_games']-1)*(df['home_ps_var_away']+df['away_ps_var_away'])+((df['awayteam_games']/2)*((df['home_ps_away']-df['away_ps_away'])**2)))/(2*df['awayteam_games']-1)
    df.drop(['home_ps_var_home', 'away_games_home', 'away_ps_home','away_ps_var_home', 'home_games_away', 'home_ps_away','home_ps_var_away', 'away_games_away','away_ps_var_away','home_games_home'],axis=1, inplace=True)
    return df

def misc_features(df):
    ''' Input: Dataframe with each row being a game and both home and away team season previous stats calculated
        Output: Dataframe in same form with other stats calculated for home and away teams
        '''
    #Turnover percent - one of four factors
    df['home_tovpct'] = df['home_topg']/(df['home_fgapg'] + 0.44 * df['home_ftapg'] + df['home_topg'])
    df['away_tovpct'] = df['away_topg']/(df['away_fgapg'] + 0.44 * df['away_ftapg'] + df['away_topg'])
    #Free throw factor - another predictor
    df['home_ft_factor'] = df['home_ftmpg']/df['home_fgapg']
    df['away_ft_factor'] = df['away_ftmpg']/df['away_fgapg']
    #Pace - avg possessions per game
    df['home_pace'] =  df['home_fgapg'] - df['home_orebpg'] + df['home_topg']
    df['away_pace'] =  df['away_fgapg'] - df['away_orebpg'] + df['away_topg']
    df.drop(['home_ftmpg','away_ftmpg'],axis=1, inplace=True)
    return df

def split_seasons(df):
    '''
    INPUT: pandas dataframe with player stats
    OUTPUT: 2 dataframes split between regular season and March Madness games.
    =========================================================================
    This will be our train/test split for each season
    Thank you to Steve Iannaccone for this piece of code
        '''
    TourneyDates = {2007: '2007-03-11',
                    2008: '2008-03-18',
                    2009: '2009-03-17',
                    2010: '2010-03-16',
                    2011: '2011-03-15',
                    2012: '2012-03-13',
                    2013: '2013-03-19',
                    2014: '2014-03-18',
                    2015: '2015-03-17',
                    2016: '2016-03-15',
                    2017: '2017-03-14',
                    2018: '2018-03-13'}
    df['madness_date'] = df['year'].map(TourneyDates)
    df_reg = df[df['DATE_STRING'] < df['madness_date']].reset_index()
    df_tourney = df[df['DATE_STRING'] >= df['madness_date']].reset_index()
    df_reg.drop(['level_0','index','madness_date','DATE_STRING'],axis=1, inplace=True)
    df_tourney.drop(['level_0','index','madness_date','DATE_STRING'],axis=1, inplace=True)
    return df_reg, df_tourney

if __name__ == "__main__":
    df = pd.read_pickle('data/modeling_db1.pkl')
    df = name_cleanup(df)
    df = shooting(df)
    df = record(df)
    df = point_spread(df)
    df = misc_features(df)
    df.dropna(inplace=True)
    df.to_pickle('data/modeling_whole.pkl')
    df_reg, df_tourney = split_seasons(df)
    df_reg.to_pickle('data/reg_model_data_final.pkl')
    df_tourney.to_pickle('data/tourney_model_data_final.pkl')

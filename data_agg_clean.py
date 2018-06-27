import pandas as pd
import numpy as np
from datetime import datetime
import pdb

def read_data():
    ''' Input: None
        Output: Dataframe with box scores for every game in array of years, players as rows
        '''
    games_df = pd.read_csv('data/2017-2018_gamedata.csv')
    games_df['year']= 2018
    years = np.arange(2007,2018)
    for year in years:
        df = pd.read_csv('data/{}-{}_gamedata.csv'.format(year-1,year))
        df['year'] = year
        games_df = pd.concat((games_df,df),ignore_index=True, sort=True)
    return games_df

def clean_game_data(games_df):
    ''' Input: DF of box scores
        Output: Cleaned DF of box scores
        '''
    #Aggregating 3PA/M into single column
    games_df['3PM-A'].fillna(games_df['3GM-A'], inplace=True)
    games_df.drop(['3GM-A','Unnamed: 0'], axis = 1,inplace=True)
    #Dropping N/A value rows (only 15)
    games_df.dropna(inplace=True)
    #Creating columns for float values of 3PA, 3PM, FGA, FGM, FTA, FTM
    games_df['3PM'] = games_df['3PM-A'].str.split("-").str.get(0).astype(float)
    games_df['3PA'] = games_df['3PM-A'].str.split("-").str.get(1).astype(float)
    games_df['FGM'] = games_df['FGM-A'].str.split("-").str.get(0).astype(float)
    games_df['FGA'] = games_df['FGM-A'].str.split("-").str.get(1).astype(float)
    games_df['FTM'] = games_df['FTM-A'].str.split("-").str.get(0).astype(float)
    games_df['FTA'] = games_df['FTM-A'].str.split("-").str.get(1).astype(float)
    #Cleaning Rebound names
    games_df.rename(index = str, columns={'OFF':'OFF_REB', 'DEF':'DEF_REB','TOT':'TOT_REB'}, inplace=True)
    #Creating home team and away team columns
    games_df['Away'] = games_df['MATCH'].str.split(" vs. ").str.get(0).str.strip('\n')
    games_df['Home'] = games_df['MATCH'].str.split(" vs. ").str.get(1).str.strip('\n')
    #Stripping team of new line
    games_df['TEAM'] = games_df['TEAM'].str.strip('\n')
    #Cleaning up minutes
    games_df['MINUTES']=games_df['MIN'].str.split(":").str.get(0).astype(float)
    games_df.drop(['MIN'], axis = 1,inplace=True)
    #Replacing date/time
    games_df.rename(index = str, columns={'DATE':'DATE_STRING'}, inplace=True)
    games_df['DATE'] = pd.to_datetime(games_df['DATE_STRING'], format='%Y-%m-%d')
    games_df['FGP'] = games_df['FGM']/games_df['FGA']
    return games_df

def create_matchups(df):
    ''' Input: DF of boxscores
        Output: DF (aggregated of matchups)
        '''
    matchups = df.groupby(['year','DATE','DATE_STRING','MATCH','TEAM','Home','Away']).agg('sum').reset_index()
    matchups['home_team'] = (matchups['TEAM']==matchups['Home']).astype(int)
    return matchups

def error(df):
    ''' Input: DF of matchups
        Outpu: DF of matchups (with unclean games removed)
        '''
    df_temp = df.copy().sort_values(['DATE','MATCH','home_team']).reset_index()
    df_temp.replace({'home_team': {0: -1}},inplace=True)
    error_check = df_temp.groupby(['year','DATE_STRING','MATCH']).agg({'home_team':'sum'})
    error_check.groupby('year').agg('sum') #About 400 games per year with no away team listed
    error_check.rename(index = str, columns={'home_team':'h_sum'}, inplace=True)
    error_check['h_sum'].sum()#Overall about 4300 games (out of 60k) with no away team
    game_outcomes = pd.merge(df_temp, error_check, how='left',on=['DATE_STRING','MATCH'])
    game_outcomes = game_outcomes[game_outcomes['h_sum']==0]
    game_outcomes.replace({'home_team': {-1: 0}},inplace=True)
    return game_outcomes

def games_finaldb(df):#Creating a db of games and outcomes of those games for later reference
    ''' Input: DF of matchups
        Output: DF of winners and point spread
        '''
    game_outcomes = df.copy().sort_values(['DATE','MATCH','home_team']).reset_index()
    game_outcomes['home_point_spread'] = game_outcomes['PTS'].diff() * game_outcomes['home_team']
    games_final = game_outcomes[['year','DATE','DATE_STRING','Home','Away','TEAM','home_point_spread']].copy()
    games_final['home_winner'] = (games_final['home_point_spread']>0).astype(int)
    games_final = games_final[games_final['TEAM']==games_final['Home']]#removing all rows with away team as row
    games_final['month'] = pd.DatetimeIndex(games_final['DATE']).month
    games_final.drop('TEAM', axis=1,inplace=True)
    return games_final

def get_season_stats(row):
    #Home df
    df = game_stats[(game_stats['year'] == row.year) & (game_stats['DATE']<row.DATE) & (game_stats['TEAM']==row.Home)]
    #pdb.set_trace()
    dfmean_home = df.loc[:, ['BLK','DEF_REB','OFF_REB','TOT_REB','PF','PTS','STL','TO','3PM','3PA','FGM','FGA','FTM','FTA','A']].mean(axis=0)
    dfvar_home = df.loc[:,['FGP','PTS']].var(axis=0)
    df_home = pd.concat([dfmean_home,dfvar_home])
    #Away df
    df2 = game_stats[(game_stats['year'] == row.year) & (game_stats['DATE']<row.DATE) & (game_stats['TEAM']==row.Away)]
    dfmean_away = df2.loc[:, ['BLK','DEF_REB','OFF_REB','TOT_REB','PF','PTS','STL','TO','3PM','3PA','FGM','FGA','FTM','FTA','A']].mean(axis=0)
    dfvar_away = df2.loc[:,['FGP','PTS']].var(axis=0)
    df_away = pd.concat([dfmean_away,dfvar_away])
    df_all = pd.concat([df_home, df_away])
    return df_all

def get_record(row):
    #pdb.set_trace()
    #aggregating home games for home team
    df = games_final[(games_final['year'] == row.year) & (games_final['DATE']<row.DATE) & (games_final['Home']==row.Home)]
    home_wins_home = df.loc[:,['home_winner']].sum(axis=0)
    home_games_home = df.loc[:,['home_winner']].count(axis=0)
    home_ps_home = df.loc[:,['home_point_spread']].mean(axis=0)
    home_ps_var_home = df.loc[:,['home_point_spread']].var(axis=0)
    #aggregating away games for home team
    df = games_final[(games_final['year'] == row.year) & (games_final['DATE']<row.DATE) & (games_final['Home']==row.Away)]
    away_losses_home = df.loc[:,['home_winner']].sum(axis=0)
    away_games_home = df.loc[:,['home_winner']].count(axis=0)
    away_ps_home = df.loc[:,['home_point_spread']].mean(axis=0)#Will have to multiply time -1 to put in same units as home point_spread home
    away_ps_var_home = df.loc[:,['home_point_spread']].var(axis=0)
    #aggregating home games for away team
    df = games_final[(games_final['year'] == row.year) & (games_final['DATE']<row.DATE) & (games_final['Away']==row.Home)]
    home_wins_away = df.loc[:,['home_winner']].sum(axis=0)
    home_games_away = df.loc[:,['home_winner']].count(axis=0)
    home_ps_away = df.loc[:,['home_point_spread']].mean(axis=0)
    home_ps_var_away = df.loc[:,['home_point_spread']].var(axis=0)
    #aggregating away games for home team
    df = games_final[(games_final['year'] == row.year) & (games_final['DATE']<row.DATE) & (games_final['Away']==row.Away)]
    away_losses_away = df.loc[:,['home_winner']].sum(axis=0)
    away_games_away = df.loc[:,['home_winner']].count(axis=0)
    away_ps_away = df.loc[:,['home_point_spread']].mean(axis=0)#Will have to multiply time -1 to put in same units as home point_spread home
    away_ps_var_away = df.loc[:,['home_point_spread']].var(axis=0)
    df_all = pd.concat([home_wins_home,home_games_home,home_ps_home,home_ps_var_home,away_losses_home,away_games_home,away_ps_home,away_ps_var_home,home_wins_away,home_games_away,home_ps_away,home_ps_var_away,away_losses_away,away_games_away,away_ps_away,away_ps_var_away])
    return df_all

# def ind_players(df):
#     playerdf =

if __name__ == "__main__":
    #Read in data
    df = read_data()

    #Clean up datafram
    cleandf = clean_game_data(df)
    cleandf.to_pickle('data/initial_clean_df.pkl')
    #cleandf=pd.read_pickle('data/initial_clean_df.pkl')

    #Create matchup dataframe, game_outcomedf (with one row per team in matchup and point spread), and games_final (with one row per game and point spread)
    matchupdf = create_matchups(cleandf)
    game_outcomedf = error(matchupdf)
    game_outcomedf.to_pickle('data/games.pkl')
    games_finaldf = games_finaldb(game_outcomedf)
    #games_finaldf.to_pickle('data/games_final.pkl')
    games_final = pd.read_pickle('data/games_final.pkl')
    #game_stats = pd.read_pickle('data/games.pkl')

    #creating set of games for predicting (removing first two months - November (11) and December (12) as not enough season history)
    games_test = games_final[(games_final['month']<4)]

    #Pulling team stats up until game in games_test dataframe
    df_allstats = games_test.apply(get_season_stats,axis=1)
    df_allstats.columns=['home_bpg','home_drebpg','home_orebpg','home_trebpg','home_foulpg','home_ppg','home_stlpg','home_topg','home_3pmpg','home_3papg','home_fgmpg','home_fgapg','home_ftmpg','home_ftapg','home_apg','home_fgp_var','home_ppg_var','away_bpg','away_drebpg','away_orebpg','away_trebpg','away_foulpg','away_ppg','away_stlpg','away_topg','away_3pmpg','away_3papg','away_fgmpg','away_fgapg','away_ftmpg','away_ftapg','away_apg','away_fgp_var','away_ppg_var']
    df_allstats.to_pickle('data/df_all.pkl')
    #df_all_stats = pd.read_pickle('data/df_all.pkl')

    #Pulling team wins and point spreads up until game in games_test dataframe
    df_wins_pts = games_test.apply(get_record,axis=1)
    df_wins_pts.columns = ['home_wins_home','home_games_home','home_ps_home','home_ps_var_home','away_losses_home','away_games_home','away_ps_home','away_ps_var_home','home_wins_away','home_games_away','home_ps_away','home_ps_var_away','away_losses_away','away_games_away','away_ps_away','away_ps_var_away']
    df_wins_pts.to_pickle('data/df_wins_pts.pkl')
    #df_wins_pts = pd.read_pickle('data/df_wins_pts.pkl')

    #Creating modeling database with team stats, game history, and outcome of game to be predicted
    modeling_db1 = pd.concat([games_test, df_all_stats, df_wins_pts],axis=1).reset_index()
    modeling_db1.to_pickle('data/modeling_db1.pkl')

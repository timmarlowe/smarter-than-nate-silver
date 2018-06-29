# (Not Necessarily) Smarter Than Nate Silver
### Predicting the NCAA Tournament Using Team Boxscore Data
##### Tim Marlowe -- Date of Last Update: 6/28/2018

_Data and original idea thanks to Steve Iannaccone, whose project "Smarter Than Nate Silver" was the original fork for this repo_

![Kris Jenkins hits a buzzer beater](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/images/kris-jenkins-villanova-buzzer-beat-shot-4516-getty-ftrjpg_fxtkh1vphf341td70sn7qpiql.jpg)
Photo Credit: [The Sporting News](http://www.sportingnews.com/ncaa-basketball/news/ncaa-tournament-greatest-games-duke-kentucky-north-carolina-unc-villanova-georgetown/3za49mgurf091mjxj0j7abz0t)

## Problem Statement:
Building models to predict the NCAA tournament is often done by using advanced metrics of team success over the course of the season. These include metrics like
[Elo rating](https://en.wikipedia.org/wiki/Elo_rating_system), [BPI and Strength of Record](http://www.espn.com/blog/statsinfo/post/_/id/125994/bpi-and-strength-of-record-what-are-they-and-how-are-they-derived).

My question was whether I could build a predictor off of a data set that used none of the aggregated metrics to predict point spread and thus outcomes of final four games. Given that there are over 3000 games a season, if a true signal could be found just in the box scores of individual matchups and the aggregation of team stats for each matchup up to that point in the season, it may be a more powerful and specific predictor than a team's advanced stats at the end of the season going into March Madness.

## Data Source
#### Aggregation Methods
Following Steve's lead, I used data from individual game box scores from the website http://sportsdata.wfmz.com. I aggregated 11 years of this data (from the 2006-2007 season to the 2017-2018 season). In order to do this, I used Steve Iacconne's [scrapey.py](https://github.com/timmarlowe/smarter-than-nate-silver/edit/master/src/scrapey.py) code. With the exception of a dictionary with the dates of the start of March Madness, this was the only code of his I used).

Upon compiling the data, I was left with a data set of 1.3 million rows, in which each row was a player's individual stat-line in a game. I aggregated up to the matchup and then game level using groupby. Then given that I wanted to predict on record up to a certain game, the task was to aggregate the stats of each team up to that game. In the example below, in order to predict Louisville's outcome agains VA Tech, I needed their team stats up until VA Tech.

![Louisville Schedule](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/images/Louisville%20Cardinals%20Schedule.png)

I created the following code in [data_agg_clean.py](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/src/data_agg_clean.py) to aggregate a team's stats (mean and variance) up to the game in question and merge that data with my games result database (games_test):
```python
def get_season_stats(row):
    #Create Home df
    df = game_stats[(game_stats['year'] == row.year) & (game_stats['DATE']<row.DATE) & (game_stats['TEAM']==row.Home)]
    #pdb.set_trace()
    dfmean_home = df.loc[:, ['BLK','DEF_REB','OFF_REB','TOT_REB','PF','PTS','STL','TO','3PM','3PA','FGM','FGA','FTM','FTA','A']].mean(axis=0)
    dfvar_home = df.loc[:,['FGP','PTS']].var(axis=0)
    df_home = pd.concat([dfmean_home,dfvar_home])
    #Create Away df
    df2 = game_stats[(game_stats['year'] == row.year) & (game_stats['DATE']<row.DATE) & (game_stats['TEAM']==row.Away)]
    dfmean_away = df2.loc[:, ['BLK','DEF_REB','OFF_REB','TOT_REB','PF','PTS','STL','TO','3PM','3PA','FGM','FGA','FTM','FTA','A']].mean(axis=0)
    dfvar_away = df2.loc[:,['FGP','PTS']].var(axis=0)
    df_away = pd.concat([dfmean_away,dfvar_away])
    #concatenate the two
    df_all = pd.concat([df_home, df_away])
    return df_all

df_allstats = games_test.apply(get_season_stats,axis=1)
    ```

Next, I attached win-loss record and average point spread up to that point in the season through a similar process.

#### Feature Creation
I cleaned up the data and added advanced stats such as [Turnover Percent, Free Throw Factor and Effective Field Goal Percentage](https://www.basketball-reference.com/about/glossary.html) using [feature_eng.py](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/src/feature_eng.py).

___Turnover percent___:
```python
df['home_tovpct'] = df['home_topg']/(df['home_fgapg'] + 0.44 * df['home_ftapg'] + df['home_topg'])
```
___Free Throw Factor___:
```python
df['home_ft_factor'] = df['home_ftmpg']/df['home_fgapg']
```
___Effective Field Goal Percent___:
```python
df['home_efgpct'] = (df['home_fgmpg']+.5*df['home_3pmpg'])/df['home_fgapg']
```
I also included an amateur calculation of ___number of possessions___ in a game as a pace metric:
```python
df['home_pace'] =  df['home_fgapg'] - df['home_orebpg'] + df['home_topg']
```
And a calculation of the ___pythagorean expectation___ for difference between games won and expected games won each team:
```python
df['home_pyth_wd'] = df['hometeam_wins'] - df['hometeam_games'] * (df['home_ppg']**14)/((df['home_ppg']**14)+(df['hometeam_opp_ppg']**14))
```

Unfortunately, because of the method of aggregation and lack of data from the box scores, I was not able to calculate most defensive statistics.

#### Dataframes and Train-Test Split
As a last step prior to train-test-split, I created three dataframes, each one with its own approach to the data.

- [DF1](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/data/reg_model_data_final1.pkl) contains multiple fields for home teams and those same fields for away teams. It has the most fields and therefore, likely the highest risk of high variance in initial regression results.

- [DF2](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/data/reg_model_data_final_homeaway.pkl) holds mostly fields representing differential in these stats between home and away teams, foregoing absolute magnitude in favor of relative magnitude.

- [DF3](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/data/reg_model_data_final3.pkl) is like DF1 in that it contains metrics for both the home and the away team, but I have standardized applicable statistics by the possessions metric, so that most fields represent statistics per possession instead of per game. I also added a squared terms for home and away point spread.

Lastly, I created a train-test split between regular season and tournament games. This decision was to ensure that the analysis was truthful to a situation in which one did not know what the results of the tournament would be. I will discuss this decision further in my results section.

## Exploratory Data Analysis
All EDA was completed using the
The following histogram of variables from DF1 demonstrates the distributions of the individual features, as well as of the labels.![Histogram plot](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/images/features_labels_histplot.png)
Most data is fairly normally distributed across rows (a row being a team up until a matchup within that season). One item of note is that home team point spread and winning percentage are skewed to the left, while away team winning percentage and point spread are skewed to the right. The sample mean for the label I was predicting on (point spread of the actual game) was __3.66__, meaning the home team on average wins by 3.66 points. This is not surprising for the regular season, but it is an issue for prediction in the post-season, when home floor is in name only.

The following scatter matrix of the features on each other (using DF2 home-away aggregated features for sake of space) demonstrates collinear relationships between some of the features, including between effective field goal percentage and points per game, pace and field goal attempts, points per game and assists per game. ![Scatter Matrix](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/images/features_scatter_matrix.png)
While most still seemed to contain some information of their own, I dropped a few, such as field goal percentage, as it was so closely related to effective field goal percentage.

Finally, scatter plots of features on the label demonstrated that a few of our features had promising relationships with point spread. ![EFG and Point Spread](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/images/diff_efgpct_and_home_point_spread.png) 
![Turnover % and Point Spread](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/images/diff_tovpct_and_home_point_spread.png)

However, others seemed to have little to no relationship to point spread 
![3-points attempted and Point Spread](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/images/diff_3papg_and_home_point_spread.png) 
![Offensive Rebounds per game and point spread](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/images/diff_orebpg_and_home_point_spread.png)




|    | Model Variables         |   Model Coefficients |
|---:|:------------------|---------------:|
|  0 | year              |        -0.0213 |
|  1 | home_fgp_var      |         0.0024 |
|  2 | home_ppg_var      |        -0.0051 |
|  3 | away_fgp_var      |        -0.0128 |
|  4 | away_ppg_var      |        -0.0063 |
|  5 | home_ps_home      |         0.29   |
|  6 | away_ps_away      |         0.2508 |
|  7 | home_3ppct        |        -0.0428 |
|  8 | away_3ppct        |         0.0478 |
|  9 | home_ftpct        |         0.0509 |
| 10 | away_ftpct        |        -0.0114 |
| 11 | home_efgpct       |        -0.1863 |
| 12 | away_efgpct       |         0.2495 |
| 13 | hometeam_wp       |         0.0512 |
| 14 | hometeam_homewp   |        -0.0421 |
| 15 | awayteam_wp       |        -0.042  |
| 16 | awayteam_awaywp   |         0.0164 |
| 17 | hometeam_pt_sprd  |        -0.1518 |
| 18 | hometeam_opp_ppg  |        -0.022  |
| 19 | awayteam_pt_sprd  |        -0.1398 |
| 20 | awayteam_opp_ppg  |         0.0497 |
| 21 | home_pyth_wd      |         0.0459 |
| 22 | away_pyth_wd      |         0.0006 |
| 23 | hometeam_ps_var   |         0.0136 |
| 24 | awayteam_ps_var   |        -0.0052 |
| 25 | home_tovpct       |        -0.1709 |
| 26 | away_tovpct       |         0.2696 |
| 27 | home_ft_factor    |        -0.2454 |
| 28 | away_ft_factor    |         0.1702 |
| 29 | home_dreb_perposs |         0.0367 |
| 30 | away_dreb_perposs |        -0.0306 |
| 31 | home_oreb_perposs |         0.1397 |
| 32 | away_oreb_perposs |        -0.2191 |
| 33 | home_foul_perposs |         0.0224 |
| 34 | away_foul_perposs |        -0.0187 |
| 35 | home_p_perposs    |         0.4522 |
| 36 | away_p_perposs    |        -0.5374 |
| 37 | home_stl_perposs  |         0.0348 |
| 38 | away_stl_perposs  |        -0.0356 |
| 39 | home_3pa_perposs  |         0.0072 |
| 40 | away_3pa_perposs  |        -0.0023 |
| 41 | home_fga_perposs  |        -0.3774 |
| 42 | away_fga_perposs  |         0.5197 |
| 43 | home_fta_perposs  |         0.0504 |
| 44 | away_fta_perposs  |         0.0819 |
| 45 | home_a_perposs    |         0.0267 |
| 46 | away_a_perposs    |        -0.0174 |
| 47 | home_b_perposs    |         0.0399 |
| 48 | away_b_perposs    |        -0.031  |
| 49 | hpsq              |        -0.0081 |
| 50 | apsq              |         0.0149 |


|    |   Precision |   Recall |   Accuracy |   Specificity |   False Positive Rate |
|---:|------------:|---------:|-----------:|--------------:|----------------------:|
| -5 |       0.704 |    0.998 |      0.703 |         0.008 |                 0.992 |
| -4 |       0.706 |    0.998 |      0.706 |         0.018 |                 0.982 |
| -3 |       0.706 |    0.992 |      0.704 |         0.023 |                 0.977 |
| -2 |       0.709 |    0.982 |      0.704 |         0.048 |                 0.952 |
| -1 |       0.716 |    0.96  |      0.704 |         0.099 |                 0.901 |
|  0 |       0.718 |    0.927 |      0.693 |         0.142 |                 0.858 |
|  1 |       0.729 |    0.873 |      0.683 |         0.234 |                 0.766 |
|  2 |       0.743 |    0.811 |      0.67  |         0.338 |                 0.662 |
|  3 |       0.757 |    0.712 |      0.637 |         0.459 |                 0.541 |
|  4 |       0.764 |    0.608 |      0.592 |         0.556 |                 0.444 |

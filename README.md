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

## Feature Engineering
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

Finally, scatter plots of features on the label demonstrated that a few of our features had promising relationships with point spread.
![EFG and Point Spread](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/images/diff_efgpct_and_home_point_spread.png)

However, others seemed to have little to no relationship to point spread
![3-points attempted and Point Spread](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/images/diff_3papg_and_home_point_spread.png)

More can be found in the [images](https://github.com/timmarlowe/smarter-than-nate-silver/tree/master/images) folder. Further feature engineering is definitely recommended to make this product viable.

## Modeling
I used three different linear regression models to estimate point spread on each of the databases (OLS, and two regularization models: Lasso and Ridge). Code for modeling can be found in [linear.py](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/src/linear.py).

Each model was created using 10-fold cross-validation on the training data.

For OLS, I used backwards selection to remove coefficients over time. However, likely due to a high bias low variance model to begin with, the removal of features that were not statistically significant created very little change in RMSE.

Similarly, Lasso models for all three dataframes settled on the extremely small coefficients in the range provided, likely because the model is underfit.
![Lasso Model DF1](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/images/Lasso_RMSE_vA.png)

Ridge faired no better:
![Ridge Model DF1]()

In fact, all three models hovered around the exact same RMSE, no matter the dataframe or adjustment to the exogenous variables:
![RMSE by Model]()

In the end, because regularization and addition of terms were both ineffective at increasing the explanatory power of the model, I chose an OLS model with fewer terms based on the DF3, the per possession database. This model was as follows when trained on the entire training dataset.

|    | Variables         |   Coefficients |
|---:|:------------------|---------------:|
|  0 | home_efgpct       |      -123.072  |
|  1 | away_efgpct       |        85.1522 |
|  2 | hometeam_wp       |        11.6049 |
|  3 | awayteam_wp       |        -9.6597 |
|  4 | home_tovpct       |       -90.1705 |
|  5 | away_tovpct       |       156.767  |
|  6 | home_ft_factor    |       -60.0435 |
|  7 | away_ft_factor    |        36.0149 |
|  8 | home_dreb_perposs |        33.9899 |
|  9 | away_dreb_perposs |       -36.7376 |
| 10 | home_oreb_perposs |        61.553  |
| 11 | away_oreb_perposs |       -93.3587 |
| 12 | home_foul_perposs |        14.5427 |
| 13 | away_foul_perposs |       -15.0109 |
| 14 | home_p_perposs    |        95.4668 |
| 15 | away_p_perposs    |       -71.0021 |
| 16 | home_stl_perposs  |        50.3138 |
| 17 | away_stl_perposs  |       -50.2611 |
| 18 | home_fga_perposs  |      -140.855  |
| 19 | away_fga_perposs  |       156.897  |
| 20 | home_fta_perposs  |       -28.0587 |
| 21 | away_fta_perposs  |        35.862  |
| 22 | home_a_perposs    |        13.2161 |
| 23 | away_a_perposs    |        -8.4026 |
| 24 | home_b_perposs    |        27.5975 |
| 25 | away_b_perposs    |       -22.3028 |

It is still clearly both over-fit and under-fit.



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

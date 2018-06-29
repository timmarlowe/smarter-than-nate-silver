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
Following Steve's lead, I used data from individual game box scores from the website http://sportsdata.wfmz.com. I aggregated 11 years of this data (from the 2006-2007 season to the 2017-2018 season). In order to do this, I used Steve Iacconne's [scrapey.py](https://github.com/timmarlowe/smarter-than-nate-silver/edit/master/src/scrapey.py) code.

This resulted in a data set of 1.3 million rows, in which each row was a player's individual stat-line in a game. I aggregated up to the matchup (2 rows per game) and then game level (1 row per game) using groupby. Then given that I wanted to predict on record up to a certain game, I aggregated the stats of each team in the matchup up to that game. In the example below, in order to predict Louisville's outcome agains VA Tech, I needed their team stats up until VA Tech.

![Louisville Schedule](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/images/Louisville%20Cardinals%20Schedule.png)

The following code in [data_agg_clean.py](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/src/data_agg_clean.py) aggregates a team's stats (mean and variance) up to the game in question and merge that data with my games result database (games_test):

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

The win-loss record and average point spread up to that point in the season were attached through a similar process.

## Feature Engineering
Advanced stats such as [Turnover Percent, Free Throw Factor and Effective Field Goal Percentage](https://www.basketball-reference.com/about/glossary.html) were added using [feature_eng.py](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/src/feature_eng.py).

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

Unfortunately, because of the method of aggregation and lack of data from the box scores, most defensive statistics were not calculable.

#### Dataframes and Train-Test Split
Prior to train-test-split I created three dataframes, each one with its own approach to the data.

- [DF1](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/data/reg_model_data_final1.pkl) contains multiple fields for home teams and those same fields for away teams. It has the most fields and therefore, likely the highest risk of high variance in initial regression results.

- [DF2](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/data/reg_model_data_final_homeaway.pkl) holds mostly fields representing differential in these stats between home and away teams, foregoing absolute magnitude in favor of relative magnitude.

- [DF3](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/data/reg_model_data_final3.pkl) is like DF1 in that it contains metrics for both the home and the away team, but I have standardized applicable statistics by the possessions metric, so that most fields represent statistics per possession instead of per game. I also added a squared terms for home and away point spread.

The train-test split was between regular season and tournament games. This decision was to ensure that the analysis was truthful to a situation in which one did not know what the results of the tournament would be. I will discuss this decision further in my results section.

## Exploratory Data Analysis
All EDA was completed using [data_viz.py](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/src/data_viz.py)

Most data is fairly normally distributed across rows (a row being a team up until a matchup within that season). One item of note is that home team point spread and winning percentage are skewed to the left, while away team winning percentage and point spread are skewed to the right.
![Histogram plot](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/images/features_labels_histplot.png)
The sample mean for the target I was predicting on (point spread of the actual game) was __3.66__, meaning the home team on average wins by 3.66 points (not surprising for the regular season, but it is an issue for prediction in the post-season).

The following scatter matrix of the features on each other (using DF2 home-away aggregated features for sake of space) demonstrates collinear relationships between some of the features, including between effective field goal percentage and points per game, pace and field goal attempts, points per game and assists per game. ![Scatter Matrix](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/images/features_scatter_matrix.png)
Some closely related fields, including field goal percent were dropped due to their collinear relationship with another variable (effective field goal percent in this case).

Finally, scatter plots of features on the label demonstrated that a few of our features had promising relationships with point spread.

![EFG and Point Spread](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/images/diff_efgpct_and_home_point_spread.png)

However, others seemed to have little to no relationship to point spread.

![3-points attempted and Point Spread](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/images/diff_3papg_and_home_point_spread.png)

More can be found in the [images](https://github.com/timmarlowe/smarter-than-nate-silver/tree/master/images) folder. Further feature engineering is definitely recommended to make this product viable.

## Modeling
I've used three different linear models to estimate point spread on each of the databases (OLS, and two regularization models: Lasso and Ridge). Code for modeling can be found in [linear.py](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/src/linear.py). Each model was created using 10-fold cross-validation of the training data.

For OLS, I used backwards selection to remove coefficients over time. However, likely due to a high bias, low variance model to begin with, the removal of features that were not statistically significant created very little change in RMSE.

Similarly, Lasso models for all three dataframes settled on small coefficients in the range provided, likely because the model is underfit.
![Lasso Model DF1](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/images/Lasso_RMSE_vA.png)

Ridge faired no better:
![Ridge Model DF1]()

In fact, all three models hovered around the exact same RMSE, no matter the dataframe or adjustment to the exogenous variables:
![RMSE by Model](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/images/Ridge_RMSE_vA.png)

Because regularization and addition of terms were both ineffective at increasing the explanatory power of the model, I chose an OLS model with fewer terms based on the per possession database (DF3). The coefficients for the model are as follows:

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

It is still clearly both over-fit (see effective field goal percentage) and under-fit.

## Results
At a score threshold of 0 point spread between home and away, this model has a 69.3% cross validated accuracy. As can be seen in the table below, the model is mostly predicting home teams will win, and extremely underpredicting on away teams.

| Threshold   |   Precision |   Recall |   Accuracy |   Specificity |   False Positive Rate |
|---:|------------:|---------:|-----------:|--------------:|----------------------:|
| -5 |       0.703 |    0.998 |      0.702 |         0.005 |                 0.995 |
| -4 |       0.705 |    0.998 |      0.705 |         0.013 |                 0.987 |
| -3 |       0.706 |    0.998 |      0.707 |         0.02  |                 0.98  |
| -2 |       0.707 |    0.985 |      0.702 |         0.036 |                 0.964 |
| -1 |       0.709 |    0.973 |      0.701 |         0.058 |                 0.942 |
|  0 |       0.712 |    0.945 |      0.693 |         0.096 |                 0.904 |
|  1 |       0.718 |    0.901 |      0.682 |         0.165 |                 0.835 |
|  2 |       0.729 |    0.825 |      0.662 |         0.277 |                 0.723 |
|  3 |       0.738 |    0.739 |      0.632 |         0.381 |                 0.619 |
|  4 |       0.74  |    0.637 |      0.588 |         0.472 |                 0.528 |

This is likely due to the fact that Away teams in March Madness are different than Away teams in the regular season, in that no one is playing on their home floor.

The ROC curve demonstrates that the model predicts above random chance at all thresholds, but could certainly be improved upon
![ROC curve](https://github.com/timmarlowe/smarter-than-nate-silver/blob/master/images/ROC%20of%20Final%20NCAA%20Game%20Prediction%20Using%20Pointspread.png)

The final RMSE of the model on the final test data was __11.53__ points - an extremely high spread.

## Conclusions and Future Work
While 70% accuracy may seem high, simply picking the higher seeded team in each game in the 2018 tournament would have yielded an accuracy of __()__%.

The clear conclusion for this model is that it did not perform well on the desired outcome. Prior to moving forward with other suggested next steps, one should consider completely revamping the dataset. It seems clear now that these aggregate season factors such as strength of schedule, conference strength, and RPI are extremely important. Not all wins (not even all wins with high Effective Field Goal Percentage) are made equal. It would be wise to return to those metrics when further pursuing tournament modeling.

However, next steps for any tournament model of this sort would be to:
- Use bootstrapping to create confidence intervals for the point spread and then assign probabilities to each of the teams winning given those intervals and the predicted point spread.
- Logistically model on the binary outcome of winning a game.
- Create a Markov chain framework that would enable you to predict the probability of any team in the field reaching any point in the tournament.
- If sticking with this dataset, follow the lead of Steve Iannaconne and use non-parametric methods, especially when gauging the effect of how individual players interact with each other on a team.

## Citations and Thanks
- Many thanks to Steve Iannaccone for initiating this repo and having this idea before I did. Hopefully he was able to execute it slightly better using neural networks.
- Thanks to the galvanize staff and solutions repo, for access to many snippets of code that I adapted here for my own purposes.
- All data was accessed from http://sportsdata.wfmz.com

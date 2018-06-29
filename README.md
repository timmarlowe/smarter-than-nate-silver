# (Not Necessarily) Smarter Than Nate Silver
### Predicting the NCAA Tournament Using Team Boxscore Data
##### Tim Marlowe -- Date of Last Update: 6/28/2018

Data and original idea thanks to Steve Iannaccone, whose project "Smarter Than Nate Silver" was the original fork for this repo

## Problem Statement:
Building models to predict the NCAA tournament is often done by using advanced metrics of team success over the course of the season. These include metrics like
[Elo rating](https://en.wikipedia.org/wiki/Elo_rating_system), [BPI and Strength of Record](http://www.espn.com/blog/statsinfo/post/_/id/125994/bpi-and-strength-of-record-what-are-they-and-how-are-they-derived).

My question was whether I could build a predictor off of a data set that used none of the aggregated metrics to predict point spread and thus outcomes of final four games. Given that there are over 3000 games a season, if a true signal could be found just in the box scores of individual matchups and the aggregation of team stats for each matchup up to that point in the season, it may be a more powerful and specific predictor than a team's advanced stats at the end of the season going into March Madness.



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

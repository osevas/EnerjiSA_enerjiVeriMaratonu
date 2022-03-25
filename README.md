# EnerjiSA Enerji Veri Maratonu
EnerjiSA Kaggle competition
Date: Feb 15, 2022
Link: https://www.kaggle.com/competitions/enerjisa-enerji-veri-maratonu/overview

EnerjiSA is an energy company in Turkey.  They operate two main business lines; power distribution and retail, with operational excellence.  They reach 10.1 million customers in 14 provinces to provide distribution services to over 21 million users.

In this Kaggle competition, EnerjiSA shares hourly data on their solar panels.  The data are from dates between Jan 1 2019 and Nov 30 2021.  The objective is to predict power generation on the dates between Dec 1 2021 and Dec 31 2021 for each hour.

I missed the deadline due to my full time job, but later did late submissions.  I approached the problem with time-series analyses and received approximately 45 as RMSE.  The first prize winner has shared his code.  It is seen that he approached the problem with regression models and received much better results.

## Lessons learned
Regression is more suitable for this problem.  Time-series have to forecast 744 units into the future.  It is probable that this timeframe is too long for time-series to perform good predictions.

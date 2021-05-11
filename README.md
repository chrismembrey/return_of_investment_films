<h1> Predicting Film Return On Investment using TMDB and IMDB </h1>

<img src = "/readme/movie_collage" width="700">

**Look at the full presentation here:**
https://docs.google.com/presentation/d/143nyULpnLyzoOyFYceJvd_ZPJrZUcaTPaFIGM6gTPUE/edit#slide=id.p


## Table of Contents
- [Introduction](#introduction)
- [Data Collection & Pre-Processing](#data-collection---pre-processing)
- [Regression Modelling](#regression-modelling)
- [Classification Modelling](#classification-modelling)
- [Future Work](#future-work)
- [Notebook Order](#notebook-order)


## Introduction
According to Statista, the film industry generates roughly $10Bn a year in revenue. If its possible to find variables that can predict whether films will have high return of investment (ROI) or profit, a model can be created to help production companies come to a decision when giving a movie the green light or not.

TMDB & IMDB are the most commonly used open-source websites where users can input infomation on films they have seen, the latter having infomation on ~7 Million titles.


## Data Collection & Pre-Processing
TMDB was primarily used to collect the data since it had fast API calls using the tmdbsimple package. Conversly, although IMDB has more trustworthy infomation (amazon own IMDB and so the infomation given is consistenly monitored), the API calls were incredibly slow so only a few varibales could be collected using this API. Movies were only collected in a dataframe if they had both budgets and revenues greater than $0.


The cleaning of these movies was acheived using the following steps:
- Replacing null values with medians/means where appropriate 
- Removal of films with budgets and revenues less than $10,000
- Dummification (categories)
- Tokenisation (blurbs)
- Count Vectorization (blurbs)
- Data Binning
- Standardisation (Continuous Features)
- Hypothesis testing was used to determine if ROI means were significantly different between categories of variables, indicating whether the variable had predictive power for ROI

NB - Initially, not all of the outliers could be removed since I needed a dataset as close to 10,000 observations as possible.

After cleaning, around 9,500 datapoints were left and the following columns were used for ROI/profit prediction:

- Directors (binary)
- Composers (binary)
- Runtime (continuous)
- Genre (binary)
- Day of week (binary)
- Month released (binary)
- Inflation budget (continuous)
- Original language bins (binary)
- Blurb length (continuous)
- From collection (binary)
- Blurb objectivity (continuous)
- Blurb polarity (coninuous)
- Unique genre pairing (binary)
- Blurb words with top 100 roi means (binary)
- Actors with more than 3 films to thier name (binary)


## Regression Modelling
Linear Regression, a Support Vector Machine (linear kernel only) and a Decision Tree Regression was used to predict the ROI of films. The best model was an ElasticNet linear regression model. 

<img src = "/Final_Movies/readmeregression_models_original.png" width="700">

As outliers were removed the training and test scores were ameliorated, giving increased confidence in the effect/importance of the coefficients (variables).

<img src = "/Final_Movies/readme/ElasticNet_Models_with_Various_Preprocessing_Methods.png" width="700">

The root mean squared error was also used as a metric to compare the models' performance.

<img src = "/Final_Movies/readme/ElasticNet_Models_with_Various_Preprocessing_Methods_rmse.png" width="700">


## Classification Modelling
ElasticNet Logistic Regression, Decision Tree Classification, XGBoost (ensemble method) and a Support Vector Machine were used to predict whether films would make a profit or not. Each film performed better than randomly guessing if the film would make a profit (above the baseline accuracy) with the best being an ElasticNet logistic regression model (l1_ratio of 0.21 , C of 51.79). The ElasticNet model had a cross validated accuacy of ~0.66 and had a similar score to the Support Vetor Machine.


<img src = "/Final_Movies/readme/regression_models_original.png" width="700">


A multi-class classification model was also carried out. The ROI variable was split into quintiles (baseline accuracy of 0.20) and an ElastiNet Logistic Regression produced an accuracy of 0.33.  

## Future Work
- Use an alternative data source for film budgets and revenues (e.g. the-numbers.com). 

- Instead of looking at the impact of all actors/directors, creating a variable that confirms if the actors/directors contained in the film are popular at that current point in time will be more beneficial for a productionisable model.

- Additional feature - from_bestselling_book? - If the story was based off a bestselling book, would that increase the chances of higher ROI?

- Using statsmodels to be able to see the significance of the features in the model.


## Notebook Order
1. move_scraping_final.ipynb  
2. cleaning & EDA.ipynb  / BLURB SPACY ANALYSIS.ipynb
3. modelling to see feature importance using regression techniques.ipynb
4. classifying_profit_for_comparison_against_the_regression_models.ipynb

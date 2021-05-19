<h2> Predicting Film Return On Investment using TMDB and IMDB </h2>

<img src = "/readme/movie_collage.png" width="700">

**Look at the full presentation here:**
https://docs.google.com/presentation/d/143nyULpnLyzoOyFYceJvd_ZPJrZUcaTPaFIGM6gTPUE/edit#slide=id.p


## Table of Contents
- [Introduction](#introduction)
- [Data Collection & Pre-Processing](#data-collection---pre-processing)
- [Regression Modelling](#regression-modelling)
- [Classification Modelling](#classification-modelling)
- [Conclusion](#conclusion)
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

Here is a small taste of the data frame appearance.

![Screenshot 2021-05-13 at 17 29 34](https://user-images.githubusercontent.com/76961031/118156130-e9cdff00-b410-11eb-9ee9-ffd0f6a0b44c.png)


## Regression Modelling
Linear Regression, a Support Vector Machine (linear kernel only) and a Decision Tree Regression was used to predict the ROI of films. The best model was an ElasticNet linear regression model for data where ROI was less than 500% and the model only yeilded a score of 0.155 showing limited explainability in ROI variance.

<img src = "/readme/regression_models_original.png" width="700">

As outliers were removed the training and test scores were ameliorated, giving increased confidence in the effect/importance of the coefficients (variables).

<img src = "/readme/ElasticNet_Models_with_Various_Preprocessing_Methods.png" width="700">

The different feature coefficients can be observed below.

![ElasticNet_Model_Regression_Coefficients](https://user-images.githubusercontent.com/76961031/118154259-bc805180-b40e-11eb-92ea-59a2bcceed14.jpeg)


The root mean squared error was also used as a metric to compare the models' performance.

<img src = "/readme/ElasticNet_Models_with_Various_Preprocessing_Methods_rmse.jpeg" width="700">


## Classification Modelling
ElasticNet Logistic Regression, Decision Tree Classification, XGBoost (ensemble method) and a Support Vector Machine were used to predict whether films would make a profit or not. Each film performed better than randomly guessing if the film would make a profit (above the baseline accuracy) with the best being an ElasticNet logistic regression model (l1_ratio of 0.21 , C of 51.79). The ElasticNet model had a cross validated accuacy of ~0.66 and had a similar score to the Support Vetor Machine.


<img src = "/readme/Classification_Models_Train_scores.png" width="700">

Here are the feature importances of the best model, the ElasticNet Logistic Regression.

![ElasticNet_logistic_regression_model_coefficients](https://user-images.githubusercontent.com/76961031/118154095-9064d080-b40e-11eb-83ac-5eacba8bb909.png)


A multi-class classification model was also carried out. The ROI variable was split into quintiles (baseline accuracy of 0.20) and an ElastiNet Logistic Regression produced an accuracy of 0.33.

Below you can see the effect of the top variables across the five classes.

![Screenshot 2021-05-13 at 17 25 22](https://user-images.githubusercontent.com/76961031/118155596-4b419e00-b410-11eb-898b-13b8a4f9a8b6.png)



## Conclusion

Although the models are not productionable (based on low r-squared scores and minimal improvement on baseline accuracy), through them I have found one feature which can certainly be used to predict film ROI in future models, from_collection, which appeared to be the highest coefficient in every model. 


## Future Work
- Use an alternative data source for film budgets and revenues (e.g. the-numbers.com). 

- Instead of looking at the impact of all actors/directors, creating a variable that confirms if the actors/directors contained in the film are popular at that current point in time will be more beneficial for a productionisable model.

- Additional feature - from_bestselling_book? - If the story was based off a bestselling book, would that increase the chances of higher ROI?

- Using statsmodels to be able to see the significance of the features in the model.


## Notebook Order
1. scraper.ipynb  
2. cleaning & EDA.ipynb  / BLURB SPACY ANALYSIS.ipynb
3. modelling to see feature importance using regression techniques.ipynb
4. classifying_profit_for_comparison_against_the_regression_models.ipynb

<h2> Predicting Film Return On Investment using TMDB and IMDB </h2>

<img src = "/readme/movie_collage.png" width="700">

**Look at the full presentation here:**
https://docs.google.com/presentation/d/143nyULpnLyzoOyFYceJvd_ZPJrZUcaTPaFIGM6gTPUE/edit?usp=sharing


## Table of Contents
- [Introduction](#introduction)
- [Data Collection & Pre-Processing](#data-collection---pre-processing)
- [Regression Modelling](#regression-modelling)
- [Classification Modelling](#classification-modelling)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Notebook Order](#notebook-order)


## Introduction
According to Statista, the film industry generates roughly $10Bn a year in revenue. If it's possible to find variables that can predict whether films will have high return of investment (ROI) or profit, a model can be created to help production companies come to a decision when giving a movie the green light or not.

TMDB & IMDB are the most commonly used open-source websites where users can input information on films they have seen, the latter having information on ~7 Million titles.

NB - Before you read any further I recommend turning your github screen display to white, if it isn't so already, so that the visualisations are easier to interpret.

## Data Collection & Pre-Processing
TMDB was primarily used to collect the data since it had fast API calls using the tmdbsimple package. Conversely, although IMDB has more trustworthy information (Amazon owns IMDB and so the information given is constantly monitored), the API calls were incredibly slow so only a few variables could be collected using this API. Movies were only collected in a dataframe if they had both budgets and revenues greater than $0.


The cleaning of these movies was achieved using the following steps:
- Replacing null values with medians/means where appropriate 
- Removal of films with budgets and revenues less than $10,000
- Dummification (categories)
- Tokenization (blurbs)
- Count Vectorization (blurbs)
- Data Binning
- Standardisation (continuous features)
- Hypothesis testing was used to determine if ROI means were significantly different between categories of variables, indicating whether the variable had predictive power for ROI

NB - Initially, not all of the outliers could be removed since I needed a dataset as close to 10,000 observations as possible.

After cleaning, around 9,500 data points were left and the following columns were used for ROI/profit prediction:

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
- Blurb polarity (continuous)
- Unique genre pairing (binary)
- Blurb words with top 100 roi means (binary)
- Actors with more than 3 films to their name (binary)

Here is a small taste of the data frame appearance.

![Screenshot 2021-05-13 at 17 29 34](https://user-images.githubusercontent.com/76961031/118156130-e9cdff00-b410-11eb-9ee9-ffd0f6a0b44c.png)

To see the full cleaning and EDA of the scraped data, please see the jupyter notebook in the folder titled 'cleaning_eda'.


## Regression Modelling
Linear Regression, a Support Vector Machine (linear kernel only) and a Decision Tree Regression was used to predict the ROI of films. The best model was an ElasticNet linear regression model for data where ROI was less than 500% and the model only yielded a score of 0.155 showing limited explainability in ROI variance.

![image](https://user-images.githubusercontent.com/76961031/118816969-a52be300-b8aa-11eb-8d0f-ccaa7b3988ce.png)

As outliers were removed the training and test scores were ameliorated, giving increased confidence in the effect/importance of the coefficients (variables).

![image](https://user-images.githubusercontent.com/76961031/118816713-5c742a00-b8aa-11eb-874f-8d66b89f07d4.png)

The different feature coefficients can be observed below.

![image](https://user-images.githubusercontent.com/76961031/118817113-c8569280-b8aa-11eb-8e89-543023ee53e2.png)


The root mean squared error was also used as a metric to compare the models' performance.

![image](https://user-images.githubusercontent.com/76961031/118816751-672ebf00-b8aa-11eb-8f42-0dcd08a3dac3.png)

To see the full method behind the regression analysis, please see the notebook titled 'modelling to see feature importance using regression techniques' in the 'modelling' folder.

## Classification Modelling
ElasticNet Logistic Regression, Decision Tree Classification, XGBoost (ensemble method) and a Support Vector Machine were used to predict whether films would make a profit or not. Each film performed better than randomly guessing if the film would make a profit (above the baseline accuracy) with the best being an ElasticNet logistic regression model (l1_ratio of 0.21 , C of 51.79). The ElasticNet model had a cross validated accuracy of ~0.66 and had a similar score to the Support Vector Machine.


![image](https://user-images.githubusercontent.com/76961031/118816000-ac062600-b8a9-11eb-8272-206a66bfd4f6.png)

Here are the feature importances of the best model, the ElasticNet Logistic Regression.

![image](https://user-images.githubusercontent.com/76961031/118817206-e2907080-b8aa-11eb-8718-96d3c56cfa90.png)


A multi-class classification model was also carried out. The ROI variable was split into quintiles (baseline accuracy of 0.20) and an ElastiNet Logistic Regression produced an accuracy of 0.33.

Below you can see the effect of the top variables across the five classes.

![image](https://user-images.githubusercontent.com/76961031/118819332-284e3880-b8ad-11eb-9897-204ae5ae7bdc.png)

To see the complete classification analysis, including the classification matrix, ROC curve and precision/recall curve, please see the notebook titled 'classifying_profit_for_comparison_against_the_regression_models' in the modelling folder.


## Conclusion

Although the models are not productionable (based on low r-squared scores and minimal improvement on baseline accuracy), through them I have found one feature which can certainly be used to predict film ROI in future models, from_collection, which appeared to be the highest coefficient in every regression and classification model. For the classification model in particular, the coefficient for the from_collection variable indicated that the odds of making a profit for films that are from a collection are 464% higher than those that are not from a collection.
Other notable features that had a positive effect on ROI were actors Tom Cruise, Tom Hanks and Brad Pitt as well as day_wednesday, which suggests that films released on a wednesday will have improved ROI. 

## Improvments to be Made
- Use an alternative data source for film budgets and revenues (e.g. the-numbers.com). 

- Instead of looking at the impact of all actors/directors, creating a variable that confirms if the actors/directors contained in the film are popular at that current point in time will be more beneficial for a productionisable model.

- Additional feature - from_bestselling_book? - If the story was based off a bestselling book, would that increase the chances of higher ROI?

- Using statsmodels to be able to see the significance of the features in the model.

- Instead of studying the blurbs of films, sentiment analysis may be more beneficial for an entire film plot. Work could also be done to map the plot sentiment with the expected time within the film. By doing this we could observe the sentiment pattern of films with high ROI to see if a new film matches a recognisable pattern here.


## Notebook Order
1. scraper.ipynb  
2. cleaning & EDA.ipynb  / BLURB SPACY ANALYSIS.ipynb
3. modelling to see feature importance using regression techniques.ipynb
4. classifying_profit_for_comparison_against_the_regression_models.ipynb


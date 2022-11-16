# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 2: Ames Housing Price Prediction

## Executive Summary

For roughly 98% of the last 2,500 years of Western intellectual history, science was a part of philosophy. It was then called natural philosophy, but science deviated from philosophy in the 17th century and emerged as a separate study or domain. As moderators of Science & Philosophy subreddits with substantial number of members, 28.4 million & 16.9 million respectively, our mission are to:

1. Develop a classification model that predicts which category a post belongs to. This will be a great help for us in making sure that topics are posted in the correct subreddit, as well as improving users experience when reading the posts.
2. Conduct sentiment analysis to evaluate user's posts. As Science & Philosophy are both factual based subreddits, a neutral and unopinionated posts are to be expected.
3. Identify trending topics for each subreddits so that we can pin it on top of our landing page.

To achieve the mission as set in the above statement, 25000 posts each are scraped from Science Subreddit & Philosophy Subreddit. Next, data cleaning & EDA are done to examine any missing data, outliers, etc as well as to identify trending topics for each subreddits. The trending topic in Science is Covid19, whereas the trending topic in Philosophy is Meaning of Life. Next, stop words are removed from the texts, and tokenization of texts are executed using Stemming. Baseline model are then developed using Multinomial Naive Bayes with hyperparameter tuning of CountVectorizer and TFIDF Vectorizer. TF-IDF are selected as the Vectorizer and will be used to develop the Final Model. Four types of model with hyperparameter tuning are developed: Multinomial Naive Bayes, Logistic Regression, Random Forest, and SVM. Logistic Regression is selected as the final model as it the model with highest accuracy model that is not underfitting / overfitting.

Next, Sentiment analysis of the overall Science & Philosophy are developed using HuggingFace analysis to highlight posts are delivered in non-neutral tone. Additional sentiment analysis will be developed using HuggingFace for the trending topics from each Science (Covid 19) & Philosophy (Meaning of Life). Over 60% of the posts are flagged as neutral in both Science & Philosophy subreddits.

## Background

Is science a part of philosophy or are they two totally different subjects? Although in current days many people assume that science and philosophy are concepts contradictory to each other, but both subjects share a more positive relationship rather than an animosity. In fact, for roughly 98% of the last 2,500 years of Western intellectual history, science was a part of philosophy. It was then called natural philosophy, but science deviated from philosophy in the 17th century and emerged as a separate study or domain ([*source*](https://archive.nytimes.com/opinionator.blogs.nytimes.com/2012/04/05/philosophy-is-not-a-science/)).

The definition of science and philosophy are as follows ([*source*](https://1000wordphilosophy.com/2018/02/13/philosophy-and-its-contrast-with-science/#:~:text=Science%20is%20about%20descriptive%20facts,objects%20(if%20they%20exist))):
- Science is about empirical knowledge; philosophy is often about that but is also about a priori knowledge (if it exists).
- Science is about contingent facts or truths; philosophy is often about that but is also about necessary truths (if they exist).
- Science is about descriptive facts; philosophy is often about that but is also about normative and evaluative truths (if such truths exist).
- Science is about physical objects; philosophy is often about that but is also about abstract objects (if they exist).

## Problem Statement

As moderators of Science & Philosophy subreddits with substantial number of members, 28.4 million & 16.9 million respectively, our mission are to:

1. Develop a classification model that predicts which category a post belongs to. This will be a great help for us in making sure that topics are posted in the correct subreddit, as well as improving users experience when reading the posts.
2. Conduct sentiment analysis to evaluate user's posts. As Science & Philosophy are both factual based subreddits, a neutral and unopinionated posts are to be expected.
3. Identify trending topics for each subreddits so that we can pin it on top of our landing page.

The procedure are as follows:
1. Scrape 25000 posts each from Science Subreddit & Philosophy Subreddit.
2. Data cleaning & EDA to examine any missing data, outliers, etc as well as to identify trending topics for each subreddits.
3. Tokenization of texts using both Stemming & Lemmatizing to obtain the best model. The model that are able to converge the most words will be selected for subsequent analysis.
4. Develop a baseline model using Multinomial Naive Bayes with hyperparameter tuning of CountVectorizer and TFIDF Vectorizer. Vectorizer that results in better performance in the baseline model will be selected for Final Model.
5. Develop the final model using: Multinomial Naive Bayes, Logistic Regression, Random Forest, and SVM with hyperparameter tuning on both the vectorizer and the model. Model that is not substantially overfit/underfit with highest accuracy will be selected as the final model.
6. Develop a Sentiment analysis of the overall Science & Philosophy using Vader. Additional sentiment analysis will be developed using HuggingFace for the trending topics from each Science & Philosophy subreddits identified in no 2.

## Datasets

The data is taken from the following subreddits:
1. Science ([*source*](https://www.reddit.com/r/science/))
2. Philosophy ([*source*](https://www.reddit.com/r/philosophy/))

[*Pushshift API*](https://github.com/pushshift/api) are used to scrape 25000 posts of each subreddits, starting from 4th October 4 2022 0:00:00 SGT backwards.

## EDA, Data Cleaning & Pre-Processing
Data cleaning & EDA are done to examine any missing data, outliers, etc as well as to identify trending topics for each subreddits. Stop words such as 'a', 'the', 'of', as well as special characters shuch as '?', '!', '@' are removed from the texts. Tokenization of texts are executed using Multinomial Naive Bayes with hyperparameter tuning on both Stemming & Lemmatization. Stemming was chosen as it is able to condense more words than lemmatization.

Common words analysis were conducted, The top single words in Science are: studi, new, research, find, covid, and the top single words in Philosophy are philosophi, philosoph, life, human, moral. The top double words in science are: covid 19, whereas the top double words for philosophy are: mean life.

## Baseline Model
Baseline model are then developed using Multinomial Naive Bayes with hyperparameter tuning of CountVectorizer and TFIDF Vectorizer. The results are shown in the table below:

| Model No | Model | Vectorizer | Accuracy | Train Score | Test Score | % Difference |
|---|---|---|---|---|---|---|
| Model 1 | Multinomial Naive Bayes | Count Vectorizer | 0.915 | 0.936 | 0.913 | 2.3% |
| Model 2 | Multinomial Naive Bayes | TFIDF Vectorizer | 0.912 | 0.930 | 0.912 | 1.8% |

As shown above, Count Vectorizer & TF-IDF produce relatively same results: ~0.91 Accuracy, ~0.93 Train Score and ~0.91 Test Score. CountVectorizer and TFIDF performs relatively the same because there are little overlaps of the same words in Science & Philosophy Reddit. However as TF-IDF are generally more powerful, it will be selected as the vectorizer in the subsequent analysis.


## Final Model
Four types of model with hyperparameter tuning are developed: Multinomial Naive Bayes, Logistic Regression, Random Forest, and SVM. 

| No | Vectorizer | Models | Accuracy | Train Score | Test Score | % Difference | Remarks |
|---|---|---|---|---|---|---|---|
| Model 1 | TF-IDF | Multinomial Naive Beyes | 0.917 | 0.942 | 0.915 | 2.7% | - |
| Model 2 | TF-IDF | Logistic Regression | 0.926 | 0.954 | 0.921 | 3.3% | Selected Model | 
| Model 3 | TF-IDF | Random Forest | 0.912 | 0.947 | 0.909 | 3.8% | - |
| Model 4 | TF-IDF | SVC | 0.931 | 0.997 | 0.929 | 6.8% | - |

The summary of hyperparameter tuning model are summarized in table above. SVC has the highest accuracy of 0.931, followed by Logistic Regression with the accuracy of 0.926, Multinomial Naive Bayes with the accuracy of 0.921, and Random Forest with the accuracy of 0.912. Except for SVC, all the models has less than 5% difference between train & test score, and hence are not considered as overfitting.

Comparing SVC and Linear Regression: SVC accuracy is higer by 0.005 than Logistic Regression, but Logistic Regression model is not overfitting and has a better interpretability as compared to SVC. With the consideration mentioned, Logistic Regression is selected as the final model.

## Sentiment Analysis
Next, Sentiment analysis of the overall Science & Philosophy are developed using HuggingFace analysis to highlight posts are delivered in non-neutral tone. Additional sentiment analysis will be developed using HuggingFace for the trending topics from each Science (Covid 19) & Philosophy (Meaning of Life). Over 60% of the posts are flagged as neutral in both Science & Philosophy subreddits.

## Summary
To achieve the mission as set in the above statement, 25000 posts each are scraped from Science Subreddit & Philosophy Subreddit. Next, data cleaning & EDA are done to examine any missing data, outliers, etc as well as to identify trending topics for each subreddits. The trending topic in Science is Covid19, whereas the trending topic in Philosophy is Meaning of Life. Next, stop words are removed from the texts, and tokenization of texts are executed using Stemming. Baseline model are then developed using Multinomial Naive Bayes with hyperparameter tuning of CountVectorizer and TFIDF Vectorizer. TF-IDF are selected as the Vectorizer and will be used to develop the Final Model. Four types of model with hyperparameter tuning are developed: Multinomial Naive Bayes, Logistic Regression, Random Forest, and SVM. Logistic Regression is selected as the final model as it the model with highest accuracy model that is not underfitting / overfitting.

Next, Sentiment analysis of the overall Science & Philosophy are developed using HuggingFace analysis to highlight posts are delivered in non-neutral tone. Additional sentiment analysis will be developed using HuggingFace for the trending topics from each Science (Covid 19) & Philosophy (Meaning of Life). Over 60% of the posts are flagged as neutral in both Science & Philosophy subreddits.

By implementing all the steps mentioned above, if users post in the appropriate subreddits, deliver their posts in a neutral and objective tone, it will greatly reduce the task of Science & Philosophy moderators, as well as improving the overall user experience of reddit readers.

## Recommendation & Limitation
- Classification Models: Although the final model has more than 90% accuracy, it is only developed using 4 types of classification models. Running Pycaret would be ideal to ensure that we have exhausted all classification models
- Sentiment Analysis: We should also manually classify the posts sentiments in order to gauge the accuracy of the models. As shown in earlier segment, there are posts that should be classified as neutral, but are classified as negative. Due to time limitation, only 1 model are developed to analyse the sentiments of the posts. Several huggingface models should be explored and model that yields the best results should be used as the final sentiment analysis model.
- Futher studies: To do a multiclass classification on both Science & Philosophy category based on the topics, as well as expanding to other fact-based subreddits such as: r/Economics, r/Astronomy, etc.
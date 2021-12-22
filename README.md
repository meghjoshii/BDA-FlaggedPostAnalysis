# BDA-FlaggedPostAnalysis


 [![Build Status](https://travis-ci.org/github/BDA-FlaggedPostAnalysis.svg?branch=master)](https://travis-ci.org/github/BDA-FlaggedPostAnalysis)
[![Language](https://img.shields.io/badge/Python-14354C?style=plastic&colorB=68B7EB)]()


[![License](https://img.shields.io/github/license/vhesener/Closures.svg?style=plastic&colorB=68B7EB)]()
[![Release](https://img.shields.io/github/release/vhesener/Closures.svg?style=plastic&colorB=68B7EB)]()

[![Language](https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)]()


## Big Data Analytics Project
## Flagged Post Analysis - Group 42

### Team Members:
1. Meghana Joshi (mmj2169)
2. Mohammed Aqid Khatkhatay (mk4427)
3. Shantanu Jain (slj2142)

### Introduction
The goal of this project is to analyze and identify the features of poor quality posts, so as to build an efficient and highly reliable machine learning model to classify posts on the basis of its quality.

Our project addresses this goal by utilizing techniques of data visualization to observe underlying patterns, feature selection and extraction to strengthen the classification model by feeding it with reliable features. The features that will be given as input to the model are a combination of user-based features, community-based features and textual features extracted from the body of the posts. 

### Repository Structure
- /dataset
    - posts_dataset.csv
- /src
    - /cleaning_feature_extraction
        - data_cleaning_feature_extraction.ipynb
    - /logistic_regression
        - logistic_regression_model.ipynb
    - /lstm_semi_supervised
        - TODO
- /report
    - TODO
- /sql_queries
    - TODO
- README.md


### Instructions on running the flask server
~~~python

!pip install requirements.txt
python app.py


~~~

### Architecture Diagram
![architecture.png](images/architecture.png)

### Dataset
- Dataset obtained from StackExchange Data Dump containing 96M posts. 
- Uploaded the same on big query and used SQL queries for merging Posts, Users, Answers and Views together.

TODO - SQL Query Images

### Semi Supervised - LSTM Encoder Decoder
TODO

### Logistic Regression
Logistic Regression is mainly used for predicting binary labels based on logits (log of odds). We leveraged the pyspark logistic regression library for creating transformers and estimator for fitting our dataset. 

Applied a grid-based hyperparameter tuner to fetch the following parameters:

- Train Test split: 90:10
- Train Validation split: 80:20
- Training samples: 25849, Test samples: 2912
- Weight balancing ratio: 0.77
- regParam: 0.01
- elasticNetParam: 0.0
- fitIntercept: True

### Results

| Evaluation Metric      | Score |
| ----------- | ----------- |
| Accuracy      | 73%       |
| Area under ROC   | 0.71        |
| Area under PR   | 0.89       |
| F1-Score   | 0.73       |
| Cohen-Kappa Metric   | 0.26        |
| Mathews Correlation Coefficient   | 0.25  |


### Conclusion
TODO

### References
TODO

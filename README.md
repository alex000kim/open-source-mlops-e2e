# E2E ML project using open-source MLOps tools

## Problem Description and Dataset
This dataset contains 10,000 records, each of which corresponds to a different bank's user. The target is `Exited`, a binary variable that describes whether the user decided to leave the bank. There are row and customer identifiers, four columns describing personal information about the user (surname, location, gender and age), and some other columns containing information related to the loan (such as credit score, the current balance in the user's account and whether they are an active member among others).

Dataset source: https://www.kaggle.com/datasets/filippoo/deep-learning-az-ann

## Use Case
The objective is to train an ML model that returns the probability of a customer churning. This is a binary classification task, therefore F1-score is a good metric to evaluate the performance of this dataset as it weights recall and precision equally, and a good retrieval algorithm will maximize both precision and recall simultaneously.


## Setup
Python 3.8+ is required to run code from this repo.
```
$ git clone https://github.com/alex000kim/open-source-mlops-e2e
$ cd open-source-mlops-e2e
$ virtualenv -p python3 .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

## What's inside
```bash
$ tree 
.
├── Churn_Modelling_France.csv # raw data file
├── Churn_Modelling_Spain.csv # raw data file
├── LICENSE
├── README.md # this file
├── TODO.md # describes next steps
├── TrainChurnModel.ipynb # jupyter notebook with model training code
├── clf-model.joblib # saved model
└── feat_imp.csv # table with feature importance values
```

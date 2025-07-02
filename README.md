# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Customer Churn Prediction helps to identify the customers who will likely leave in a while.
This can help businesses either not to focus on them or offer better offers to save them.
Both of the methods lead to the business growth, therefore it's quite helpful 0_0

## Files and data description
```
Customer-Churn-Prediction
|   README.md — You are reading this file btw :)
│   churn_notebook.ipynb — a notebook with an initial solution
|   churn_library.py — the main file with refactored code
|   test_churn_library.py — a file with tests for every function in churn_library.py
|   requirements_py3.13.txt — requirements to be able to run this code
|
└───data
|   |   bank_data.csv — a csv file with the dataset used for training the model (taken from Udacity)
│
└───models
│   │   logistic_model.pkl — Logistic Regression Model
│   │   rfc_model.pkl — Random Forest Classifier (the best)
│  
│   
└───images
|   └───eda — EDA images
|   |
|   └───results — after-training analysis (such as feature importance etc.)
```

## Running Files
For running the main code use ipython `churn_library.py`,
after that you should see new files in images and models

To run the unit tests use `pytest`, you will see a report in your command line (5 tests).




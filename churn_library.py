'''
Author: Kristian Komarov
Date Created: 02.07.2025

This module creates models to predict customer churn.
It analyzes the data, performs EDA, trains models and analyzes the performance and the feature importance.
'''

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from joblib import dump

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(path: str) -> pd.DataFrame:
    '''
    Returns DataFrame for the csv found at path

    input:
        path: a path to the csv

    output:
        df: pandas DataFrame
    '''
    return pd.read_csv(path)


def perform_eda(df: pd.DataFrame) -> None:
    '''
    Perform EDA on df and save figures to images folder

    input:
        df: pandas dataframe

    output:
        None
    '''
    # transform the target variable into int (create a new column Churn)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == 'Existing Customer' else 1)
    df.drop(['Attrition_Flag'], axis=1, inplace=True)

    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig('images/eda/Churn_hist.png')

    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig('images/eda/Customer_Age_hist.png')

    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('images/eda/Martial_Status_hist.png')

    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained
    # using a kernel density estimate
    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('images/eda/Total_Trans_Ct_hist.png')

    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(numeric_only=True), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('images/eda/heatmap.png')


def encoder_helper(
        df: pd.DataFrame,
        category_list: list) -> pd.DataFrame:
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features

    output:
            df: pandas dataframe with new columns for
    '''
    for cat in category_list:
        encoded_col = []
        groups = df.groupby(cat)['Churn'].mean()

        for val in df[cat]:
            encoded_col.append(groups.loc[val])

        df[f'{cat}_Churn'] = encoded_col

    return df


def perform_feature_engineering(df: pd.DataFrame):
    '''
    input:
              df: pandas dataframe

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    cat_col_list = df.select_dtypes(include='object').columns.tolist()
    df = encoder_helper(df, cat_col_list)

    y = df['Churn']

    x = pd.DataFrame()
    keep_col_list = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    x[keep_col_list] = df[keep_col_list]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_pred_lr,
                                y_train_pred_rf,
                                y_test_pred_lr,
                                y_test_pred_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_pred_lr: training predictions from logistic regression
            y_train_pred_rf: training predictions from random forest
            y_test_pred_lr: test predictions from logistic regression
            y_test_pred_rf: test predictions from random forest

    output:
             None
    '''
    plt.figure(figsize=(20, 10))
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train Results'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_pred_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.savefig('images/results/rf_train.png')

    plt.figure(figsize=(20, 10))
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 0.6, str('Random Forest Test Results'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_test, y_test_pred_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.savefig('images/results/rf_test.png')

    plt.figure(figsize=(20, 10))
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 0.6, str('Logistic Regression Train Results'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_pred_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.savefig('images/results/lr_train.png')

    plt.figure(figsize=(20, 10))
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 0.6, str('Logistic Regression Test Results'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_test, y_test_pred_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.savefig('images/results/lr_test.png')


def feature_importance_plot(model, x_data, output_path):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    plt.savefig(output_path)


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['log2', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy'],
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    y_train_pred_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_pred_rf = cv_rfc.best_estimator_.predict(x_test)

    lrc = LogisticRegression(solver='newton-cg', max_iter=3000)

    lrc.fit(x_train, y_train)

    y_train_pred_lr = lrc.predict(x_train)
    y_test_pred_lr = lrc.predict(x_test)

    classification_report_image(
        y_train,
        y_test,
        y_train_pred_lr,
        y_train_pred_rf,
        y_test_pred_lr,
        y_test_pred_rf)

    dump(cv_rfc.best_estimator_, 'models/rfc_model.pkl')
    dump(lrc, 'models/logistic_model.pkl')

    feature_importance_plot(
        cv_rfc,
        x_train,
        'images/results/feature_importance.png')


if __name__ == '__main__':
    dataframe = import_data('data/bank_data.csv')

    perform_eda(dataframe)

    train_data, test_data, train_pred, test_pred = perform_feature_engineering(
        dataframe)

    train_models(train_data, test_data, train_pred, test_pred)

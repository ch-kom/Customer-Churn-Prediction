'''
Unit Tests for churn_library
'''

import logging
import pytest
from churn_library import *


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


@pytest.fixture(scope="module")
def path():
    return 'data/bank_data.csv'


@pytest.fixture(scope="module")
def dataset(path):
    return import_data(path)


def test_perform_eda(dataset):
    '''
    test perform eda function
    '''
    try:
        perform_eda(dataset)
        assert len([file for file in os.listdir(
            'images/eda') if '.png' in file]) == 5
        logging.info("Testing perform_eda: SUCCESS")
    except AttributeError as err:
        logging.error("Testing perform_eda: Input should be a dataframe")
        raise err
    except SyntaxError as err:
        logging.error("Testing perform_eda: Input should be a dataframe")
        raise err
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: The number of images saved doesn't match")
        raise err


def test_encoder_helper(dataset):
    '''
    test encoder helper
    '''
    cat_column_list = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    try:
        initial_df = dataset.copy()
        df = encoder_helper(dataset, cat_column_list)

        assert (len(df.columns) - len(initial_df.columns)) == 5

        logging.info("Testing encoder_helper: SUCCESS")

    except KeyError as err:
        logging.error(
            "Testing encoder_helper: There are column names that doesn't exist in your dataframe")
        raise err

    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The number of columns added by the encoder_helper isn't correct")
        raise err


def test_perform_feature_engineering(dataset):
    '''
    test perform_feature_engineering
    '''
    try:
        output = perform_feature_engineering(dataset)

        assert len(output) == 4

        logging.info("Testing perform_feature_engineering: SUCCESS")

    except KeyError as err:
        logging.error(
            "Testing perform_feature_engineering: Target column names doesn't exist in dataframe")
        raise err

    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: The number of items in the output list isn't correct")
        raise err

    try:
        for item in output:
            assert item.shape[0] > 0

    except AssertionError as err:
        logging.error("Testing import_data: The output seem to be empty")
        raise err


def test_train_models(dataset):
    '''
    test train_models
    '''
    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(dataset)
        train_models(x_train, x_test, y_train, y_test)
        assert len([file for file in os.listdir(
            'models') if '.pkl' in file]) == 2
        logging.info("Testing train_models: SUCCESS")
    except MemoryError as err:
        logging.error(
            "Testing train_models: Out of memory while train the models")
        raise err
    except AssertionError as err:
        logging.error(
            "Testing train_models: The number of models saved doesn't match")
        raise err

    try:
        assert len([file for file in os.listdir(
            'images/eda') if '.png' in file]) == 5
        logging.info("Testing feature_importance_plot: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing feature_importance_plot: The number of images saved doesn't match")
        raise err

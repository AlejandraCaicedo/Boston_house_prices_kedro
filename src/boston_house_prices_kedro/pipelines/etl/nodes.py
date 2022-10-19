"""
This is a boilerplate pipeline 'etl'
generated using Kedro 0.18.3
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, Tuple
import mlflow
from sklearn.model_selection import train_test_split
from deepchecks.tabular.suites import data_integrity
import logging

logger = logging.getLogger(__name__)


def drop_colums(data: pd.DataFrame) -> pd.DataFrame:
    data.drop(['ftyrgv','Unnamed: 0', 'nbgde','ID','index'], axis= 1, inplace = True)
    return data

def replace_nan_symbol(data: pd.DataFrame) -> pd.DataFrame:
    data.replace(np.nan, '?', inplace = True)
    return data

def drop_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    data.drop_duplicates(inplace=True)
    return data

def drop_wrong_values(data: pd.DataFrame) -> pd.DataFrame:
    data.replace(['xxxxx', '-26543765432345.0', 'gdhg7u8whui 784gryb', '88876599956788.0'], np.nan, inplace = True)
    data.dropna(inplace=True)
    return data

def replace_symbol_nan(data: pd.DataFrame) -> pd.DataFrame:
    data.replace('?', np.nan, inplace = True)
    data.reset_index(inplace = True, drop = True)
    return data

def change_data_types(data: pd.DataFrame) -> pd.DataFrame:
    columns_names = data.columns.values
    for colum in columns_names:
        data[colum] = data[colum].astype('float64')
    return data

def drop_nan(data: pd.DataFrame, col:str) -> pd.DataFrame:
    data.drop(data[data[col].notnull() == False].index, inplace=True)
    data.reset_index(inplace = True, drop = True)
    return data

def reset_index(data: pd.DataFrame) -> pd.DataFrame:
    data.reset_index(inplace = True, drop = True)
    return data

def etl_processing(data: pd.DataFrame) -> pd.DataFrame:
    mlflow.set_experiment('house-pricing')
    mlflow.log_param("shape",data.shape)

    data = (data
            .pipe(drop_colums)
            .pipe(replace_nan_symbol)
            .pipe(drop_duplicates)
            .pipe(drop_wrong_values)
            .pipe(replace_symbol_nan)
            .pipe(change_data_types)
            .pipe(drop_duplicates)
            .pipe(drop_nan, col= "medv") 
    )

    return data

def data_integrity_validation(data: pd.DataFrame,
                              parameters: Dict) -> pd.DataFrame:

    # Run Suite:
    integ_suite = data_integrity()
    suite_result = integ_suite.run(data)
    mlflow.set_experiment('house-pricing')
    mlflow.log_param(f"data integrity validation", str(suite_result.passed()))
    if not suite_result.passed():
        # save report in data/08_reporting
        suite_result.save_as_html('data/08_reporting/data_integrity_check.html')
        logger.error("data integrity not pass validation tests")
        #raise Exception("data integrity not pass validation tests")
    return data

def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.
    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_processing.yml.
    Returns:
        Split data.
    """
    mlflow.set_experiment('house-pricing')
    mlflow.log_param("split random_state", parameters['split']['random_state'])
    mlflow.log_param("split test_size", parameters['split']['test_size'])

    x_features = data[parameters['features']]
    y_target = data[parameters['target_column']]

    x_train, x_test, y_train, y_test = train_test_split(
        x_features,
        y_target,
        test_size=parameters['split']['test_size'],
        random_state=parameters['split']['random_state']
    )

    mlflow.log_param(f"shape train", x_train.shape)
    mlflow.log_param(f"shape test", x_test.shape)

    return x_train, x_test, y_train, y_test
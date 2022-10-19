"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.3
"""

import logging

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn import svm

from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

def train_model(x: pd.DataFrame,
                   y:pd.Series):
    """Trains the classification model.
    Args:
        x_features: Training data
        y_target: target column name
    Returns:
        Trained model.
    """

    y_train = y
    x_train = x
    

    logger.info("training model")

    SVR_pipeline = svm.SVR(kernel='rbf',C=885)

    mlflow.set_experiment('house-pricing')
    mlflow.set_tag("mlflow.runName", SVR_pipeline.__class__.__name__)

    SVR_pipeline.fit(x_train,y_train)

    return SVR_pipeline


def test_transform(x_test: pd.DataFrame, y:pd.Series,
                   train_transformer: Pipeline) -> np.ndarray:
    """ predictions for dataframe"""
    logger.info("transform X_test")
    data = pd.concat([x_test,y],axis=1)
    print(x_test.info())
    x_test_transformed = train_transformer.transform(data)
    mlflow.set_experiment('house-pricing')
    mlflow.log_param(f"shape test_transformed", x_test_transformed.shape)
    x_test_transformed_final = x_test_transformed[:, :-1]
    y_test_validation = x_test_transformed[:, -1]
    print("x_test_transformed")
    print(x_test_transformed)

    print("y_test_transformed")
    print(y_test_validation)
    return x_test_transformed_final,y_test_validation

def predict(model,
            data: np.ndarray) -> np.ndarray:
    """ predictions for dataframe"""
    logger.info("predicting ")

    pred = model.predict(data)
    return pred
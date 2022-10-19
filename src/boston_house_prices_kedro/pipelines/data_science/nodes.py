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

def train_model(x_features: np.ndarray, y_target: np.ndarray):
    """Trains the classification model.
    Args:
        x_features: Training data
        y_target: target column name
    Returns:
        Trained model.
    """

    x_train = x_features
    y_train = y_target

    logger.info("training model")

    SVR_pipeline = svm.SVR(kernel='rbf',C=885)

    mlflow.set_experiment('house-pricing')
    mlflow.set_tag("mlflow.runName", SVR_pipeline.__class__.__name__)

    SVR_pipeline.fit(x_train,y_train)

    return SVR_pipeline


def test_transform(x_test: pd.DataFrame,
                   train_transformer: Pipeline) -> np.ndarray:
    """ predictions for dataframe"""
    logger.info("transform X_test")
    x_test_transformed = train_transformer.transform(x_test)
    mlflow.set_experiment('readmission')
    mlflow.log_param(f"shape test_transformed", x_test_transformed.shape)
    return x_test_transformed


def predict(model,
            data: np.ndarray) -> np.ndarray:
    """ predictions for dataframe"""
    logger.info("predicting ")

    pred = model.predict(data)
    return pred
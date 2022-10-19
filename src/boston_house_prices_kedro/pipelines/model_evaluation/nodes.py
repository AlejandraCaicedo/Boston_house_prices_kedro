"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.18.3
"""

import logging

import numpy as np
import pandas as pd
from typing import Any, Dict
from sklearn.metrics import r2_score
import mlflow

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import model_evaluation

logger = logging.getLogger(__name__)

def evaluate_model(predictions: np.ndarray,
                   test_labels: pd.Series,
                   name: str):
    """
    Evaluate the model by calculating the accuracy score.
    """

    score = r2_score(test_labels, predictions)

    logger.info(f"Model accuracy {name}= {score}")

    mlflow.set_experiment('house-pricing')
    mlflow.log_metric(f"r2_score {name}", score)

    # parse the score to a string with only 4 decimal places
    return f"{score:.4f}"


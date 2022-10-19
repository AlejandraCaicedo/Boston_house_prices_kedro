"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import predict, train_model, test_transform

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            
        ]
    )
    
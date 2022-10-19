"""
This is a boilerplate pipeline 'etl'
generated using Kedro 0.18.3
"""

from .nodes import data_integrity_validation, etl_processing,split_data
from kedro.pipeline import Pipeline, node, pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(func=etl_processing,
             inputs="data_raw",
             name="etl_processing",
             outputs="data_preprocessed"
        ),node(
            func=data_integrity_validation,
             inputs=["data_preprocessed","parameters"],
             outputs="data_integrity_check",
             name="data_integrity_validation",
             
        ),node(
            func=split_data,
                inputs=["data_integrity_check", "parameters"],
                outputs=["x_train",
                         "x_test",
                         "y_train",
                         "y_test"],
                name="split-train_test",
        )
    ])
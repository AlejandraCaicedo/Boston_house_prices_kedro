"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import data_type_splits, fill_nan_mean, fill_nan_mode, first_processing, last_processing, post_processing


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
         node(
                func=first_processing,
                inputs=["x_train", "parameters"],
                outputs=["data_first", "first_processing_pipline"],
                name="first_processing",
            ),
            node(
                func=data_type_splits,
                inputs="parameters",
                outputs=["coll_fill_mode", "coll_fill_mean"],
                name="data_type_split",
            ),
            node(
                func=fill_nan_mode,
                inputs="coll_fill_mode",
                outputs="coll_fill_mode_pipeline",
                name="coll_fill_mode_pipeline_transforms",
            ),
            node(
                func=fill_nan_mean,
                inputs="coll_fill_mean",
                outputs="coll_fill_mean_pipeline",
                name="coll_fill_mean_pipeline_transforms",
            ),
            node(
                func=last_processing,
                inputs=["x_train","y_train",
                        "first_processing_pipline",
                        "coll_fill_mode_pipeline",
                        "coll_fill_mean_pipeline"],
                outputs=["column_transformers_pipeline", "x_train_transformed"],
                name="cols_transforms_pipeline",
            ),
            node(
                func=post_processing,
                inputs="x_train_transformed",
                outputs=["x_train_model_input",
                         "y_train_model_input"],
                name="post_processing",
            )
    ])
